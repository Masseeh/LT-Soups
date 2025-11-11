import torch
import numpy as np
from contextlib import contextmanager
import copy
from .samplers import ClassAwareSampler

class DummyWriter:
    def __init__(self, _writer) -> None:
        self._writer = _writer
    def add_scalar(self, *args, **kwargs):
        if self._writer is not None:
            self._writer.add_scalar(*args, **kwargs)
    def close(self):
        if self._writer is not None:
            self._writer.close()

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers, cb=False):
        super().__init__()

        self.dataset = dataset
        if weights is None: 
            num_samples = len(dataset)
        else:
            num_samples = len([w for w in weights if w > 0])
        
        num_samples = num_samples if num_samples > batch_size else batch_size

        if weights:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, replacement=False, num_samples=num_samples
            )
        elif cb:
            sampler = ClassAwareSampler(dataset)
        else:
            replacement = False if num_samples > batch_size else True
            sampler = torch.utils.data.RandomSampler(self.dataset, replacement=replacement, num_samples=num_samples) 
 

        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size=batch_size, drop_last=False
        )

        self._infinite_iterator = iter(
            torch.utils.data.DataLoader(
                self.dataset,
                num_workers=num_workers,
                pin_memory=True,
                batch_sampler=_InfiniteSampler(batch_sampler),
            )
        )

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError

class ModelSelection:
    def __init__(self, device="cuda", key='mean_acc'):
        self.best_model = None
        self.best_metric = None
        self.best_step = None
        self.key = key
        self.patience = 0
        self.device = device
    
    def update(self, model, metric, step):
        if self.key == 'last':
            self.best_model = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            self.best_metric = metric
            self.best_step = step + 1
            return True

        s_metric = metric[self.key]
        if self.best_metric is None or s_metric > self.best_metric[self.key]:
            self.best_model = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            self.best_metric = metric
            self.best_step = step + 1
            self.patience = 0

            return True
        else:
            self.patience += 1

            return False
    
    def load(self, model):
        if self.best_model is not None:
            model.load_state_dict({k: v.to(self.device) for k, v in self.best_model.items()})
            info = f"(mean {self.best_metric['mean_acc']:.4f} many {self.best_metric['many_acc']:.4f} med {self.best_metric['med_acc']:.4f} few {self.best_metric['few_acc']:.4f})"
            return self.best_step, info

def safe_load(model, state_dict):
    missing_keys, unexpected_keys =  model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0

def my_print(*args, file, **kwargs):
    print(*args, **kwargs, file=file)

# Function to print memory usage
def print_memory_usage(phase):
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
    cached_memory = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert to MB
    print(f"{phase}: Allocated Memory: {allocated_memory:.2f} MB, Cached Memory: {cached_memory:.2f} MB")

def linear_interpolate(frac, start, end):
    return frac * end + (1 - frac) * start

def log_interpolate(frac, start, end):
    gen = np.exp(np.log(start) * (1 - frac) + np.log(end) * frac)
    gen = np.where(gen < end, end, gen)
    return gen

def cosine_interpolate(frac, start, end):
    return start + (end - start) * (1 - np.cos(np.pi * frac)) / 2

def exp_interpolate(frac, start, end, k=10):
    return end + (start - end) * np.exp(-k * frac)

@contextmanager
def set_seed(seed):
    # Save the current random state
    current_state = torch.get_rng_state()
    
    # Set the seed for torch
    torch.manual_seed(seed)
    
    try:
        # Yield control back to the block of code using this context manager
        yield
    finally:
        # Restore the original random state
        torch.set_rng_state(current_state)

def T(w):
    return w.transpose(0, 1)

@torch.no_grad()
def merge_lora_weights(trainer, alpha=1.0):    
    # Merging LORA weights
    for blk_ix, blk in enumerate(trainer.model.image_encoder.blocks):
        
        # Merge MLP weights
        lora_mlps = trainer.tuner.lora_mlp_list[blk_ix]
        mlp_we1 = T(lora_mlps['1'].lora_A.data @ lora_mlps['1'].lora_B.data) * lora_mlps['1'].scaling
        mlp_we2 = T(lora_mlps['2'].lora_A.data @ lora_mlps['2'].lora_B.data) * lora_mlps['2'].scaling

        blk.mlp[0].weight.data += (mlp_we1 * alpha)
        blk.mlp[2].weight.data += (mlp_we2 * alpha)

        # Merge Attention weights
        lora_att = trainer.tuner.lora_list[blk_ix]
        att_weq = T(lora_att['q'].lora_A.data @ lora_att['q'].lora_B.data) * lora_att['q'].scaling
        att_wev = T(lora_att['v'].lora_A.data @ lora_att['v'].lora_B.data) * lora_att['v'].scaling
        att_dim = att_weq.shape[0]
        blk.attn.in_proj_weight[:att_dim].data += (att_weq * alpha)
        blk.attn.in_proj_weight[att_dim*2:].data += (att_wev * alpha)

        # print(f"Block {blk_ix} weights merged.")
    
    print("LORA weights merged.")

@torch.no_grad()
def merge_lora_weights_with_sd(trainer, sd, alpha=1.0, verbose=False):   
    # Merging LORA weights
    if trainer.cfg.backbone.startswith("CLIP-ViT"):
        clip = True
    else:
        clip = False

    for blk_ix, blk in enumerate(trainer.model.image_encoder.blocks):
        
        # Merge MLP weights
        lora_mlp = f"lora_mlp_list.{blk_ix}"
        mlp_we1 = T(sd[f"{lora_mlp}.1.lora_A"] @ sd[f"{lora_mlp}.1.lora_B"]) * (1.0 / sd[f"{lora_mlp}.1.lora_A"].shape[1])
        mlp_we2 = T(sd[f"{lora_mlp}.2.lora_A"] @ sd[f"{lora_mlp}.2.lora_B"]) * (1.0 / sd[f"{lora_mlp}.2.lora_A"].shape[1])

        if clip:
            blk.mlp[0].weight.data += (mlp_we1 * alpha)
            blk.mlp[2].weight.data += (mlp_we2 * alpha)
        else:
            blk.mlp.fc1.weight.data += (mlp_we1 * alpha)
            blk.mlp.fc2.weight.data += (mlp_we2 * alpha)

        # Merge Attention weights
        lora_att = f"lora_list.{blk_ix}"
        att_weq = T(sd[f"{lora_att}.q.lora_A"] @ sd[f"{lora_att}.q.lora_B"]) * (1.0 / sd[f"{lora_att}.q.lora_A"].shape[1])
        att_wev = T(sd[f"{lora_att}.v.lora_A"] @ sd[f"{lora_att}.v.lora_B"]) * (1.0 / sd[f"{lora_att}.v.lora_A"].shape[1])
        att_dim = att_weq.shape[0]

        if clip:
            blk.attn.in_proj_weight[:att_dim].data += (att_weq * alpha)
            blk.attn.in_proj_weight[att_dim*2:].data += (att_wev * alpha)
        else:
            blk.attn.qkv.weight[:att_dim].data += (att_weq * alpha)
            blk.attn.qkv.weight[att_dim*2:].data += (att_wev * alpha)

        if verbose: print(f"Block {blk_ix} weights merged.")