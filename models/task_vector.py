import abc
import torch
import copy

def T(w):
    return w.transpose(0, 1)


def lerp(anchor, task_vectors):
    """Linear interpolation of multiple task vectors."""
    assert len(task_vectors) >= 2
    w0 = anchor
    ws = task_vectors
    offsets = [w - w0 for w in ws]
    wh = sum(offsets) / len(offsets)

    return w0 + wh

class TaskVector(abc.ABC):
    def __init__(
        self, vector, device="cpu"
    ):  
        self.vector = copy.deepcopy(vector)
        self.to(device)
        self.device = device

    def to(self, device):
        with torch.no_grad():
            for key in self.vector:
                self.vector[key] = self.vector[key].to(device)
        
        self.device = device
    
    @torch.no_grad()
    def full_rank(self, num_blocks, alpha=None):
        assert 'lora' in list(self.vector.keys())[0]

        sd = self.vector
        new_vector = {}

        for blk_ix in range(num_blocks):

            lora_mlp = f"lora_mlp_list.{blk_ix}"

            rank = sd[f"{lora_mlp}.1.lora_A"].shape[1]
            if alpha is None:
                alpha = 1.0 / rank
            
            mlp_we1 = T(sd[f"{lora_mlp}.1.lora_A"] @ sd[f"{lora_mlp}.1.lora_B"])
            mlp_we2 = T(sd[f"{lora_mlp}.2.lora_A"] @ sd[f"{lora_mlp}.2.lora_B"])

            new_vector[f"{lora_mlp}.1"] = mlp_we1.data * alpha
            new_vector[f"{lora_mlp}.2"] = mlp_we2.data * alpha

            lora_att = f"lora_list.{blk_ix}"
            att_weq = T(sd[f"{lora_att}.q.lora_A"] @ sd[f"{lora_att}.q.lora_B"])
            att_wev = T(sd[f"{lora_att}.v.lora_A"] @ sd[f"{lora_att}.v.lora_B"])

            new_vector[f"{lora_att}.q"] = att_weq.data * alpha
            new_vector[f"{lora_att}.v"] = att_wev.data * alpha
        
        return self.__class__(vector=new_vector, device=self.device)
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return self.__class__(vector=new_vector, device=self.device)

    def __sub__(self, other):
        """Subtract two task vectors."""
        return self.__add__(-other)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return self.__class__(vector=new_vector, device=self.device)

    def __pow__(self, power):
        """Power of a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] ** power
        return self.__class__(vector=new_vector, device=self.device)

    def __mul__(self, other):
        """Multiply a task vector by a scalar."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = other * self.vector[key]
        return self.__class__(vector=new_vector, device=self.device)

    def __truediv__(self, other):
        """Divide a task vector by a scalar."""
        return self.__mul__(1 / other)

    def dot(self, other):
        """Dot product of two task vectors."""
        with torch.no_grad():
            dot_product = 0.0
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                dot_product += torch.sum(self.vector[key] * other.vector[key])
        return dot_product

    def norm(self):
        """Norm of a task vector."""
        return torch.sqrt(self.dot(self))