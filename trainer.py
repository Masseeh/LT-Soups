import os
import copy
import random
import shutil
from pathlib import Path
import json
import time
import datetime
import numpy as np
from functools import partial
from tqdm import tqdm
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms as tv_v1
import torchvision.transforms.v2 as transforms
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn

from clip import clip
from timm import create_model as timm_create_model
from timm.models.vision_transformer import vit_base_patch16_224, vit_base_patch16_384, vit_large_patch16_224

import datasets
from models import *
from models.clip_text import CLIP_Text

from utils.meter import AverageMeter
from utils.samplers import DownSampler
from utils.losses import *
from utils.evaluator import Evaluator
from utils.templates import ZEROSHOT_TEMPLATES
from utils.util import my_print
from utils.util import ModelSelection, safe_load, merge_lora_weights_with_sd, DummyWriter, InfiniteDataLoader
from models import TaskVector

def load_clip_to_cpu(backbone_name, prec, pretrained=True):
    backbone_name = backbone_name.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu").eval()

    model = clip.build_model(state_dict or model.state_dict(), pretrained=pretrained)

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp32" or prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model

def load_vit_to_cpu(backbone_name, prec, pretrained=True):
    if backbone_name == "IN21K-ViT-B/16":
        model = vit_base_patch16_224(pretrained=pretrained).eval()
    elif backbone_name == "IN1K-ViT-B/16":
        model = timm_create_model('vit_base_patch16_224', pretrained=pretrained)
    elif backbone_name == "IN21K-ViT-B/16@384px":
        model = vit_base_patch16_384(pretrained=pretrained).eval()
    elif backbone_name == "IN21K-ViT-L/16":
        model = vit_large_patch16_224(pretrained=pretrained).eval()
    elif backbone_name == "IN1K-ViT-S/16":
        model = timm_create_model('vit_small_patch16_224', pretrained=pretrained)


    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp16":
        # ViT's default precision is fp32
        model.half()
    
    return model

class Trainer:
    def __init__(self, cfg, logger):

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        self.cfg = cfg

        self.logger = logger
        self._print = partial(my_print, file=self.logger)
        self._writer = None

        self.build_data_loader()
        self.build_model()

        self.test_evaluator = Evaluator(cfg, logger, self.many_idxs, self.med_idxs, self.few_idxs, stout=True)
        self.val_evaluator = Evaluator(cfg, logger, self.many_idxs, self.med_idxs, self.few_idxs, stout=False)

        self.zs_head_dict = copy.deepcopy(self.head.state_dict())
        self.zs_encoder_dict = copy.deepcopy(self.model.image_encoder.state_dict())


    def build_data_loader(self):
        cfg = self.cfg
        root = cfg.root
        resolution = cfg.resolution
        expand = cfg.expand

        if cfg.backbone.startswith("CLIP"):
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        self._print("mean:", mean)
        self._print("std:", std)

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std),
        ])

        transform_plain = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std),
        ])

        _to_dtype = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True)
            ])

        if cfg.tte:
            if cfg.tte_mode == "fivecrop":
                transform_test = transforms.Compose([
                    transforms.Resize(resolution + expand),
                    transforms.FiveCrop(resolution),
                    tv_v1.Lambda(lambda crops: torch.stack([_to_dtype(crop) for crop in crops])),
                    transforms.Normalize(mean, std),
                ])
            elif cfg.tte_mode == "tencrop":
                transform_test = transforms.Compose([
                    transforms.Resize(resolution + expand),
                    transforms.TenCrop(resolution),
                    tv_v1.Lambda(lambda crops: torch.stack([_to_dtype(crop) for crop in crops])),
                    transforms.Normalize(mean, std),
                ])
            elif cfg.tte_mode == "randaug":
                _resize_and_flip = transforms.Compose([
                    transforms.RandomResizedCrop(resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                ])
                transform_test = transforms.Compose([
                    tv_v1.Lambda(lambda image: torch.stack([_resize_and_flip(image) for _ in range(cfg.randaug_times)])),
                    transforms.Normalize(mean, std),
                ])
        else:
            transform_test = transforms.Compose([
                transforms.Resize(resolution * 8 // 7),
                transforms.CenterCrop(resolution),
                tv_v1.Lambda(lambda crop: torch.stack([_to_dtype(crop)])),
                transforms.Normalize(mean, std),
            ])
        
        train_dataset = getattr(datasets, cfg.dataset)(root, split='train', transform=transform_train, cfg=cfg)
        train_init_dataset = getattr(datasets, cfg.dataset)(root, split='train', transform=transform_plain, cfg=cfg)
        train_test_dataset = getattr(datasets, cfg.dataset)(root, split='train', transform=transform_test, cfg=cfg)
        test_dataset = getattr(datasets, cfg.dataset)(root, split='test', transform=transform_test, cfg=cfg)
        val_dataset = getattr(datasets, cfg.dataset)(root, split='val', transform=transform_test, cfg=cfg)

        self.num_classes = train_dataset.num_classes
        self.cls_num_list = train_dataset.cls_num_list
        self.classnames = train_dataset.classnames

        split_cls_num_list = self.cls_num_list

        self.many_idxs = (np.array(split_cls_num_list) > cfg.many_threshold).nonzero()[0]
        self.med_idxs = ((np.array(split_cls_num_list) >= cfg.few_threshold) & (np.array(split_cls_num_list) <= cfg.many_threshold)).nonzero()[0]
        self.few_idxs = (np.array(split_cls_num_list) < cfg.few_threshold).nonzero()[0]

        if cfg.init_head == "1_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=1)
        elif cfg.init_head == "10_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=10)
        elif cfg.init_head == "100_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=100)
        else:
            init_sampler = None

        total_train_points_before = sum(self.cls_num_list)
        weights = None
        cb = False
        if cfg.balanced_sampling:
            if cfg.subsample:
                self._print("Subsampling")

                minority_indx = np.argmin(split_cls_num_list)

                divisor = 10 if cfg.fraction_bf else 1
                min_samples = int(split_cls_num_list[minority_indx] * float(cfg.balance_factor) / divisor)

                # Step 1: Group the indices by class
                class_indices = {label: [] for label in set(train_dataset.labels)}
                cls_num_list = [0] * self.num_classes
                for idx, label in enumerate(train_dataset.labels):
                    class_indices[label].append(idx)

                rng = np.random.default_rng(cfg.sub_sample_seed)

                # Step 2: Randomly pick samples from each class
                selected_indices = []
                for label, indices in class_indices.items():
                    chc = rng.choice(indices, min(min_samples, len(indices)), replace=False)
                    selected_indices.extend(chc.tolist())
                    cls_num_list[label] = len(chc)

                weights = np.zeros(len(train_dataset.labels))
                weights[selected_indices] = 1
                weights = weights.tolist()

                self.cls_num_list = cls_num_list
            else:
                self._print("Balanced sampling with class frequencies")
                cb = True


        self._print("Imbalance ratio:", max(self.cls_num_list) / min(self.cls_num_list))

        self.train_loader = InfiniteDataLoader(train_dataset, weights=weights,
            batch_size=cfg.micro_batch_size, num_workers=cfg.num_workers, cb=cb)

        eval_batch_size = cfg.test_batch_size
        eval_num_workers = cfg.num_workers * 2

        self.train_init_loader = DataLoader(train_init_dataset,
            batch_size=eval_batch_size, sampler=init_sampler, shuffle=False,
            num_workers=eval_num_workers, pin_memory=True)

        self.train_test_loader = DataLoader(train_test_dataset,
            batch_size=eval_batch_size, shuffle=False,
            num_workers=eval_num_workers, pin_memory=True)

        self.test_loader = DataLoader(test_dataset,
            batch_size=eval_batch_size, shuffle=False,
            num_workers=eval_num_workers, pin_memory=True)

        self.val_loader = DataLoader(val_dataset,
            batch_size=eval_batch_size, shuffle=False,
            num_workers=eval_num_workers, pin_memory=True)
        
        self.accum_step = 1

        cls_num_list = self.cls_num_list
        
        np_cls_num_list = np.array(cls_num_list)
        self._print(f"{len(self.many_idxs)} Many-shot classes with {np_cls_num_list[self.many_idxs].sum() if len(self.many_idxs) > 0 else 0} samples")
        self._print(f"{len(self.med_idxs)} Medium-shot classes with {np_cls_num_list[self.med_idxs].sum() if len(self.med_idxs) > 0 else 0} samples")
        self._print(f"{len(self.few_idxs)} Few-shot classes with {np_cls_num_list[self.few_idxs].sum() if len(self.few_idxs) > 0 else 0} samples")

        self._print(f"Total training points: {sum(cls_num_list)} -> {sum(cls_num_list) / total_train_points_before * 100:.2f}% of original dataset")
        self.one_epoch = (sum(cls_num_list) // cfg.micro_batch_size) + 1
        self._print("Number of steps per epoch:", self.one_epoch)
        self.n_steps = self.one_epoch * cfg.num_epochs
        self.n_steps = max(self.n_steps, 100)
        self._print("Number of total steps: {}".format(self.n_steps))
        self._print("Evaluation steps: {}".format([step for step in range(self.n_steps) if ((step + 1) % (self.n_steps // cfg.eval_freq)) == 0]))

    def build_model(self):
        cfg = self.cfg
        num_classes = self.num_classes

        self._print("Building model")
        if cfg.random_init:
            self._print("Random initialization")

        if cfg.backbone.startswith("CLIP"):
            self._print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec, pretrained=not cfg.random_init)
            self.model = PeftModelFromCLIP(cfg, self.logger, clip_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

            self.text_encoder = CLIP_Text(clip_model)

        elif cfg.backbone.startswith("IN21K-ViT"):
            self._print(f"Loading ViT (backbone: {cfg.backbone})")
            vit_model = load_vit_to_cpu(cfg.backbone, cfg.prec, pretrained=not cfg.random_init)
            self.model = PeftModelFromViT(cfg, self.logger, vit_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

        elif cfg.backbone.startswith("IN1K-ViT"):
            self._print(f"Loading ViT (backbone: {cfg.backbone})")
            vit_model = load_vit_to_cpu(cfg.backbone, cfg.prec, pretrained=not cfg.random_init)
            self.model = PeftModelFromViT(cfg, self.logger, vit_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

        if cfg.zero_shot:
            if cfg.init_head == "text_feat":
                self.init_head_text_feat()
            elif cfg.init_head in ["class_mean", "1_shot", "10_shot", "100_shot"]:
                self.init_head_class_mean()

        if not (cfg.zero_shot or cfg.test_train or cfg.test_only or cfg.val_only):
            if cfg.init_head == "text_feat":
                self.init_head_text_feat()
            elif cfg.init_head in ["class_mean", "1_shot", "10_shot", "100_shot"]:
                self.init_head_class_mean()
            elif cfg.init_head == "linear_probe":
                self.init_head_linear_probe()
            else:
                self._print("No initialization with head")

            self.build_trainbles()
            self.build_criterion()
            
            torch.cuda.empty_cache()
        
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1 and cfg.gpu is None:
            self._print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_lr_scheduler(self, n_steps, optimizer):
        cfg = self.cfg
        wu_steps = 0

        if cfg.scheduler == "cosine":
            if cfg.warmup:
                wu_steps = int(max(20, 0.01 * n_steps))

            warmup = np.interp(np.arange(1+wu_steps), [0, wu_steps], [1e-6, 1])
            max_mult = 0.1

            ni = n_steps - wu_steps
            xx = np.arange(ni)/ni
            cosine = max_mult + (1 - max_mult) * (np.cos(np.pi * xx) + 1) / 2
            lr_schedule = np.concatenate([warmup, cosine])
            lr_lambda = lambda x: lr_schedule[x]
            
            self._print(f"warmup_steps={wu_steps} with max lr: {cfg.lr}, min lr: {cfg.lr * max_mult}")
        
        else:
            lr_lambda = lambda _: 1
        
        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return sched

    def build_optimizer(self, params_to_optimize):
        cfg = self.cfg

        if cfg.adam:
            optim = torch.optim.AdamW(params_to_optimize,
                                        lr=cfg.lr, weight_decay=cfg.wd, betas=(cfg.momentum, 0.999))
        else:
            optim = torch.optim.SGD(params_to_optimize,
                                        lr=cfg.lr, weight_decay=cfg.wd, momentum=cfg.momentum)
        
        return optim

    def build_trainbles(self):
        cfg = self.cfg

        self._print(f"Turning off gradients in the model")
        for param in self.model.parameters():
            param.requires_grad_(False)
        
        if cfg.full_tuning and cfg.partial is None:
            self._print("Turning on gradients in the image encoder")
            for param in self.model.image_encoder.parameters():
                param.requires_grad_(True)
            
            params_to_optimize = [{"params": [p for p in self.model.image_encoder.parameters() if p.requires_grad == True]}]

        else:
            self._print("Turning on gradients in the tuner")
            for param_name, param in self.tuner.named_parameters():
                param.requires_grad_(True)
            
            params_to_optimize = [{"params": [p for p in self.tuner.parameters() if p.requires_grad == True]}]

        if cfg.head_tuning:
            self._print(f"Turning on gradients in the head")
            for param in self.head.optim_params():
                param.requires_grad_(True)
            # for param in self.head.parameters():
            #     param.requires_grad_(True)
            
            params_to_optimize.extend([{"params": [p for p in self.head.parameters() if p.requires_grad == True], "lr": cfg.lr * cfg.lrh_factor}])

        # print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad == True)
        self._print(f"Total params: {total_params}")
        self._print(f"Tuned params: {trainable_params} -> {trainable_params / total_params * 100:.2f}% , rank {self.tuner.adapter_dim}")

        self.optim = self.build_optimizer(params_to_optimize)
        self.sched = self.build_lr_scheduler(self.n_steps, self.optim)
        self.scaler = GradScaler(enabled=cfg.prec == "amp")

    def build_criterion(self):
        cfg = self.cfg

        cls_num_list = torch.Tensor(self.cls_num_list).to(self.device)

        if cfg.loss_type == "CE":
            self.criterion = nn.CrossEntropyLoss()
        elif cfg.loss_type == "WCE":
            self.criterion = nn.CrossEntropyLoss(weight=cls_num_list)
        elif cfg.loss_type == "Focal": # https://arxiv.org/abs/1708.02002
            self.criterion = FocalLoss()
        elif cfg.loss_type == "LDAM": # https://arxiv.org/abs/1906.07413
            self.criterion = LDAMLoss(cls_num_list=cls_num_list, s=cfg.scale)
        elif cfg.loss_type == "CB": # https://arxiv.org/abs/1901.05555
            self.criterion = ClassBalancedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "GRW": # https://arxiv.org/abs/2103.16370
            self.criterion = GeneralizedReweightLoss(cls_num_list=cls_num_list, exp_scale=cfg.exp_scale)
        elif cfg.loss_type == "BS": # https://arxiv.org/abs/2007.10740
            self.criterion = BalancedSoftmaxLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LA": # https://arxiv.org/abs/2007.07314
            self.criterion = LogitAdjustedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LADE": # https://arxiv.org/abs/2012.00321
            self.criterion = LADELoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "VS":
            self.criterion = VSLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "BCE":
            self.criterion = SigLIPLoss(num_classes=self.num_classes)
        
    def get_tokenized_prompts(self, classnames, template):
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # self._print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        return prompts

    @torch.no_grad()
    def init_head_text_feat(self):
        cfg = self.cfg
        classnames = self.classnames

        assert cfg.backbone.startswith("CLIP"), "Text feature initialization is only available for CLIP"

        self.text_encoder.to(self.device)

        self._print("Initialize head with text features")
        if cfg.prompt == "ensemble":
            all_text_features = []
            for template in tqdm(ZEROSHOT_TEMPLATES['imagenet']):
                prompts = self.get_tokenized_prompts(classnames, template)
                text_features = self.text_encoder.encode_text(prompts)
                text_features = F.normalize(text_features, dim=-1)
                all_text_features.append(text_features)
            all_text_features = torch.stack(all_text_features)
            text_features = all_text_features.mean(dim=0)
        elif cfg.prompt == "descriptor":
            with open("utils/descriptors_imagenet.json") as f:
                descriptors = json.load(f)
            template = "{}"
            all_class_features = []
            for cn in tqdm(classnames):
                prompts = self.get_tokenized_prompts(descriptors[cn], template)
                text_features = self.text_encoder.encode_text(prompts)
                text_features = F.normalize(text_features, dim=-1)
                all_class_features.append(text_features.mean(dim=0))
            text_features = torch.stack(all_class_features)
        elif cfg.prompt == "classname":
            template = "{}"
            prompts = self.get_tokenized_prompts(classnames, template)
            text_features = self.text_encoder.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)
        elif cfg.prompt == "default":
            template = "a photo of a {}."
            prompts = self.get_tokenized_prompts(classnames, template)
            text_features = self.text_encoder.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)
        if cfg.backbone.startswith("CLIP-ViT") and not cfg.use_proj:
            text_features = text_features @ self.model.image_encoder.proj.t()
            text_features = F.normalize(text_features, dim=-1)

        self.head.apply_weight(text_features)

        del self.text_encoder

    @torch.no_grad()
    def init_head_class_mean(self):
        self._print("Initialize head with class means")

        possible_ckp = os.path.join("class_means", self.cfg.dataset.lower() + "_" + self.cfg.backbone.lower().replace("/", "").replace("-","_") + ".pth")
        if os.path.exists(possible_ckp):
            self._print(f"Loading class means from {possible_ckp}")
            class_means = torch.load(possible_ckp, map_location=self.device, weights_only=True)
            self.head.apply_weight(class_means['class_means'])
            return

        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            _, feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        sorted_index = all_labels.argsort()
        all_features = all_features[sorted_index]
        all_labels = all_labels[sorted_index]

        unique_labels, label_counts = torch.unique(all_labels, return_counts=True)

        class_means = [None] * self.num_classes
        idx = 0
        for i, cnt in zip(unique_labels, label_counts):
            class_means[i] = all_features[idx: idx+cnt].mean(dim=0, keepdim=True)
            idx += cnt
        class_means = torch.cat(class_means, dim=0)
        class_means = F.normalize(class_means, dim=-1)

        if self.cfg.save_head_class_mean:
            ckp = {"class_means": class_means.cpu()}
            torch.save(ckp, os.path.join("class_means", self.cfg.dataset.lower() + "_" + self.cfg.backbone.lower().replace("/", "").replace("-","_") + ".pth"))
    
        self.head.apply_weight(class_means)

    @torch.no_grad()
    def init_head_linear_probe(self):
        self._print("Initialize head with linear probing")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()

        clf = LogisticRegression(solver="lbfgs", max_iter=100, penalty="l2", class_weight="balanced").fit(all_features, all_labels)
        class_weights = torch.from_numpy(clf.coef_).to(all_features.dtype).to(self.device)
        class_weights = F.normalize(class_weights, dim=-1)

        self.head.apply_weight(class_weights)

    def train(self):
        cfg = self.cfg
        data_loader = self.train_loader

        # Initialize summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)

        if cfg.tensorboard:
            self._print(f"Initialize tensorboard (log_dir={writer_dir})")
            self._writer = DummyWriter(SummaryWriter(log_dir=writer_dir))
        else:
            self._print("Tensorboard is not used")
            self._writer = DummyWriter(None)

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)
        cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]

        # Remember the starting time (for computing the elapsed time)
        time_start = time.time()
        
        self.model.train()

        best_mean = ModelSelection(device=self.device, key="mean_acc")

        if cfg.ema > 0:
            self._print("Use EMA with decay:", cfg.ema)
            averaged_model = AveragedModel(self.model, avg_fn=get_ema_avg_fn(cfg.ema), device=self.device)
            averaged_model.eval()
            best_avg = ModelSelection(device=self.device, key="mean_acc")

        val_stats = []

        data_loader = iter(data_loader)
        
        for step in range(self.n_steps):
            step_start_time = time.time()

            batch = next(data_loader)

            image = batch[0]
            label = batch[1]
            image = image.to(self.device)
            label = label.to(self.device)

            with autocast(device_type='cuda', enabled=cfg.prec == "amp"):
                output, _ = self.model(image, return_feature=False)
                loss = self.criterion(output, label)

                loss_micro = loss / self.accum_step

                self.scaler.scale(loss_micro).backward()

                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad()

            with torch.no_grad():
                pred = output.argmax(dim=1)
                correct = pred.eq(label).float()
                acc = correct.mean().mul_(100.0)

            current_lr = self.optim.param_groups[0]["lr"]
            loss_meter.update(loss.item())

            acc_meter.update(acc.item())
            batch_time.update(time.time() - step_start_time)

            for _c, _y in zip(correct, label):
                cls_meters[_y].update(_c.mul_(100.0).item(), n=1)
            cls_accs = [cls_meters[i].avg for i in range(self.num_classes)]

            mean_acc = np.mean(np.array(cls_accs))
            many_acc = np.mean(np.array(cls_accs)[self.many_idxs]) if len(self.many_idxs) > 0 else -1
            med_acc = np.mean(np.array(cls_accs)[self.med_idxs]) if len(self.med_idxs) > 0 else -1
            few_acc = np.mean(np.array(cls_accs)[self.few_idxs]) if len(self.few_idxs) > 0 else -1

            if cfg.ema > 0:
                averaged_model.update_parameters(self.model)

            meet_freq = (step + 1) % cfg.print_freq == 0
            if meet_freq:
                info = []
                info += [f"step [{step + 1}/{self.n_steps}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                info += [f"(mean {mean_acc:.4f} many {many_acc:.4f} med {med_acc:.4f} few {few_acc:.4f})"]
                info += [f"lr {current_lr:.4e}"]
                self._print(" ".join(info))

            meet_eval_freq = (step + 1) % (self.n_steps // cfg.eval_freq) == 0
            if meet_eval_freq and cfg.eval_on_val:
                eval_results = self.eval("val")

                eval_info = [f"validation @ step {step + 1}:"]
                eval_info += [f"(mean {eval_results['mean_acc']:.4f} many {eval_results['many_acc']:.4f} med {eval_results['med_acc']:.4f} few {eval_results['few_acc']:.4f})"]
                self._print(" ".join(eval_info))

                tb_step = (step + 1) // self.one_epoch

                self._writer.add_scalar("val/mean_acc", eval_results['mean_acc'], tb_step)
                self._writer.add_scalar("val/many_acc", eval_results['many_acc'], tb_step)
                self._writer.add_scalar("val/med_acc", eval_results['med_acc'], tb_step)
                self._writer.add_scalar("val/few_acc", eval_results['few_acc'], tb_step)

                val_stats.append(eval_info)

                ### BEST MODEL
                updated = best_mean.update(self.model, eval_results, step)
                if cfg.save_model and updated:
                    self.save_model(cfg.output_dir, image_encoder=cfg.full_tuning, post_fix=best_mean.key)

                    if not cfg.skip_test:
                        test_results = self.eval("test")
                        test_info = [f"test @ step {step + 1}:"]
                        test_info += [f"(mean {test_results['mean_acc']:.4f} many {test_results['many_acc']:.4f} med {test_results['med_acc']:.4f} few {test_results['few_acc']:.4f})"]
                        self._print(" ".join(test_info))
                        self.save_results_json(cfg.output_dir, test_results, post_fix=best_mean.key)

                # AVG EVAL
                ##########################
                if cfg.ema > 0:
                    eval_results = self.eval("val", ema_model=averaged_model)
                    eval_info = [f"EMA validation @ step {step + 1}:"]
                    eval_info += [f"(mean {eval_results['mean_acc']:.4f} many {eval_results['many_acc']:.4f} med {eval_results['med_acc']:.4f} few {eval_results['few_acc']:.4f})"]
                    self._print(" ".join(eval_info))

                    val_stats.append(eval_info)

                    self._writer.add_scalar("val/ema_mean_acc", eval_results['mean_acc'], tb_step)
                    self._writer.add_scalar("val/ema_many_acc", eval_results['many_acc'], tb_step)
                    self._writer.add_scalar("val/ema_med_acc", eval_results['med_acc'], tb_step)
                    self._writer.add_scalar("val/ema_few_acc", eval_results['few_acc'], tb_step)

                    updated = best_avg.update(averaged_model.module, eval_results, step)
                    if cfg.save_model and updated:
                        self.save_model(cfg.output_dir, image_encoder=cfg.full_tuning, ema_model=averaged_model.module, post_fix="ema_" + best_mean.key)

                        if not cfg.skip_test:
                            test_results = self.eval("test")
                            test_info = [f"EMA test @ step {step + 1}:"]
                            test_info += [f"(mean {test_results['mean_acc']:.4f} many {test_results['many_acc']:.4f} med {test_results['med_acc']:.4f} few {test_results['few_acc']:.4f})"]
                            self._print(" ".join(test_info))
                            self.save_results_json(cfg.output_dir, test_results, post_fix="ema_" + best_mean.key)


                if cfg.early_stop > 0:
                    if best_mean.patience > 0:
                        self._print(f"patience {best_mean.patience}/{cfg.early_stop}")

                    if best_mean.patience >= cfg.early_stop:
                        self._print(f"Early stopping at iteration {step + 1}")
                        break
            
            self._writer.add_scalar("train/lr", current_lr, step)
            self._writer.add_scalar("train/loss.val", loss_meter.val, step)
            self._writer.add_scalar("train/loss.avg", loss_meter.avg, step)
            self._writer.add_scalar("train/acc.val", acc_meter.val, step)
            self._writer.add_scalar("train/acc.avg", acc_meter.avg, step)
            self._writer.add_scalar("train/mean_acc", mean_acc, step)
            self._writer.add_scalar("train/many_acc", many_acc, step)
            self._writer.add_scalar("train/med_acc", med_acc, step)
            self._writer.add_scalar("train/few_acc", few_acc, step)
            
            
            self.sched.step()
            torch.cuda.empty_cache()

        self._print("Finish training")
        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        self._print(f"Time elapsed: {elapsed}")

        for val_stat in val_stats:
            self._print(f"stats: {val_stat}")

        if cfg.eval_on_val:
            
            ### BEST MODEL
            best_step, max_acc = best_mean.load(self.model)
            self._print(f"Best {best_mean.key} at round {best_step}: {max_acc}")
            self._print("Best model loaded")
            self.test("test", post_fix=best_mean.key)

            if cfg.ema > 0:
                best_step, max_acc = best_avg.load(self.model)
                self._print(f"Best EMA at round {best_step}: {max_acc}")
                self._print("Best model loaded")
                self.test("test", post_fix="ema_" + best_avg.key)

        else:
            if cfg.save_model:
                self.save_model(cfg.output_dir, image_encoder=cfg.full_tuning)

            self.test("test")

        # Close writer
        self._writer.close()

    def train_soups(self):
        cfg = self.cfg
        data_loader = self.train_loader

        # Initialize summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)

        if cfg.tensorboard:
            self._print(f"Initialize tensorboard (log_dir={writer_dir})")
            self._writer = DummyWriter(SummaryWriter(log_dir=writer_dir))
        else:
            self._print("Tensorboard is not used")
            self._writer = DummyWriter(None)

        tr_ds = data_loader.dataset
        class_indices = {label: [] for label in set(tr_ds.labels)}
        for idx, label in enumerate(tr_ds.labels):
            class_indices[label].append(idx)
        for label in class_indices.keys():
            class_indices[label] = np.array(class_indices[label])

        zs_head_vector = TaskVector(vector=self.zs_head_dict, device=self.device)
        zs_encoder_vector = TaskVector(vector=self.zs_encoder_dict, device=self.device)

        last_out = ModelSelection(key="last")

        meta_bsz = cfg.meta_bsz
        val_stats = []

        if meta_bsz == 1:
            branch_merge_fn = lambda anchor, tk: tk[-1]
        else:
            branch_merge_fn = lerp
            
        merge_fn = lambda w0, w1: w0 * (1 - cfg.merge_lm) + w1 * cfg.merge_lm

        convert = lambda i: 2**i if i >= 0 else -1
        im_factors = [convert(i) for i in cfg.factors]
        factor_num = len(im_factors)
        self._print(f"Total factors: {im_factors} = {factor_num}")

        sub_head_vectors = []
        sub_encoder_vectors = []

        for iteration in range(factor_num):
            running_head_vectors = []
            running_encoder_vectors = []
            
            minority_num, cls_num_list = self.sample_balance_factor(balance_factor=int(im_factors[iteration]))

            for branch in range(meta_bsz):

                self._print(f"Re-init")
                safe_load(self.head, zs_head_vector.vector)
                safe_load(self.model.image_encoder, zs_encoder_vector.vector)

                # zero-grad the parameters
                for p in self.model.parameters():
                    if p.requires_grad:
                        p.grad = torch.zeros_like(p.data)

                self._print(f"branch {branch+1}/{meta_bsz} of factor {im_factors[iteration]}")

                loader = self.sample_loader(iteration, cls_num_list, class_indices, tr_ds, minority_num)

                opt = self.build_optimizer([p for p in self.model.parameters() if p.requires_grad == True])
                scheduler = self.build_lr_scheduler(self.n_steps, opt)
                scaler = GradScaler(enabled=cfg.prec == "amp")

                self.fast_adapt(self.model, loader, self.n_steps, scaler, opt, scheduler)

                running_head_vectors.append(TaskVector(vector=self.head.state_dict(), device=self.device))
                running_encoder_vectors.append(TaskVector(vector=self.model.image_encoder.state_dict(), device=self.device))

            head_anchor = zs_head_vector
            encoder_anchor = zs_encoder_vector

            # MERGE RESAMPLES
            stage_head_vector = branch_merge_fn(head_anchor, running_head_vectors)
            stage_encoder_vector = branch_merge_fn(encoder_anchor, running_encoder_vectors)

            running_head_vectors = []
            running_encoder_vectors = []

            sub_head_vectors.append(stage_head_vector)
            sub_encoder_vectors.append(stage_encoder_vector)

            eval_head = merge_fn(zs_head_vector, sub_head_vectors[0])
            for s_vector in sub_head_vectors[1:]:
                eval_head = merge_fn(eval_head, s_vector)

            eval_encoder = merge_fn(zs_encoder_vector, sub_encoder_vectors[0])    
            for s_vector in sub_encoder_vectors[1:]:
                eval_encoder = merge_fn(eval_encoder, s_vector)

            safe_load(self.head, eval_head.vector)  
            safe_load(self.model.image_encoder, eval_encoder.vector)

            merge_eval_results = self.eval("val")
            merge_eval_info = [f"validation @ factor {im_factors[iteration]}:"]
            merge_eval_info += [f"(mean {merge_eval_results['mean_acc']:.4f} many {merge_eval_results['many_acc']:.4f} med {merge_eval_results['med_acc']:.4f} few {merge_eval_results['few_acc']:.4f})"]
            self._print(" ".join(merge_eval_info))
            
            val_stats.append(merge_eval_info)

            self._writer.add_scalar("val/mean_acc", merge_eval_results['mean_acc'], iteration)
            self._writer.add_scalar("val/many_acc", merge_eval_results['many_acc'], iteration)
            self._writer.add_scalar("val/med_acc", merge_eval_results['med_acc'], iteration)
            self._writer.add_scalar("val/few_acc", merge_eval_results['few_acc'], iteration)

            if cfg.save_every_factors:
                self.save_model(cfg.output_dir, image_encoder=True, post_fix=str(cfg.factors[:iteration+1]).replace(" ",""))

            updated = last_out.update(self.model, merge_eval_results, iteration)
            if cfg.save_model and updated:
                self.save_model(cfg.output_dir, image_encoder=True, post_fix=last_out.key)      

                if not cfg.skip_test:
                    test_results = self.eval("test")
                    test_info = [f"test @ step {im_factors[iteration]}:"]
                    test_info += [f"(mean {test_results['mean_acc']:.4f} many {test_results['many_acc']:.4f} med {test_results['med_acc']:.4f} few {test_results['few_acc']:.4f})"]
                    self._print(" ".join(test_info))
                    self.save_results_json(cfg.output_dir, test_results, post_fix=last_out.key)


        for idx, val_stat in enumerate(val_stats):
            self._print(f"Round {idx} stats: {val_stat}")

        best_step, max_acc = last_out.load(self.model)
        self._print(f"Best {last_out.key} at round {best_step}: {max_acc}")
        self._print("Best model loaded")
        self.test("test", post_fix=last_out.key)

        # Close writer
        self._writer.close()

    def sample_balance_factor(self, balance_factor):
        cfg = self.cfg

        ## BALANCE FACTOR
        minority_indx = np.argmin(self.cls_num_list)
        if balance_factor != -1:
            minority_num = int(self.cls_num_list[minority_indx] * balance_factor)
            cls_num_list = [min(cnl, minority_num) for cnl in self.cls_num_list]
        else:
            # use full training set
            minority_num = int(self.cls_num_list[minority_indx])
            cls_num_list = [cnl for cnl in self.cls_num_list]
        

        if balance_factor == 1:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.ls)
        else:
            if cfg.loss_type == "LA":
                self.criterion = LogitAdjustedLoss(cls_num_list=torch.Tensor(cls_num_list).to(self.device))
            elif cfg.loss_type == "GRW":
                self.criterion = GeneralizedReweightLoss(cls_num_list=torch.Tensor(cls_num_list).to(self.device))
            elif cfg.loss_type == "CE":
                self.criterion = nn.CrossEntropyLoss()

        self._print("Loss function:", self.criterion)
        self._print("Imbalance ratio:", max(cls_num_list) / min(cls_num_list))

        self._print(f"Total training points: {sum(cls_num_list)} -> {(sum(cls_num_list) / sum(self.cls_num_list)) * 100:.4f}%")
        self.one_epoch = (sum(cls_num_list) // cfg.micro_batch_size) + 1
        self._print("Number of steps per epoch:", self.one_epoch)
        self.n_steps = self.one_epoch * cfg.num_epochs
        self.n_steps = max(self.n_steps, 100)
        self._print("Number of total steps: {}".format(self.n_steps))
        self._print("Evaluation steps: {}".format([step for step in range(self.n_steps) if ((step + 1) % (self.n_steps // cfg.eval_freq)) == 0]))

        return minority_num, cls_num_list

    def sample_loader(self, seed, cls_num_list, class_indices, train_dataset, minority_num):
        cfg = self.cfg

        rng = np.random.default_rng(seed)

        selected_indices = []
        for key, indices in class_indices.items():
            chc = rng.choice(indices, min(minority_num, len(indices)), replace=False)
            selected_indices.extend(chc.tolist())

        weights = np.zeros(len(train_dataset.labels))
        weights[selected_indices] = 1

        data_loader = InfiniteDataLoader(train_dataset, weights=weights.tolist(),
            batch_size=cfg.micro_batch_size, num_workers=cfg.num_workers)

        loader = iter(data_loader)

        self._print("Sample a new loader")

        return loader

    def fast_adapt(self, model, loader, n_steps, scaler, optim, sched):
        cfg = self.cfg

        task_start_time = time.time()

        model.train()

        best_in = ModelSelection(key=f"mean_acc")

        if cfg.ema > 0:
            self._print("Use EMA with decay:", cfg.ema)
            averaged_model = AveragedModel(self.model, avg_fn=get_ema_avg_fn(cfg.ema), device=self.device)
            averaged_model.eval()
            best_avg = ModelSelection(device=self.device, key="mean_acc")

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)
        cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]

        for step in range(n_steps):
            step_start_time = time.time()

            batch = next(loader)

            image = batch[0]
            label = batch[1]
            image = image.to(self.device)
            label = label.to(self.device)

            with autocast(device_type='cuda', enabled=cfg.prec == "amp"):

                output, _ = model(image, return_feature=False)
                loss = self.criterion(output, label)

                loss_micro = loss / self.accum_step
                
                scaler.scale(loss_micro).backward()
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

            with torch.no_grad():
                pred = output.argmax(dim=1)
                correct = pred.eq(label).float()
                acc = correct.mean().mul_(100.0)

            current_lr = optim.param_groups[0]["lr"]

            loss_meter.update(loss.item())

            acc_meter.update(acc.item())
            batch_time.update(time.time() - step_start_time)

            if cfg.ema > 0:
                averaged_model.update_parameters(model)

            for _c, _y in zip(correct, label):
                cls_meters[_y].update(_c.mul_(100.0).item(), n=1)
            cls_accs = [cls_meters[i].avg for i in range(self.num_classes)]

            mean_acc = np.mean(np.array(cls_accs))
            many_acc = np.mean(np.array(cls_accs)[self.many_idxs]) if len(self.many_idxs) > 0 else -1
            med_acc = np.mean(np.array(cls_accs)[self.med_idxs]) if len(self.med_idxs) > 0 else -1
            few_acc = np.mean(np.array(cls_accs)[self.few_idxs]) if len(self.few_idxs) > 0 else -1

            meet_freq = (step + 1) % cfg.print_freq == 0
            if meet_freq:
                info = []
                info += [f"step [{step + 1}/{n_steps}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]

                info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                info += [f"(mean {mean_acc:.4f} many {many_acc:.4f} med {med_acc:.4f} few {few_acc:.4f})"]
                info += [f"lr {current_lr:.4e}"]
                self._print(" ".join(info))

            meet_eval_freq = (step + 1) % (self.n_steps // cfg.eval_freq) == 0
            if meet_eval_freq and cfg.eval_on_val:
                if cfg.ema > 0:
                    eval_results = self.eval("val", ema_model=averaged_model)
                    eval_info = [f"EMA validation @ step {step + 1}:"]
                    eval_info += [f"(mean {eval_results['mean_acc']:.4f} many {eval_results['many_acc']:.4f} med {eval_results['med_acc']:.4f} few {eval_results['few_acc']:.4f})"]
                    self._print(" ".join(eval_info))
                    best_avg.update(averaged_model.module, eval_results, step)
                else:
                    eval_results = self.eval("val", ema_model=None)
                    eval_info = [f"validation @ step {step + 1}:"]
                    eval_info += [f"(mean {eval_results['mean_acc']:.4f} many {eval_results['many_acc']:.4f} med {eval_results['med_acc']:.4f} few {eval_results['few_acc']:.4f})"]
                    self._print(" ".join(eval_info))
                    best_in.update(model, eval_results, step)

                if cfg.early_stop > 0:
                    if best_in.patience > 0:
                        self._print(f"patience {best_in.patience}/{cfg.early_stop}")

                    if best_in.patience >= cfg.early_stop:
                        self._print(f"Early stopping at iteration {step + 1}")
                        break
            
            sched.step()
            torch.cuda.empty_cache()

        self._print(f"Finish training")
        elapsed = round(time.time() - task_start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        self._print(f"Time elapsed: {elapsed}")

        if cfg.eval_on_val:
            if cfg.ema > 0:
                best_step, max_acc = best_avg.load(model)
                self._print(f"Best EMA {best_avg.key} at step {best_step}: {max_acc}")
                self._print("Best model loaded")
            else:
                best_step, max_acc = best_in.load(model)
                self._print(f"Best {best_in.key} at step {best_step}: {max_acc}")
                self._print("Best model loaded")

    @torch.no_grad()
    def test(self, mode="test", post_fix="", ema_model=None):

        if ema_model is None:
            model = self.model
        else:
            model = ema_model

        model.eval()

        self.test_evaluator.reset()

        if mode == "train":
            self._print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "val":
            self._print(f"Evaluate on the validation set")
            data_loader = self.val_loader
        elif mode == "test":
            self._print(f"Evaluate on the test set")
            data_loader = self.test_loader
        
        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            with autocast(device_type='cuda', enabled=self.cfg.prec == "amp"):
                if _ncrops <= 5:
                    output = model(image)[0]
                    output = output.view(_bsz, _ncrops, -1).mean(dim=1)
                else:
                    # CUDA out of memory
                    output = []
                    image = image.view(_bsz, _ncrops, _c, _h, _w)
                    for k in range(_ncrops):
                        output.append(model(image[:, k])[0])
                    output = torch.stack(output).mean(dim=0)
            
            self.test_evaluator.process(output, label)

        results = self.test_evaluator.evaluate()

        if ema_model is not None:
            model.train()

        for k, v in results.items():
            tag = f"test/{k}"
            if self._writer is not None:
                self._writer.add_scalar(tag, v)

        if self.cfg.save_results:
            self.save_results_json(self.cfg.output_dir, results, post_fix=post_fix)

        return results

    def save_results_json(self, output_dir, results, post_fix=""):
        if post_fix != "":
            post_fix = "_" + post_fix

        timestamp = self.logger.file.name.split("/")[-1].lstrip("log").rstrip(".txt")
        with open(os.path.join(output_dir, f"results{timestamp}{post_fix}.json"), "w") as f:
            json.dump(results, f, indent=4)

    @torch.no_grad()
    def eval(self, mode="val", ema_model=None):
        
        if ema_model is None:
            model = self.model
        else:
            model = ema_model

        model.eval()

        self.val_evaluator.reset()

        if mode == "train":
            data_loader = self.train_test_loader
        elif mode == "val":
            data_loader = self.val_loader
        elif mode == "test":
            data_loader = self.test_loader

        for batch in data_loader:
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            with autocast(device_type='cuda', enabled=self.cfg.prec == "amp"):
                if _ncrops <= 5:
                    output = model(image)[0]
                    output = output.view(_bsz, _ncrops, -1).mean(dim=1)
                else:
                    # CUDA out of memory
                    output = []
                    image = image.view(_bsz, _ncrops, _c, _h, _w)
                    for k in range(_ncrops):
                        output.append(model(image[:, k])[0])
                    output = torch.stack(output).mean(dim=0)
            
            self.val_evaluator.process(output, label)

        results = self.val_evaluator.evaluate()

        if ema_model is not None:
            model.train()

        return results

    def save_model(self, directory, image_encoder=False, skip_head=False, post_fix="", ema_model=None):

        if ema_model is None:
            model = self.model
        else:
            model = ema_model

        tuner_dict = copy.deepcopy(model.tuner.state_dict())
        head_dict = copy.deepcopy(model.head.state_dict())
        image_encoder_dict = copy.deepcopy(model.image_encoder.state_dict())

        if not image_encoder:
            checkpoint = {
                "tuner": tuner_dict
            }
            self._print("Adding tuner weights to checkpoint")
        else:
            checkpoint = {
                "image_encoder": image_encoder_dict
            }
            self._print("Adding image_encoder weights to checkpoint")

        if not skip_head:
            checkpoint["head"] = head_dict
            self._print("Adding head weights to checkpoint")

        keys = list(checkpoint.keys())
        # remove 'module.' in state_dict's keys
        for key in keys:
            state_dict = checkpoint[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            checkpoint[key] = new_state_dict

        # save model
        if post_fix != "":
            post_fix = "_" + post_fix

        save_path = os.path.join(directory, f"checkpoint{post_fix}.pth.tar")
        torch.save(checkpoint, save_path)
        self._print(f"Checkpoint saved to {save_path}")

    def load_model(self, directory, tuner=True, skip_head=False, device=None, post_fix=""):
        if post_fix != "":
            post_fix = "_" + post_fix

        load_path = os.path.join(directory, f"checkpoint{post_fix}.pth.tar")
        loaded = False

        if not os.path.exists(load_path):
            raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

        checkpoint = torch.load(load_path, map_location=self.device if device==None else device, weights_only=True)
        
        if 'image_encoder' in checkpoint:
            image_encoder_dict = checkpoint["image_encoder"]
            self._print("Loading image_encoder weights to from {}".format(load_path))
            self.model.image_encoder.load_state_dict(image_encoder_dict, strict=True)
            loaded = True

        if tuner:
            tuner_dict = checkpoint["tuner"]
            self._print("Loading tuner weights to from {}".format(load_path))
            self.tuner.load_state_dict(tuner_dict, strict=False)
            loaded = True

        if 'head' in checkpoint and not skip_head:
            head_dict = checkpoint["head"]
            self._print("Loading head weights to from {}".format(load_path))

            if self.cfg.classifier == "DisAlignClassifier":
                if head_dict["weight"].shape == self.head.base_classifier.weight.shape:
                    safe_load(self.head.base_classifier, head_dict)
            else:
                if head_dict["weight"].shape == self.head.weight.shape:
                    safe_load(self.head, head_dict)
            
            loaded = True
        
        if 'lora' in directory.name and tuner == False:
            tuner_dict = checkpoint["tuner"]
            if len(tuner_dict) == 0:
                raise ValueError("No tuner weights found in the checkpoint")
            tuner_vector = TaskVector(vector=tuner_dict, device=self.device)
            merge_lora_weights_with_sd(self, tuner_vector.vector)
            self._print("Merging tuner weights from {}".format(load_path))
            loaded = True
        

        assert loaded == True, "No weights loaded"
