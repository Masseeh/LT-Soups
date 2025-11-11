import torch
import torch.nn as nn
from torch.nn import functional as F

from .peft_vit import Peft_ViT, ViT_Tuner
from .peft_rn import Peft_RN, RN_Tuner
from .classifiers import *

class PeftModelFromCLIP(nn.Module):
    def __init__(self, cfg, logger, clip_model, num_classes):
        super().__init__()

        if cfg.backbone.startswith("CLIP-ViT"):
            self.image_encoder = Peft_ViT(logger, clip_model.visual, cfg.use_proj)
            self.tuner = ViT_Tuner(cfg, logger, clip_model.visual, num_classes)
        elif cfg.backbone.startswith("CLIP-RN"):
            self.image_encoder = Peft_RN(clip_model.visual)
            self.tuner = RN_Tuner(cfg, clip_model.visual, num_classes)

        feat_dim = self.image_encoder.out_dim
        dtype = self.image_encoder.dtype

        self.head = eval(cfg.classifier)(feat_dim, num_classes, dtype)
    
    def forward(self, image, use_tuner=True, return_feature=False):
        tuner = self.tuner if use_tuner else None
        head = self.head
        return self.image_encoder(image, tuner, head, return_feature)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.image_encoder.train(mode)
        self.head.train(mode)
        self.tuner.train(mode)


class PeftModelFromViT(nn.Module):
    def __init__(self, cfg, logger, vit_model, num_classes):
        super().__init__()

        if cfg.backbone.startswith("IN21K-ViT") or cfg.backbone.startswith("IN1K-ViT"):
            self.image_encoder = Peft_ViT(logger, vit_model)
            self.tuner = ViT_Tuner(cfg, logger, vit_model, num_classes)
        
        feat_dim = self.image_encoder.out_dim
        dtype = self.image_encoder.dtype
        self.head = eval(cfg.classifier)(feat_dim, num_classes, dtype)

    def forward(self, image, use_tuner=True, return_feature=False):
        tuner = self.tuner if use_tuner else None
        head = self.head
        return self.image_encoder(image, tuner, head, return_feature)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.image_encoder.train(mode)
        self.head.train(mode)
        self.tuner.train(mode)
