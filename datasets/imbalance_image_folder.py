import os
from utils.util import log_interpolate, exp_interpolate
import torchvision
import numpy as np
from collections import defaultdict

class IMBALANCEImageFolder(torchvision.datasets.ImageFolder):
    classnames_txt = None
    cls_num = 200
    val_per_class = 20
    random_seed = 0

    def __init__(self, root, imb_factor=0.01, random_seed=0, split="train",
                 transform=None, few_shot=100, cls_split_f=50, target_transform=None):
    
        cls_split_f = [f/100 for f in [cls_split_f]]
        self.cls_split = [int(self.cls_num * f) for f in cls_split_f]
        self.cls_split.extend([self.cls_num - sum(self.cls_split)])

        self.few_shot = few_shot

        train = True if split == "train" else False
        if train:
            nroot = os.path.join(root, "train")
        else:
            nroot = os.path.join(root, "val")

        super().__init__(nroot, transform, target_transform)

        if train:
            self.random_seed = random_seed
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_factor)
            self.gen_imbalanced_data(img_num_list)

        if split == "val" or split == "test":
            np_targets = np.array(self.targets)
            targets_idx = []
            for cls_idx in range(len(self.classes)):
                
                if split == "val":
                    f_cls_idx = np.where(np_targets == cls_idx)[0][:self.val_per_class]
                else:
                    f_cls_idx = np.where(np_targets == cls_idx)[0][self.val_per_class:]

                targets_idx.extend(f_cls_idx)
            
            self.targets = np_targets[targets_idx].tolist()
            self.imgs = [self.imgs[i] for i in targets_idx]
        
        self.classnames = self.read_classnames()
        self.labels = self.targets
        self.samples = self.imgs
        self.cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)

    def get_img_num_per_cls(self, cls_num, imb_factor):
        targets_np = np.array(self.targets, dtype=np.int64)
        _, counts = np.unique(targets_np, return_counts=True)
        img_max = counts[0]
        img_min = img_max * imb_factor
        K=5

        many_cls_num = exp_interpolate(np.linspace(0, 1, self.cls_split[0], endpoint=True), img_max, self.few_shot + 1, k=K).astype(int).tolist()
        few_cls_num = exp_interpolate(np.linspace(0, 1, self.cls_split[1], endpoint=True), self.few_shot, img_min, k=K).astype(int).tolist()

        img_num_per_cls = many_cls_num + few_cls_num
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        rng = np.random.default_rng(self.random_seed)

        new_imgs = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            rng.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_imgs.extend([self.imgs[i] for i in selec_idx])
            new_targets.extend([the_class, ] * the_img_num)
        self.imgs = new_imgs
        self.targets = new_targets
        
    def get_cls_num_list(self):
        counter = defaultdict(int)
        for label in self.labels:
            counter[label] += 1
        labels = list(counter.keys())
        labels.sort()
        cls_num_list = [counter[label] for label in labels]
        return cls_num_list

    def read_classnames(self):
        set_class = set(self.classes)

        if self.classnames_txt is None:
            return self.classes

        classnames = []
        with open(self.classnames_txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])

                if folder in set_class:
                    classnames.append(classname)

        return classnames

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


class TinyImageNet_IR100(IMBALANCEImageFolder):
    classnames_txt = "./datasets/ImageNet_LT/classnames.txt"
    cls_num = 200

    def __init__(self, root, split="train", transform=None, cfg=None):
        super().__init__(root, imb_factor=0.01, split=split, transform=transform)

    def get_img_num_per_cls(self, cls_num, imb_factor):
        targets_np = np.array(self.targets, dtype=np.int64)
        _, counts = np.unique(targets_np, return_counts=True)
        img_max = counts[0]

        img_num_per_cls = []
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
        return img_num_per_cls


class TinyImageNet_F(IMBALANCEImageFolder):
    classnames_txt = "./datasets/ImageNet_LT/classnames.txt"
    cls_num = 200

    def __init__(self, root, split="train", transform=None, cfg=None):
        super().__init__(root, imb_factor=0.01, split=split, transform=transform, cls_split_f=cfg.cls_split_f, few_shot=200)