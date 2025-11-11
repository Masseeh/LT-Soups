from collections import defaultdict
import torchvision
import numpy as np
from utils.util import exp_interpolate

class IMBALANCECIFAR100(torchvision.datasets.CIFAR100):
    cls_num = 100
    val_per_class = 10
    random_seed = 0

    def __init__(self, root, imb_factor=None, random_seed=0, split="train",
                 transform=None, target_transform=None, download=True):
        
        train = True if split == "train" else False
        super().__init__(root, train, transform, target_transform, download)

        if train and imb_factor is not None:
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
            self.data = self.data[targets_idx]
        
        self.classnames = self.classes
        self.labels = self.targets
        self.cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)
        
    def get_img_num_per_cls(self, cls_num, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        rng = np.random.default_rng(self.random_seed)

        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            rng.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        counter = defaultdict(int)
        for label in self.labels:
            counter[label] += 1
        labels = list(counter.keys())
        labels.sort()
        cls_num_list = [counter[label] for label in labels]
        return cls_num_list
    
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


class CIFAR100(IMBALANCECIFAR100):
    def __init__(self, root, split="train", transform=None, cfg=None):
        super().__init__(root, imb_factor=None, split=split, transform=transform)

class CIFAR100_IR10(IMBALANCECIFAR100):
    def __init__(self, root, split="train", transform=None, cfg=None):
        super().__init__(root, imb_factor=0.1, split=split, transform=transform)
class CIFAR100_IR50(IMBALANCECIFAR100):
    def __init__(self, root, split="train", transform=None, cfg=None):
        super().__init__(root, imb_factor=0.02, split=split, transform=transform)

class CIFAR100_IR100(IMBALANCECIFAR100):
    def __init__(self, root, split="train", transform=None, cfg=None):
        super().__init__(root, imb_factor=0.01, split=split, transform=transform)

class CIFAR100_IRF(IMBALANCECIFAR100):
    def __init__(self, root, split="train", transform=None, cfg=None):

        cls_split_f = [f/100 for f in [cfg.cls_split_f]]
        self.cls_split = [int(self.cls_num * f) for f in cls_split_f]
        self.cls_split.extend([self.cls_num - sum(self.cls_split)])

        self.few_shot = 100

        super().__init__(root, imb_factor=0.01, split=split, transform=transform)

    def get_img_num_per_cls(self, cls_num, imb_factor):
        img_max = len(self.data) / cls_num
        img_min = img_max * imb_factor
        K=5

        many_cls_num = exp_interpolate(np.linspace(0, 1, self.cls_split[0], endpoint=True), img_max, self.few_shot + 1, k=K).astype(int).tolist()
        few_cls_num = exp_interpolate(np.linspace(0, 1, self.cls_split[1], endpoint=True), self.few_shot, img_min, k=K).astype(int).tolist()

        img_num_per_cls = many_cls_num + few_cls_num
        return img_num_per_cls