import os
from .lt_data import LT_Dataset
from utils.util import cosine_interpolate, linear_interpolate, log_interpolate, exp_interpolate
import torchvision
import numpy as np
from collections import defaultdict

class Places_LT(LT_Dataset):
    classnames_txt = "./datasets/Places_LT/classnames.txt"
    train_txt = "./datasets/Places_LT/Places_LT_train.txt"
    test_txt = "./datasets/Places_LT/Places_LT_test.txt"
    val_txt = "./datasets/Places_LT/Places_LT_val.txt"

    def __init__(self, root, split='train', transform=None, cfg=None):
        super().__init__(root, split, transform)

        self.classnames = self.read_classnames()

        self.names = []
        with open(self.txt) as f:
            for line in f:
                self.names.append(self.classnames[int(line.split()[1])])

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        name = self.names[index]
        return image, label, name

    @classmethod
    def read_classnames(self):
        classnames = []
        with open(self.classnames_txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames.append(classname)
        return classnames