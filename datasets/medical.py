import os
from PIL import Image
import numpy as np
import pandas as pd
import torch


class NIH_CXR_LT(torch.utils.data.Dataset):
    label_dir = 'dir'

    def __init__ (self, root, split='train', transform=None, cfg=None):

        self.transform = transform
        self.root = root
        self.split = split

        self.classnames = [
            'No Finding', 'Infiltration', 'Atelectasis', 'Effusion', 'Nodule',
            'Mass', 'Pneumothorax', 'Consolidation', 'Pleural_Thickening',
            'Cardiomegaly', 'Fibrosis', 'Edema', 'Tortuous Aorta', 'Emphysema',
            'Pneumonia', 'Calcification of the Aorta', 'Pneumoperitoneum', 'Hernia',
            'Subcutaneous Emphysema', 'Pneumomediastinum'
        ]

        if split == 'train':
            name = f'nih-cxr-lt_single-label_{split}.csv'
        else:
            name = f'nih-cxr-lt_single-label_balanced-{split}.csv'

        self.label_df = pd.read_csv(os.path.join(self.label_dir, name))
        
        self.img_paths = self.label_df['id'].apply(lambda x: os.path.join(self.root, x)).values.tolist()
        self.labels = self.label_df[self.classnames].idxmax(axis=1).apply(lambda x: self.classnames.index(x)).values

        self.cls_num_list = self.label_df[self.classnames].sum(0).values.tolist()
        self.num_classes = len(self.cls_num_list)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        with open(path, 'rb') as f:
            x = Image.open(f).convert('RGB')

        if self.transform is not None:
            x = self.transform(x)

        y = np.array(self.labels[idx])

        return x.float(), torch.from_numpy(y).long()