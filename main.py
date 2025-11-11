import os
os.environ["TORCH_HUB"] = "cache/"
os.environ['HF_HOME'] = "cache/"

import random
import math
import shutil
import argparse
import numpy as np
import torch
from pathlib import Path

from utils.config import _C as cfg
from utils.logger import setup_logger

from trainer import Trainer
from utils.util import my_print
from functools import partial


def main(args):
    cfg_data_file = os.path.join("./configs/data", args.data + ".yaml")
    cfg_model_file = os.path.join("./configs/model", args.model + ".yaml")

    cfg.defrost()
    cfg.merge_from_file(cfg_model_file)
    cfg.merge_from_file(cfg_data_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    if cfg.output_dir is None:
        cfg_name = "_".join([args.data, args.model])
        opts_name = "".join(["_" + item for item in args.opts])
        cfg.output_dir = os.path.join("./output", cfg_name + opts_name)
    else:
        cfg.output_dir = os.path.join("./output", cfg.output_dir)
    print("Output directory: {}".format(cfg.output_dir))
    logger = setup_logger(cfg.output_dir)
    _print = partial(my_print, file=logger)

    _print("** Config **")
    _print(cfg)
    _print("************")
    
    if cfg.seed is not None:
        seed = cfg.seed
        _print("Setting fixed seed: {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    trainer = Trainer(cfg, logger)
    
    if cfg.model_dir is not None:
        model_dir = Path(cfg.model_dir)
        if model_dir.is_dir():
            post_fix = "mean_acc"
        elif model_dir.is_file():
            post_fix = model_dir.name.split('.')[0].split('checkpoint_')[-1]
            model_dir = model_dir.parent
        else:
            raise ValueError(f"model_dir {model_dir} is not a valid file or directory")

        trainer.load_model(model_dir.absolute(), tuner=True if cfg.adapter_dim != None else False, skip_head=cfg.re_head, post_fix=post_fix)
    

    if cfg.zero_shot:
        mode = "test"
        if cfg.test_train:
            mode = "train"
        elif cfg.val_only:
            mode = "val"
            
        trainer.test(mode, post_fix="mean_acc")
        return

    if cfg.test_train == True:
        trainer.test("train")
        return

    if cfg.test_only == True:
        trainer.test("test")
        return

    if cfg.val_only == True:
        trainer.test("val")
        return
        
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="", help="data config file")
    parser.add_argument("--model", "-m", type=str, default="", help="model config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")
    args = parser.parse_args()
    main(args)
