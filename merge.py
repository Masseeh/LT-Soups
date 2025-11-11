import os
os.environ["TORCH_HUB"] = "cache/"
os.environ['HF_HOME'] = "cache/"

import random
from pathlib import Path
import argparse
import numpy as np
import torch
import copy

from utils.config import _C as cfg
from utils.logger import setup_logger
from utils.util import my_print
from functools import partial

from trainer import Trainer
from models import TaskVector
from utils.util import safe_load

def main(args):
    cfg_data_file = os.path.join("./configs/data", args.data + ".yaml")
    cfg_model_file = os.path.join("./configs/model", args.model + ".yaml")

    cfg.defrost()
    cfg.merge_from_file(cfg_data_file)
    cfg.merge_from_file(cfg_model_file)
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

    assert cfg.lora == False

    trainer = Trainer(cfg, logger)
    zs_all = copy.deepcopy(trainer.model.state_dict())
    zs_vector = TaskVector(vector=zs_all, device=trainer.device)

    checkpoints = []
    checkpoints = [f"output/INAT2018_IND/{args.data}_IND_[{i}]_b1/checkpoint_last.pth.tar" for i in range(9)]

    # 1 model
    _print(f"model {checkpoints[0]}")
    checkpoint = torch.load(checkpoints[0], map_location=trainer.device, weights_only=True)

    head = checkpoint["head"]
    safe_load(trainer.model.head, head)
    
    img_encoder = checkpoint["image_encoder"]
    safe_load(trainer.model.image_encoder, img_encoder)

    f_vector = TaskVector(vector=trainer.model.state_dict(), device=trainer.device)

    task_vectors = [f_vector]

    if len(checkpoints) > 1:
        for ckpt in checkpoints[1:]:
            trainer.model.load_state_dict(zs_all)
            _print(f"model {ckpt}")
            checkpoint = torch.load(ckpt, map_location=trainer.device, weights_only=True)

            head = checkpoint["head"]
            safe_load(trainer.model.head, head)

            img_encoder = checkpoint["image_encoder"]
            safe_load(trainer.model.image_encoder, img_encoder)

            s_vector = TaskVector(vector=trainer.model.state_dict(), device=trainer.device)
            task_vectors.append(s_vector)

    lm = cfg.merge_lm
    new_vector = zs_vector * lm +  task_vectors[0] * (1 - lm)
    for s_vector in task_vectors[1:]:
        new_vector = new_vector * lm + s_vector * (1 - lm)

    safe_load(trainer.model, new_vector.vector)
    for split in ["test"]:
        # eval_results = trainer.eval(split)
        eval_results = trainer.test(split, post_fix="mean_acc")
        eval_info = [f"(mean {eval_results['mean_acc']:.4f} many {eval_results['many_acc']:.4f} med {eval_results['med_acc']:.4f} few {eval_results['few_acc']:.4f})"]
        _print(" ".join(eval_info))
    
    trainer.save_model(cfg.output_dir, image_encoder=True, post_fix="last")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="", help="data config file")
    parser.add_argument("--model", "-m", type=str, default="", help="model config file")
    parser.add_argument("--ckpt", nargs='+', type=str, default=[], help="checkpoint paths to merge")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")
    args = parser.parse_args()
    main(args)
