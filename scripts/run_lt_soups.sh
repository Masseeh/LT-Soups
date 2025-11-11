#!/bin/bash

# abort on error
set -e

model=clip_vit_b16
gpu=1
num_epochs=10
lr0=3e-5
lr1=3e-4
wd=1e-2
seed=0
meta_bsz=2
rank=64
ema=0.99
eval_freq=5

dataset=cifar100_ir100
merge_lm=0.7
meta_bsz=2
factors="[0,1,2,3,4,5,6]"
base_model=WA
python main_soups.py -d $dataset -m $model output_dir ${dataset}_${base_model} \
    loss_type "LA" num_epochs $num_epochs head_tuning True full_tuning True ema $ema \
    gpu $gpu eval_on_val True lr $lr0 adam True wd $wd scheduler cosine warmup True eval_freq $eval_freq merge_lm $merge_lm \
    tensorboard True seed $seed meta_bsz $meta_bsz factors $factors

base_model=LP-WA_ep${num_epochs}
python main.py -d $dataset -m $model output_dir ${dataset}_${base_model} \
    loss_type "LA" num_epochs $num_epochs head_tuning True ema $ema \
    gpu $gpu eval_on_val True lr $lr0 adam True wd $wd scheduler cosine warmup True eval_freq $eval_freq \
    tensorboard True seed $seed model_dir output/${dataset}_WA/checkpoint_last.pth.tar