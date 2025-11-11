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

dataset=cifar100_f
for imb_factor in 0.02 0.01 0.005
do  
    for cls_factor in 95 90 80 70 60 50 40 30 20 10 5
    do 
        base_model=FullFT-${cls_factor}-${imb_factor}
        python main.py -d $dataset -m $model output_dir CIFAR100/${dataset}_${base_model} \
            loss_type "LA" num_epochs $num_epochs head_tuning True full_tuning True ema $ema \
            gpu $gpu eval_on_val True lr $lr0 adam True wd $wd scheduler cosine warmup True eval_freq $eval_freq \
            tensorboard False seed $seed cls_split_f $cls_factor imb_factor $imb_factor
    done
done