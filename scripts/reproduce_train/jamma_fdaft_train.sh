#!/bin/bash
# JamMa-FDAFT Training Script

# Outdoor training (MegaDepth)
python train_jamma_fdaft.py \
    configs/data/megadepth_trainval_832.py \
    configs/jamma_fdaft/outdoor/final.py \
    --exp_name jamma_fdaft_outdoor \
    --batch_size 2 \
    --gpus 1 \
    --max_epochs 30

# Indoor training (ScanNet)  
python train_jamma_fdaft.py \
    configs/data/scannet_trainval.py \
    configs/jamma_fdaft/indoor/final.py \
    --exp_name jamma_fdaft_indoor \
    --batch_size 2 \
    --gpus 1 \
    --max_epochs 25
