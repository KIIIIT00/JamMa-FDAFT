#!/bin/bash
# JamMa-FDAFT Testing Script

# Test on MegaDepth
python test_jamma_fdaft.py \
    configs/data/megadepth_test_1500.py \
    configs/jamma_fdaft/outdoor/test.py \
    --ckpt_path weight/jamma_fdaft_weight.ckpt \
    --detailed_analysis \
    --save_visualizations

# Test on ScanNet
python test_jamma_fdaft.py \
    configs/data/scannet_test_1500.py \
    configs/jamma_fdaft/indoor/test.py \
    --ckpt_path weight/jamma_fdaft_weight.ckpt \
    --detailed_analysis
