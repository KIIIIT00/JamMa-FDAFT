```
# JamMa-FDAFT Project Structure
# Integrated model combining FDAFT feature extraction with JamMa's Joint Mamba and C2F matching

jamma_fdaft/
├── README.md
├── requirements.txt
├── setup.py
├── train.py
├── test.py
├── demo_jamma_fdaft.py
├── configs/
│   ├── jamma_fdaft/
│   │   ├── indoor/
│   │   │   ├── final.py
│   │   │   └── test.py
│   │   └── outdoor/
│   │       ├── final.py
│   │       └── test.py
│   └── data/
│       ├── base.py
│       ├── megadepth_trainval_832.py
│       ├── megadepth_test_1500.py
│       ├── scannet_trainval.py
│       └── scannet_test_1500.py
├── src/
│   ├── __init__.py
│   ├── config/
│   │   └── default.py
│   ├── jamma_fdaft/
│   │   ├── __init__.py
│   │   ├── backbone_fdaft.py          # FDAFT Encoder
│   │   ├── jamma_fdaft.py             # Main integrated model
│   │   ├── mamba_module.py            # Joint Mamba (JEGO) from JamMa
│   │   ├── matching_module.py         # C2F Matching from JamMa
│   │   └── components/
│   │       ├── __init__.py
│   │       ├── scale_space.py         # FDAFT scale space
│   │       ├── feature_detector.py    # FDAFT feature detector
│   │       ├── gloh_descriptor.py     # FDAFT GLOH descriptor
│   │       └── utils.py               # Shared utilities
│   ├── lightning/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   └── lightning_jamma_fdaft.py   # Lightning module for training
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── megadepth.py
│   │   ├── scannet.py
│   │   └── sampler.py
│   ├── losses/
│   │   ├── __init__.py
│   │   └── loss.py
│   ├── optimizers/
│   │   ├── __init__.py
│   │   └── __init__.py
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       ├── plotting.py
│       ├── dataset.py
│       ├── misc.py
│       ├── comm.py
│       ├── dataloader.py
│       ├── augment.py
│       ├── profiler.py
│       └── visualization.py
├── scripts/
│   ├── reproduce_train/
│   │   └── outdoor.sh
│   └── reproduce_test/
│       ├── indoor.sh
│       └── outdoor.sh
├── docs/
│   ├── TRAINING.md
│   └── ARCHITECTURE.md
├── assets/
│   └── structured_forests/
│       └── model.yml.gz
└── weight/
    └── jamma_fdaft_weight.ckpt
```