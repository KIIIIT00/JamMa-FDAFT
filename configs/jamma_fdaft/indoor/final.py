"""
Configuration for JamMa-FDAFT indoor scene matching (ScanNet dataset)
"""

from src.config.default import _CN as cfg

# FDAFT Encoder Configuration (Indoor optimized)
cfg.FDAFT = cfg.__class__()
cfg.FDAFT.NUM_LAYERS = 3
cfg.FDAFT.SIGMA_0 = 1.2  # Slightly larger scale for indoor scenes
cfg.FDAFT.USE_STRUCTURED_FORESTS = True
cfg.FDAFT.MAX_KEYPOINTS = 1500  # Adjusted for indoor scenes
cfg.FDAFT.NMS_RADIUS = 4

# JamMa Configuration
cfg.JAMMA.RESOLUTION = (8, 2)
cfg.JAMMA.FINE_WINDOW_SIZE = 5
cfg.JAMMA.COARSE.D_MODEL = 256
cfg.JAMMA.FINE.D_MODEL = 64

# Training Setup (Indoor specific)
cfg.TRAINER.SCHEDULER = 'CosineAnnealing'
cfg.TRAINER.COSA_TMAX = 25  # Shorter for indoor
cfg.TRAINER.CANONICAL_BS = 2
cfg.TRAINER.CANONICAL_LR = 8e-5  # Lower learning rate for indoor
cfg.TRAINER.WARMUP_STEP = 48600

# Pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5
cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1

# Indoor-specific FDAFT settings
cfg.FDAFT.FEATURE_DETECTOR = cfg.__class__()
cfg.FDAFT.FEATURE_DETECTOR.FAST_THRESHOLD = 0.04  # More sensitive for indoor
cfg.FDAFT.FEATURE_DETECTOR.KAZE_THRESHOLD = 0.0008  # More sensitive

cfg.FDAFT.PLANETARY_SPECIFIC = cfg.__class__()
cfg.FDAFT.PLANETARY_SPECIFIC.CRATER_DETECTION = False  # Not applicable for indoor
cfg.FDAFT.PLANETARY_SPECIFIC.TEXTURE_MASK_THRESHOLD = 0.015  # Lower threshold

cfg.TRAINER.SEED = 66
