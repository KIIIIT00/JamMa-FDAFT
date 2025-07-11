"""
Test configuration for JamMa-FDAFT indoor scenes
"""

from src.config.default import _CN as cfg

# Pose estimation
cfg.TRAINER.POSE_ESTIMATION_METHOD = 'LO-RANSAC'
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5
cfg.TRAINER.EPI_ERR_THR = 5e-4  # For ScanNet

# FDAFT Configuration for Indoor Testing
cfg.FDAFT = cfg.__class__()
cfg.FDAFT.NUM_LAYERS = 3
cfg.FDAFT.SIGMA_0 = 1.2
cfg.FDAFT.USE_STRUCTURED_FORESTS = True
cfg.FDAFT.MAX_KEYPOINTS = 800  # Reduced for faster testing
cfg.FDAFT.NMS_RADIUS = 4

# JamMa Test Configuration
cfg.JAMMA.MP = False
cfg.JAMMA.EVAL_TIMES = 1
cfg.JAMMA.MATCH_COARSE.INFERENCE = True
cfg.JAMMA.FINE.INFERENCE = True
cfg.JAMMA.MATCH_COARSE.USE_SM = True
cfg.JAMMA.MATCH_COARSE.THR = 0.2
cfg.JAMMA.FINE.THR = 0.1
cfg.JAMMA.MATCH_COARSE.BORDER_RM = 2