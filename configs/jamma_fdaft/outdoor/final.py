"""
Configuration for JamMa-FDAFT outdoor scene matching (MegaDepth dataset)
Integrates FDAFT feature extraction with JamMa's Joint Mamba and C2F matching
"""

from src.config.default import _CN as cfg

# =============================================================================
# JAMMA-FDAFT Pipeline Configuration
# =============================================================================

# FDAFT Encoder Configuration
cfg.FDAFT = cfg.__class__()
cfg.FDAFT.NUM_LAYERS = 3  # Scale space layers
cfg.FDAFT.SIGMA_0 = 1.0   # Initial scale parameter
cfg.FDAFT.USE_STRUCTURED_FORESTS = True  # Use pre-trained Structured Forests
cfg.FDAFT.MAX_KEYPOINTS = 2000  # Maximum keypoints for feature detection
cfg.FDAFT.NMS_RADIUS = 5  # Non-maximum suppression radius
cfg.FDAFT.STRUCTURED_FORESTS_MODEL = "assets/structured_forests/model.yml"

# JamMa Configuration (Modified for FDAFT)
cfg.JAMMA.RESOLUTION = (8, 2)  # Multi-scale resolution
cfg.JAMMA.FINE_WINDOW_SIZE = 5  # Fine-level window size (must be odd)
cfg.JAMMA.COARSE.D_MODEL = 256  # FDAFT features dimension
cfg.JAMMA.FINE.D_MODEL = 64     # Fine-level features dimension

# Optimizer and Scheduler
cfg.TRAINER.SCHEDULER = 'CosineAnnealing'
cfg.TRAINER.COSA_TMAX = 30
cfg.TRAINER.CANONICAL_BS = 2
cfg.TRAINER.CANONICAL_LR = 1e-4
cfg.TRAINER.WARMUP_STEP = 58200  # 3 epochs

# Pose Estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

# Training Setup
cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.JAMMA.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
cfg.JAMMA.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 20
cfg.TRAINER.N_VAL_PAIRS_TO_PLOT = 1

# Reproducibility
cfg.TRAINER.SEED = 66

# =============================================================================
# FDAFT-Specific Planetary Image Optimizations
# =============================================================================

# Scale Space Configuration
cfg.FDAFT.SCALE_SPACE = cfg.__class__()
cfg.FDAFT.SCALE_SPACE.ECM_PATCH_SIZE = 16
cfg.FDAFT.SCALE_SPACE.STEERABLE_FILTER_SIGMA = 2.0
cfg.FDAFT.SCALE_SPACE.PHASE_CONGRUENCY = cfg.__class__()
cfg.FDAFT.SCALE_SPACE.PHASE_CONGRUENCY.ORIENTATIONS = [0, 30, 60, 90, 120, 150]  # degrees
cfg.FDAFT.SCALE_SPACE.PHASE_CONGRUENCY.SCALES = [1, 2, 4]
cfg.FDAFT.SCALE_SPACE.PHASE_CONGRUENCY.NOISE_THRESHOLD = 0.1

# Feature Detection Configuration
cfg.FDAFT.FEATURE_DETECTOR = cfg.__class__()
cfg.FDAFT.FEATURE_DETECTOR.FAST_THRESHOLD = 0.05
cfg.FDAFT.FEATURE_DETECTOR.BLOB_THRESHOLD = 0.05
cfg.FDAFT.FEATURE_DETECTOR.CORNER_K = 0.04  # Harris parameter
cfg.FDAFT.FEATURE_DETECTOR.USE_KAZE = True
cfg.FDAFT.FEATURE_DETECTOR.KAZE_THRESHOLD = 0.001
cfg.FDAFT.FEATURE_DETECTOR.KAZE_OCTAVES = 4
cfg.FDAFT.FEATURE_DETECTOR.KAZE_OCTAVE_LAYERS = 4

# GLOH Descriptor Configuration
cfg.FDAFT.DESCRIPTOR = cfg.__class__()
cfg.FDAFT.DESCRIPTOR.GLOH_RADIAL_BINS = 3
cfg.FDAFT.DESCRIPTOR.GLOH_ANGULAR_BINS = 8
cfg.FDAFT.DESCRIPTOR.GLOH_ORIENTATION_BINS = 16
cfg.FDAFT.DESCRIPTOR.GRADIENT_THRESHOLD = 0.1
cfg.FDAFT.DESCRIPTOR.LAMBDA_ORI = 1.5
cfg.FDAFT.DESCRIPTOR.LAMBDA_DESC = 6.0

# Planetary-Specific Optimizations
cfg.FDAFT.PLANETARY_SPECIFIC = cfg.__class__()
cfg.FDAFT.PLANETARY_SPECIFIC.CRATER_DETECTION = True
cfg.FDAFT.PLANETARY_SPECIFIC.CIRCULAR_STRUCTURE_RADIUS_RANGE = [10, 100]
cfg.FDAFT.PLANETARY_SPECIFIC.TEXTURE_MASK_THRESHOLD = 0.02
cfg.FDAFT.PLANETARY_SPECIFIC.CONTRAST_ENHANCEMENT = True

# Matching Configuration
cfg.FDAFT.MATCHING = cfg.__class__()
cfg.FDAFT.MATCHING.RATIO_THRESHOLD = 0.8  # Lowe's ratio test
cfg.FDAFT.MATCHING.RANSAC_THRESHOLD = 3.0  # pixels
cfg.FDAFT.MATCHING.RANSAC_MAX_ITERS = 1000