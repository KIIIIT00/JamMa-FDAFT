"""
Updated default configuration including FDAFT settings
"""

from yacs.config import CfgNode as CN
_CN = CN()

# Original JamMa configuration (unchanged)
_CN.JAMMA = CN()
_CN.JAMMA.RESOLUTION = (8, 2)
_CN.JAMMA.FINE_WINDOW_SIZE = 5

_CN.JAMMA.COARSE = CN()
_CN.JAMMA.COARSE.D_MODEL = 256

_CN.JAMMA.MATCH_COARSE = CN()
_CN.JAMMA.MATCH_COARSE.USE_SM = True
_CN.JAMMA.MATCH_COARSE.THR = 0.2
_CN.JAMMA.MATCH_COARSE.BORDER_RM = 2
_CN.JAMMA.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.JAMMA.MATCH_COARSE.SKH_ITERS = 3
_CN.JAMMA.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2
_CN.JAMMA.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200
_CN.JAMMA.MATCH_COARSE.INFERENCE = False

_CN.JAMMA.FINE = CN()
_CN.JAMMA.FINE.D_MODEL = 128
_CN.JAMMA.FINE.DENSER = False
_CN.JAMMA.FINE.INFERENCE = False
_CN.JAMMA.FINE.DSMAX_TEMPERATURE = 0.1
_CN.JAMMA.FINE.THR = 0.1

_CN.JAMMA.LOSS = CN()
_CN.JAMMA.LOSS.COARSE_WEIGHT = 1.0
_CN.JAMMA.LOSS.FOCAL_ALPHA = 0.25
_CN.JAMMA.LOSS.FOCAL_GAMMA = 2.0
_CN.JAMMA.LOSS.POS_WEIGHT = 1.0
_CN.JAMMA.LOSS.NEG_WEIGHT = 1.0
_CN.JAMMA.LOSS.SUB_WEIGHT = 1 * 10**4
_CN.JAMMA.LOSS.FINE_WEIGHT = 1.0

_CN.JAMMA.MP = False
_CN.JAMMA.EVAL_TIMES = 1

# FDAFT Configuration (New)
_CN.FDAFT = CN()
_CN.FDAFT.NUM_LAYERS = 3
_CN.FDAFT.SIGMA_0 = 1.0
_CN.FDAFT.USE_STRUCTURED_FORESTS = True
_CN.FDAFT.MAX_KEYPOINTS = 1000
_CN.FDAFT.NMS_RADIUS = 5
_CN.FDAFT.STRUCTURED_FORESTS_MODEL = None

# FDAFT Scale Space
_CN.FDAFT.SCALE_SPACE = CN()
_CN.FDAFT.SCALE_SPACE.ECM_PATCH_SIZE = 16
_CN.FDAFT.SCALE_SPACE.STEERABLE_FILTER_SIGMA = 2.0

# FDAFT Feature Detector
_CN.FDAFT.FEATURE_DETECTOR = CN()
_CN.FDAFT.FEATURE_DETECTOR.FAST_THRESHOLD = 0.05
_CN.FDAFT.FEATURE_DETECTOR.USE_KAZE = True
_CN.FDAFT.FEATURE_DETECTOR.KAZE_THRESHOLD = 0.001

# FDAFT Descriptor
_CN.FDAFT.DESCRIPTOR = CN()
_CN.FDAFT.DESCRIPTOR.GLOH_RADIAL_BINS = 3
_CN.FDAFT.DESCRIPTOR.GLOH_ANGULAR_BINS = 8
_CN.FDAFT.DESCRIPTOR.GLOH_ORIENTATION_BINS = 16

# FDAFT Planetary Optimizations
_CN.FDAFT.PLANETARY_SPECIFIC = CN()
_CN.FDAFT.PLANETARY_SPECIFIC.CRATER_DETECTION = True
_CN.FDAFT.PLANETARY_SPECIFIC.TEXTURE_MASK_THRESHOLD = 0.02

# Original Dataset and Trainer configurations (unchanged)
_CN.DATASET = CN()
# ... (rest of the configuration remains the same)

_CN.TRAINER = CN()
# ... (rest of the configuration remains the same)

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for JamMa-FDAFT."""
    return _CN.clone()