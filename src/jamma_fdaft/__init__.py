"""
JamMa-FDAFT: FDAFT Encoder + Joint Mamba + C2F Matching

Fast Double-Channel Aggregated Feature Transform (FDAFT) integrated with
Joint Mamba and Coarse-to-Fine matching for planetary remote sensing.
"""

from .backbone_fdaft import FDAFTEncoder
from .jamma_fdaft import JamMaFDAFT

__version__ = "1.0.0"
__author__ = "JamMa-FDAFT Team"

__all__ = [
    'FDAFTEncoder',
    'JamMaFDAFT',
]