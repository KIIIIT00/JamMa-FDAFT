"""
JamMa-FDAFT: Integrated model combining FDAFT feature extraction with JamMa's Joint Mamba and C2F matching

Architecture: Input Images → FDAFT Encoder → Joint Mamba (JEGO) → C2F Matching

This implementation integrates:
- FDAFT: Fast Double-Channel Aggregated Feature Transform for robust planetary image feature extraction
- JamMa: Joint Mamba for efficient long-range feature interaction
- C2F Matching: Coarse-to-Fine hierarchical matching with sub-pixel refinement
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops.einops import rearrange
from .mamba_module import JointMamba
from .matching_module import CoarseMatching, FineSubMatching
from .components.utils import KeypointEncoder_wo_score, up_conv4, MLPMixerEncoderLayer, normalize_keypoints
from ..utils.profiler import PassThroughProfiler

torch.backends.cudnn.deterministic = True
INF = 1E9


class JamMaFDAFT(nn.Module):
    """
    JamMa-FDAFT: Integrated model for planetary remote sensing image matching
    
    Combines FDAFT's robust feature extraction with JamMa's efficient matching pipeline
    """
    
    def __init__(self, config, profiler=None):
        super().__init__()
        self.config = config
        self.profiler = profiler or PassThroughProfiler()
        self.d_model_c = self.config['coarse']['d_model']
        self.d_model_f = self.config['fine']['d_model']

        # Keypoint encoder for position embedding
        self.kenc = KeypointEncoder_wo_score(self.d_model_c, [32, 64, 128, self.d_model_c])
        
        # Joint Mamba for feature interaction (JEGO)
        self.joint_mamba = JointMamba(
            self.d_model_c, 4, 
            rms_norm=True, 
            residual_in_fp32=True, 
            fused_add_norm=True, 
            profiler=self.profiler
        )
        
        # Coarse matching module
        self.coarse_matching = CoarseMatching(config['match_coarse'], self.profiler)

        # Fine-level feature processing network
        self.act = nn.GELU()
        dim = [256, 128, 64]
        self.up2 = up_conv4(dim[0], dim[1], dim[1])  # 1/8 -> 1/4
        self.conv7a = nn.Conv2d(2*dim[1], dim[1], kernel_size=3, stride=1, padding=1)
        self.conv7b = nn.Conv2d(dim[1], dim[1], kernel_size=3, stride=1, padding=1)
        self.up3 = up_conv4(dim[1], dim[2], dim[2])  # 1/4 -> 1/2
        self.conv8a = nn.Conv2d(dim[2], dim[2], kernel_size=3, stride=1, padding=1)
        self.conv8b = nn.Conv2d(dim[2], dim[2], kernel_size=3, stride=1, padding=1)

        # Fine-level encoder
        W = self.config['fine_window_size']
        self.fine_enc = nn.ModuleList([MLPMixerEncoderLayer(2*W**2, 64) for _ in range(4)])
        
        # Fine matching and sub-pixel refinement
        self.fine_matching = FineSubMatching(config, self.profiler)

    def coarse_match(self, data):
        """
        Perform coarse-level matching using FDAFT features and Joint Mamba
        
        Args:
            data: Dictionary containing FDAFT features and metadata
        """
        # Get FDAFT features (already processed by backbone)
        desc0, desc1 = data['feat_8_0'].flatten(2, 3), data['feat_8_1'].flatten(2, 3)
        kpts0, kpts1 = data['grid_8'], data['grid_8']
        
        # Keypoint normalization for position encoding
        kpts0 = normalize_keypoints(kpts0, data['imagec_0'].shape[-2:])
        kpts1 = normalize_keypoints(kpts1, data['imagec_1'].shape[-2:])

        kpts0, kpts1 = kpts0.transpose(1, 2), kpts1.transpose(1, 2)
        
        # Add position encoding to FDAFT features
        desc0 = desc0 + self.kenc(kpts0)
        desc1 = desc1 + self.kenc(kpts1)
        
        data.update({
            'feat_8_0': desc0,
            'feat_8_1': desc1,
        })

        # Joint Mamba feature interaction (JEGO)
        with self.profiler.profile("coarse interaction"):
            self.joint_mamba(data)

        # Coarse-level matching
        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)

        with self.profiler.profile("coarse matching"):
            self.coarse_matching(
                data['feat_8_0'].transpose(1,2), 
                data['feat_8_1'].transpose(1,2), 
                data, 
                mask_c0=mask_c0, 
                mask_c1=mask_c1
            )

    def inter_fpn(self, feat_8, feat_4):
        """
        Feature Pyramid Network for multi-scale feature fusion
        
        Args:
            feat_8: Features at 1/8 resolution
            feat_4: Features at 1/4 resolution
            
        Returns:
            feat_2: Features at 1/2 resolution
        """
        d2 = self.up2(feat_8)  # 1/4
        d2 = self.act(self.conv7a(torch.cat([feat_4, d2], 1)))
        feat_4 = self.act(self.conv7b(d2))

        d1 = self.up3(feat_4)  # 1/2
        d1 = self.act(self.conv8a(d1))
        feat_2 = self.conv8b(d1)
        
        return feat_2

    def fine_preprocess(self, data, profiler):
        """
        Preprocess features for fine-level matching
        
        Args:
            data: Dictionary containing coarse matches and features
            profiler: Profiler for timing analysis
            
        Returns:
            feat0, feat1: Preprocessed fine-level features
        """
        data['resolution1'] = 8
        stride = data['resolution1'] // self.config['resolution'][1]
        W = self.config['fine_window_size']
        
        # Get multi-scale features from FDAFT backbone
        feat_8 = torch.cat([data['feat_8_0'], data['feat_8_1']], 0).view(
            2*data['bs'], data['c'], data['h_8'], -1
        )
        feat_4 = torch.cat([data['feat_4_0'], data['feat_4_1']], 0)

        if data['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, W ** 2, self.d_model_f, device=feat_4.device)
            feat1 = torch.empty(0, W ** 2, self.d_model_f, device=feat_4.device)
            return feat0, feat1

        # Feature pyramid network for multi-scale fusion
        feat_f = self.inter_fpn(feat_8, feat_4)
        feat_f0, feat_f1 = torch.chunk(feat_f, 2, dim=0)
        data.update({'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]})

        # Unfold (crop) all local windows around coarse matches
        pad = 0 if W % 2 == 0 else W//2
        feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=pad)
        feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)
        feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=pad)
        feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)

        # Select only the predicted matches from coarse level
        feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids']]  # [n, ww, cf]
        feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']]  # [n, ww, cf]

        # Process through fine-level encoder
        feat_f = torch.cat([feat_f0_unfold, feat_f1_unfold], 1).transpose(1, 2)
        for layer in self.fine_enc:
            feat_f = layer(feat_f)
        feat_f0_unfold, feat_f1_unfold = feat_f[:, :, :W**2], feat_f[:, :, W**2:]
        
        return feat_f0_unfold, feat_f1_unfold

    def forward(self, data, mode='test'):
        """
        Forward pass of JamMa-FDAFT model
        
        Args:
            data: Dictionary containing FDAFT features and metadata
            mode: 'train', 'val', or 'test'
            
        Note: FDAFT backbone processing should be done before calling this
        """
        self.mode = mode
        
        # Update data with image dimensions
        data.update({
            'hw0_i': data['imagec_0'].shape[2:],
            'hw1_i': data['imagec_1'].shape[2:],
            'hw0_c': [data['h_8'], data['w_8']],
            'hw1_c': [data['h_8'], data['w_8']],
        })

        # Coarse-level matching with Joint Mamba
        self.coarse_match(data)

        # Fine-level matching and sub-pixel refinement
        with self.profiler.profile("fine matching"):
            # Preprocess features for fine-level matching
            feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(data, self.profiler)

            # Fine-level matching and sub-pixel refinement
            self.fine_matching(
                feat_f0_unfold.transpose(1, 2), 
                feat_f1_unfold.transpose(1, 2), 
                data
            )

    def get_model_info(self):
        """
        Get information about the integrated model
        
        Returns:
            Dictionary with model architecture information
        """
        return {
            "model_name": "JamMa-FDAFT",
            "architecture": "FDAFT Encoder → Joint Mamba (JEGO) → C2F Matching",
            "feature_extractor": "FDAFT (Fast Double-Channel Aggregated Feature Transform)",
            "feature_interaction": "Joint Mamba with JEGO scanning strategy",
            "matching_strategy": "Hierarchical Coarse-to-Fine with sub-pixel refinement",
            "optimizations": {
                "planetary_images": "FDAFT optimized for weak textures and gradual transitions",
                "efficient_attention": "Joint Mamba for linear complexity long-range modeling", 
                "hierarchical_matching": "C2F strategy for robust matching at multiple scales"
            },
            "parameters": {
                "coarse_d_model": self.d_model_c,
                "fine_d_model": self.d_model_f,
                "fine_window_size": self.config['fine_window_size'],
                "resolution": self.config['resolution']
            }
        }