"""
FDAFT Backbone Encoder for JamMa-FDAFT

This module replaces the ConvNext backbone in original JamMa with FDAFT-based feature extraction.
It provides multi-scale features optimized for planetary remote sensing images.

Architecture:
Input Images → Double-Frequency Scale Space → Feature Detection → Feature Processing → Multi-scale Features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.utils import create_meshgrid
from einops import rearrange
from .components.scale_space import DoubleFrequencyScaleSpace
from .components.feature_detector import FeatureDetector
from .components.gloh_descriptor import GLOHDescriptor
import numpy as np


class FDAFTEncoder(nn.Module):
    """
    FDAFT-based backbone encoder for planetary image feature extraction
    
    Replaces ConvNext backbone with FDAFT components optimized for:
    - Weak surface textures in planetary images
    - Gradual illumination changes
    - Crater-like circular structures
    - Multi-scale feature extraction
    """
    
    def __init__(self, 
                 num_layers: int = 3,
                 sigma_0: float = 1.0,
                 use_structured_forests: bool = True,
                 max_keypoints: int = 2000,
                 nms_radius: int = 5):
        """
        Initialize FDAFT encoder
        
        Args:
            num_layers: Number of scale space layers
            sigma_0: Initial scale parameter for scale space
            use_structured_forests: Whether to use pre-trained Structured Forests
            max_keypoints: Maximum keypoints for feature detection
            nms_radius: Non-maximum suppression radius
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.sigma_0 = sigma_0
        self.use_structured_forests = use_structured_forests
        self.max_keypoints = max_keypoints
        
        # FDAFT components
        self.scale_space = DoubleFrequencyScaleSpace(num_layers, sigma_0)
        self.detector = FeatureDetector(nms_radius, max_keypoints, use_kaze=True)
        
        # Feature processing networks
        # Convert FDAFT features to JamMa-compatible feature maps
        self.feature_conv_8 = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=3, padding=1),  # 2 channels from FDAFT
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Match JamMa dimensions
            nn.BatchNorm2d(256),
            nn.GELU()
        )
        
        self.feature_conv_4 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Match JamMa dimensions
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        
        # Adaptive pooling for consistent feature map sizes
        self.adaptive_pool_8 = nn.AdaptiveAvgPool2d((None, None))  # Will be set dynamically
        self.adaptive_pool_4 = nn.AdaptiveAvgPool2d((None, None))  # Will be set dynamically
        
    def preprocess_images(self, image_batch):
        """
        Preprocess batch of images for FDAFT processing
        
        Args:
            image_batch: Batch of images [B, C, H, W]
            
        Returns:
            Preprocessed images ready for FDAFT
        """
        # Convert to grayscale if needed for FDAFT processing
        if image_batch.shape[1] == 3:  # RGB
            # Convert RGB to grayscale using luminance weights
            gray_batch = 0.299 * image_batch[:, 0] + 0.587 * image_batch[:, 1] + 0.114 * image_batch[:, 2]
            gray_batch = gray_batch.unsqueeze(1)  # Add channel dimension
        else:
            gray_batch = image_batch
            
        # Normalize to [0, 1] range
        gray_batch = gray_batch.float()
        if gray_batch.max() > 1.0:
            gray_batch = gray_batch / 255.0
            
        return gray_batch
    
    def extract_fdaft_features(self, image):
        """
        Extract FDAFT features from a single image
        
        Args:
            image: Single image tensor [1, H, W] or [H, W]
            
        Returns:
            Dictionary containing scale space features and feature points
        """
        # Convert to numpy for FDAFT processing
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                img_np = image.squeeze(0).cpu().numpy()
            else:
                img_np = image.cpu().numpy()
        else:
            img_np = image
            
        # Ensure proper range [0, 255] for FDAFT
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        
        # Build double-frequency scale spaces
        low_freq_space = self.scale_space.build_low_frequency_scale_space(img_np)
        high_freq_space = self.scale_space.build_high_frequency_scale_space(img_np)
        
        # Extract feature points (optional, for debugging/visualization)
        corner_points, corner_scores = self.detector.extract_corner_points(low_freq_space)
        blob_points, blob_scores = self.detector.extract_blob_points(high_freq_space)
        
        return {
            'low_freq_space': low_freq_space,
            'high_freq_space': high_freq_space,
            'corner_points': corner_points,
            'blob_points': blob_points,
            'corner_scores': corner_scores,
            'blob_scores': blob_scores
        }
    
    def create_feature_maps(self, low_freq_space, high_freq_space, target_size):
        """
        Create feature maps from FDAFT scale spaces
        
        Args:
            low_freq_space: Low-frequency scale space layers
            high_freq_space: High-frequency scale space layers
            target_size: Target size for feature maps (H, W)
            
        Returns:
            Combined feature maps for different scales
        """
        device = next(self.parameters()).device
        
        # Combine first two layers of each scale space
        low_freq_combined = (low_freq_space[0] + low_freq_space[1]) / 2.0
        high_freq_combined = (high_freq_space[0] + high_freq_space[1]) / 2.0
        
        # Normalize to [0, 1]
        low_freq_norm = (low_freq_combined - low_freq_combined.min()) / (
            low_freq_combined.max() - low_freq_combined.min() + 1e-8
        )
        high_freq_norm = (high_freq_combined - high_freq_combined.min()) / (
            high_freq_combined.max() - high_freq_combined.min() + 1e-8
        )
        
        # Convert to torch tensors
        low_freq_tensor = torch.from_numpy(low_freq_norm).float().to(device)
        high_freq_tensor = torch.from_numpy(high_freq_norm).float().to(device)
        
        # Resize to target size
        low_freq_resized = F.interpolate(
            low_freq_tensor.unsqueeze(0).unsqueeze(0), 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        high_freq_resized = F.interpolate(
            high_freq_tensor.unsqueeze(0).unsqueeze(0), 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # Combine low and high frequency features
        combined_features = torch.cat([low_freq_resized, high_freq_resized], dim=0)  # [2, H, W]
        
        return combined_features
    
    def forward(self, data):
        """
        Forward pass of FDAFT encoder
        
        Args:
            data: Dictionary containing image batch and metadata
            
        Updates data with:
            - feat_8_0, feat_8_1: Features at 1/8 resolution
            - feat_4_0, feat_4_1: Features at 1/4 resolution  
            - grid_8: Coordinate grids for position encoding
            - Feature dimensions and metadata
        """
        B, _, H, W = data['imagec_0'].shape
        
        # Preprocess images
        image_batch = torch.cat([data['imagec_0'], data['imagec_1']], 0)  # [2B, C, H, W]
        preprocessed = self.preprocess_images(image_batch)
        
        # Process each image through FDAFT
        all_features_8 = []
        all_features_4 = []
        
        for i in range(2 * B):  # Process each image individually
            img = preprocessed[i]  # [C, H, W]
            
            # Extract FDAFT features
            fdaft_features = self.extract_fdaft_features(img)
            
            # Create feature maps at different scales
            # Scale 8: 1/8 resolution for coarse matching
            target_8 = (H // 8, W // 8)
            features_8 = self.create_feature_maps(
                fdaft_features['low_freq_space'],
                fdaft_features['high_freq_space'],
                target_8
            )
            
            # Scale 4: 1/4 resolution for fine matching
            target_4 = (H // 4, W // 4)
            features_4 = self.create_feature_maps(
                fdaft_features['low_freq_space'],
                fdaft_features['high_freq_space'],
                target_4
            )
            
            all_features_8.append(features_8)
            all_features_4.append(features_4)
        
        # Stack features for batch processing
        features_8_batch = torch.stack(all_features_8, dim=0)  # [2B, 2, H/8, W/8]
        features_4_batch = torch.stack(all_features_4, dim=0)  # [2B, 2, H/4, W/4]
        
        # Process through feature convolution networks
        feat_8_processed = self.feature_conv_8(features_8_batch)  # [2B, 256, H/8, W/8]
        feat_4_processed = self.feature_conv_4(features_4_batch)  # [2B, 128, H/4, W/4]
        
        # Split back into two images
        feat_8_0, feat_8_1 = torch.chunk(feat_8_processed, 2, dim=0)  # [B, 256, H/8, W/8]
        feat_4_0, feat_4_1 = torch.chunk(feat_4_processed, 2, dim=0)  # [B, 128, H/4, W/4]
        
        # Create coordinate grids for position encoding
        scale = 8
        h_8, w_8 = H // scale, W // scale
        device = data['imagec_0'].device
        grid = [rearrange(
            (create_meshgrid(h_8, w_8, False, device) * scale).squeeze(0), 
            'h w t->(h w) t'
        )] * B
        grid_8 = torch.stack(grid, 0)
        
        # Update data with FDAFT features
        data.update({
            'bs': B,
            'c': feat_8_0.shape[1],  # Feature dimension
            'h_8': h_8,
            'w_8': w_8,
            'hw_8': h_8 * w_8,
            'feat_8_0': feat_8_0,
            'feat_8_1': feat_8_1,
            'feat_4_0': feat_4_0,
            'feat_4_1': feat_4_1,
            'grid_8': grid_8,
        })
        
    def get_fdaft_info(self):
        """
        Get information about FDAFT configuration
        
        Returns:
            Dictionary with FDAFT encoder details
        """
        return {
            "encoder_type": "FDAFT (Fast Double-Channel Aggregated Feature Transform)",
            "scale_space_layers": self.num_layers,
            "initial_scale": self.sigma_0,
            "structured_forests": self.use_structured_forests,
            "max_keypoints": self.max_keypoints,
            "optimizations": {
                "edge_confidence_map": "Machine learning-based structured edge detection",
                "phase_congruency": "Weighted phase congruency for texture analysis",
                "double_frequency": "Separate low/high frequency processing",
                "planetary_specific": "Optimized for weak textures and gradual transitions"
            },
            "output_features": {
                "feat_8": "256-dim features at 1/8 resolution (coarse)",
                "feat_4": "128-dim features at 1/4 resolution (fine)",
                "scale_spaces": "Low/high frequency scale space representations"
            }
        }