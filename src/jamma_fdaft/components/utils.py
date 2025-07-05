"""
FDAFT Utility Functions

Shared utility functions for FDAFT components including image processing,
geometric transformations, and planetary-specific optimizations.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union


# Copy utilities from JamMa for consistency
def normalize_keypoints(
        kpts: torch.Tensor,
        size: torch.Tensor) -> torch.Tensor:
    """Normalize keypoints to [-1, 1] range"""
    if not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts


class KeypointEncoder_wo_score(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([2] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        return self.encoder(kpts)


class TransLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(x.transpose(1,2)).transpose(1,2)


def MLP(channels: list, do_bn=True):
    """Multi-layer perceptron"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(TransLN(channels[i]))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class up_conv4(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out):
        super(up_conv4, self).__init__()
        self.lin = nn.Conv2d(dim_in, dim_mid, kernel_size=3, stride=1, padding=1)
        self.inter = nn.Upsample(scale_factor=2, mode='bilinear')
        self.transconv = nn.ConvTranspose2d(dim_in, dim_mid, kernel_size=2, stride=2)
        self.cbr = nn.Sequential(
            nn.Conv2d(dim_mid, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_inter = self.inter(self.lin(x))
        x_conv = self.transconv(x)
        x = self.cbr(x_inter+x_conv)
        return x


class MLPMixerEncoderLayer(nn.Module):
    def __init__(self, dim1, dim2, factor=1):
        super(MLPMixerEncoderLayer, self).__init__()

        self.mlp1 = nn.Sequential(nn.Linear(dim1, dim1*factor),
                                  nn.GELU(),
                                  nn.Linear(dim1*factor, dim1))
        self.mlp2 = nn.Sequential(nn.Linear(dim2, dim2*factor),
                                  nn.LayerNorm(dim2*factor),
                                  nn.GELU(),
                                  nn.Linear(dim2*factor, dim2))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [N, L, C]
        """
        x = x + self.mlp1(x)
        x = x.transpose(1, 2)
        x = x + self.mlp2(x)
        return x.transpose(1, 2)


# FDAFT-specific utility functions

def adaptive_threshold(image: np.ndarray, factor: float = 0.5) -> float:
    """
    Compute adaptive threshold for planetary images
    
    Args:
        image: Input image
        factor: Threshold factor
        
    Returns:
        Adaptive threshold value
    """
    # Use Otsu's method as base
    threshold, _ = cv2.threshold(
        (image * 255).astype(np.uint8), 0, 255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # Adapt for planetary images (typically lower contrast)
    adaptive_threshold = (threshold / 255.0) * factor
    
    return adaptive_threshold


def enhance_planetary_contrast(image: np.ndarray, 
                             clip_limit: float = 2.0,
                             tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Enhance contrast for planetary images using CLAHE
    
    Args:
        image: Input image
        clip_limit: CLAHE clip limit
        tile_grid_size: Grid size for CLAHE
        
    Returns:
        Contrast-enhanced image
    """
    if image.dtype != np.uint8:
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image_uint8 = image
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image_uint8)
    
    # Convert back to float
    enhanced = enhanced.astype(np.float32) / 255.0
    
    return enhanced


def suppress_illumination_gradients(image: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """
    Suppress gradual illumination changes typical in planetary images
    
    Args:
        image: Input image
        sigma: Gaussian blur sigma for background estimation
        
    Returns:
        Illumination-corrected image
    """
    # Estimate background illumination
    background = cv2.GaussianBlur(image, (0, 0), sigma)
    
    # Subtract background (with epsilon to avoid division by zero)
    corrected = image - background + 0.5
    corrected = np.clip(corrected, 0, 1)
    
    return corrected


def detect_texture_regions(image: np.ndarray, 
                          window_size: int = 15,
                          texture_threshold: float = 0.01) -> np.ndarray:
    """
    Detect regions with sufficient texture for feature extraction
    
    Args:
        image: Input image
        window_size: Window size for texture analysis
        texture_threshold: Minimum texture strength
        
    Returns:
        Binary mask of textured regions
    """
    # Compute local variance as texture measure
    kernel = np.ones((window_size, window_size), np.float32) / (window_size**2)
    
    # Local mean and variance
    local_mean = cv2.filter2D(image, -1, kernel)
    local_var = cv2.filter2D(image**2, -1, kernel) - local_mean**2
    
    # Create texture mask
    texture_mask = local_var > texture_threshold
    
    return texture_mask.astype(np.uint8)


def polar_transform(image: np.ndarray, center: Tuple[int, int], 
                   max_radius: int) -> np.ndarray:
    """
    Apply polar transformation for circular structure analysis
    
    Args:
        image: Input image
        center: Center point (x, y)
        max_radius: Maximum radius for transformation
        
    Returns:
        Polar-transformed image
    """
    h, w = image.shape[:2]
    cx, cy = center
    
    # Create polar coordinate mapping
    polar_h, polar_w = max_radius, 360
    polar_image = np.zeros((polar_h, polar_w), dtype=image.dtype)
    
    for r in range(polar_h):
        for theta in range(polar_w):
            # Convert polar to Cartesian
            angle_rad = theta * np.pi / 180.0
            x = int(cx + r * np.cos(angle_rad))
            y = int(cy + r * np.sin(angle_rad))
            
            # Check bounds and copy pixel
            if 0 <= x < w and 0 <= y < h:
                polar_image[r, theta] = image[y, x]
    
    return polar_image


def compute_image_statistics(image: np.ndarray) -> dict:
    """
    Compute comprehensive image statistics for planetary image analysis
    
    Args:
        image: Input image
        
    Returns:
        Dictionary of image statistics
    """
    stats = {
        'mean': np.mean(image),
        'std': np.std(image),
        'min': np.min(image),
        'max': np.max(image),
        'contrast': np.std(image) / (np.mean(image) + 1e-8),
        'dynamic_range': np.max(image) - np.min(image)
    }
    
    # Compute gradient statistics
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    stats['gradient_mean'] = np.mean(gradient_magnitude)
    stats['gradient_std'] = np.std(gradient_magnitude)
    
    # Texture strength
    texture_mask = detect_texture_regions(image)
    stats['texture_coverage'] = np.mean(texture_mask)
    
    return stats


def planetary_image_quality_score(image: np.ndarray) -> float:
    """
    Compute quality score for planetary images
    
    Args:
        image: Input image
        
    Returns:
        Quality score (0-1, higher is better)
    """
    stats = compute_image_statistics(image)
    
    # Quality factors
    contrast_score = min(stats['contrast'] * 2, 1.0)  # Prefer moderate contrast
    texture_score = stats['texture_coverage']
    dynamic_range_score = min(stats['dynamic_range'] * 2, 1.0)
    
    # Combine scores
    quality_score = (contrast_score + texture_score + dynamic_range_score) / 3.0
    
    return quality_score


def apply_planetary_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    Apply comprehensive preprocessing for planetary images
    
    Args:
        image: Input image
        
    Returns:
        Preprocessed image
    """
    # Ensure float32 format
    if image.dtype != np.float32:
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
    
    # Step 1: Suppress illumination gradients
    image = suppress_illumination_gradients(image)
    
    # Step 2: Enhance contrast adaptively
    image = enhance_planetary_contrast(image)
    
    # Step 3: Noise reduction (gentle)
    image = cv2.GaussianBlur(image, (3, 3), 0.5)
    
    return image


def create_multiscale_pyramid(image: np.ndarray, 
                             num_levels: int = 4,
                             scale_factor: float = 0.5) -> List[np.ndarray]:
    """
    Create multi-scale image pyramid
    
    Args:
        image: Input image
        num_levels: Number of pyramid levels
        scale_factor: Scale factor between levels
        
    Returns:
        List of pyramid levels
    """
    pyramid = [image]
    current_image = image
    
    for i in range(num_levels - 1):
        # Gaussian blur and downsample
        blurred = cv2.GaussianBlur(current_image, (5, 5), 1.0)
        
        new_size = (
            int(current_image.shape[1] * scale_factor),
            int(current_image.shape[0] * scale_factor)
        )
        
        downsampled = cv2.resize(blurred, new_size, interpolation=cv2.INTER_AREA)
        pyramid.append(downsampled)
        current_image = downsampled
    
    return pyramid


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy image
    
    Args:
        tensor: Input tensor [C, H, W] or [H, W]
        
    Returns:
        Numpy image
    """
    if tensor.dim() == 3:
        # Convert CHW to HWC
        image = tensor.permute(1, 2, 0).cpu().numpy()
        if image.shape[2] == 1:
            image = image.squeeze(2)
    else:
        image = tensor.cpu().numpy()
    
    # Ensure proper range
    if image.max() <= 1.0:
        image = np.clip(image, 0, 1)
    else:
        image = np.clip(image / 255.0, 0, 1)
    
    return image.astype(np.float32)


def numpy_to_tensor_image(image: np.ndarray) -> torch.Tensor:
    """
    Convert numpy image to PyTorch tensor
    
    Args:
        image: Input numpy image [H, W] or [H, W, C]
        
    Returns:
        PyTorch tensor [C, H, W] or [1, H, W]
    """
    if image.ndim == 2:
        # Add channel dimension
        tensor = torch.from_numpy(image).unsqueeze(0)
    else:
        # Convert HWC to CHW
        tensor = torch.from_numpy(image).permute(2, 0, 1)
    
    return tensor.float()