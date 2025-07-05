"""
FDAFT Double-Frequency Scale Space Implementation

This module implements the core scale space operations for FDAFT,
including double-frequency decomposition and multi-scale analysis
optimized for planetary remote sensing images.
"""

import numpy as np
import cv2
from scipy import ndimage, signal
from skimage import filters
import torch


class DoubleFrequencyScaleSpace:
    """
    Double-frequency scale space for FDAFT feature extraction
    
    Separates images into low and high frequency components at multiple scales
    to better handle weak textures and gradual illumination changes in planetary images.
    """
    
    def __init__(self, num_layers: int = 3, sigma_0: float = 1.0):
        """
        Initialize double-frequency scale space
        
        Args:
            num_layers: Number of scale space layers
            sigma_0: Initial scale parameter
        """
        self.num_layers = num_layers
        self.sigma_0 = sigma_0
        self.scale_factor = np.sqrt(2)
        
    def build_low_frequency_scale_space(self, image: np.ndarray) -> list:
        """
        Build low-frequency scale space using Gaussian smoothing
        
        Args:
            image: Input grayscale image [H, W]
            
        Returns:
            List of low-frequency scale space layers
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
            
        scale_space = []
        current_sigma = self.sigma_0
        
        for i in range(self.num_layers):
            # Apply Gaussian smoothing for low-frequency component
            smoothed = cv2.GaussianBlur(image, (0, 0), current_sigma)
            scale_space.append(smoothed)
            current_sigma *= self.scale_factor
            
        return scale_space
    
    def build_high_frequency_scale_space(self, image: np.ndarray) -> list:
        """
        Build high-frequency scale space using difference of Gaussians
        
        Args:
            image: Input grayscale image [H, W]
            
        Returns:
            List of high-frequency scale space layers
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
            
        scale_space = []
        current_sigma = self.sigma_0
        
        for i in range(self.num_layers):
            # Create high-frequency component using DoG
            sigma1 = current_sigma
            sigma2 = current_sigma * 1.6  # Standard DoG ratio
            
            blur1 = cv2.GaussianBlur(image, (0, 0), sigma1)
            blur2 = cv2.GaussianBlur(image, (0, 0), sigma2)
            
            dog = blur1 - blur2
            scale_space.append(dog)
            current_sigma *= self.scale_factor
            
        return scale_space
    
    def compute_phase_congruency(self, image: np.ndarray, scales: list = [1, 2, 4], 
                                orientations: list = [0, 30, 60, 90, 120, 150]) -> np.ndarray:
        """
        Compute phase congruency for edge detection
        
        Args:
            image: Input image
            scales: List of scales for analysis
            orientations: List of orientations in degrees
            
        Returns:
            Phase congruency map
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
            
        h, w = image.shape
        pc_map = np.zeros((h, w))
        
        for scale in scales:
            for orientation in orientations:
                # Create oriented filter
                angle_rad = np.radians(orientation)
                
                # Simple oriented filter (Gabor-like)
                ksize = int(8 * scale) | 1  # Ensure odd size
                kernel = cv2.getGaborKernel((ksize, ksize), scale, angle_rad, 
                                          2*np.pi/scale, 0.5, 0, ktype=cv2.CV_32F)
                
                # Apply filter
                filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
                pc_map += np.abs(filtered)
        
        # Normalize
        pc_map = pc_map / (len(scales) * len(orientations))
        return pc_map
    
    def enhance_weak_textures(self, low_freq: np.ndarray, high_freq: np.ndarray) -> np.ndarray:
        """
        Enhance weak textures using adaptive processing
        
        Args:
            low_freq: Low-frequency component
            high_freq: High-frequency component
            
        Returns:
            Enhanced image with improved weak textures
        """
        # Compute local variance for texture strength estimation
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(low_freq, -1, kernel)
        local_var = cv2.filter2D(low_freq**2, -1, kernel) - local_mean**2
        
        # Adaptive enhancement based on local texture strength
        texture_strength = np.sqrt(local_var + 1e-6)
        enhancement_factor = 1.0 + (1.0 / (texture_strength + 0.1))
        
        # Apply enhancement
        enhanced = low_freq + enhancement_factor * high_freq
        
        return enhanced
    
    def compute_steerable_filters(self, image: np.ndarray, sigma: float = 2.0) -> dict:
        """
        Compute steerable filter responses for oriented feature detection
        
        Args:
            image: Input image
            sigma: Filter scale parameter
            
        Returns:
            Dictionary containing filter responses
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
            
        responses = {}
        
        # First-order steerable filters (derivatives)
        sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        
        # Gradient magnitude and orientation
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        orientation = np.arctan2(sobel_y, sobel_x)
        
        responses['magnitude'] = magnitude
        responses['orientation'] = orientation
        responses['dx'] = sobel_x
        responses['dy'] = sobel_y
        
        # Second-order steerable filters
        # Compute second derivatives for blob detection
        laplacian = cv2.Laplacian(image, cv2.CV_32F)
        responses['laplacian'] = laplacian
        
        return responses
    
    def build_enhanced_scale_space(self, image: np.ndarray) -> dict:
        """
        Build complete enhanced scale space with all components
        
        Args:
            image: Input grayscale image
            
        Returns:
            Dictionary containing all scale space components
        """
        # Build basic scale spaces
        low_freq_space = self.build_low_frequency_scale_space(image)
        high_freq_space = self.build_high_frequency_scale_space(image)
        
        # Compute additional features
        phase_congruency = self.compute_phase_congruency(image)
        steerable_responses = self.compute_steerable_filters(image)
        
        # Enhanced textures for each scale
        enhanced_space = []
        for low, high in zip(low_freq_space, high_freq_space):
            enhanced = self.enhance_weak_textures(low, high)
            enhanced_space.append(enhanced)
        
        return {
            'low_frequency': low_freq_space,
            'high_frequency': high_freq_space,
            'enhanced': enhanced_space,
            'phase_congruency': phase_congruency,
            'steerable': steerable_responses,
            'gradient_magnitude': steerable_responses['magnitude'],
            'gradient_orientation': steerable_responses['orientation']
        }