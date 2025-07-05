"""
FDAFT GLOH (Gradient Location and Orientation Histogram) Descriptor Implementation

This module implements GLOH descriptors optimized for planetary remote sensing images,
providing robust feature descriptions for weak textures and gradual illumination changes.
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import feature
import math


class GLOHDescriptor:
    """
    GLOH descriptor for FDAFT feature extraction
    
    Provides robust feature descriptions using gradient location and orientation histograms,
    optimized for planetary remote sensing applications.
    """
    
    def __init__(self, 
                 radial_bins: int = 3,
                 angular_bins: int = 8, 
                 orientation_bins: int = 16,
                 patch_size: int = 64,
                 gradient_threshold: float = 0.1):
        """
        Initialize GLOH descriptor
        
        Args:
            radial_bins: Number of radial bins in the descriptor
            angular_bins: Number of angular bins per radial ring
            orientation_bins: Number of orientation bins for histogram
            patch_size: Size of the patch around each keypoint
            gradient_threshold: Threshold for gradient magnitude
        """
        self.radial_bins = radial_bins
        self.angular_bins = angular_bins
        self.orientation_bins = orientation_bins
        self.patch_size = patch_size
        self.gradient_threshold = gradient_threshold
        
        # Pre-compute bin assignments for efficiency
        self._precompute_bin_assignments()
        
    def _precompute_bin_assignments(self):
        """Pre-compute spatial bin assignments for the descriptor patch"""
        center = self.patch_size // 2
        
        # Create coordinate grids
        y, x = np.mgrid[-center:center+1, -center:center+1]
        
        # Convert to polar coordinates
        self.distances = np.sqrt(x**2 + y**2)
        self.angles = np.arctan2(y, x)
        
        # Normalize distances to [0, radial_bins]
        max_distance = center
        self.radial_indices = np.floor(
            self.distances / max_distance * self.radial_bins
        ).astype(int)
        self.radial_indices = np.clip(self.radial_indices, 0, self.radial_bins - 1)
        
        # Convert angles to [0, 2π] and bin them
        self.angles = (self.angles + np.pi) % (2 * np.pi)
        self.angular_indices = np.floor(
            self.angles / (2 * np.pi) * self.angular_bins
        ).astype(int)
        self.angular_indices = np.clip(self.angular_indices, 0, self.angular_bins - 1)
        
    def compute_keypoint_orientation(self, image: np.ndarray, keypoint: tuple) -> float:
        """
        Compute dominant orientation for a keypoint
        
        Args:
            image: Input image
            keypoint: Keypoint coordinates (x, y)
            
        Returns:
            Dominant orientation in radians
        """
        x, y = int(keypoint[0]), int(keypoint[1])
        center = self.patch_size // 2
        
        # Extract patch around keypoint
        y1, y2 = y - center, y + center + 1
        x1, x2 = x - center, x + center + 1
        
        # Handle boundary conditions
        if (y1 < 0 or y2 >= image.shape[0] or 
            x1 < 0 or x2 >= image.shape[1]):
            return 0.0
        
        patch = image[y1:y2, x1:x2].astype(np.float32)
        
        # Compute gradients
        grad_x = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        
        # Compute gradient magnitude and orientation
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x)
        
        # Create orientation histogram
        hist = np.zeros(self.orientation_bins)
        
        # Weight by magnitude and distance from center
        weights = magnitude * np.exp(-self.distances / (2 * (center / 3)**2))
        
        # Bin orientations
        orientation_indices = np.floor(
            (orientation + np.pi) / (2 * np.pi) * self.orientation_bins
        ).astype(int)
        orientation_indices = np.clip(orientation_indices, 0, self.orientation_bins - 1)
        
        # Accumulate weighted votes
        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                if magnitude[i, j] > self.gradient_threshold:
                    hist[orientation_indices[i, j]] += weights[i, j]
        
        # Find dominant orientation
        dominant_bin = np.argmax(hist)
        dominant_orientation = (dominant_bin / self.orientation_bins) * 2 * np.pi - np.pi
        
        return dominant_orientation
    
    def compute_descriptor(self, image: np.ndarray, keypoint: tuple, 
                          orientation: float = None) -> np.ndarray:
        """
        Compute GLOH descriptor for a keypoint
        
        Args:
            image: Input image
            keypoint: Keypoint coordinates (x, y)
            orientation: Keypoint orientation (if None, will be computed)
            
        Returns:
            GLOH descriptor vector
        """
        x, y = int(keypoint[0]), int(keypoint[1])
        center = self.patch_size // 2
        
        # Compute orientation if not provided
        if orientation is None:
            orientation = self.compute_keypoint_orientation(image, keypoint)
        
        # Extract patch around keypoint
        y1, y2 = y - center, y + center + 1
        x1, x2 = x - center, x + center + 1
        
        # Handle boundary conditions
        if (y1 < 0 or y2 >= image.shape[0] or 
            x1 < 0 or x2 >= image.shape[1]):
            # Return zero descriptor for boundary keypoints
            total_bins = self.radial_bins * self.angular_bins * self.orientation_bins
            return np.zeros(total_bins)
        
        patch = image[y1:y2, x1:x2].astype(np.float32)
        
        # Compute gradients
        grad_x = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        
        # Rotate gradients to align with keypoint orientation
        cos_theta = np.cos(-orientation)  # Negative for alignment
        sin_theta = np.sin(-orientation)
        
        grad_x_rot = grad_x * cos_theta - grad_y * sin_theta
        grad_y_rot = grad_x * sin_theta + grad_y * cos_theta
        
        # Compute rotated gradient magnitude and orientation
        magnitude = np.sqrt(grad_x_rot**2 + grad_y_rot**2)
        grad_orientation = np.arctan2(grad_y_rot, grad_x_rot)
        
        # Initialize descriptor
        descriptor = np.zeros(
            (self.radial_bins, self.angular_bins, self.orientation_bins)
        )
        
        # Apply Gaussian weighting
        sigma = self.patch_size / 6.0
        gaussian_weights = np.exp(-self.distances**2 / (2 * sigma**2))
        
        # Bin gradient orientations
        grad_orientation_indices = np.floor(
            (grad_orientation + np.pi) / (2 * np.pi) * self.orientation_bins
        ).astype(int)
        grad_orientation_indices = np.clip(
            grad_orientation_indices, 0, self.orientation_bins - 1
        )
        
        # Accumulate gradients into spatial and orientation bins
        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                if magnitude[i, j] > self.gradient_threshold:
                    # Get spatial bin indices
                    rad_bin = self.radial_indices[i, j]
                    ang_bin = self.angular_indices[i, j]
                    ori_bin = grad_orientation_indices[i, j]
                    
                    # Weight by magnitude and Gaussian
                    weight = magnitude[i, j] * gaussian_weights[i, j]
                    
                    # Accumulate into descriptor
                    descriptor[rad_bin, ang_bin, ori_bin] += weight
        
        # Flatten descriptor
        descriptor = descriptor.flatten()
        
        # Normalize descriptor
        descriptor = self._normalize_descriptor(descriptor)
        
        return descriptor
    
    def _normalize_descriptor(self, descriptor: np.ndarray) -> np.ndarray:
        """
        Normalize descriptor to achieve illumination invariance
        
        Args:
            descriptor: Raw descriptor vector
            
        Returns:
            Normalized descriptor
        """
        # L2 normalization
        norm = np.linalg.norm(descriptor)
        if norm > 1e-8:
            descriptor = descriptor / norm
        
        # Clip values to reduce influence of large gradients
        descriptor = np.clip(descriptor, 0, 0.2)
        
        # Re-normalize
        norm = np.linalg.norm(descriptor)
        if norm > 1e-8:
            descriptor = descriptor / norm
        
        return descriptor
    
    def compute_descriptors_batch(self, image: np.ndarray, 
                                keypoints: np.ndarray) -> np.ndarray:
        """
        Compute GLOH descriptors for multiple keypoints
        
        Args:
            image: Input image
            keypoints: Array of keypoints [N, 2] or [N, 3+]
            
        Returns:
            Array of descriptors [N, descriptor_length]
        """
        if len(keypoints) == 0:
            total_bins = self.radial_bins * self.angular_bins * self.orientation_bins
            return np.zeros((0, total_bins))
        
        descriptors = []
        
        for i, keypoint in enumerate(keypoints):
            # Extract x, y coordinates
            x, y = keypoint[0], keypoint[1]
            
            # Compute descriptor
            descriptor = self.compute_descriptor(image, (x, y))
            descriptors.append(descriptor)
        
        return np.array(descriptors)
    
    def enhance_descriptor_for_planetary_images(self, descriptor: np.ndarray) -> np.ndarray:
        """
        Apply planetary-specific enhancements to descriptor
        
        Args:
            descriptor: Original GLOH descriptor
            
        Returns:
            Enhanced descriptor
        """
        # Apply contrast enhancement for weak textures
        enhanced = descriptor.copy()
        
        # Adaptive histogram equalization effect
        # Enhance low-magnitude components
        low_magnitude_mask = enhanced < 0.1
        enhanced[low_magnitude_mask] *= 1.5
        
        # Suppress noise in very small components
        noise_mask = enhanced < 0.02
        enhanced[noise_mask] *= 0.5
        
        # Re-normalize
        enhanced = self._normalize_descriptor(enhanced)
        
        return enhanced
    
    def compute_enhanced_descriptor(self, image: np.ndarray, keypoint: tuple,
                                  orientation: float = None) -> np.ndarray:
        """
        Compute enhanced GLOH descriptor optimized for planetary images
        
        Args:
            image: Input image
            keypoint: Keypoint coordinates (x, y)
            orientation: Keypoint orientation
            
        Returns:
            Enhanced GLOH descriptor
        """
        # Compute standard GLOH descriptor
        descriptor = self.compute_descriptor(image, keypoint, orientation)
        
        # Apply planetary-specific enhancements
        enhanced_descriptor = self.enhance_descriptor_for_planetary_images(descriptor)
        
        return enhanced_descriptor
    
    def get_descriptor_length(self) -> int:
        """
        Get the length of the descriptor vector
        
        Returns:
            Descriptor length
        """
        return self.radial_bins * self.angular_bins * self.orientation_bins
    
    def match_descriptors(self, desc1: np.ndarray, desc2: np.ndarray, 
                         ratio_threshold: float = 0.8) -> np.ndarray:
        """
        Match descriptors using ratio test
        
        Args:
            desc1: First set of descriptors [N, D]
            desc2: Second set of descriptors [M, D]
            ratio_threshold: Lowe's ratio test threshold
            
        Returns:
            Array of matches [K, 2] where each row is [idx1, idx2]
        """
        if len(desc1) == 0 or len(desc2) == 0:
            return np.array([]).reshape(0, 2)
        
        # Compute pairwise distances
        distances = np.linalg.norm(
            desc1[:, np.newaxis] - desc2[np.newaxis, :], axis=2
        )
        
        matches = []
        
        for i in range(len(desc1)):
            # Find two nearest neighbors
            sorted_indices = np.argsort(distances[i])
            
            if len(sorted_indices) >= 2:
                nearest_dist = distances[i, sorted_indices[0]]
                second_nearest_dist = distances[i, sorted_indices[1]]
                
                # Apply ratio test
                if nearest_dist < ratio_threshold * second_nearest_dist:
                    matches.append([i, sorted_indices[0]])
        
        return np.array(matches) if matches else np.array([]).reshape(0, 2)