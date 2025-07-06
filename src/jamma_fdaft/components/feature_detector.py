"""
FDAFT Feature Detector Implementation

This module implements feature detection algorithms optimized for planetary remote sensing images,
including corner detection, blob detection, and structured edge detection.
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import feature, measure
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import maximum_filter
import pickle
import os


def find_local_maxima(image, min_distance=5, threshold_abs=0.01):
    """
    Find local maxima in an image - scikit-image independent implementation
    
    Args:
        image: Input image
        min_distance: Minimum distance between peaks
        threshold_abs: Minimum threshold for peak detection
        
    Returns:
        List of (row, col) coordinates of local maxima
    """
    # Apply threshold
    mask = image > threshold_abs
    
    # Find local maxima using maximum filter
    local_maxima = maximum_filter(image, size=min_distance) == image
    
    # Combine threshold and local maxima
    peaks = mask & local_maxima
    
    # Get coordinates
    coords = np.where(peaks)
    if len(coords[0]) > 0:
        return list(zip(coords[0], coords[1]))
    else:
        return []


class FeatureDetector:
    """
    FDAFT feature detector for planetary remote sensing images
    
    Combines multiple detection strategies optimized for weak textures,
    circular structures (craters), and gradual illumination changes.
    """
    
    def __init__(self, nms_radius: int = 5, max_keypoints: int = 2000, 
                 use_kaze: bool = True, fast_threshold: float = 0.05):
        """
        Initialize feature detector
        
        Args:
            nms_radius: Non-maximum suppression radius
            max_keypoints: Maximum number of keypoints to extract
            use_kaze: Whether to use KAZE detector
            fast_threshold: Threshold for FAST detector
        """
        self.nms_radius = nms_radius
        self.max_keypoints = max_keypoints
        self.use_kaze = use_kaze
        self.fast_threshold = fast_threshold
        
        # Initialize detectors
        self._setup_detectors()
        
        # Structured Forests model (placeholder - would load pre-trained model)
        self.structured_forests_model = None
        
    def _setup_detectors(self):
        """Setup various feature detectors"""
        # FAST detector for corner detection
        self.fast_detector = cv2.FastFeatureDetector_create(
            threshold=int(self.fast_threshold * 255),
            nonmaxSuppression=True
        )
        
        # KAZE detector for robust features
        if self.use_kaze:
            self.kaze_detector = cv2.KAZE_create(
                threshold=0.001,
                nOctaves=4,
                nOctaveLayers=4
            )
        
        # ORB detector as backup
        self.orb_detector = cv2.ORB_create(nfeatures=self.max_keypoints)
    
    def load_structured_forests_model(self, model_path: str):
        """
        Load pre-trained Structured Forests model
        
        Args:
            model_path: Path to the pre-trained model file
        """
        if os.path.exists(model_path):
            try:
                # For OpenCV structured forests
                self.structured_forests_model = cv2.ximgproc.createStructuredEdgeDetection(model_path)
                print(f"Loaded Structured Forests model from {model_path}")
            except:
                print(f"Failed to load Structured Forests model from {model_path}")
                self.structured_forests_model = None
        else:
            print(f"Structured Forests model not found at {model_path}")
            self.structured_forests_model = None
    
    def extract_corner_points(self, scale_space: list, use_structured_forests: bool = True) -> tuple:
        """
        Extract corner points from scale space
        
        Args:
            scale_space: List of scale space layers
            use_structured_forests: Whether to use structured forests for refinement
            
        Returns:
            Tuple of (corner_points, corner_scores)
        """
        all_corners = []
        all_scores = []
        
        for i, layer in enumerate(scale_space):
            # Convert to uint8 for OpenCV detectors
            if layer.dtype != np.uint8:
                layer_uint8 = (np.clip(layer, 0, 1) * 255).astype(np.uint8)
            else:
                layer_uint8 = layer
            
            # FAST corner detection
            fast_kpts = self.fast_detector.detect(layer_uint8)
            
            # Harris corner detection for comparison
            harris_response = cv2.cornerHarris(layer_uint8, 2, 3, 0.04)
            harris_response = cv2.dilate(harris_response, None)
            
            # KAZE features if enabled
            kaze_kpts = []
            if self.use_kaze and hasattr(self, 'kaze_detector'):
                try:
                    kaze_kpts = self.kaze_detector.detect(layer_uint8)
                except:
                    kaze_kpts = []
            
            # Combine corner points
            corners = []
            scores = []
            
            # Process FAST corners
            for kpt in fast_kpts:
                x, y = int(kpt.pt[0]), int(kpt.pt[1])
                if self._is_valid_point(x, y, layer.shape):
                    corners.append([x, y, i])  # Include scale index
                    scores.append(kpt.response)
            
            # Process KAZE corners
            for kpt in kaze_kpts:
                x, y = int(kpt.pt[0]), int(kpt.pt[1])
                if self._is_valid_point(x, y, layer.shape):
                    corners.append([x, y, i])
                    scores.append(kpt.response)
            
            # Harris corners (local maxima)
            if len(corners) < self.max_keypoints // len(scale_space):
                # Use our custom local maxima finder
                harris_corners = find_local_maxima(
                    harris_response, 
                    min_distance=self.nms_radius,
                    threshold_abs=0.01 * harris_response.max()
                )
                
                for y, x in harris_corners:
                    if self._is_valid_point(x, y, layer.shape):
                        corners.append([x, y, i])
                        scores.append(harris_response[y, x])
            
            if corners:
                all_corners.extend(corners)
                all_scores.extend(scores)
        
        # Apply structured forests refinement if available
        if use_structured_forests and self.structured_forests_model is not None:
            all_corners, all_scores = self._refine_with_structured_forests(
                all_corners, all_scores, scale_space[0]
            )
        
        # Non-maximum suppression and selection of top corners
        if all_corners:
            corners_array = np.array(all_corners)
            scores_array = np.array(all_scores)
            
            # Sort by score
            sorted_idx = np.argsort(scores_array)[::-1]
            corners_array = corners_array[sorted_idx]
            scores_array = scores_array[sorted_idx]
            
            # Apply NMS
            final_corners, final_scores = self._apply_nms(
                corners_array, scores_array, self.nms_radius
            )
            
            # Limit to max keypoints
            if len(final_corners) > self.max_keypoints:
                final_corners = final_corners[:self.max_keypoints]
                final_scores = final_scores[:self.max_keypoints]
                
            return final_corners, final_scores
        
        return np.array([]), np.array([])
    
    def extract_blob_points(self, scale_space: list) -> tuple:
        """
        Extract blob points for circular structure detection (e.g., craters)
        
        Args:
            scale_space: List of high-frequency scale space layers
            
        Returns:
            Tuple of (blob_points, blob_scores)
        """
        all_blobs = []
        all_scores = []
        
        for i, layer in enumerate(scale_space):
            # Convert to appropriate range
            if layer.dtype != np.float32:
                layer = layer.astype(np.float32)
            
            # Normalize layer
            layer_norm = (layer - layer.min()) / (layer.max() - layer.min() + 1e-8)
            
            # Laplacian of Gaussian for blob detection
            sigma = 1.0 * (2 ** i)  # Scale-dependent sigma
            log_response = -sigma**2 * ndimage.gaussian_laplace(layer_norm, sigma)
            
            # Find local maxima using our custom implementation
            blob_candidates = find_local_maxima(
                log_response,
                min_distance=self.nms_radius,
                threshold_abs=0.01
            )
            
            # SIFT-like blob detector
            try:
                layer_uint8 = (np.clip(layer_norm, 0, 1) * 255).astype(np.uint8)
                sift = cv2.SIFT_create()
                sift_kpts = sift.detect(layer_uint8)
                
                for kpt in sift_kpts:
                    x, y = int(kpt.pt[0]), int(kpt.pt[1])
                    if self._is_valid_point(x, y, layer.shape):
                        all_blobs.append([x, y, i, kpt.size])
                        all_scores.append(kpt.response)
                        
            except:
                # Fallback to LoG blobs
                for y, x in blob_candidates:
                    if self._is_valid_point(x, y, layer.shape):
                        all_blobs.append([x, y, i, sigma])
                        all_scores.append(log_response[y, x])
        
        # Select and refine blob points
        if all_blobs:
            blobs_array = np.array(all_blobs)
            scores_array = np.array(all_scores)
            
            # Sort by score
            sorted_idx = np.argsort(scores_array)[::-1]
            blobs_array = blobs_array[sorted_idx]
            scores_array = scores_array[sorted_idx]
            
            # Apply NMS
            final_blobs, final_scores = self._apply_nms(
                blobs_array, scores_array, self.nms_radius * 2
            )
            
            # Limit to max keypoints
            max_blobs = self.max_keypoints // 2  # Reserve half for blobs
            if len(final_blobs) > max_blobs:
                final_blobs = final_blobs[:max_blobs]
                final_scores = final_scores[:max_blobs]
                
            return final_blobs, final_scores
        
        return np.array([]), np.array([])
    
    def _refine_with_structured_forests(self, corners: list, scores: list, 
                                      reference_image: np.ndarray) -> tuple:
        """
        Refine corner points using Structured Forests edge detection
        
        Args:
            corners: List of corner points
            scores: List of corner scores
            reference_image: Reference image for edge detection
            
        Returns:
            Refined corners and scores
        """
        if self.structured_forests_model is None:
            return corners, scores
        
        try:
            # Convert image for structured forests
            if reference_image.dtype != np.float32:
                img = reference_image.astype(np.float32) / 255.0
            else:
                img = reference_image
            
            # Detect edges using structured forests
            edges = self.structured_forests_model.detectEdges(img)
            
            # Refine corners based on edge proximity
            refined_corners = []
            refined_scores = []
            
            for i, corner in enumerate(corners):
                x, y = int(corner[0]), int(corner[1])
                
                # Check local edge strength
                patch_size = 5
                y1, y2 = max(0, y - patch_size), min(edges.shape[0], y + patch_size + 1)
                x1, x2 = max(0, x - patch_size), min(edges.shape[1], x + patch_size + 1)
                
                local_edge_strength = np.mean(edges[y1:y2, x1:x2])
                
                # Keep corners with sufficient edge support
                if local_edge_strength > 0.1:  # Threshold for edge strength
                    refined_corners.append(corner)
                    refined_scores.append(scores[i] * local_edge_strength)
            
            return refined_corners, refined_scores
            
        except Exception as e:
            print(f"Structured Forests refinement failed: {e}")
            return corners, scores
    
    def _apply_nms(self, points: np.ndarray, scores: np.ndarray, radius: int) -> tuple:
        """
        Apply non-maximum suppression to remove nearby points
        
        Args:
            points: Array of points [N, >=2]
            scores: Array of scores [N]
            radius: Suppression radius
            
        Returns:
            Filtered points and scores
        """
        if len(points) == 0:
            return points, scores
        
        # Sort by score (descending)
        sorted_idx = np.argsort(scores)[::-1]
        sorted_points = points[sorted_idx]
        sorted_scores = scores[sorted_idx]
        
        keep = []
        suppressed = np.zeros(len(sorted_points), dtype=bool)
        
        for i in range(len(sorted_points)):
            if suppressed[i]:
                continue
                
            keep.append(i)
            
            # Suppress nearby points
            for j in range(i + 1, len(sorted_points)):
                if suppressed[j]:
                    continue
                    
                dist = np.sqrt(
                    (sorted_points[i, 0] - sorted_points[j, 0])**2 + 
                    (sorted_points[i, 1] - sorted_points[j, 1])**2
                )
                
                if dist < radius:
                    suppressed[j] = True
        
        return sorted_points[keep], sorted_scores[keep]
    
    def _is_valid_point(self, x: int, y: int, shape: tuple, border: int = 10) -> bool:
        """
        Check if point is within valid image bounds
        
        Args:
            x, y: Point coordinates
            shape: Image shape (H, W)
            border: Border margin
            
        Returns:
            True if point is valid
        """
        h, w = shape[:2]
        return (border <= x < w - border and 
                border <= y < h - border)
    
    def detect_crater_structures(self, image: np.ndarray, 
                               min_radius: int = 10, max_radius: int = 100) -> list:
        """
        Detect circular structures (craters) using Hough transform
        
        Args:
            image: Input image
            min_radius: Minimum crater radius
            max_radius: Maximum crater radius
            
        Returns:
            List of detected circles [(x, y, radius), ...]
        """
        if image.dtype != np.uint8:
            image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        # Edge detection for Hough transform
        edges = cv2.Canny(image_uint8, 50, 150)
        
        # Hough circle detection
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_radius * 2,
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        crater_list = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                crater_list.append((x, y, r))
        
        return crater_list