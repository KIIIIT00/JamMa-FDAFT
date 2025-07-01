import numpy as np
import cv2
from skimage.feature import corner_fast
from typing import List, Tuple

# Try different import paths for peak_local_maxima
try:
    from skimage.feature.peak import peak_local_maxima
except ImportError:
    try:
        from skimage.feature import peak_local_maxima
    except ImportError:
        # Fallback: implement our own peak detection
        def peak_local_maxima(image, min_distance=1, threshold_abs=None, num_peaks=np.inf):
            """Fallback implementation of peak_local_maxima"""
            from scipy.ndimage import maximum_filter
            
            if threshold_abs is None:
                threshold_abs = 0.0
            
            # Apply threshold
            mask = image >= threshold_abs
            
            # Find local maxima using maximum filter
            local_maxima = maximum_filter(image, size=min_distance*2+1) == image
            
            # Combine masks
            peaks_mask = mask & local_maxima
            
            # Get coordinates
            coords = np.where(peaks_mask)
            
            if len(coords[0]) == 0:
                return []
            
            # Sort by intensity and limit number
            intensities = image[coords]
            sorted_indices = np.argsort(intensities)[::-1]
            
            if num_peaks < len(sorted_indices):
                sorted_indices = sorted_indices[:int(num_peaks)]
            
            return list(zip(coords[0][sorted_indices], coords[1][sorted_indices]))
        
class FeatureDetector:
    """
    Feature point detection for FDAFT
    Extracts corner points from low-frequency space and blob points from high-frequency space
    """
    
    def __init__(self, nms_radius: int = 5, max_keypoints: int = 1000):
        """
        Initialize feature detector
        
        Args:
            nms_radius: Radius for non-maximum suppression
            max_keypoints: Maximum number of keypoints to retain
        """
        self.nms_radius = nms_radius
        self.max_keypoints = max_keypoints
        
    def extract_corner_points(self, low_freq_space: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract corner points using FAST detector from low-frequency images
        
        Args:
            low_freq_space: List of low-frequency scale space layers
            
        Returns:
            corner_points: Array of corner point coordinates [N, 2]
            corner_scores: Array of corner point scores [N]
        """
        all_corners = []
        all_scores = []
        
        for layer_idx, layer in enumerate(low_freq_space):
            # Convert to uint8 for corner detection
            layer_uint8 = ((layer - layer.min()) / 
                          (layer.max() - layer.min()) * 255).astype(np.uint8)
            
            # Try FAST corner detection first
            corners = self._extract_fast_corners(layer_uint8)
            
            # If FAST fails, fallback to goodFeaturesToTrack
            if len(corners) == 0:
                corners = self._extract_good_features(layer_uint8)
            
            if len(corners) > 0:
                # Compute corner scores (Harris-like response)
                scores = self._compute_corner_scores(layer, corners)
                
                # Weight by layer (higher layers get higher weights)
                layer_weight = layer_idx + 1
                scores = scores * layer_weight
                
                all_corners.extend(corners)
                all_scores.extend(scores)
        
        if not all_corners:
            return np.empty((0, 2)), np.array([])
            
        corners = np.array(all_corners)
        scores = np.array(all_scores)
        
        # Apply non-maximum suppression
        corners, scores = self._non_maximum_suppression(corners, scores)
        
        # Keep top keypoints
        if len(corners) > self.max_keypoints:
            top_indices = np.argsort(scores)[-self.max_keypoints:]
            corners = corners[top_indices]
            scores = scores[top_indices]
            
        return corners, scores
    
    def _extract_fast_corners(self, image: np.ndarray) -> np.ndarray:
        """Extract FAST corners with robust error handling"""
        try:
            corners = corner_fast(image, n=9, threshold=0.05)
            
            # Handle different return formats from corner_fast
            if isinstance(corners, tuple):
                # Some versions return (y_coords, x_coords)
                if len(corners) == 2:
                    y_coords, x_coords = corners
                    if len(y_coords) > 0 and len(x_coords) > 0:
                        corners = np.column_stack((y_coords, x_coords))
                    else:
                        corners = np.empty((0, 2))
                else:
                    corners = np.empty((0, 2))
            elif isinstance(corners, np.ndarray):
                # Some versions return array of coordinates
                if corners.ndim == 1:
                    # If 1D, convert appropriately
                    corners = np.empty((0, 2))
                elif corners.ndim == 2 and corners.shape[1] == 2:
                    # Correct format
                    pass
                else:
                    corners = np.empty((0, 2))
            else:
                corners = np.empty((0, 2))
                
        except Exception as e:
            print(f"Warning: FAST corner detection failed: {e}")
            corners = np.empty((0, 2))
        
        return corners
    
    def _extract_good_features(self, image: np.ndarray) -> np.ndarray:
        """Fallback corner detection using OpenCV goodFeaturesToTrack"""
        try:
            # Use OpenCV's goodFeaturesToTrack as fallback
            max_corners = self.max_keypoints // 3  # Divide by number of layers
            corners = cv2.goodFeaturesToTrack(
                image,
                maxCorners=max_corners,
                qualityLevel=0.01,
                minDistance=self.nms_radius,
                useHarrisDetector=True,
                k=0.04
            )
            
            if corners is not None:
                # Convert from (x, y) to (y, x) format and reshape
                corners = corners.reshape(-1, 2)
                corners = corners[:, [1, 0]]  # Swap x,y to y,x
            else:
                corners = np.empty((0, 2))
                
        except Exception as e:
            print(f"Warning: goodFeaturesToTrack failed: {e}")
            corners = np.empty((0, 2))
        
        return corners
        
    def extract_blob_points(self, high_freq_space: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract blob points using KAZE-like detector from high-frequency images
        
        Args:
            high_freq_space: List of high-frequency scale space layers
            
        Returns:
            blob_points: Array of blob point coordinates [N, 2]
            blob_scores: Array of blob point scores [N]
        """
        all_blobs = []
        all_scores = []
        
        for layer_idx, layer in enumerate(high_freq_space):
            # Normalize layer
            normalized = (layer - layer.min()) / (layer.max() - layer.min() + 1e-8)
            
            # Find local maxima with robust handling
            try:
                coordinates = peak_local_maxima(
                    normalized, 
                    min_distance=self.nms_radius, 
                    threshold_abs=0.05,
                    num_peaks=self.max_keypoints // len(high_freq_space)
                )
                
                # Handle different return formats
                if isinstance(coordinates, list):
                    if len(coordinates) > 0:
                        blob_points = np.array(coordinates)
                    else:
                        blob_points = np.empty((0, 2))
                elif isinstance(coordinates, tuple) and len(coordinates) == 2:
                    # Some versions return (y_coords, x_coords)
                    y_coords, x_coords = coordinates
                    if len(y_coords) > 0 and len(x_coords) > 0:
                        blob_points = np.column_stack((y_coords, x_coords))
                    else:
                        blob_points = np.empty((0, 2))
                else:
                    blob_points = np.empty((0, 2))
                    
            except Exception as e:
                print(f"Warning: Peak detection failed: {e}")
                blob_points = np.empty((0, 2))
            
            if len(blob_points) > 0:
                # Ensure blob_points has correct shape
                if blob_points.ndim == 1:
                    blob_points = blob_points.reshape(-1, 2)
                
                # Compute blob scores (response values)
                scores = []
                for point in blob_points:
                    y, x = int(point[0]), int(point[1])
                    if 0 <= y < normalized.shape[0] and 0 <= x < normalized.shape[1]:
                        scores.append(normalized[y, x])
                    else:
                        scores.append(0.0)
                
                scores = np.array(scores)
                
                # Weight by layer
                layer_weight = layer_idx + 1
                scores = scores * layer_weight
                
                # Convert to list for extending
                all_blobs.extend(blob_points.tolist())
                all_scores.extend(scores.tolist())
        
        if not all_blobs:
            return np.empty((0, 2)), np.array([])
            
        blobs = np.array(all_blobs)
        scores = np.array(all_scores)
        
        # Apply non-maximum suppression
        blobs, scores = self._non_maximum_suppression(blobs, scores)
        
        # Keep top keypoints
        if len(blobs) > self.max_keypoints:
            top_indices = np.argsort(scores)[-self.max_keypoints:]
            blobs = blobs[top_indices]
            scores = scores[top_indices]
            
        return blobs, scores
        
    def _compute_corner_scores(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Compute Harris-like corner response scores
        
        Args:
            image: Input image layer
            corners: Corner point coordinates [N, 2] in format [y, x]
            
        Returns:
            Corner response scores
        """
        if len(corners) == 0:
            return np.array([])
        
        # Compute gradients
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        
        scores = []
        window_size = 3
        k = 0.04  # Harris parameter
        
        for corner in corners:
            # Ensure corner is in correct format and within bounds
            if len(corner) != 2:
                scores.append(0.0)
                continue
                
            y, x = int(corner[0]), int(corner[1])
            
            # Check bounds
            if (y < window_size or y >= image.shape[0] - window_size or
                x < window_size or x >= image.shape[1] - window_size):
                scores.append(0.0)
                continue
            
            # Extract window around corner
            y_start = y - window_size
            y_end = y + window_size + 1
            x_start = x - window_size
            x_end = x + window_size + 1
            
            # Compute structure tensor
            Ix = grad_x[y_start:y_end, x_start:x_end]
            Iy = grad_y[y_start:y_end, x_start:x_end]
            
            Ixx = np.sum(Ix * Ix)
            Iyy = np.sum(Iy * Iy)
            Ixy = np.sum(Ix * Iy)
            
            # Harris corner response
            det_M = Ixx * Iyy - Ixy * Ixy
            trace_M = Ixx + Iyy
            
            if trace_M != 0:
                corner_response = det_M - k * (trace_M ** 2)
            else:
                corner_response = 0.0
            
            scores.append(max(0.0, corner_response))  # Ensure non-negative
        
        return np.array(scores)
    
    def _non_maximum_suppression(self, points: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply non-maximum suppression to remove overlapping feature points
        
        Args:
            points: Feature point coordinates [N, 2]
            scores: Feature point scores [N]
            
        Returns:
            Filtered points and scores after NMS
        """
        if len(points) == 0 or len(scores) == 0:
            return np.empty((0, 2)), np.array([])
        
        # Ensure points and scores have same length
        if len(points) != len(scores):
            min_len = min(len(points), len(scores))
            points = points[:min_len]
            scores = scores[:min_len]
            
        if len(points) == 0:
            return np.empty((0, 2)), np.array([])
            
        # Sort by scores in descending order
        sorted_indices = np.argsort(scores)[::-1]
        
        keep_indices = []
        suppressed = np.zeros(len(points), dtype=bool)
        
        for idx in sorted_indices:
            if idx >= len(suppressed) or suppressed[idx]:
                continue
                
            keep_indices.append(idx)
            
            # Suppress nearby points
            if idx < len(points):
                distances = np.linalg.norm(points - points[idx], axis=1)
                suppress_mask = distances < self.nms_radius
                
                # Ensure suppress_mask doesn't exceed suppressed length
                suppress_mask = suppress_mask[:len(suppressed)]
                suppressed[:len(suppress_mask)] |= suppress_mask
                suppressed[idx] = False  # Don't suppress the current point
            
        return points[keep_indices], scores[keep_indices]