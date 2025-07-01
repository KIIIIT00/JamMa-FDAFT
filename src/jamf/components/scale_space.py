import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from typing import List, Tuple
from skimage import feature

class DoubleFrequencyScaleSpace:
    """
    Double-Frequency Scale Space Construction for FDAFT
    Implements both low-frequency (ECM-based) and high-frequency (phase-based) scale spaces
    """
    
    def __init__(self, num_layers: int = 3, sigma_0: float = 1.0):
        """
        Initialize scale space parameters
        
        Args:
            num_layers: Number of layers in scale space
            sigma_0: Initial scale parameter
        """
        self.num_layers = num_layers
        self.sigma_0 = sigma_0
        
    def compute_edge_confidence_map(self, image: np.ndarray) -> np.ndarray:
        """
        Compute Edge Confidence Map using machine learning-based structured edge detection
        
        This method tries to use the pre-trained Structured Forests model if available,
        otherwise falls back to a simplified implementation.
        
        Args:
            image: Input image
            
        Returns:
            Edge confidence map with values in [0, 1]
        """
        # Try to use OpenCV's Structured Forests if available
        try:
            ecm = self._compute_structured_forests_opencv(image)
            return ecm
        except Exception as e:
            print(f"Warning: OpenCV Structured Forests not available ({e}), using fallback implementation")
            pass
            
        # Fallback to simplified implementation
        if len(image.shape) == 3:
            rgb_image = image.astype(np.float32) / 255.0
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.float32)
            rgb_image = np.stack([gray, gray, gray], axis=2) / 255.0
            
        gray = gray.astype(np.float32) / 255.0
        
        # Feature extraction for structured edge detection
        ecm = self._compute_structured_edge_features(rgb_image, gray)
        
        return ecm
    
    def _compute_structured_forests_opencv(self, image: np.ndarray) -> np.ndarray:
        """
        Compute ECM using OpenCV's pre-trained Structured Forests model
        
        Args:
            image: Input image
            
        Returns:
            Edge confidence map
        """
        # Check if OpenCV contrib modules are available
        if not hasattr(cv2, 'ximgproc'):
            raise ImportError("OpenCV ximgproc module not available")
        
        # Convert image to proper format
        if len(image.shape) == 3:
            # RGB to BGR for OpenCV
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            # Convert grayscale to BGR
            bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Normalize to [0, 1] float32
        bgr_image = bgr_image.astype(np.float32) / 255.0
        
        # Try to load pre-trained model
        model_paths = [
            "model.yml",  # Current directory
            "models/model.yml",  # Models subdirectory
            "models/structured_forests.yml",  # Alternative name
            "assets/model.yml",  # Assets directory
            "/usr/local/share/opencv4/model.yml",  # System path
        ]
        
        edge_detector = None
        for model_path in model_paths:
            try:
                edge_detector = cv2.ximgproc.createStructuredEdgeDetection(model_path)
                print(f"Loaded Structured Forests model from: {model_path}")
                break
            except:
                continue
        
        if edge_detector is None:
            # Try to download model if not found
            try:
                self._download_structured_forests_model()
                edge_detector = cv2.ximgproc.createStructuredEdgeDetection("model.yml")
            except:
                raise FileNotFoundError("Structured Forests model not found and could not be downloaded")
        
        # Detect edges
        edges = edge_detector.detectEdges(bgr_image)
        
        # Compute orientation map
        orientation_map = edge_detector.computeOrientation(edges)
        
        # Apply non-maximum suppression
        edges_nms = edge_detector.edgesNms(edges, orientation_map)
        
        # Apply additional post-processing for planetary images
        ecm = self._postprocess_structured_edges(edges_nms, image)
        
        return ecm
    
    def _download_structured_forests_model(self):
        """
        Download the pre-trained Structured Forests model from OpenCV
        """
        import urllib.request
        import gzip
        import os
        
        model_url = "https://github.com/opencv/opencv_extra/raw/master/testdata/cv/ximgproc/model.yml.gz"
        
        print("Downloading Structured Forests model...")
        
        try:
            # Download compressed model
            urllib.request.urlretrieve(model_url, "model.yml.gz")
            
            # Decompress
            with gzip.open("model.yml.gz", 'rb') as f_in:
                with open("model.yml", 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # Clean up
            os.remove("model.yml.gz")
            
            print("Model downloaded successfully!")
            
        except Exception as e:
            print(f"Failed to download model: {e}")
            raise
    
    def _postprocess_structured_edges(self, edges: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """
        Post-process structured forest edges for planetary images
        
        Args:
            edges: Raw edge map from Structured Forests
            original_image: Original input image
            
        Returns:
            Enhanced edge confidence map
        """
        # Convert to grayscale if needed
        if len(original_image.shape) == 3:
            gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = original_image.copy()
        
        gray = gray.astype(np.float32) / 255.0
        
        # Enhance edges based on planetary image characteristics
        
        # 1. Suppress noise in low-contrast regions
        contrast_mask = self._compute_local_contrast(gray)
        edges_enhanced = edges * contrast_mask
        
        # 2. Boost edges in crater-like circular structures
        circular_enhancement = self._detect_circular_structures(gray)
        edges_enhanced = edges_enhanced + 0.3 * circular_enhancement * edges
        
        # 3. Suppress edges in very smooth regions (sky/space)
        texture_mask = self._compute_texture_mask(gray)
        edges_enhanced = edges_enhanced * texture_mask
        
        # 4. Apply adaptive thresholding
        edges_enhanced = self._adaptive_edge_enhancement(edges_enhanced, gray)
        
        # Normalize to [0, 1]
        edges_enhanced = np.clip(edges_enhanced, 0, 1)
        
        return edges_enhanced
    
    def _compute_local_contrast(self, image: np.ndarray, window_size: int = 15) -> np.ndarray:
        """Compute local contrast for noise suppression"""
        kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        
        # Local mean
        local_mean = cv2.filter2D(image, -1, kernel)
        
        # Local variance
        local_var = cv2.filter2D(image * image, -1, kernel) - local_mean * local_mean
        local_std = np.sqrt(local_var + 1e-8)
        
        # Contrast mask (higher values for higher contrast regions)
        contrast_mask = np.tanh(local_std * 10)
        
        return contrast_mask
    
    def _detect_circular_structures(self, image: np.ndarray) -> np.ndarray:
        """Detect circular structures (craters) using Hough transform"""
        # Convert to uint8 for HoughCircles
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Apply median filter to reduce noise
        image_filtered = cv2.medianBlur(image_uint8, 5)
        
        # Detect circles
        try:
            circles = cv2.HoughCircles(
                image_filtered,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=30,
                param1=50,
                param2=30,
                minRadius=10,
                maxRadius=100
            )
        except:
            circles = None
        
        # Create circular structure map
        circular_map = np.zeros_like(image)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Create circular mask
                Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
                dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)
                
                # Enhance edge region around circle
                circle_edge_mask = np.logical_and(dist_from_center >= r-5, dist_from_center <= r+5)
                circular_map[circle_edge_mask] = 1.0
        
        return circular_map
    
    def _compute_texture_mask(self, image: np.ndarray) -> np.ndarray:
        """Compute texture mask to suppress edges in smooth regions"""
        # Compute local texture using standard deviation
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        local_mean = cv2.filter2D(image, -1, kernel)
        local_var = cv2.filter2D(image * image, -1, kernel) - local_mean * local_mean
        local_std = np.sqrt(local_var + 1e-8)
        
        # Create texture mask (suppress very smooth regions)
        texture_threshold = 0.02
        texture_mask = np.tanh(local_std / texture_threshold)
        
        return texture_mask
    
    def _adaptive_edge_enhancement(self, edges: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Apply adaptive enhancement based on local image properties"""
        # Compute image gradient for reference
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient magnitude
        grad_mag_norm = grad_mag / (grad_mag.max() + 1e-8)
        
        # Enhance edges where gradient is strong
        enhancement_factor = 1.0 + 0.5 * grad_mag_norm
        edges_enhanced = edges * enhancement_factor
        
        # Apply bilateral filter to preserve strong edges while smoothing weak ones
        edges_enhanced = cv2.bilateralFilter(
            edges_enhanced.astype(np.float32), 
            d=5, 
            sigmaColor=0.1, 
            sigmaSpace=5
        )
        
        return edges_enhanced
    
    def _compute_structured_edge_features(self, rgb_image: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """
        Compute structured edge features using local patch analysis
        This simulates the structured forest approach without the actual trained model
        
        Args:
            rgb_image: RGB image [H, W, 3]
            gray: Grayscale image [H, W]
            
        Returns:
            Edge confidence map
        """
        h, w = gray.shape
        patch_size = 16  # Standard patch size for structured forests
        half_patch = patch_size // 2
        
        # Initialize edge confidence map
        ecm = np.zeros_like(gray)
        
        # Compute multi-scale features
        # 1. Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_ori = np.arctan2(grad_y, grad_x)
        
        # 2. Color gradient features
        color_grads = []
        for c in range(3):
            gx = cv2.Sobel(rgb_image[:,:,c], cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(rgb_image[:,:,c], cv2.CV_32F, 0, 1, ksize=3)
            color_grads.append(np.sqrt(gx**2 + gy**2))
        
        # 3. Local Binary Pattern features
        lbp = self._compute_lbp(gray)
        
        # 4. Texture features using Gabor filters
        gabor_responses = self._compute_gabor_features(gray)
        
        # Process each pixel with local patch context
        for y in range(half_patch, h - half_patch):
            for x in range(half_patch, w - half_patch):
                # Extract local patch features
                patch_features = self._extract_patch_features(
                    rgb_image, gray, grad_mag, grad_ori, lbp, gabor_responses,
                    y, x, patch_size
                )
                
                # Simplified decision tree ensemble (normally would use trained model)
                edge_confidence = self._predict_edge_confidence(patch_features)
                ecm[y, x] = edge_confidence
        
        # Apply smoothing to reduce noise
        ecm = cv2.GaussianBlur(ecm, (3, 3), 0.5)
        
        # Normalize to [0, 1]
        ecm = (ecm - ecm.min()) / (ecm.max() - ecm.min() + 1e-8)
        
        return ecm
    
    def _compute_lbp(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Compute Local Binary Pattern"""
        try:
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(image, n_points, radius, method='uniform')
            return lbp / n_points  # Normalize
        except ImportError:
            # Fallback: simple LBP implementation
            h, w = image.shape
            lbp = np.zeros_like(image)
            
            # Simple 8-neighbor LBP
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = image[i, j]
                    code = 0
                    
                    # Check 8 neighbors
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j+1], image[i+1, j+1], image[i+1, j],
                        image[i+1, j-1], image[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)
                    
                    lbp[i, j] = code
            
            return lbp / 255.0
    
    def _compute_gabor_features(self, image: np.ndarray) -> np.ndarray:
        """Compute Gabor filter responses for texture analysis"""
        responses = []
        
        # Multiple orientations and frequencies
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        frequencies = [0.1, 0.3, 0.5]
        
        for theta in orientations:
            for freq in frequencies:
                try:
                    kernel = cv2.getGaborKernel((15, 15), sigma=2, theta=theta, 
                                              lambd=1.0/freq, gamma=0.5, psi=0)
                    response = cv2.filter2D(image, cv2.CV_32F, kernel)
                    responses.append(np.abs(response))
                except:
                    # If Gabor fails, create zero response
                    responses.append(np.zeros_like(image))
        
        return np.stack(responses, axis=2)
    
    def _extract_patch_features(self, rgb_image: np.ndarray, gray: np.ndarray,
                               grad_mag: np.ndarray, grad_ori: np.ndarray,
                               lbp: np.ndarray, gabor_responses: np.ndarray,
                               y: int, x: int, patch_size: int) -> np.ndarray:
        """Extract features from local patch around pixel (y, x)"""
        half_patch = patch_size // 2
        
        # Extract patches
        rgb_patch = rgb_image[y-half_patch:y+half_patch, x-half_patch:x+half_patch]
        gray_patch = gray[y-half_patch:y+half_patch, x-half_patch:x+half_patch]
        grad_patch = grad_mag[y-half_patch:y+half_patch, x-half_patch:x+half_patch]
        ori_patch = grad_ori[y-half_patch:y+half_patch, x-half_patch:x+half_patch]
        lbp_patch = lbp[y-half_patch:y+half_patch, x-half_patch:x+half_patch]
        gabor_patch = gabor_responses[y-half_patch:y+half_patch, x-half_patch:x+half_patch]
        
        features = []
        
        # 1. Color statistics
        features.extend([
            np.mean(rgb_patch[:,:,0]), np.std(rgb_patch[:,:,0]),
            np.mean(rgb_patch[:,:,1]), np.std(rgb_patch[:,:,1]),
            np.mean(rgb_patch[:,:,2]), np.std(rgb_patch[:,:,2])
        ])
        
        # 2. Gradient statistics
        features.extend([
            np.mean(grad_patch), np.std(grad_patch),
            np.mean(gray_patch), np.std(gray_patch)
        ])
        
        # 3. Orientation histogram
        ori_hist, _ = np.histogram(ori_patch.flatten(), bins=8, range=(-np.pi, np.pi))
        features.extend(ori_hist / (ori_hist.sum() + 1e-8))
        
        # 4. LBP histogram
        lbp_hist, _ = np.histogram(lbp_patch.flatten(), bins=10)
        features.extend(lbp_hist / (lbp_hist.sum() + 1e-8))
        
        # 5. Gabor response statistics
        for i in range(gabor_patch.shape[2]):
            features.extend([
                np.mean(gabor_patch[:,:,i]),
                np.std(gabor_patch[:,:,i])
            ])
        
        return np.array(features)
    
    def _predict_edge_confidence(self, features: np.ndarray) -> float:
        """
        Predict edge confidence using Structured Forest approach
        """
        
        # Simulate structured forest decision tree ensemble
        predictions = []
        
        # Tree 1: Focus on gradient-based features
        tree1_pred = self._decision_tree_1(features)
        predictions.append(tree1_pred)
        
        # Tree 2: Focus on color-based features  
        tree2_pred = self._decision_tree_2(features)
        predictions.append(tree2_pred)
        
        # Tree 3: Focus on texture-based features
        tree3_pred = self._decision_tree_3(features)
        predictions.append(tree3_pred)
        
        # Tree 4: Focus on orientation features
        tree4_pred = self._decision_tree_4(features)
        predictions.append(tree4_pred)
        
        # Ensemble prediction (average of all trees)
        edge_confidence = np.mean(predictions)
        
        return np.clip(edge_confidence, 0.0, 1.0)
    
    def _decision_tree_1(self, features: np.ndarray) -> float:
        """Decision Tree 1: Gradient-based splits"""
        grad_mean = features[6] if len(features) > 6 else 0.0
        grad_std = features[7] if len(features) > 7 else 0.0
        
        if grad_mean > 0.15:  # Strong gradient
            if grad_std > 0.1:  # High variation
                return 0.9
            else:
                return 0.7
        else:  # Weak gradient
            if grad_std > 0.05:
                return 0.3
            else:
                return 0.1
    
    def _decision_tree_2(self, features: np.ndarray) -> float:
        """Decision Tree 2: Color-based splits"""
        color_var_r = features[1] if len(features) > 1 else 0.0
        color_var_g = features[3] if len(features) > 3 else 0.0
        color_var_b = features[5] if len(features) > 5 else 0.0
        
        total_color_var = color_var_r + color_var_g + color_var_b
        
        if total_color_var > 0.2:  # High color variation
            if max(color_var_r, color_var_g, color_var_b) > 0.1:
                return 0.8
            else:
                return 0.6
        else:  # Low color variation
            return 0.2
    
    def _decision_tree_3(self, features: np.ndarray) -> float:
        """Decision Tree 3: Texture-based splits"""
        if len(features) > 28:
            lbp_hist = features[18:28]
            lbp_entropy = -np.sum(lbp_hist * np.log(lbp_hist + 1e-8))
            
            # Gabor response statistics
            gabor_features = features[28:]
            gabor_mean = np.mean(gabor_features[::2]) if len(gabor_features) > 0 else 0.0
            gabor_std = np.mean(gabor_features[1::2]) if len(gabor_features) > 1 else 0.0
            
            if lbp_entropy > 1.5:  # Complex texture
                if gabor_mean > 0.1:
                    return 0.8
                else:
                    return 0.5
            else:  # Simple texture
                if gabor_std > 0.05:
                    return 0.4
                else:
                    return 0.1
        else:
            return 0.3
    
    def _decision_tree_4(self, features: np.ndarray) -> float:
        """Decision Tree 4: Orientation-based splits"""
        if len(features) > 18:
            ori_hist = features[10:18]
            
            # Calculate orientation consistency
            ori_entropy = -np.sum(ori_hist * np.log(ori_hist + 1e-8))
            dominant_ori = np.max(ori_hist)
            
            if dominant_ori > 0.3:  # Strong dominant orientation
                if ori_entropy < 1.0:  # Low entropy (consistent)
                    return 0.9
                else:
                    return 0.6
            else:  # No dominant orientation
                if ori_entropy > 2.0:  # High entropy (chaotic)
                    return 0.2
                else:
                    return 0.4
        else:
            return 0.3
        
    def compute_weighted_phase_congruency(self, image: np.ndarray) -> np.ndarray:
        """
        Compute Weighted Phase Congruency (PCw)
        
        Args:
            image: Input image
            
        Returns:
            Weighted phase congruency map
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Normalize image
        gray = (gray - gray.mean()) / (gray.std() + 1e-8)
        
        # Parameters for Log-Gabor filters
        orientations = np.array([0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6])
        scales = [1, 2, 4]
        
        # Initialize response arrays
        responses_real = []
        responses_imag = []
        
        for scale in scales:
            for theta in orientations:
                try:
                    # Create Log-Gabor filter
                    kernel_real = cv2.getGaborKernel((31, 31), scale, theta, 2*np.pi/3, 0.5, 0)
                    kernel_imag = cv2.getGaborKernel((31, 31), scale, theta, 2*np.pi/3, 0.5, np.pi/2)
                    
                    # Apply filters
                    response_real = cv2.filter2D(gray, cv2.CV_32F, kernel_real)
                    response_imag = cv2.filter2D(gray, cv2.CV_32F, kernel_imag)
                    
                    responses_real.append(response_real)
                    responses_imag.append(response_imag)
                except:
                    # If filter fails, add zero response
                    responses_real.append(np.zeros_like(gray))
                    responses_imag.append(np.zeros_like(gray))
        
        # Compute phase congruency
        responses_real = np.array(responses_real)
        responses_imag = np.array(responses_imag)
        
        # Amplitude and phase
        amplitude = np.sqrt(responses_real**2 + responses_imag**2)
        
        # Phase congruency calculation
        sum_amplitude = np.sum(amplitude, axis=0)
        sum_responses = np.sqrt(np.sum(responses_real, axis=0)**2 + 
                               np.sum(responses_imag, axis=0)**2)
        
        # Phase congruency with noise threshold
        noise_threshold = 0.1
        pc = np.maximum(sum_responses - noise_threshold, 0) / (sum_amplitude + 1e-8)
        
        return pc
        
    def compute_maximum_moment_map(self, pc: np.ndarray) -> np.ndarray:
        """
        Compute Maximum Moment Map from phase congruency
        Based on equation (3) in the paper
        
        Args:
            pc: Phase congruency map
            
        Returns:
            Maximum moment map
        """
        # Compute gradients
        grad_y, grad_x = np.gradient(pc)
        
        # Compute structure tensor components
        a = grad_x * grad_x
        b = 2 * grad_x * grad_y
        c = grad_y * grad_y
        
        # Compute maximum eigenvalue (equation 3)
        # Mmax = 1/2 * (c + a + sqrt(b^2 + (a-c)^2))
        discriminant = b*b + (a - c)*(a - c)
        mmax = 0.5 * (c + a + np.sqrt(discriminant + 1e-8))
        
        return mmax
        
    def apply_steerable_gaussian_filter(self, mmax: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply second-order steerable Gaussian filter
        Based on equation (5) in the paper
        
        Args:
            mmax: Maximum moment map
            sigma: Filter scale
            
        Returns:
            Filtered response
        """
        # Define orientations
        orientations = np.array([0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6])
        
        responses = []
        
        for theta in orientations:
            try:
                # Create steerable filter kernels
                size = int(6 * sigma + 1)
                if size % 2 == 0:
                    size += 1
                    
                x, y = np.meshgrid(np.arange(-size//2, size//2+1), 
                                  np.arange(-size//2, size//2+1))
                
                # Second-order Gaussian derivatives
                exp_term = np.exp(-(x*x + y*y)/(2*sigma*sigma))
                
                # Base filters (equation 5)
                G_xx = (-1/(2*np.pi*sigma**4)) * (1 - x*x/sigma**2) * exp_term
                G_yy = (-1/(2*np.pi*sigma**4)) * (1 - y*y/sigma**2) * exp_term
                G_xy = (x*y/(2*np.pi*sigma**6)) * exp_term
                
                # Steerable combination
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                
                G_theta = (cos_theta**2 * G_xx + sin_theta**2 * G_yy - 
                          2*cos_theta*sin_theta * G_xy)
                
                # Apply filter
                response = cv2.filter2D(mmax, -1, G_theta)
                responses.append(response)
            except:
                # If filter fails, add zero response
                responses.append(np.zeros_like(mmax))
        
        # Sum all orientation responses
        result = np.sum(responses, axis=0)
        
        return result
        
    def build_low_frequency_scale_space(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Build low-frequency scale space using ECM
        
        Args:
            image: Input image
            
        Returns:
            List of scale space layers
        """
        # Compute ECM
        ecm = self.compute_edge_confidence_map(image)
        
        # Build scale space
        scale_space = []
        
        for n in range(self.num_layers):
            # Compute scale (equation 1)
            sigma_n = self.sigma_0 * (np.sqrt(3/2))**n
            
            # Apply Gaussian filtering
            filtered = gaussian_filter(ecm, sigma=sigma_n)
            scale_space.append(filtered)
            
        return scale_space
        
    def build_high_frequency_scale_space(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Build high-frequency scale space using phase features
        
        Args:
            image: Input image
            
        Returns:
            List of scale space layers
        """
        # Compute weighted phase congruency
        pc = self.compute_weighted_phase_congruency(image)
        
        # Compute maximum moment map
        mmax = self.compute_maximum_moment_map(pc)
        
        # Apply steerable Gaussian filter
        filtered_mmax = self.apply_steerable_gaussian_filter(mmax, sigma=2.0)
        
        # Build scale space
        scale_space = []
        
        for n in range(self.num_layers):
            # Compute scale
            sigma_n = self.sigma_0 * (np.sqrt(3/2))**n
            
            # Apply Gaussian filtering
            filtered = gaussian_filter(filtered_mmax, sigma=sigma_n)
            scale_space.append(filtered)
            
        return scale_space