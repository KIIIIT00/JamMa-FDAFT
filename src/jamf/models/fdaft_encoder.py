"""
FDAFT特徴抽出エンコーダ

JamMaのConvNeXtエンコーダを置き換える
双周波数特徴抽出によりJamMaに特徴を供給
"""

import numpy as np
import cv2
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn

from components.scale_space import DoubleFrequencyScaleSpace
from components.feature_detector import FeatureDetector  
from components.gloh_descriptor import GLOHDescriptor
from components.structured_forests import StructuredForestsECM


class FDAFTEncoder(nn.Module):
    """
    FDAFT特徴抽出エンコーダ
    
    JamMaのエンコーダ部分を置き換え
    双周波数（ECM + Phase Congruency）特徴を抽出
    """
    
    def __init__(self,
                 num_layers: int = 3,
                 sigma_0: float = 1.0,
                 descriptor_radius: int = 48,
                 max_keypoints: int = 1000,
                 mode: str = "planetary",
                 structured_forests_model_path: str = "assets/structured_forests_model.yml"):
        """
        Initialize FDAFT encoder
        
        Args:
            num_layers: スケール空間の層数
            sigma_0: 初期スケールパラメータ  
            descriptor_radius: GLOH記述子の半径
            max_keypoints: 最大特徴点数
            mode: モード（惑星画像に最適化）
            structured_forests_model_path: Structured Forestsモデルパス
        """
        super(FDAFTEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.sigma_0 = sigma_0
        self.descriptor_radius = descriptor_radius
        self.max_keypoints = max_keypoints
        self.mode = mode
        
        # 双周波数スケール空間構築
        self.scale_space = DoubleFrequencyScaleSpace(
            num_layers=num_layers,
            sigma_0=sigma_0
        )
        
        # Structured Forests ECM
        self.structured_forests = StructuredForestsECM(
            model_path=structured_forests_model_path
        )
        
        # 特徴点検出器
        self.feature_detector = FeatureDetector(
            nms_radius=5,
            max_keypoints=max_keypoints
        )
        
        # GLOH記述子
        self.gloh_descriptor = GLOHDescriptor(
            patch_size=descriptor_radius * 2 + 1,
            num_radial_bins=3,
            num_angular_bins=8,
            num_orientation_bins=16
        )
        
        # モード別パラメータ調整
        self._adjust_parameters_for_mode(mode)
        
        # 出力次元
        self._output_dim = self._calculate_output_dim()
    
    def _adjust_parameters_for_mode(self, mode: str):
        """
        モードに応じてパラメータを調整
        """
        if mode == "planetary":
            # 惑星画像用：照明変化と弱テクスチャに対応
            self.phase_threshold = 0.05  # 低い閾値でより多くの位相特徴を検出
            self.ecm_enhancement = 1.2   # ECM強化係数
            self.contrast_enhancement = True
            
        elif mode == "indoor":
            # 屋内環境用：幾何学的構造に重点
            self.phase_threshold = 0.1
            self.ecm_enhancement = 1.0
            self.contrast_enhancement = False
            
        elif mode == "outdoor":
            # 屋外環境用：バランス型
            self.phase_threshold = 0.08
            self.ecm_enhancement = 1.1
            self.contrast_enhancement = False
        
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    
    def extract_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        画像から双周波数特徴を抽出
        
        Args:
            image: 入力画像 (grayscale)
            
        Returns:
            Dictionary containing coarse and fine features
        """
        # グレースケール変換
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()
        
        # コントラスト強化（惑星画像モード）
        if self.mode == "planetary" and self.contrast_enhancement:
            gray_image = self._enhance_contrast_for_planetary(gray_image)
        
        # 双周波数スケール空間構築
        low_freq_space = self._build_enhanced_low_freq_space(gray_image)
        high_freq_space = self._build_enhanced_high_freq_space(gray_image)
        
        # 特徴点抽出
        corner_points, corner_scores = self.feature_detector.extract_corner_points(low_freq_space)
        blob_points, blob_scores = self.feature_detector.extract_blob_points(high_freq_space)
        
        # 記述子計算
        corner_descriptors = self.gloh_descriptor.describe(gray_image, corner_points)
        blob_descriptors = self.gloh_descriptor.describe(gray_image, blob_points)
        
        # 特徴マップ構築
        coarse_features = self._build_feature_maps(
            gray_image, low_freq_space, corner_points, corner_descriptors
        )
        fine_features = self._build_feature_maps(
            gray_image, high_freq_space, blob_points, blob_descriptors
        )
        
        return {
            'coarse_features': coarse_features,
            'fine_features': fine_features,
            'corner_points': corner_points,
            'blob_points': blob_points,
            'corner_descriptors': corner_descriptors,
            'blob_descriptors': blob_descriptors,
            'corner_scores': corner_scores,
            'blob_scores': blob_scores
        }
    
    def _enhance_contrast_for_planetary(self, image: np.ndarray) -> np.ndarray:
        """
        惑星画像用のコントラスト強化
        """
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # ガンマ補正で暗部を強調
        gamma = 0.8
        enhanced = np.power(enhanced / 255.0, gamma) * 255.0
        enhanced = enhanced.astype(np.uint8)
        
        return enhanced
    
    def _build_enhanced_low_freq_space(self, image: np.ndarray) -> List[np.ndarray]:
        """
        強化された低周波スケール空間構築
        """
        # Structured Forests ECMを使用
        ecm = self.structured_forests.compute_ecm(image)
        
        # ECM強化（モード依存）
        ecm = ecm * self.ecm_enhancement
        ecm = np.clip(ecm, 0, 1)
        
        # スケール空間構築
        low_freq_space = []
        for n in range(self.num_layers):
            sigma_n = self.sigma_0 * (np.sqrt(3/2))**n
            filtered = cv2.GaussianBlur(ecm, (0, 0), sigma_n)
            low_freq_space.append(filtered)
        
        return low_freq_space
    
    def _build_enhanced_high_freq_space(self, image: np.ndarray) -> List[np.ndarray]:
        """
        強化された高周波スケール空間構築
        """
        # Phase Congruency計算
        pc = self.scale_space.compute_weighted_phase_congruency(image)
        
        # 閾値処理（モード依存）
        pc_thresholded = np.where(pc > self.phase_threshold, pc, 0)
        
        # Maximum Moment Map
        mmax = self.scale_space.compute_maximum_moment_map(pc_thresholded)
        
        # Steerable Gaussian Filter
        filtered_mmax = self.scale_space.apply_steerable_gaussian_filter(mmax, sigma=2.0)
        
        # スケール空間構築
        high_freq_space = []
        for n in range(self.num_layers):
            sigma_n = self.sigma_0 * (np.sqrt(3/2))**n
            filtered = cv2.GaussianBlur(filtered_mmax, (0, 0), sigma_n)
            high_freq_space.append(filtered)
        
        return high_freq_space
    
    def _build_feature_maps(self, image: np.ndarray, scale_space: List[np.ndarray],
                           keypoints: np.ndarray, descriptors: np.ndarray) -> np.ndarray:
        """
        特徴マップを構築
        
    JamMaのJoint Mambaが処理できる形式に変換
        """
        h, w = image.shape
        
        # スケール空間の平均を計算
        avg_scale_space = np.mean(scale_space, axis=0)
        
        # 特徴点情報を統合
        if len(keypoints) > 0:
            # 特徴点位置での記述子情報を画像に埋め込み
            feature_map = self._embed_descriptors_in_image(
                avg_scale_space, keypoints, descriptors
            )
        else:
            # 特徴点がない場合はスケール空間をそのまま使用
            feature_map = avg_scale_space
        
        # 3チャンネルに拡張（元画像 + スケール空間 + 特徴情報）
        normalized_image = image.astype(np.float32) / 255.0
        normalized_scale = (avg_scale_space - avg_scale_space.min()) / (avg_scale_space.max() - avg_scale_space.min() + 1e-8)
        normalized_feature = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        # [H, W, 3]形式で結合
        combined_features = np.stack([
            normalized_image,
            normalized_scale, 
            normalized_feature
        ], axis=2)
        
        return combined_features
    
    def _embed_descriptors_in_image(self, base_image: np.ndarray, 
                                   keypoints: np.ndarray, 
                                   descriptors: np.ndarray) -> np.ndarray:
        """
        記述子情報を画像に埋め込み
        """
        feature_map = base_image.copy()
        
        if len(keypoints) == 0:
            return feature_map
        
        # 記述子の主成分を計算
        descriptor_strength = np.linalg.norm(descriptors, axis=1)
        
        # 特徴点周辺に強度を反映
        for i, (keypoint, strength) in enumerate(zip(keypoints, descriptor_strength)):
            y, x = int(keypoint[0]), int(keypoint[1])
            
            # 画像境界チェック
            if 0 <= y < feature_map.shape[0] and 0 <= x < feature_map.shape[1]:
                # ガウシアンカーネルで重み付け
                radius = 5
                y_start = max(0, y - radius)
                y_end = min(feature_map.shape[0], y + radius + 1)
                x_start = max(0, x - radius)
                x_end = min(feature_map.shape[1], x + radius + 1)
                
                # ガウシアン重み
                Y, X = np.ogrid[y_start:y_end, x_start:x_end]
                gaussian_weight = np.exp(-((X - x)**2 + (Y - y)**2) / (2 * (radius/2)**2))
                
                # 特徴強度を反映
                feature_map[y_start:y_end, x_start:x_end] += gaussian_weight * strength * 0.1
        
        return feature_map
    
    def _calculate_output_dim(self) -> int:
        """
        出力次元を計算
        """
        # 3チャンネル特徴マップ + GLOH記述子次元
        base_dim = 3  # [原画像, スケール空間, 特徴情報]
        gloh_dim = self.gloh_descriptor.get_descriptor_size()
        
        return base_dim + gloh_dim // 10  # 記述子次元を圧縮
    
    def get_output_dim(self) -> int:
        """
        出力次元を取得
        """
        return self._output_dim
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        PyTorchモジュールとしてのforward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Dictionary with extracted features
        """
        batch_size = x.shape[0]
        device = x.device
        
        # バッチ内の各画像を処理
        batch_features = {
            'coarse': [],
            'fine': []
        }
        
        for i in range(batch_size):
            # テンソルをnumpy配列に変換
            image_np = x[i].permute(1, 2, 0).cpu().numpy()
            if image_np.shape[2] == 1:
                image_np = image_np.squeeze(2)
            elif image_np.shape[2] == 3:
                image_np = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                image_np = (image_np * 255).astype(np.uint8)
            
            # FDAFT特徴抽出
            features = self.extract_features(image_np)
            
            # テンソルに変換して追加
            coarse_tensor = torch.from_numpy(features['coarse_features']).float()
            fine_tensor = torch.from_numpy(features['fine_features']).float()
            
            # [H, W, C] → [C, H, W]
            if len(coarse_tensor.shape) == 3:
                coarse_tensor = coarse_tensor.permute(2, 0, 1)
            if len(fine_tensor.shape) == 3:
                fine_tensor = fine_tensor.permute(2, 0, 1)
            
            batch_features['coarse'].append(coarse_tensor)
            batch_features['fine'].append(fine_tensor)
        
        # バッチテンソルに変換
        coarse_features = torch.stack(batch_features['coarse']).to(device)
        fine_features = torch.stack(batch_features['fine']).to(device)
        
        return {
            'coarse_features': coarse_features,
            'fine_features': fine_features
        }