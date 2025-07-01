"""
JamMa-FDAFT (JamF) - 統合特徴マッチングモデル

JamMaのJoint Mamba処理とFDAFTの特徴抽出を組み合わせた
惑星画像および困難な条件下での特徴マッチング用モデル
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Tuple, Dict, Optional, List
import time

from .fdaft_encoder import FDAFTEncoder
from .joint_mamba import JEGOMamba
from .c2f_matcher import CoarseToFineMatcher


class JamF(nn.Module):
    """
    JamMa-FDAFT統合モデル
    
    アーキテクチャ:
    Input Images → FDAFT Encoder → Joint Mamba (JEGO) → C2F Matcher → Matches
    
    特徴:
    - FDAFTの双周波数特徴抽出 (ECM + Phase Congruency)
    - JamMaのJEGO戦略 (Joint Efficient Global Omnidirectional)
    - 線形複雑度 O(N) の効率的処理
    - 惑星画像の照明変化・弱テクスチャに対応
    """
    
    def __init__(self,
                 # FDAFT encoder parameters
                 num_scale_layers: int = 3,
                 sigma_0: float = 1.0,
                 descriptor_radius: int = 48,
                 max_keypoints: int = 1000,
                 # Joint Mamba parameters
                 mamba_dim: int = 256,
                 mamba_layers: int = 4,
                 # Matching parameters
                 coarse_resolution: int = 8,
                 fine_resolution: int = 2,
                 # Model mode
                 mode: str = "planetary",  # "planetary", "indoor", "outdoor"
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize JamMa-FDAFT統合モデル
        
        Args:
            num_scale_layers: スケール空間の層数
            sigma_0: 初期スケールパラメータ
            descriptor_radius: GLOH記述子の半径
            max_keypoints: 最大特徴点数
            mamba_dim: Mamba隠れ次元
            mamba_layers: Mambaレイヤー数
            coarse_resolution: 粗いマッチングの解像度比率
            fine_resolution: 細かいマッチングの解像度比率
            mode: モデルモード（惑星/室内/屋外）
            device: 計算デバイス
        """
        super(JamF, self).__init__()
        
        self.mode = mode
        self.device = device
        self.coarse_resolution = coarse_resolution
        self.fine_resolution = fine_resolution
        
        # FDAFT特徴抽出エンコーダ
        self.fdaft_encoder = FDAFTEncoder(
            num_layers=num_scale_layers,
            sigma_0=sigma_0,
            descriptor_radius=descriptor_radius,
            max_keypoints=max_keypoints,
            mode=mode
        )
        
        # Joint Mamba (JEGO strategy)
        self.joint_mamba = JEGOMamba(
            input_dim=mamba_dim,
            hidden_dim=mamba_dim,
            num_layers=mamba_layers,
            skip_steps=2  # EVMambaのスキップスキャン
        )
        
        # Coarse-to-Fine Matcher
        self.c2f_matcher = CoarseToFineMatcher(
            feature_dim=mamba_dim,
            coarse_resolution=coarse_resolution,
            fine_resolution=fine_resolution
        )
        
        # 特徴次元調整用プロジェクション
        self.feature_projection = nn.Linear(
            self.fdaft_encoder.get_output_dim(), 
            mamba_dim
        )
        
        self.to(device)
    
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            data: {
                'image0': torch.Tensor [B, C, H, W]
                'image1': torch.Tensor [B, C, H, W]
            }
            
        Returns:
            Dictionary containing matches and confidence scores
        """
        image0 = data['image0'].to(self.device)
        image1 = data['image1'].to(self.device)
        
        batch_size = image0.shape[0]
        
        # Step 1: FDAFT特徴抽出
        features0 = self._extract_fdaft_features(image0)
        features1 = self._extract_fdaft_features(image1)
        
        # Step 2: 特徴次元の調整
        features0 = self._project_features(features0)
        features1 = self._project_features(features1)
        
        # Step 3: Joint Mamba処理 (JEGO strategy)
        enhanced_features = self._apply_joint_mamba(features0, features1)
        
        # Step 4: Coarse-to-Fine マッチング
        results = self._coarse_to_fine_matching(
            enhanced_features['features0'], 
            enhanced_features['features1'],
            features0, features1  # Fine特徴として元の特徴も使用
        )
        
        return results
    
    def _extract_fdaft_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        FDAFT特徴抽出
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Dictionary with coarse and fine features
        """
        batch_size = images.shape[0]
        all_coarse_features = []
        all_fine_features = []
        
        for i in range(batch_size):
            # テンソルをnumpy配列に変換
            image_np = images[i].permute(1, 2, 0).cpu().numpy()
            if image_np.shape[2] == 1:
                image_np = image_np.squeeze(2)
            elif image_np.shape[2] == 3:
                image_np = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                image_np = (image_np * 255).astype(np.uint8)
            
            # FDAFT特徴抽出
            fdaft_features = self.fdaft_encoder.extract_features(image_np)
            
            # 特徴マップをテンソルに変換
            coarse_features = self._convert_features_to_tensor(
                fdaft_features['coarse_features'], 
                target_size=(images.shape[2] // self.coarse_resolution, 
                           images.shape[3] // self.coarse_resolution)
            )
            fine_features = self._convert_features_to_tensor(
                fdaft_features['fine_features'],
                target_size=(images.shape[2] // self.fine_resolution, 
                           images.shape[3] // self.fine_resolution)
            )
            
            all_coarse_features.append(coarse_features)
            all_fine_features.append(fine_features)
        
        return {
            'coarse': torch.stack(all_coarse_features).to(self.device),
            'fine': torch.stack(all_fine_features).to(self.device)
        }
    
    def _convert_features_to_tensor(self, features: np.ndarray, 
                                  target_size: Tuple[int, int]) -> torch.Tensor:
        """
        FDAFT特徴をテンソル形式に変換
        """
        if len(features.shape) == 2:
            # 2D特徴を3D (C, H, W)に変換
            features = features[np.newaxis, :, :]
        elif len(features.shape) == 3 and features.shape[2] > features.shape[0]:
            # (H, W, C) → (C, H, W)
            features = features.transpose(2, 0, 1)
        
        # サイズ調整
        features_tensor = torch.from_numpy(features).float()
        if features_tensor.shape[1:] != target_size:
            features_tensor = torch.nn.functional.interpolate(
                features_tensor.unsqueeze(0), 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        return features_tensor
    
    def _project_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        特徴をMamba用の次元に投影
        """
        coarse_features = features['coarse']  # [B, C, H, W]
        
        # (B, C, H, W) → (B, H*W, C)
        batch_size, channels, height, width = coarse_features.shape
        features_flat = coarse_features.view(batch_size, channels, -1).permute(0, 2, 1)
        
        # 次元投影
        projected_features = self.feature_projection(features_flat)
        
        return projected_features
    
    def _apply_joint_mamba(self, features0: torch.Tensor, 
                          features1: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Joint Mamba (JEGO strategy) 適用
        
        Args:
            features0, features1: Projected features [B, N, D]
            
        Returns:
            Enhanced features after Joint Mamba processing
        """
        # Joint scan: 2つの画像の特徴を交互にスキャン
        joint_features = self.joint_mamba(features0, features1)
        
        return {
            'features0': joint_features['enhanced_features0'],
            'features1': joint_features['enhanced_features1']
        }
    
    def _coarse_to_fine_matching(self, 
                                coarse_features0: torch.Tensor,
                                coarse_features1: torch.Tensor,
                                fine_features0: torch.Tensor,
                                fine_features1: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Coarse-to-Fine マッチング
        """
        return self.c2f_matcher(
            coarse_features0, coarse_features1,
            fine_features0, fine_features1
        )
    
    def match_images(self, image0: np.ndarray, image1: np.ndarray) -> Dict:
        """
        2つの画像をマッチング（推論用）
        
        Args:
            image0, image1: Input images as numpy arrays
            
        Returns:
            Matching results dictionary
        """
        self.eval()
        
        with torch.no_grad():
            # 前処理
            tensor0 = self._preprocess_image(image0)
            tensor1 = self._preprocess_image(image1)
            
            data = {
                'image0': tensor0.unsqueeze(0),
                'image1': tensor1.unsqueeze(0)
            }
            
            # Forward pass
            start_time = time.time()
            results = self.forward(data)
            matching_time = time.time() - start_time
            
            # 後処理
            processed_results = self._postprocess_results(results, image0.shape, image1.shape)
            processed_results['matching_time'] = matching_time
            
            return processed_results
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        画像を前処理してテンソルに変換
        """
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # RGB → Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 正規化
        image = image.astype(np.float32) / 255.0
        
        # [H, W] → [C, H, W]
        if len(image.shape) == 2:
            image = image[np.newaxis, :, :]
        
        return torch.from_numpy(image).float()
    
    def _postprocess_results(self, results: Dict[str, torch.Tensor], 
                           shape0: tuple, shape1: tuple) -> Dict:
        """
        結果を後処理
        """
        processed = {}
        
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                processed[key] = value.cpu().numpy()
            else:
                processed[key] = value
        
        return processed
    
    def get_model_info(self) -> Dict[str, any]:
        """
        モデル情報を取得
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'JamF (JamMa-FDAFT)',
            'mode': self.mode,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': self.device,
            'components': {
                'fdaft_encoder': type(self.fdaft_encoder).__name__,
                'joint_mamba': type(self.joint_mamba).__name__,
                'c2f_matcher': type(self.c2f_matcher).__name__
            }
        }