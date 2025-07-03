"""
Coarse-to-Fine Matcher

JamMaのCoarse-to-Fine マッチング戦略を実装
FDAFT特徴とJoint Mamba処理後の特徴を使用してマッチングを行う
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class CoarseToFineMatcher(nn.Module):
    """
    Coarse-to-Fine Matching Module
    
    1. Coarse matching: Joint Mamba処理後の特徴でグローバルマッチング
    2. Fine matching: FDAFT細部特徴で精密マッチング  
    3. Sub-pixel refinement: サブピクセル精度の調整
    """
    
    def __init__(self,
                 feature_dim: int = 256,
                 coarse_resolution: int = 8,
                 fine_resolution: int = 2,
                 temperature: float = 0.1,
                 coarse_threshold: float = 0.2,
                 fine_window_size: int = 5):
        """
        Initialize Coarse-to-Fine matcher
        
        Args:
            feature_dim: 特徴次元
            coarse_resolution: 粗いマッチングの解像度比率
            fine_resolution: 細かいマッチングの解像度比率
            temperature: Softmax温度パラメータ
            coarse_threshold: 粗いマッチングの閾値
            fine_window_size: 細かいマッチングのウィンドウサイズ
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.coarse_resolution = coarse_resolution
        self.fine_resolution = fine_resolution
        self.temperature = temperature
        self.coarse_threshold = coarse_threshold
        self.fine_window_size = fine_window_size
        
        # Coarse matching layers
        self.coarse_attention = CrossAttentionLayer(feature_dim)
        
        # Fine matching layers (MLP-Mixer style)
        self.fine_mixer = MLPMixer(
            patch_size=fine_window_size,
            feature_dim=feature_dim // 4,  # 細部特徴用に次元削減
            num_layers=2
        )
        
        # Sub-pixel refinement
        self.refinement_head = nn.Sequential(
            nn.Linear(feature_dim // 4, feature_dim // 8),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 8, 2),  # [delta_x, delta_y]
            nn.Tanh()  # [-1, 1] range
        )
        
    def forward(self, 
                coarse_features0: torch.Tensor,
                coarse_features1: torch.Tensor,
                fine_features0: torch.Tensor,
                fine_features1: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for coarse-to-fine matching
        
        Args:
            coarse_features0, coarse_features1: Coarse features [B, N, D]
            fine_features0, fine_features1: Fine features [B, N, D]
            
        Returns:
            Dictionary with matching results
        """
        batch_size = coarse_features0.shape[0]
        
        # Step 1: Coarse matching
        coarse_matches = self.coarse_matching(coarse_features0, coarse_features1)
        
        # Step 2: Fine matching
        fine_matches = self.fine_matching(
            fine_features0, fine_features1, 
            coarse_matches['matches'], coarse_matches['confidence']
        )
        
        # Step 3: Sub-pixel refinement
        refined_matches = self.sub_pixel_refinement(fine_matches)
        
        return {
            'coarse_matches': coarse_matches['matches'],
            'coarse_confidence': coarse_matches['confidence'],
            'fine_matches': fine_matches['matches'],
            'fine_confidence': fine_matches['confidence'],
            'refined_matches': refined_matches['matches'],
            'refined_offsets': refined_matches['offsets'],
            'final_confidence': refined_matches['confidence']
        }
    
    def coarse_matching(self, features0: torch.Tensor, features1: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Coarse matching using global features
        
        Args:
            features0, features1: Coarse features [B, N, D]
            
        Returns:
            Coarse matches and confidence scores
        """
        batch_size, seq_len, feature_dim = features0.shape
        
        # Cross-attention for mutual interaction
        enhanced_feat0, enhanced_feat1 = self.coarse_attention(features0, features1)
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(enhanced_feat0, enhanced_feat1)
        
        # Dual softmax for matching probabilities
        prob_0to1, prob_1to0 = self.dual_softmax(similarity_matrix)
        
        # Extract matches using row/column maximum strategy
        matches, confidence = self.extract_coarse_matches(prob_0to1, prob_1to0)
        
        return {
            'matches': matches,
            'confidence': confidence,
            'similarity_matrix': similarity_matrix,
            'prob_0to1': prob_0to1,
            'prob_1to0': prob_1to0
        }
    
    def fine_matching(self, 
                     fine_features0: torch.Tensor,
                     fine_features1: torch.Tensor,
                     coarse_matches: torch.Tensor,
                     coarse_confidence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Fine matching using local windows around coarse matches
        
        Args:
            fine_features0, fine_features1: Fine features [B, N, D]
            coarse_matches: Coarse matches [B, M, 2]
            coarse_confidence: Coarse confidence [B, M]
            
        Returns:
            Fine matches and confidence scores
        """
        batch_size, num_matches = coarse_matches.shape[:2]
        
        if num_matches == 0:
            return {
                'matches': torch.empty(batch_size, 0, 2, device=coarse_matches.device),
                'confidence': torch.empty(batch_size, 0, device=coarse_matches.device)
            }
        
        fine_matches_list = []
        fine_confidence_list = []
        
        for b in range(batch_size):
            batch_fine_matches = []
            batch_fine_confidence = []
            
            for m in range(num_matches):
                if coarse_confidence[b, m] > self.coarse_threshold:
                    # Extract local windows around coarse match
                    match_0 = coarse_matches[b, m, 0].long()
                    match_1 = coarse_matches[b, m, 1].long()
                    
                    window_0 = self.extract_local_window(fine_features0[b], match_0)
                    window_1 = self.extract_local_window(fine_features1[b], match_1)
                    
                    if window_0 is not None and window_1 is not None:
                        # Fine matching within windows
                        fine_match, fine_conf = self.match_local_windows(window_0, window_1)
                        
                        if fine_match is not None:
                            # Convert local coordinates to global coordinates
                            global_match_0 = self.local_to_global_coords(match_0, fine_match[0])
                            global_match_1 = self.local_to_global_coords(match_1, fine_match[1])
                            
                            batch_fine_matches.append(torch.stack([global_match_0, global_match_1]))
                            batch_fine_confidence.append(fine_conf)
            
            if batch_fine_matches:
                fine_matches_list.append(torch.stack(batch_fine_matches))
                fine_confidence_list.append(torch.stack(batch_fine_confidence))
            else:
                fine_matches_list.append(torch.empty(0, 2, device=coarse_matches.device))
                fine_confidence_list.append(torch.empty(0, device=coarse_matches.device))
        
        # Pad sequences to same length for batching
        max_matches = max(len(matches) for matches in fine_matches_list)
        if max_matches == 0:
            return {
                'matches': torch.empty(batch_size, 0, 2, device=coarse_matches.device),
                'confidence': torch.empty(batch_size, 0, device=coarse_matches.device)
            }
        
        padded_matches = []
        padded_confidence = []
        
        for matches, confidence in zip(fine_matches_list, fine_confidence_list):
            if len(matches) < max_matches:
                padding_matches = torch.zeros(max_matches - len(matches), 2, device=matches.device)
                padding_confidence = torch.zeros(max_matches - len(confidence), device=confidence.device)
                matches = torch.cat([matches, padding_matches], dim=0)
                confidence = torch.cat([confidence, padding_confidence], dim=0)
            
            padded_matches.append(matches)
            padded_confidence.append(confidence)
        
        return {
            'matches': torch.stack(padded_matches),
            'confidence': torch.stack(padded_confidence)
        }
    
    def sub_pixel_refinement(self, fine_matches: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Sub-pixel refinement for precise localization
        
        Args:
            fine_matches: Fine matching results
            
        Returns:
            Refined matches with sub-pixel accuracy
        """
        matches = fine_matches['matches']  # [B, M, 2]
        confidence = fine_matches['confidence']  # [B, M]
        
        batch_size, num_matches = matches.shape[:2]
        
        if num_matches == 0:
            return {
                'matches': matches,
                'offsets': torch.zeros_like(matches),
                'confidence': confidence
            }
        
        # この例では簡単な実装：実際にはlocal featuresを使用する
        # Sub-pixel offsets (regression-based refinement)
        refined_offsets = torch.zeros_like(matches, dtype=torch.float32)
        
        # ここで実際のsub-pixel refinementを実装
        # 例：local gradient-based refinement, parabolic fitting等
        
        refined_matches = matches.float() + refined_offsets
        
        return {
            'matches': refined_matches,
            'offsets': refined_offsets,
            'confidence': confidence
        }
    
    def compute_similarity_matrix(self, features0: torch.Tensor, features1: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity matrix between two feature sets
        """
        # L2 normalize features
        feat0_norm = F.normalize(features0, p=2, dim=-1)
        feat1_norm = F.normalize(features1, p=2, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.bmm(feat0_norm, feat1_norm.transpose(1, 2))
        
        # Apply temperature scaling
        similarity = similarity / self.temperature
        
        return similarity
    
    def dual_softmax(self, similarity_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply dual softmax for mutual nearest neighbor matching
        """
        # Row-wise softmax (image0 -> image1)
        prob_0to1 = F.softmax(similarity_matrix, dim=2)
        
        # Column-wise softmax (image1 -> image0)
        prob_1to0 = F.softmax(similarity_matrix, dim=1)
        
        return prob_0to1, prob_1to0
    
    def extract_coarse_matches(self, prob_0to1: torch.Tensor, prob_1to0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract coarse matches using mutual nearest neighbor strategy
        """
        batch_size, seq_len0, seq_len1 = prob_0to1.shape
        
        # Find mutual nearest neighbors
        max_prob_0to1, indices_0to1 = torch.max(prob_0to1, dim=2)
        max_prob_1to0, indices_1to0 = torch.max(prob_1to0, dim=1)
        
        matches_list = []
        confidence_list = []
        
        for b in range(batch_size):
            batch_matches = []
            batch_confidence = []
            
            for i in range(seq_len0):
                j = indices_0to1[b, i]
                if indices_1to0[b, j] == i:  # Mutual nearest neighbor
                    conf = (max_prob_0to1[b, i] + max_prob_1to0[b, j]) / 2
                    if conf > self.coarse_threshold:
                        batch_matches.append(torch.tensor([i, j], device=prob_0to1.device))
                        batch_confidence.append(conf)
            
            if batch_matches:
                matches_list.append(torch.stack(batch_matches))
                confidence_list.append(torch.stack(batch_confidence))
            else:
                matches_list.append(torch.empty(0, 2, device=prob_0to1.device))
                confidence_list.append(torch.empty(0, device=prob_0to1.device))
        
        # Pad to same length for batching
        max_matches = max(len(matches) for matches in matches_list)
        if max_matches == 0:
            return (torch.empty(batch_size, 0, 2, device=prob_0to1.device),
                   torch.empty(batch_size, 0, device=prob_0to1.device))
        
        padded_matches = []
        padded_confidence = []
        
        for matches, confidence in zip(matches_list, confidence_list):
            if len(matches) < max_matches:
                padding_matches = torch.zeros(max_matches - len(matches), 2, device=matches.device)
                padding_confidence = torch.zeros(max_matches - len(confidence), device=confidence.device)
                matches = torch.cat([matches, padding_matches], dim=0)
                confidence = torch.cat([confidence, padding_confidence], dim=0)
            
            padded_matches.append(matches)
            padded_confidence.append(confidence)
        
        return torch.stack(padded_matches), torch.stack(padded_confidence)
    
    def extract_local_window(self, features: torch.Tensor, center_idx: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract local window around a feature point
        """
        seq_len, feature_dim = features.shape
        half_window = self.fine_window_size // 2
        
        start_idx = max(0, center_idx - half_window)
        end_idx = min(seq_len, center_idx + half_window + 1)
        
        if end_idx - start_idx >= 3:  # Minimum window size
            return features[start_idx:end_idx]
        else:
            return None
    
    def match_local_windows(self, window_0: torch.Tensor, window_1: torch.Tensor) -> Tuple[Optional[torch.Tensor], float]:
        """
        Match two local windows using MLP-Mixer
        """
        # Expand dimensions for MLP-Mixer
        window_0_expanded = window_0.unsqueeze(0).unsqueeze(0)  # [1, 1, W, D]
        window_1_expanded = window_1.unsqueeze(0).unsqueeze(0)  # [1, 1, W, D]
        
        # Process with MLP-Mixer
        processed_0 = self.fine_mixer(window_0_expanded)
        processed_1 = self.fine_mixer(window_1_expanded)
        
        # Compute similarity and find best match
        similarity = F.cosine_similarity(
            processed_0.flatten(1), 
            processed_1.flatten(1), 
            dim=1
        )
        
        confidence = similarity.item()
        
        if confidence > 0.5:  # Threshold for fine matching
            # Return center indices (simplified)
            center_0 = torch.tensor(window_0.shape[0] // 2)
            center_1 = torch.tensor(window_1.shape[0] // 2)
            return torch.stack([center_0, center_1]), confidence
        else:
            return None, 0.0
    
    def local_to_global_coords(self, global_center: torch.Tensor, local_offset: torch.Tensor) -> torch.Tensor:
        """
        Convert local coordinates to global coordinates
        """
        half_window = self.fine_window_size // 2
        return global_center - half_window + local_offset


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for feature enhancement
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.norm0 = nn.LayerNorm(feature_dim)
        self.norm1 = nn.LayerNorm(feature_dim)
    
    def forward(self, features0: torch.Tensor, features1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention between two feature sets
        """
        # Self-attention + Cross-attention for each feature set
        enhanced_0 = self.cross_attention(features0, features1)
        enhanced_1 = self.cross_attention(features1, features0)
        
        return enhanced_0, enhanced_1
    
    def cross_attention(self, query_features: torch.Tensor, key_value_features: torch.Tensor) -> torch.Tensor:
        """
        Perform cross-attention
        """
        batch_size, seq_len, feature_dim = query_features.shape
        
        # Residual connection
        residual = query_features
        
        # Layer norm
        query_features = self.norm0(query_features)
        key_value_features = self.norm1(key_value_features)
        
        # Project to Q, K, V
        Q = self.q_proj(query_features)
        K = self.k_proj(key_value_features)
        V = self.v_proj(key_value_features)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, feature_dim
        )
        output = self.out_proj(attended_values)
        
        # Residual connection
        return residual + output


class MLPMixer(nn.Module):
    """
    MLP-Mixer for fine-level feature processing
    Adapted for local window matching
    """
    
    def __init__(self, patch_size: int, feature_dim: int, num_layers: int = 2):
        super().__init__()
        
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        
        self.layers = nn.ModuleList([
            MLPMixerLayer(patch_size, feature_dim) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP-Mixer layers
        
        Args:
            x: Input tensor [B, 1, W, D]
            
        Returns:
            Processed tensor [B, 1, W, D]
        """
        batch_size, num_patches, seq_len, feature_dim = x.shape
        
        # Flatten for processing
        x = x.view(batch_size * num_patches, seq_len, feature_dim)
        
        # Apply MLP-Mixer layers
        for layer in self.layers:
            x = layer(x)
        
        # Normalize
        x = self.norm(x)
        
        # Restore shape
        x = x.view(batch_size, num_patches, seq_len, feature_dim)
        
        return x


class MLPMixerLayer(nn.Module):
    """
    Single MLP-Mixer layer
    """
    
    def __init__(self, seq_len: int, feature_dim: int, expansion_factor: int = 4):
        super().__init__()
        
        # Token mixing (across sequence dimension)
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(seq_len, seq_len * expansion_factor),
            nn.GELU(),
            nn.Linear(seq_len * expansion_factor, seq_len)
        )
        
        # Channel mixing (across feature dimension)
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(feature_dim * expansion_factor, feature_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, L, D]
            
        Returns:
            Output tensor [B, L, D]
        """
        # Token mixing
        residual = x
        x_transposed = x.transpose(1, 2)  # [B, D, L]
        x_mixed = self.token_mixing(x_transposed)
        x = residual + x_mixed.transpose(1, 2)  # [B, L, D]
        
        # Channel mixing
        residual = x
        x_channel_mixed = self.channel_mixing(x)
        x = residual + x_channel_mixed
        
        return x