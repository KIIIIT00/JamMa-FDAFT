"""
Joint Mamba (JEGO Strategy) 実装

JamMaのJoint Efficient Global Omnidirectional戦略を実装
FDAFTから抽出された特徴に対してJoint scanとMambaブロックを適用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import math


class MambaBlock(nn.Module):
    """
    Mamba Block実装
    
    State Space Model (S6)を使用した効率的な長距離依存関係モデリング
    """
    
    def __init__(self, 
                 d_model: int,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2):
        """
        Initialize Mamba block
        
        Args:
            d_model: モデル次元
            d_state: SSM状態次元
            d_conv: 畳み込みカーネルサイズ
            expand: 中間層の拡張係数
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        
        # 入力投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 1D畳み込み
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # SSMパラメータ
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # A parameter (state transition matrix)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).repeat(self.d_inner, 1)  # [d_inner, d_state]
        A = -A  # For stability
        self.A_log = nn.Parameter(torch.log(A))
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 出力投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # LayerNorm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, L, D]
            
        Returns:
            Output tensor [B, L, D]
        """
        batch, length, dim = x.shape
        
        # 残差接続用
        residual = x
        
        # LayerNorm
        x = self.norm(x)
        
        # 入力投影 [B, L, D] -> [B, L, 2*d_inner]
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # [B, L, d_inner] each
        
        # 1D畳み込み
        x = x.transpose(1, 2)  # [B, d_inner, L]
        x = self.conv1d(x)[:, :, :length]  # Causal masking
        x = x.transpose(1, 2)  # [B, L, d_inner]
        
        # SiLU activation
        x = F.silu(x)
        
        # SSM computation
        y = self.ssm(x)
        
        # Gating
        y = y * F.silu(z)
        
        # 出力投影
        output = self.out_proj(y)
        
        # 残差接続
        return residual + output
    
    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        State Space Model computation
        """
        batch, length, d_inner = x.shape
        
        # State space parameters
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        D = self.D.float()
        
        # Project x to get B and C
        x_dbl = self.x_proj(x)  # [B, L, 2*d_state]
        delta, B, C = x_dbl[:, :, :d_inner], x_dbl[:, :, d_inner:d_inner+self.d_state], x_dbl[:, :, d_inner+self.d_state:]
        
        # Delta projection and softplus
        delta = F.softplus(self.dt_proj(delta))  # [B, L, d_inner]
        
        # Discretize A and B
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [B, L, d_inner, d_state]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # [B, L, d_inner, d_state]
        
        # SSM scan (simplified version for efficiency)
        y = self.selective_scan(x, deltaA, deltaB, C, D)
        
        return y
    
    def selective_scan(self, x: torch.Tensor, deltaA: torch.Tensor, 
                      deltaB: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        """
        Simplified selective scan implementation
        """
        batch, length, d_inner = x.shape
        d_state = deltaA.shape[-1]
        
        # Initialize hidden state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        for i in range(length):
            # Update hidden state
            h = deltaA[:, i] * h + deltaB[:, i] * x[:, i:i+1].transpose(1, 2)
            
            # Output
            y = torch.sum(h * C[:, i:i+1].transpose(1, 2), dim=-1)
            y = y + D * x[:, i]
            
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)


class JEGOMamba(nn.Module):
    """
    Joint Efficient Global Omnidirectional Mamba
    
    JamMaのJEGO戦略を実装:
    1. Joint scan: 2つの画像を交互にスキャン
    2. Efficient scan: スキップステップで効率化
    3. Global receptive field: 全方向スキャンで実現
    4. Omnidirectional: 4方向のスキャンとアグリゲータ
    """
    
    def __init__(self,
                 input_dim: int = 256,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 skip_steps: int = 2):
        """
        Initialize JEGO Mamba
        
        Args:
            input_dim: 入力特徴次元
            hidden_dim: 隠れ層次元
            num_layers: Mambaレイヤー数
            num_heads: アテンション用ヘッド数（アグリゲータ用）
            skip_steps: スキップスキャンのステップ数
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip_steps = skip_steps
        
        # 特徴投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 4方向のMambaブロック
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(hidden_dim) for _ in range(4)  # 4 directions
        ])
        
        # アグリゲータ（gated convolutional unit）
        self.aggregator = GatedConvolutionalAggregator(hidden_dim)
        
        # 出力投影
        self.output_projection = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, features0: torch.Tensor, features1: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with JEGO strategy
        
        Args:
            features0: Image 0 features [B, N, D]
            features1: Image 1 features [B, N, D]
            
        Returns:
            Enhanced features after JEGO processing
        """
        batch_size, seq_len, _ = features0.shape
        
        # Step 1: 特徴投影
        proj_features0 = self.input_projection(features0)
        proj_features1 = self.input_projection(features1)
        
        # Step 2: JEGO Scan - 4方向のJoint scan
        scanned_sequences = self.jego_scan(proj_features0, proj_features1)
        
        # Step 3: Mamba処理 - 各方向独立に処理
        processed_sequences = []
        for i, sequence in enumerate(scanned_sequences):
            processed = self.mamba_blocks[i](sequence)
            processed_sequences.append(processed)
        
        # Step 4: JEGO Merge - バランスの取れた受容野でマージ
        merged_features = self.jego_merge(processed_sequences, seq_len)
        
        # Step 5: アグリゲータで全方向情報を統合
        aggregated_features = self.aggregator(merged_features)
        
        # Step 6: 出力投影
        enhanced_features0, enhanced_features1 = self.split_joint_features(
            aggregated_features, seq_len
        )
        
        enhanced_features0 = self.output_projection(enhanced_features0)
        enhanced_features1 = self.output_projection(enhanced_features1)
        
        return {
            'enhanced_features0': enhanced_features0,
            'enhanced_features1': enhanced_features1
        }
    
    def jego_scan(self, features0: torch.Tensor, features1: torch.Tensor) -> list:
        """
        JEGO Scan: Joint + Efficient + 4-directional scanning
        
        Args:
            features0, features1: Input features [B, N, D]
            
        Returns:
            List of 4 scanned sequences
        """
        batch_size, seq_len, hidden_dim = features0.shape
        
        # 2D座標に戻すためのreshape (仮定: square image)
        H = W = int(math.sqrt(seq_len))
        if H * W != seq_len:
            # 非正方形の場合の処理
            H = int(math.sqrt(seq_len))
            W = seq_len // H
        
        # [B, N, D] → [B, H, W, D]
        feat0_2d = features0.view(batch_size, H, W, hidden_dim)
        feat1_2d = features1.view(batch_size, H, W, hidden_dim)
        
        # Joint concatenation (水平・垂直)
        joint_horizontal = torch.cat([feat0_2d, feat1_2d], dim=2)  # [B, H, 2W, D]
        joint_vertical = torch.cat([feat0_2d, feat1_2d], dim=1)    # [B, 2H, W, D]
        
        # 4方向スキャンシーケンス生成
        sequences = []
        
        # 1. Right scan (水平右方向)
        right_seq = self.scan_right_with_skip(joint_horizontal)
        sequences.append(right_seq)
        
        # 2. Left scan (水平左方向)
        left_seq = self.scan_left_with_skip(joint_horizontal)
        sequences.append(left_seq)
        
        # 3. Down scan (垂直下方向)
        down_seq = self.scan_down_with_skip(joint_vertical)
        sequences.append(down_seq)
        
        # 4. Up scan (垂直上方向)
        up_seq = self.scan_up_with_skip(joint_vertical)
        sequences.append(up_seq)
        
        return sequences
    
    def scan_right_with_skip(self, joint_features: torch.Tensor) -> torch.Tensor:
        """
        右方向スキャン（スキップ付き）with Joint scan
        """
        batch_size, H, W, D = joint_features.shape
        sequence = []
        
        for h in range(0, H, self.skip_steps):
            for w in range(0, W, self.skip_steps):
                if h < H and w < W:
                    sequence.append(joint_features[:, h, w, :])
        
        if sequence:
            return torch.stack(sequence, dim=1)  # [B, L/4, D]
        else:
            return torch.empty(batch_size, 0, D, device=joint_features.device)
    
    def scan_left_with_skip(self, joint_features: torch.Tensor) -> torch.Tensor:
        """
        左方向スキャン（スキップ付き）with Joint scan
        """
        batch_size, H, W, D = joint_features.shape
        sequence = []
        
        for h in range(0, H, self.skip_steps):
            for w in range(W-1, -1, -self.skip_steps):
                if h < H and w >= 0:
                    sequence.append(joint_features[:, h, w, :])
        
        if sequence:
            return torch.stack(sequence, dim=1)
        else:
            return torch.empty(batch_size, 0, D, device=joint_features.device)
    
    def scan_down_with_skip(self, joint_features: torch.Tensor) -> torch.Tensor:
        """
        下方向スキャン（スキップ付き）with Joint scan
        """
        batch_size, H, W, D = joint_features.shape
        sequence = []
        
        for w in range(0, W, self.skip_steps):
            for h in range(0, H, self.skip_steps):
                if h < H and w < W:
                    sequence.append(joint_features[:, h, w, :])
        
        if sequence:
            return torch.stack(sequence, dim=1)
        else:
            return torch.empty(batch_size, 0, D, device=joint_features.device)
    
    def scan_up_with_skip(self, joint_features: torch.Tensor) -> torch.Tensor:
        """
        上方向スキャン（スキップ付き）with Joint scan
        """
        batch_size, H, W, D = joint_features.shape
        sequence = []
        
        for w in range(0, W, self.skip_steps):
            for h in range(H-1, -1, -self.skip_steps):
                if h >= 0 and w < W:
                    sequence.append(joint_features[:, h, w, :])
        
        if sequence:
            return torch.stack(sequence, dim=1)
        else:
            return torch.empty(batch_size, 0, D, device=joint_features.device)
    
    def jego_merge(self, processed_sequences: list, original_seq_len: int) -> torch.Tensor:
        """
        JEGO Merge: バランスの取れた受容野でシーケンスをマージ
        
        Args:
            processed_sequences: 4方向の処理済みシーケンス
            original_seq_len: 元のシーケンス長
            
        Returns:
            マージされた特徴 [B, 2*N, D] (2つの画像の結合特徴)
        """
        batch_size = processed_sequences[0].shape[0]
        hidden_dim = processed_sequences[0].shape[-1]
        
        # 2D座標系に戻すための準備
        H = W = int(math.sqrt(original_seq_len))
        if H * W != original_seq_len:
            H = int(math.sqrt(original_seq_len))
            W = original_seq_len // H
        
        # 4方向のシーケンスを2D特徴マップに復元
        horizontal_map = torch.zeros(batch_size, H, 2*W, hidden_dim, device=processed_sequences[0].device)
        vertical_map = torch.zeros(batch_size, 2*H, W, hidden_dim, device=processed_sequences[0].device)
        
        # Right/Left scan復元
        self.restore_horizontal_scans(processed_sequences[0], processed_sequences[1], 
                                    horizontal_map, H, W)
        
        # Up/Down scan復元
        self.restore_vertical_scans(processed_sequences[2], processed_sequences[3], 
                                  vertical_map, H, W)
        
        # 水平・垂直特徴を分離
        horizontal_feat0 = horizontal_map[:, :, :W, :]
        horizontal_feat1 = horizontal_map[:, :, W:, :]
        vertical_feat0 = vertical_map[:, :H, :, :]
        vertical_feat1 = vertical_map[:, H:, :, :]
        
        # 水平・垂直特徴を合成
        merged_feat0 = horizontal_feat0 + vertical_feat0
        merged_feat1 = horizontal_feat1 + vertical_feat1
        
        # 平坦化して結合
        merged_feat0_flat = merged_feat0.view(batch_size, -1, hidden_dim)
        merged_feat1_flat = merged_feat1.view(batch_size, -1, hidden_dim)
        
        # 2つの画像の特徴を結合
        joint_features = torch.cat([merged_feat0_flat, merged_feat1_flat], dim=1)
        
        return joint_features
    
    def restore_horizontal_scans(self, right_seq: torch.Tensor, left_seq: torch.Tensor,
                               target_map: torch.Tensor, H: int, W: int):
        """
        水平スキャンを2Dマップに復元
        """
        # Right scan復元
        seq_idx = 0
        for h in range(0, H, self.skip_steps):
            for w in range(0, 2*W, self.skip_steps):
                if seq_idx < right_seq.shape[1] and h < H and w < 2*W:
                    target_map[:, h, w, :] = right_seq[:, seq_idx, :]
                    seq_idx += 1
        
        # Left scan復元（処理を簡略化）
        # 実際の実装では座標の対応をより詳細に行う
    
    def restore_vertical_scans(self, down_seq: torch.Tensor, up_seq: torch.Tensor,
                             target_map: torch.Tensor, H: int, W: int):
        """
        垂直スキャンを2Dマップに復元
        """
        # Down scan復元
        seq_idx = 0
        for w in range(0, W, self.skip_steps):
            for h in range(0, 2*H, self.skip_steps):
                if seq_idx < down_seq.shape[1] and h < 2*H and w < W:
                    target_map[:, h, w, :] = down_seq[:, seq_idx, :]
                    seq_idx += 1
        
        # Up scan復元（処理を簡略化）
    
    def split_joint_features(self, joint_features: torch.Tensor, 
                           original_seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        結合特徴を2つの画像の特徴に分離
        """
        # joint_features: [B, 2*N, D] → [B, N, D], [B, N, D]
        features0 = joint_features[:, :original_seq_len, :]
        features1 = joint_features[:, original_seq_len:, :]
        
        return features0, features1


class GatedConvolutionalAggregator(nn.Module):
    """
    Gated Convolutional Aggregator
    
    4方向からの情報を統合するためのアグリゲータ
    JamMa論文のequation (8)に対応
    """
    
    def __init__(self, hidden_dim: int, kernel_size: int = 3):
        """
        Initialize aggregator
        
        Args:
            hidden_dim: 隠れ層次元
            kernel_size: 畳み込みカーネルサイズ
        """
        super().__init__()
        
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        
        self.gelu = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features [B, L, D]
            
        Returns:
            Aggregated features [B, L, D]
        """
        # [B, L, D] → [B, D, L] for Conv1d
        x_conv = x.transpose(1, 2)
        
        # Gated convolution (equation 8)
        # σ = GELU(Conv3(F̃^c))
        sigma = self.gelu(self.conv1(x_conv))
        
        # F̂^c = Conv3(σ · Conv3(F̃^c))
        gated = sigma * self.conv2(x_conv)
        output_conv = self.conv3(gated)
        
        # [B, D, L] → [B, L, D]
        output = output_conv.transpose(1, 2)
        
        # 残差接続
        return x + output