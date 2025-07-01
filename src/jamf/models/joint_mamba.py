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
        
        # 2D座標に戻すためのreshape