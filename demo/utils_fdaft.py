"""
JamMa-FDAFT Demo Utilities

JamMaの学習済みモデルを使用するFDAFT統合版のユーティリティ
demo/utils.pyのFDAFT対応版
"""

import torch
from torch import nn
from src.jamma_fdaft.jamma_fdaft import JamMaFDAFT
from src.jamma_fdaft.backbone_fdaft import FDAFTEncoder
from loguru import logger


class JamMaFDAFTDemo(nn.Module):
    """
    JamMa-FDAFT Demo Model
    
    JamMaの学習済みモデルを使用してFDAFT特徴を処理
    demo/utils.pyのJamMaクラスと同じインターフェース
    """
    
    def __init__(self, config, pretrained='official') -> None:
        super().__init__()
        
        # FDAFTバックボーン
        self.backbone = FDAFTEncoder.from_config(self._create_fdaft_config(config))
        
        # JamMaマッチャー
        self.matcher = JamMaFDAFT(config)

        # JamMaの学習済みモデルをロード
        self._load_pretrained_weights(pretrained)

    def _create_fdaft_config(self, jamma_config):
        """JamMa設定からFDAFT設定を作成"""
        # 簡易的なFDAFT設定を作成
        class FDAFTConfig:
            class FDAFT:
                NUM_LAYERS = 3
                SIGMA_0 = 1.0
                USE_STRUCTURED_FORESTS = True
                MAX_KEYPOINTS = 2000
                NMS_RADIUS = 5
            
            class JAMMA:
                class COARSE:
                    D_MODEL = jamma_config['coarse']['d_model']
                class FINE:
                    D_MODEL = jamma_config['fine']['d_model']
                RESOLUTION = jamma_config['resolution']
                FINE_WINDOW_SIZE = jamma_config['fine_window_size']
        
        return FDAFTConfig()

    def _load_pretrained_weights(self, pretrained):
        """JamMaの学習済みモデルをロード"""
        if pretrained == 'official':
            try:
                missing_keys, unexpected_keys = self.matcher.load_jamma_pretrained_weights('official')
                logger.info(f"✓ JamMa official weights loaded successfully")
                logger.info(f"  Missing keys (FDAFT): {len(missing_keys)}")
                logger.info(f"  Unexpected keys: {len(unexpected_keys)}")
            except Exception as e:
                logger.warning(f"Failed to load official JamMa weights: {e}")
                logger.info("Using random initialization")
        elif pretrained:
            try:
                missing_keys, unexpected_keys = self.matcher.load_jamma_pretrained_weights(pretrained)
                logger.info(f"✓ JamMa weights loaded from: {pretrained}")
                logger.info(f"  Missing keys (FDAFT): {len(missing_keys)}")
                logger.info(f"  Unexpected keys: {len(unexpected_keys)}")
            except Exception as e:
                logger.warning(f"Failed to load JamMa weights from {pretrained}: {e}")
                logger.info("Using random initialization")

    def forward(self, data):
        """Forward pass - same interface as demo/utils.py JamMa class"""
        # 1. FDAFT特徴抽出
        self.backbone(data)
        
        # 2. JamMaマッチング（学習済みモデル使用）
        return self.matcher(data)


# JamMa互換の設定（demo/utils.pyと同じ形式）
cfg = {
    'coarse': {
        'd_model': 256,
    },
    'fine': {
        'd_model': 64,
        'dsmax_temperature': 0.1,
        'thr': 0.1,
        'inference': True
    },
    'match_coarse': {
        'thr': 0.2,
        'use_sm': True,
        'border_rm': 2,
        'dsmax_temperature': 0.1,
        'inference': True
    },
    'fine_window_size': 5,
    'resolution': [8, 2]
}


# エイリアス（demo/utils.pyと同じ名前でインポートできるように）
JamMa = JamMaFDAFTDemo