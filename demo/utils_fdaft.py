"""
JamMa-FDAFT Demo Utilities

JamMaの学習済みモデルを使用するFDAFT統合版のユーティリティ
demo/utils.pyのFDAFT対応版（完全互換性確保）
"""

import os
import sys
import torch
from torch import nn

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.jamma.jamma import JamMa as JamMa_
from src.jamma.backbone import CovNextV2_nano
from src.jamma_fdaft.backbone_fdaft import FDAFTEncoder
from loguru import logger


class JamMaFDAFTDemo(nn.Module):
    """
    JamMa-FDAFT Demo Model
    
    JamMaの学習済みモデルを使用してFDAFT特徴を処理
    demo/utils.pyのJamMaクラスと同じインターフェース（完全互換）
    """
    
    def __init__(self, config, pretrained='official') -> None:
        super().__init__()
        
        # FDAFTバックボーン（FDAFT設定を辞書から作成）
        self.backbone = self._create_fdaft_backbone(config)
        
        # JamMaマッチャー（オリジナルのJamMaクラスを使用）
        self.matcher = JamMa_(config)
        
        # JamMaの学習済みモデルをロード
        self._load_pretrained_weights(pretrained)

    def _create_fdaft_backbone(self, config):
        """JamMa設定からFDAFT設定を作成してFDAFTバックボーンを初期化"""
        # FDAFTに必要な設定を作成
        fdaft_config = type('FDAFTConfig', (), {})()
        fdaft_config.FDAFT = type('FDAFT', (), {})()
        fdaft_config.JAMMA = type('JAMMA', (), {})()
        fdaft_config.JAMMA.COARSE = type('COARSE', (), {})()
        fdaft_config.JAMMA.FINE = type('FINE', (), {})()
        
        # デフォルト値を設定
        fdaft_config.FDAFT.NUM_LAYERS = 3
        fdaft_config.FDAFT.SIGMA_0 = 1.0
        fdaft_config.FDAFT.USE_STRUCTURED_FORESTS = True
        fdaft_config.FDAFT.MAX_KEYPOINTS = 2000
        fdaft_config.FDAFT.NMS_RADIUS = 5
        
        # JamMa設定から次元を取得
        fdaft_config.JAMMA.COARSE.D_MODEL = config['coarse']['d_model']
        fdaft_config.JAMMA.FINE.D_MODEL = config['fine']['d_model']
        fdaft_config.JAMMA.RESOLUTION = config['resolution']
        fdaft_config.JAMMA.FINE_WINDOW_SIZE = config['fine_window_size']
        
        return FDAFTEncoder.from_config(fdaft_config)

    def _load_pretrained_weights(self, pretrained):
        """JamMaの学習済みモデルをロード"""
        if pretrained == 'official':
            try:
                # JamMaの公式学習済みモデルをロード
                state_dict = torch.hub.load_state_dict_from_url(
                    'https://github.com/leoluxxx/JamMa/releases/download/v0.1/jamma.ckpt',
                    file_name='jamma.ckpt'
                )['state_dict']
                
                # マッチャー部分のみを抽出
                matcher_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('matcher.'):
                        # matcher.xxx -> xxx に変換
                        new_key = key[8:]  # "matcher."を除去
                        matcher_state_dict[new_key] = value
                
                # マッチャーのみに重みをロード（バックボーンは除外）
                missing_keys, unexpected_keys = self.matcher.load_state_dict(
                    matcher_state_dict, strict=False
                )
                
                logger.info(f"✓ JamMa official weights loaded successfully")
                logger.info(f"  Missing keys (expected for FDAFT): {len(missing_keys)}")
                logger.info(f"  Unexpected keys: {len(unexpected_keys)}")
                
            except Exception as e:
                logger.warning(f"Failed to load official JamMa weights: {e}")
                logger.info("Using random initialization")
                
        elif pretrained:
            try:
                # カスタム学習済みモデルをロード
                state_dict = torch.load(pretrained, map_location='cpu')['state_dict']
                
                # マッチャー部分のみを抽出
                matcher_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('matcher.'):
                        new_key = key[8:]
                        matcher_state_dict[new_key] = value
                
                missing_keys, unexpected_keys = self.matcher.load_state_dict(
                    matcher_state_dict, strict=False
                )
                
                logger.info(f"✓ JamMa weights loaded from: {pretrained}")
                logger.info(f"  Missing keys (expected for FDAFT): {len(missing_keys)}")
                logger.info(f"  Unexpected keys: {len(unexpected_keys)}")
                
            except Exception as e:
                logger.warning(f"Failed to load JamMa weights from {pretrained}: {e}")
                logger.info("Using random initialization")

    def forward(self, data):
        """Forward pass - demo/utils.pyのJamMaクラスと同じインターフェース"""
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