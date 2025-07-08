"""
修正されたJamMa-FDAFT Complete Demonstration Script

主な修正点：
- FDAFTエンコーダーの出力次元とJamMaの期待次元の適切な適合
- 次元適応レイヤーの修正
- エラーハンドリングの改善
- NameErrorの修正
"""

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn

# プロジェクトルートを正しく設定
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # demo/ -> プロジェクトルート
src_path = os.path.join(project_root, 'src')

# パスを追加
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print(f"Project root: {project_root}")
print(f"Src path: {src_path}")

try:
    # JamMa-FDAFT関連のインポート
    from src.jamma_fdaft.backbone_fdaft import FDAFTEncoder
    from src.jamma.jamma import JamMa
    from src.jamma.backbone import CovNextV2_nano
    from src.config.default import get_cfg_defaults
    from src.utils.plotting import make_matching_figures
    from src.utils.dataset import read_megadepth_color
    import torch.nn.functional as F
    from src.utils.misc import lower_config
    print("✅ すべてのモジュールが正常にインポートされました")
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    sys.exit(1)


class JamMaFDAFTDemo(nn.Module):
    """
    JamMa-FDAFT統合モデル（デモ用）修正版
    FDAFTエンコーダー + JamMaの学習済みモデルを組み合わせ
    次元の不一致問題を解決
    """
    
    def __init__(self, config, pretrained_jamma='official'):
        super().__init__()
        self.config = config
        
        print("🔧 JamMa-FDAFTデモモデルを初期化中...")
        
        self.jamma_config = self._convert_config_to_dict(config)
        
        # FDAFTエンコーダーを初期化
        try:
            self.fdaft_backbone = FDAFTEncoder.from_config(config)
            print("✅ FDAFTバックボーン初期化完了")
        except Exception as e:
            print(f"❌ FDAFTバックボーン初期化失敗: {e}")
            # フォールバック: ConvNextV2を使用
            print("🔄 ConvNextV2バックボーンにフォールバック")
            self.fdaft_backbone = CovNextV2_nano()
        
        # JamMaの学習済みモデルを読み込み
        try:
            # 辞書形式の設定をJamMaに渡す
            self.jamma_matcher = JamMa(config=self.jamma_config, profiler=None)
            print("✅ JamMaマッチャー初期化完了")
        except Exception as e:
            print(f"❌ JamMaマッチャー初期化失敗: {e}")
            raise
        
        # JamMaの学習済み重みを読み込み
        if pretrained_jamma == 'official':
            try:
                print("📥 JamMa学習済みモデルをダウンロード中...")
                state_dict = torch.hub.load_state_dict_from_url(
                    'https://github.com/leoluxxx/JamMa/releases/download/v0.1/jamma.ckpt',
                    file_name='jamma.ckpt')['state_dict']
                
                # JamMa部分のみ読み込み（backboneは除外）
                jamma_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('matcher.'):
                        # matcher.xxx -> xxx に変換
                        new_key = key[8:]  # "matcher."を除去
                        jamma_state_dict[new_key] = value
                
                self.jamma_matcher.load_state_dict(jamma_state_dict, strict=False)
                print("✅ JamMa学習済みモデルを読み込み完了")
                
            except Exception as e:
                print(f"⚠️ JamMa学習済みモデルの読み込みに失敗: {e}")
                print("🔄 スクラッチから初期化します")
        
        # 動的次元適応レイヤー（実際の特徴量サイズに基づいて調整）
        # これらのレイヤーは最初のforward時に初期化される
        self.dimension_adapter_8 = None
        self.dimension_adapter_4 = None
        self._adapters_initialized = False
    
    def _convert_config_to_dict(self, yacs_config):
        """YACS設定を辞書形式に変換してJamMa互換にする"""
        try:
            # YACSオブジェクトから必要な設定を取得
            jamma_cfg = yacs_config.JAMMA
            
            config_dict = {
                'coarse': {
                    'd_model': jamma_cfg.COARSE.D_MODEL,
                },
                'fine': {
                    'd_model': jamma_cfg.FINE.D_MODEL,
                    'dsmax_temperature': getattr(jamma_cfg.FINE, 'DSMAX_TEMPERATURE', 0.1),
                    'thr': jamma_cfg.FINE.THR,
                    'inference': jamma_cfg.FINE.INFERENCE
                },
                'match_coarse': {
                    'thr': jamma_cfg.MATCH_COARSE.THR,
                    'use_sm': jamma_cfg.MATCH_COARSE.USE_SM,
                    'border_rm': jamma_cfg.MATCH_COARSE.BORDER_RM,
                    'dsmax_temperature': getattr(jamma_cfg.MATCH_COARSE, 'DSMAX_TEMPERATURE', 0.1),
                    'inference': jamma_cfg.MATCH_COARSE.INFERENCE,
                    'train_coarse_percent': getattr(jamma_cfg.MATCH_COARSE, 'TRAIN_COARSE_PERCENT', 0.3),
                    'train_pad_num_gt_min': getattr(jamma_cfg.MATCH_COARSE, 'TRAIN_PAD_NUM_GT_MIN', 20)
                },
                'fine_window_size': jamma_cfg.FINE_WINDOW_SIZE,
                'resolution': list(jamma_cfg.RESOLUTION)  # tupleをlistに変換
            }
            
            return config_dict
            
        except Exception as e:
            print(f"設定変換エラー: {e}")
            # フォールバック設定
            return {
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
                    'inference': True,
                    'train_coarse_percent': 0.3,
                    'train_pad_num_gt_min': 20
                },
                'fine_window_size': 5,
                'resolution': [8, 2]
            }
    
    def _initialize_adapters(self, feat_8_shape, feat_4_shape):
        """実際の特徴量サイズに基づいて次元適応レイヤーを初期化"""
        print(f"🔧 次元適応レイヤーを初期化中...")
        print(f"  feat_8 shape: {feat_8_shape}")
        print(f"  feat_4 shape: {feat_4_shape}")
        
        # Coarse features (1/8 resolution) の次元適応
        fdaft_8_dim = feat_8_shape[1]  # チャンネル数を取得
        jamma_8_dim = self.jamma_config['coarse']['d_model']  # 256
        
        if fdaft_8_dim != jamma_8_dim:
            self.dimension_adapter_8 = nn.Sequential(
                nn.Conv2d(fdaft_8_dim, jamma_8_dim, kernel_size=1),
                nn.BatchNorm2d(jamma_8_dim),
                nn.GELU()
            ).to(next(self.parameters()).device)
            print(f"  ✅ Coarse adapter: {fdaft_8_dim} -> {jamma_8_dim}")
        else:
            self.dimension_adapter_8 = nn.Identity()
            print(f"  ✅ Coarse adapter: Identity (dimensions match)")
        
        # Fine features (1/4 resolution) の次元適応
        fdaft_4_dim = feat_4_shape[1]
        jamma_4_dim = self.jamma_config['fine']['d_model']  # 64
        
        if fdaft_4_dim != jamma_4_dim:
            self.dimension_adapter_4 = nn.Sequential(
                nn.Conv2d(fdaft_4_dim, jamma_4_dim, kernel_size=1),
                nn.BatchNorm2d(jamma_4_dim),
                nn.GELU()
            ).to(next(self.parameters()).device)
            print(f"  ✅ Fine adapter: {fdaft_4_dim} -> {jamma_4_dim}")
        else:
            self.dimension_adapter_4 = nn.Identity()
            print(f"  ✅ Fine adapter: Identity (dimensions match)")
        
        self._adapters_initialized = True
    
    def forward(self, data):
        """統合フォワードパス"""
        try:
            print("  🔄 FDAFTによる特徴抽出開始...")
            # 1. FDAFTで特徴抽出
            self.fdaft_backbone(data)
            
            # 2. 次元適応レイヤーの初期化（最初の呼び出し時のみ）
            if not self._adapters_initialized:
                if 'feat_8_0' in data and 'feat_4_0' in data:
                    self._initialize_adapters(data['feat_8_0'].shape, data['feat_4_0'].shape)
                else:
                    raise RuntimeError("FDAFT backbone did not produce expected features")
            
            print("  🔄 次元適応処理中...")
            # 3. 次元適応
            if 'feat_8_0' in data and 'feat_8_1' in data:
                original_shape_8 = data['feat_8_0'].shape
                data['feat_8_0'] = self.dimension_adapter_8(data['feat_8_0'])
                data['feat_8_1'] = self.dimension_adapter_8(data['feat_8_1'])
                print(f"    feat_8: {original_shape_8} -> {data['feat_8_0'].shape}")
            
            if 'feat_4_0' in data and 'feat_4_1' in data:
                original_shape_4 = data['feat_4_0'].shape
                data['feat_4_0'] = self.dimension_adapter_4(data['feat_4_0'])
                data['feat_4_1'] = self.dimension_adapter_4(data['feat_4_1'])
                print(f"    feat_4: {original_shape_4} -> {data['feat_4_0'].shape}")
            
            # デバッグ: マッチング前の特徴量とマスクの形状確認
            print("  🔍 JamMaマッチング前の確認:")
            print(f"    feat_8_0: {data['feat_8_0'].shape}")
            print(f"    feat_8_1: {data['feat_8_1'].shape}")
            if 'mask0' in data and data['mask0'] is not None:
                print(f"    mask0: {data['mask0'].shape}")
                print(f"    mask1: {data['mask1'].shape}")
                
                # JamMaの処理に合わせてflat化した際のサイズを予測
                b, c, h, w = data['feat_8_0'].shape
                hw = h * w
                print(f"    予想される flatten size: {hw}")
                print(f"    mask flatten size: {data['mask0'].numel()}")
                
                # マスクのサイズが特徴量と一致するか確認
                expected_mask_shape = (b, h, w)  # [B, H, W]
                if data['mask0'].shape != expected_mask_shape:
                    print(f"    ⚠️ マスクサイズ不一致! 期待: {expected_mask_shape}, 実際: {data['mask0'].shape}")
                    # マスクを正しいサイズにリサイズ
                    
                    # マスクの次元を確認して適切に処理
                    mask0 = data['mask0']
                    mask1 = data['mask1']
                    
                    # 3次元の場合: [B, H, W] -> 必要に応じてリサイズ
                    if mask0.dim() == 3 and mask0.shape != expected_mask_shape:
                        mask0 = F.interpolate(
                            mask0.float().unsqueeze(1),  # [B, 1, H, W]
                            size=(h, w),
                            mode='nearest'
                        ).squeeze(1).bool()  # [B, H, W]
                        
                        mask1 = F.interpolate(
                            mask1.float().unsqueeze(1),  # [B, 1, H, W]
                            size=(h, w),
                            mode='nearest'
                        ).squeeze(1).bool()  # [B, H, W]
                    
                    # 2次元の場合: [H, W] -> [B, H, W]
                    elif mask0.dim() == 2:
                        # バッチ次元を追加してリサイズ
                        mask0 = F.interpolate(
                            mask0.float().unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
                            size=(h, w),
                            mode='nearest'
                        ).squeeze(0).bool()  # [H, W]
                        mask0 = mask0.unsqueeze(0).repeat(b, 1, 1)  # [B, H, W]
                        
                        mask1 = F.interpolate(
                            mask1.float().unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
                            size=(h, w),
                            mode='nearest'
                        ).squeeze(0).bool()  # [H, W]
                        mask1 = mask1.unsqueeze(0).repeat(b, 1, 1)  # [B, H, W]
                    
                    data['mask0'] = mask0
                    data['mask1'] = mask1
                    print(f"    ✅ マスクを修正: {mask0.shape}")
                else:
                    print(f"    ✅ マスクサイズ正常: {data['mask0'].shape}")
            else:
                print("    マスクなし")
            
            print("  🔄 JamMaマッチング処理開始...")
            # 4. JamMaでマッチング
            return self.jamma_matcher(data, mode='test')
            
        except Exception as e:
            print(f"❌ フォワードパス中にエラー: {e}")
            # デバッグ情報を出力
            print("  🔍 詳細なデバッグ情報:")
            for key in ['feat_8_0', 'feat_8_1', 'feat_4_0', 'feat_4_1', 'mask0', 'mask1', 'h_8', 'w_8', 'hw_8']:
                if key in data:
                    value = data[key]
                    if hasattr(value, 'shape'):
                        print(f"    {key}: {value.shape}")
                    else:
                        print(f"    {key}: {value} (type: {type(value)})")
            raise


def create_planetary_image_pair():
    """
    惑星表面画像ペアの生成（改良版）
    """
    print("  🎨 合成惑星表面画像を作成中...")
    np.random.seed(42)
    size = (512, 512)
    
    # より現実的な地形生成
    x, y = np.meshgrid(np.linspace(0, 10, size[1]), np.linspace(0, 10, size[0]))
    
    # 多スケール地形生成
    terrain1 = (
        np.sin(x) * np.cos(y) +                    # 大規模特徴
        0.5 * np.sin(2*x) * np.cos(3*y) +         # 中規模特徴  
        0.3 * np.sin(5*x) * np.cos(2*y) +         # 小規模特徴
        0.2 * np.sin(8*x) * np.cos(5*y) +         # 細部
        0.1 * np.random.normal(0, 1, size)        # ノイズ
    )
    
    # クレーター様の円形窪地を追加
    crater_positions = [
        (128, 150, 25),  # (center_x, center_y, radius)
        (300, 200, 35),
        (400, 400, 20),
        (150, 350, 30)
    ]
    
    for cx, cy, radius in crater_positions:
        y_coords, x_coords = np.ogrid[:size[0], :size[1]]
        crater_mask = (x_coords - cx)**2 + (y_coords - cy)**2 <= radius**2
        
        # 現実的なクレータープロファイル
        distance = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        crater_depth = np.exp(-distance**2 / (2 * (radius/2)**2)) * 0.4
        
        terrain1[crater_mask] -= crater_depth[crater_mask]
    
    # 2番目の画像（幾何変換適用）
    center = (size[1]//2, size[0]//2)
    angle = 12  # degrees
    scale = 0.95
    
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += 25  # translation x
    M[1, 2] += 15  # translation y
    
    image2 = cv2.warpAffine(terrain1, M, (size[1], size[0]))
    
    # 照明変化をシミュレート
    illumination_gradient_x = np.linspace(0.85, 1.15, size[1])
    illumination_gradient_y = np.linspace(1.05, 0.95, size[0])
    illumination_map = np.outer(illumination_gradient_y, illumination_gradient_x)
    
    image2 = image2 * illumination_map + 0.1
    
    # [0, 255]範囲に正規化
    image1 = ((terrain1 - terrain1.min()) / (terrain1.max() - terrain1.min()) * 255).astype(np.uint8)
    image2 = ((image2 - image2.min()) / (image2.max() - image2.min()) * 255).astype(np.uint8)
    
    print("  ✅ 合成惑星画像の作成が完了しました！")
    return image1, image2


def prepare_data_batch(image1, image2, use_masks=False):
    """
    JamMa-FDAFT処理用のデータバッチ準備（修正版）
    
    Args:
        image1, image2: 入力画像
        use_masks: マスクを使用するかどうか（デモではFalseが安全）
    """
    # RGB形式に変換（3チャンネル）
    if len(image1.shape) == 2:
        image1_rgb = np.stack([image1, image1, image1], axis=2)
        image2_rgb = np.stack([image2, image2, image2], axis=2)
    else:
        image1_rgb = image1
        image2_rgb = image2
    
    # 一時的にダミーファイルを作成してread_megadepth_colorを使用
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp1, \
         tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp2:
        
        cv2.imwrite(tmp1.name, cv2.cvtColor(image1_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(tmp2.name, cv2.cvtColor(image2_rgb, cv2.COLOR_RGB2BGR))
        
        try:
            # MegaDepthスタイルの前処理
            image1_tensor, scale1, mask1, prepad_size1 = read_megadepth_color(
                tmp1.name, resize=832, df=16, padding=True
            )
            image2_tensor, scale2, mask2, prepad_size2 = read_megadepth_color(
                tmp2.name, resize=832, df=16, padding=True
            )
        finally:
            # 一時ファイルを削除
            os.unlink(tmp1.name)
            os.unlink(tmp2.name)
    
    print(f"  📏 画像とマスクの初期サイズ:")
    print(f"    image1_tensor: {image1_tensor.shape}")
    print(f"    image2_tensor: {image2_tensor.shape}")
    if mask1 is not None:
        print(f"    mask1: {mask1.shape}")
        print(f"    mask2: {mask2.shape}")
    
    # マスク処理
    mask1_coarse = mask2_coarse = None
    
    if use_masks and mask1 is not None and mask2 is not None:
        print("  📏 マスクを使用してcoarse処理を実行")
        
        # マスクサイズを修正（coarseレベル用）
        coarse_scale = 0.125  # 1/8 scale
        
        # バッチサイズを1に設定してマスクを処理
        B = 1  # バッチサイズ
        
        # マスクを正しい形状に整形: [B, H, W]
        if mask1.dim() == 2:  # [H, W]
            mask1 = mask1.unsqueeze(0)  # [1, H, W]
        if mask2.dim() == 2:  # [H, W]
            mask2 = mask2.unsqueeze(0)  # [1, H, W]
        
        # マスクを[B, 1, H, W]形式にして処理
        mask_stack = torch.stack([mask1, mask2], dim=0).unsqueeze(1).float()  # [2, 1, H, W]
        
        # coarseスケールでリサイズ
        mask_coarse = F.interpolate(
            mask_stack,
            scale_factor=coarse_scale,
            mode='nearest',
            recompute_scale_factor=False
        ).squeeze(1).bool()  # [2, H_c, W_c]
        
        # バッチサイズが1なので、正しい次元を保持
        mask1_coarse = mask_coarse[0].unsqueeze(0)  # [1, H_c, W_c] 
        mask2_coarse = mask_coarse[1].unsqueeze(0)  # [1, H_c, W_c]
        
        print(f"    coarseマスクサイズ: {mask1_coarse.shape}, {mask2_coarse.shape}")
        
        # 確認: 正しい次元数になっているか
        assert mask1_coarse.dim() == 3, f"mask1_coarse should be 3D, got {mask1_coarse.dim()}D"
        assert mask2_coarse.dim() == 3, f"mask2_coarse should be 3D, got {mask2_coarse.dim()}D"
        
    else:
        print("  📏 マスクなしでデモを実行（より安全）")
        mask1_coarse = mask2_coarse = None
    
    # データ辞書作成
    data = {
        'imagec_0': image1_tensor,
        'imagec_1': image2_tensor,
        'dataset_name': ['JamMa-FDAFT-Demo'],
        'scene_id': 'demo_scene',
        'pair_id': 0,
        'pair_names': [('demo_image1.png', 'demo_image2.png')]
    }
    
    # マスクがある場合のみ追加
    if mask1_coarse is not None:
        data['mask0'] = mask1_coarse
        data['mask1'] = mask2_coarse
    
    # デバッグ: 最終的なデータ形状を確認
    print(f"  📏 最終データ形状:")
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"    {key}: {value.shape}")
        elif value is None:
            print(f"    {key}: None")
    
    return data


def create_demo_config():
    """
    デモ用設定の作成（JamMa互換）
    """
    # デフォルト設定取得
    config = get_cfg_defaults()
    
    # JamMa互換の設定
    config.JAMMA.RESOLUTION = (8, 2)
    config.JAMMA.FINE_WINDOW_SIZE = 5
    config.JAMMA.COARSE.D_MODEL = 256  # JamMaのデフォルト
    config.JAMMA.FINE.D_MODEL = 64    # JamMaのデフォルト
    
    # マッチング閾値
    config.JAMMA.MATCH_COARSE.USE_SM = True
    config.JAMMA.MATCH_COARSE.THR = 0.2
    config.JAMMA.MATCH_COARSE.BORDER_RM = 2
    config.JAMMA.FINE.THR = 0.1
    config.JAMMA.FINE.INFERENCE = True
    config.JAMMA.MATCH_COARSE.INFERENCE = True
    
    return config


def demonstrate_jamma_fdaft():
    """メイン実演関数"""
    print("🚀 JamMa-FDAFT統合パイプライン実演")
    print("=" * 60)
    print("アーキテクチャ: Input Images → FDAFT Encoder → Joint Mamba (JEGO) → C2F Matching")
    print("特徴: FDAFTエンコーダー + JamMaの学習済みマッチング")
    print()
    
    # ステップ1: サンプル画像作成
    print("Step 1: 合成惑星画像の作成...")
    start_time = time.time()
    try:
        image1, image2 = create_planetary_image_pair()
        creation_time = time.time() - start_time
        print(f"  ✅ 画像作成完了 {creation_time:.2f} 秒")
    except Exception as e:
        print(f"  ❌ 画像作成エラー: {e}")
        return False
    
    # 入力画像表示
    print("\n📊 入力画像を表示中...")
    try:
        # matplotlib設定
        try:
            import matplotlib
            matplotlib.use('TkAgg')
        except:
            matplotlib.use('Agg')
            print("注意: 非対話型matplotlibバックエンドを使用")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.imshow(image1, cmap='gray')
        ax1.set_title('惑星画像1 (参照)', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2.imshow(image2, cmap='gray')
        ax2.set_title('惑星画像2 (変換済み)', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"  ⚠️ 画像表示エラー: {e}")
    
    # ステップ2: JamMa-FDAFT初期化
    print("\nStep 2: JamMa-FDAFT モデル初期化...")
    
    try:
        # 一貫した設定作成
        config = create_demo_config()
        
        print("  🔧 JamMa-FDAFT統合モデル初期化中...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  📱 使用デバイス: {device}")
        
        model = JamMaFDAFTDemo(config, pretrained_jamma='official').to(device)
        model.eval()
        
        print("  ✅ JamMa-FDAFT初期化成功!")
        
    except Exception as e:
        print(f"  ❌ モデル初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # モデル情報表示
    print("\n📋 モデルアーキテクチャ情報:")
    try:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  総パラメータ数: {total_params:,}")
        print(f"  Backbone: FDAFT Encoder (惑星画像特化)")
        print(f"  Matcher: 学習済みJoint Mamba + C2F マッチング")
        print(f"  出力次元: Coarse={config.JAMMA.COARSE.D_MODEL}, Fine={config.JAMMA.FINE.D_MODEL}")
    except Exception as e:
        print(f"  ⚠️ モデル情報取得エラー: {e}")
    
    # ステップ3: データ準備と推論実行
    print("\nStep 3: JamMa パイプライン実行...")
    start_time = time.time()
    
    try:
        # データバッチ準備（マスクなしで安全に実行）
        print("  📦 データバッチを準備中...")
        data = prepare_data_batch(image1, image2, use_masks=False)
        
        # デバイスに移動
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(device)
        
        with torch.no_grad():
            print("  🔄 FDAFT + JamMa統合処理中...")
            # 統合パイプライン実行
            model(data)
        
        processing_time = time.time() - start_time
        print(f"  ✅ パイプライン完了 {processing_time:.2f} 秒")
        
    except Exception as e:
        print(f"  ❌ 処理中エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ステップ4: 結果分析
    print("\nStep 4: 結果分析...")
    
    try:
        # マッチング結果抽出
        num_matches = len(data.get('mkpts0_f', []))
        coarse_matches = len(data.get('mkpts0_c', []))
        
        print(f"  🎯 粗レベルマッチ検出: {coarse_matches}")
        print(f"  🎯 細レベルマッチ検出: {num_matches}")
        
    except Exception as e:
        print(f"  ⚠️ 結果分析エラー: {e}")
        num_matches = 0
        coarse_matches = 0
    
    # ステップ5: 結果可視化
    print(f"\nStep 5: 結果可視化...")
    try:
        if num_matches > 0:
            # マッチング可視化作成
            make_matching_figures(data, mode='evaluation')
            print("  ✅ 可視化完了")
        else:
            print("  ⚠️ 可視化用マッチなし")
            
    except Exception as e:
        print(f"  ❌ 可視化エラー: {e}")
    
    # 最終まとめ
    print("\n" + "="*60)
    print("🎊 JAMMA-FDAFT デモ実演まとめ")
    print("="*60)
    print(f"処理時間: {processing_time:.2f} 秒")
    print(f"最終マッチ数: {num_matches}")
    
    if num_matches >= 8:
        print("✅ 成功: JamMa-FDAFTが画像のマッチングに成功!")
        print("  パイプラインが実証:")
        print("  - FDAFT: 惑星画像特化の堅牢な特徴抽出")
        print("  - Joint Mamba: 効率的な長距離特徴相互作用")
        print("  - C2F マッチング: 階層的マッチングとサブピクセル精細化")
    else:
        print("⚠️ 限定的成功: 少数のマッチのみ検出")
        print("  原因として考えられるもの:")
        print("  - デモ用モデルサイズの縮小")
        print("  - 合成画像の特性が困難")
        print("  - 特定画像タイプ用のパラメータ調整の必要性")
    
    print(f"\n🚀 次のステップ:")
    print(f"  - 実際のデータセット（MegaDepth、ScanNetなど）での訓練")
    print(f"  - 実行: python train.py configs/data/megadepth_trainval_832.py configs/jamma/outdoor/final.py")
    print(f"  - テスト: python test.py configs/data/megadepth_test_1500.py configs/jamma/outdoor/test.py")
    
    return True


if __name__ == "__main__":
    """実演スクリプトのエントリーポイント"""
    try:
        success = demonstrate_jamma_fdaft()
        
        if success:
            print(f"\n🎉 JamMa-FDAFT デモが正常に完了しました!")
            input("Enterキーで終了...")
        else:
            print(f"\n❌ デモが失敗しました。上記のエラーメッセージを確認してください。")
            
    except KeyboardInterrupt:
        print(f"\n\nデモがユーザーによって中断されました。")
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        
    sys.exit(0)