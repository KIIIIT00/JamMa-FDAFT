"""
修正されたJamMa-FDAFT Complete Demonstration Script

主な修正点：
- JamMaの学習済みモデル使用対応
- 設定の整合性を確保
- FDAFTエンコーダーとJamMa次元の適合性向上
- 統合パイプラインの安定化
"""

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from jamma_fdaft.backbone_fdaft import FDAFTEncoder
    from jamma.jamma import JamMa
    from jamma.backbone import CovNextV2_nano
    from config.default import get_cfg_defaults
    from utils.plotting import make_matching_figures
    from utils.dataset import read_megadepth_color
    import torch.nn.functional as F
    from utils.misc import lower_config
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the project is properly set up.")
    sys.exit(1)


class JamMaFDAFTDemo(nn.Module):
    """
    JamMa-FDAFT統合モデル（デモ用）
    FDAFTエンコーダー + JamMaの学習済みモデルを組み合わせ
    """
    
    def __init__(self, config, pretrained_jamma='official'):
        super().__init__()
        self.config = config
        
        # FDAFTエンコーダーを初期化（JamMaの次元に合わせる）
        self.fdaft_backbone = FDAFTEncoder.from_config(config)
        
        # JamMaの学習済みモデルを読み込み
        self.jamma_backbone = CovNextV2_nano()
        self.jamma_matcher = JamMa(config=config.JAMMA, profiler=None)
        
        # JamMaの学習済み重みを読み込み
        if pretrained_jamma == 'official':
            try:
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
                print("✓ JamMa学習済みモデルを読み込み完了")
                
            except Exception as e:
                print(f"JamMa学習済みモデルの読み込みに失敗: {e}")
                print("スクラッチから初期化します")
        
        # 次元適応レイヤー（FDAFTからJamMaへの橋渡し）
        fdaft_dim = 256  # FDAFT出力次元
        jamma_dim = 256  # JamMa期待次元
        
        self.dimension_adapter_8 = nn.Sequential(
            nn.Conv2d(fdaft_dim, jamma_dim, kernel_size=1),
            nn.BatchNorm2d(jamma_dim),
            nn.ReLU(inplace=True)
        ) if fdaft_dim != jamma_dim else nn.Identity()
        
        self.dimension_adapter_4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),  # Fine level adaptation
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, data):
        """統合フォワードパス"""
        # 1. FDAFTで特徴抽出
        self.fdaft_backbone(data)
        
        # 2. 次元適応
        data['feat_8_0'] = self.dimension_adapter_8(data['feat_8_0'])
        data['feat_8_1'] = self.dimension_adapter_8(data['feat_8_1'])
        data['feat_4_0'] = self.dimension_adapter_4(data['feat_4_0'])
        data['feat_4_1'] = self.dimension_adapter_4(data['feat_4_1'])
        
        # 3. JamMaでマッチング
        return self.jamma_matcher(data, mode='test')


def create_planetary_image_pair():
    """
    惑星表面画像ペアの生成（改良版）
    """
    print("  Creating synthetic planetary surface images...")
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
    
    print("  ✓ Synthetic planetary images created successfully!")
    return image1, image2


def prepare_data_batch(image1, image2):
    """
    JamMa-FDAFT処理用のデータバッチ準備
    """
    # RGB形式に変換（3チャンネル）
    if len(image1.shape) == 2:
        image1_rgb = np.stack([image1, image1, image1], axis=2)
        image2_rgb = np.stack([image2, image2, image2], axis=2)
    else:
        image1_rgb = image1
        image2_rgb = image2
    
    # MegaDepthスタイルの前処理（JamMaと互換性を保つ）
    image1_tensor, scale1, mask1, prepad_size1 = read_megadepth_color(
        None, resize=832, df=16, padding=True
    )
    image2_tensor, scale2, mask2, prepad_size2 = read_megadepth_color(
        None, resize=832, df=16, padding=True
    )
    
    # 実際の画像データで置き換え
    image1_processed = torch.from_numpy(image1_rgb).float().permute(2, 0, 1) / 255.0
    image2_processed = torch.from_numpy(image2_rgb).float().permute(2, 0, 1) / 255.0
    
    # リサイズとパディング
    image1_processed = F.interpolate(image1_processed.unsqueeze(0), size=(832, 832), mode='bilinear').squeeze(0)
    image2_processed = F.interpolate(image2_processed.unsqueeze(0), size=(832, 832), mode='bilinear').squeeze(0)
    
    # 正規化（ImageNet統計）
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    image1_processed = (image1_processed - imagenet_mean) / imagenet_std
    image2_processed = (image2_processed - imagenet_mean) / imagenet_std
    
    # バッチ次元を追加
    image1_batch = image1_processed.unsqueeze(0)  # [1, 3, H, W]
    image2_batch = image2_processed.unsqueeze(0)  # [1, 3, H, W]
    
    # マスク生成
    mask1 = torch.ones(1, 832, 832).bool()
    mask2 = torch.ones(1, 832, 832).bool()
    
    # データ辞書作成
    data = {
        'imagec_0': image1_batch,
        'imagec_1': image2_batch,
        'mask0': mask1,
        'mask1': mask2,
        'dataset_name': ['JamMa-FDAFT-Demo'],
        'scene_id': 'demo_scene',
        'pair_id': 0,
        'pair_names': [('demo_image1.png', 'demo_image2.png')]
    }
    
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
    config.JAMMA.FINE.D_MODEL = 128    # JamMaのデフォルト
    
    # マッチング閾値
    config.JAMMA.MATCH_COARSE.USE_SM = True
    config.JAMMA.MATCH_COARSE.THR = 0.2
    config.JAMMA.MATCH_COARSE.BORDER_RM = 2
    config.JAMMA.FINE.THR = 0.1
    config.JAMMA.FINE.INFERENCE = True
    config.JAMMA.MATCH_COARSE.INFERENCE = True
    
    # FDAFT設定
    config.FDAFT = config.__class__()
    config.FDAFT.NUM_LAYERS = 3
    config.FDAFT.SIGMA_0 = 1.0
    config.FDAFT.USE_STRUCTURED_FORESTS = True
    config.FDAFT.MAX_KEYPOINTS = 1000  # デモ用に削減
    config.FDAFT.NMS_RADIUS = 5
    
    return config


def demonstrate_jamma_fdaft():
    """メイン実演関数"""
    print("JamMa-FDAFT統合パイプライン実演")
    print("=" * 60)
    print("アーキテクチャ: Input Images → FDAFT Encoder → Joint Mamba (JEGO) → C2F Matching")
    print("特徴: JamMaの学習済みモデルを使用")
    print()
    
    # ステップ1: サンプル画像作成
    print("Step 1: 合成惑星画像の作成...")
    start_time = time.time()
    image1, image2 = create_planetary_image_pair()
    creation_time = time.time() - start_time
    print(f"  ✓ 画像作成完了 {creation_time:.2f} 秒")
    
    # 入力画像表示
    print("\n入力画像を表示中...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.imshow(image1, cmap='gray')
    ax1.set_title('惑星画像1 (参照)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(image2, cmap='gray')
    ax2.set_title('惑星画像2 (変換済み)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # ステップ2: JamMa-FDAFT初期化
    print("\nStep 2: JamMa-FDAFT モデル初期化...")
    
    # 一貫した設定作成
    config = create_demo_config()
    _config = lower_config(config)
    
    print("  JamMa-FDAFT統合モデル初期化中...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = JamMaFDAFTDemo(config, pretrained_jamma='official').to(device)
    model.eval()
    
    print("  ✓ JamMa-FDAFT初期化成功!")
    
    # モデル情報表示
    print("\nモデルアーキテクチャ情報:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  総パラメータ数: {total_params:,}")
    print(f"  FDAFT Encoder: 惑星画像特化特徴抽出")
    print(f"  JamMa Matcher: 学習済みJoint Mamba + C2F マッチング")
    print(f"  出力次元: Coarse={config.JAMMA.COARSE.D_MODEL}, Fine={config.JAMMA.FINE.D_MODEL}")
    
    # ステップ3: データ準備と推論実行
    print("\nStep 3: JamMa-FDAFT パイプライン実行...")
    start_time = time.time()
    
    try:
        # データバッチ準備
        data = prepare_data_batch(image1, image2)
        
        # デバイスに移動
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(device)
        
        with torch.no_grad():
            print("  FDAFT + JamMa統合処理中...")
            # 統合パイプライン実行
            model(data)
        
        processing_time = time.time() - start_time
        print(f"  ✓ パイプライン完了 {processing_time:.2f} 秒")
        
    except Exception as e:
        print(f"  ✗ 処理中エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ステップ4: 結果分析
    print("\nStep 4: 結果分析...")
    
    # マッチング結果抽出
    num_matches = len(data.get('mkpts0_f', []))
    coarse_matches = len(data.get('mkpts0_c', []))
    
    print(f"  粗レベルマッチ検出: {coarse_matches}")
    print(f"  細レベルマッチ検出: {num_matches}")
    
    # ステップ5: 結果可視化
    print(f"\nStep 5: 結果可視化...")
    try:
        if num_matches > 0:
            # マッチング可視化作成
            make_matching_figures(data, mode='evaluation')
            print("  ✓ 可視化完了")
        else:
            print("  ⚠ 可視化用マッチなし")
            
    except Exception as e:
        print(f"  ✗ 可視化エラー: {e}")
    
    # 最終まとめ
    print("\n" + "="*60)
    print("JAMMA-FDAFT 実演まとめ")
    print("="*60)
    print(f"処理時間: {processing_time:.2f} 秒")
    print(f"最終マッチ数: {num_matches}")
    
    if num_matches >= 8:
        print("✓ 成功: JamMa-FDAFTが惑星画像のマッチングに成功!")
        print("  統合パイプラインが実証:")
        print("  - FDAFT: 弱い表面テクスチャ用の堅牢な特徴抽出")
        print("  - JamMa学習済み: 効率的な長距離特徴相互作用")
        print("  - C2F マッチング: 階層的マッチングとサブピクセル精細化")
        print("  - 惑星最適化: 困難な表面での性能向上")
    else:
        print("⚠ 限定的成功: 少数のマッチのみ検出")
        print("  原因として考えられるもの:")
        print("  - デモ用モデルサイズの縮小（完全モデルでより良い結果）")
        print("  - 合成画像の特性が困難")
        print("  - 特定画像タイプ用のパラメータ調整の必要性")
    
    print(f"\n次のステップ:")
    print(f"  - 実際の惑星データセット（火星、月など）での訓練")
    print(f"  - 実行: python train_jammf.py configs/data/megadepth_trainval_832.py configs/jamma_fdaft/outdoor/final.py")
    print(f"  - テスト: python test_jammf.py configs/data/megadepth_test_1500.py configs/jamma_fdaft/outdoor/test.py")
    
    return True


if __name__ == "__main__":
    """実演スクリプトのエントリーポイント"""
    try:
        # matplotlib設定
        try:
            import matplotlib
            matplotlib.use('TkAgg')
        except:
            matplotlib.use('Agg')
            print("注意: 非対話型matplotlibバックエンドを使用")
        
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