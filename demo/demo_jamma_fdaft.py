"""
修正されたJamMa-FDAFT Complete Demonstration Script

主な修正点：
- JamMaの学習済みモデルとの完全互換性を確保
- demo/utils_fdaft.pyを使用したシンプルなインターフェース
- 元のdemo/demo.pyと同様の使いやすさ
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import torch
import torch.nn.functional as F

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    # JamMa-FDAFT用のユーティリティをインポート
    from demo.utils_fdaft import JamMa, cfg
    from src.utils.dataset import read_megadepth_color
    from src.utils.plotting import make_matching_figures
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the project is properly set up.")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path}")
    sys.exit(1)


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
    
    print("  JamMa-FDAFT統合モデル初期化中...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # demo/utils_fdaft.pyを使用してモデルを初期化
    jamma_fdaft = JamMa(config=cfg).eval().to(device)
    
    print("  ✓ JamMa-FDAFT初期化成功!")
    
    # モデル情報表示
    print("\nモデルアーキテクチャ情報:")
    total_params = sum(p.numel() for p in jamma_fdaft.parameters())
    print(f"  総パラメータ数: {total_params:,}")
    print(f"  FDAFT Encoder: 惑星画像特化特徴抽出")
    print(f"  JamMa Matcher: 学習済みJoint Mamba + C2F マッチング")
    print(f"  出力次元: Coarse={cfg['coarse']['d_model']}, Fine={cfg['fine']['d_model']}")
    
    # ステップ3: データ準備と推論実行
    print("\nStep 3: JamMa-FDAFT パイプライン実行...")
    start_time = time.time()
    
    try:
        # 画像をPILで一時的に保存してread_megadepth_colorで読み込み
        from PIL import Image
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp1, \
             tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp2:
            
            Image.fromarray(image1).save(tmp1.name)
            Image.fromarray(image2).save(tmp2.name)
            
            # demo/demo.pyと同様の方法でデータを準備
            image0, scale0, mask0, prepad_size0 = read_megadepth_color(tmp1.name, 832, 16, True)
            image1_tensor, scale1, mask1, prepad_size1 = read_megadepth_color(tmp2.name, 832, 16, True)
            
            # マスクの処理
            mask0 = F.interpolate(mask0[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
            mask1 = F.interpolate(mask1[None, None].float(), scale_factor=0.125, mode='nearest', recompute_scale_factor=False)[0].bool()
            
            # データ辞書を作成（demo/demo.pyと同じ形式）
            data = {
                'imagec_0': image0.to(device),
                'imagec_1': image1_tensor.to(device),
                'mask0': mask0.to(device),
                'mask1': mask1.to(device),
            }
            
            # 一時ファイルを削除
            os.unlink(tmp1.name)
            os.unlink(tmp2.name)
        
        with torch.no_grad():
            print("  FDAFT + JamMa統合処理中...")
            # 統合パイプライン実行（demo/demo.pyと同じインターフェース）
            jamma_fdaft(data)
        
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