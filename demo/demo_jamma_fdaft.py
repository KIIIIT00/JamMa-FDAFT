"""
修正されたJamMa-FDAFT Complete Demonstration Script

主な修正点：
- assetsフォルダの実際の画像を使用
- src/demo/demo.pyと同様のインターフェース
- コマンドライン引数で画像パスを指定可能
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    # JamMa-FDAFT用のユーティリティをインポート
    from demo.utils_fdaft import JamMa, cfg
    from src.utils.dataset import read_megadepth_color
    from src.utils.plotting import make_matching_figures, make_confidence_figure, make_evaluation_figure_wheel
    from loguru import logger
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the project is properly set up.")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path}")
    sys.exit(1)


def load_image_from_assets(image_path: str):
    """
    assetsフォルダから画像を読み込み
    
    Args:
        image_path: 画像ファイルのパス
        
    Returns:
        読み込まれた画像（numpy array）
    """
    if not os.path.exists(image_path):
        # assetsフォルダ内を検索
        assets_path = os.path.join(project_root, 'assets', image_path)
        if os.path.exists(assets_path):
            image_path = assets_path
        else:
            # figsフォルダ内も検索
            figs_path = os.path.join(project_root, 'assets', 'figs', image_path)
            if os.path.exists(figs_path):
                image_path = figs_path
            else:
                raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"Loading image from: {image_path}")
    
    # OpenCVで画像を読み込み
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image, image_path


def display_input_images(image1, image2, image1_path, image2_path):
    """
    入力画像を表示
    
    Args:
        image1, image2: 入力画像
        image1_path, image2_path: 画像のパス
    """
    print("\n入力画像を表示中...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    ax1.imshow(image1)
    ax1.set_title(f'画像1: {os.path.basename(image1_path)}', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(image2)
    ax2.set_title(f'画像2: {os.path.basename(image2_path)}', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()


def prepare_images_for_matching(image1_path: str, image2_path: str):
    """
    マッチング用に画像を準備
    
    Args:
        image1_path, image2_path: 画像ファイルのパス
        
    Returns:
        demo/demo.pyと同じ形式のデータ辞書
    """
    print(f"画像を準備中...")
    print(f"  画像1: {image1_path}")
    print(f"  画像2: {image2_path}")
    
    # demo/demo.pyと同様の方法でデータを準備
    image0, scale0, mask0, prepad_size0 = read_megadepth_color(image1_path, 832, 16, True)
    image1_tensor, scale1, mask1, prepad_size1 = read_megadepth_color(image2_path, 832, 16, True)
    
    # マスクの処理
    if mask0 is not None:
        mask0 = F.interpolate(mask0[None, None].float(), scale_factor=0.125, 
                             mode='nearest', recompute_scale_factor=False)[0].bool()
    if mask1 is not None:
        mask1 = F.interpolate(mask1[None, None].float(), scale_factor=0.125, 
                             mode='nearest', recompute_scale_factor=False)[0].bool()
    
    return image0, image1_tensor, mask0, mask1, scale0, scale1, prepad_size0, prepad_size1


def demonstrate_jamma_fdaft_with_assets(image1_path: str, image2_path: str, output_dir: str = 'output/'):
    """
    assets画像を使用したJamMa-FDAFTデモ
    
    Args:
        image1_path: 最初の画像のパス
        image2_path: 2番目の画像のパス
        output_dir: 出力ディレクトリ
    """
    print("JamMa-FDAFT統合パイプライン実演 (Assets Images)")
    print("=" * 60)
    print("アーキテクチャ: Input Images → FDAFT Encoder → Joint Mamba (JEGO) → C2F Matching")
    print("特徴: JamMaの学習済みモデルを使用")
    print()
    
    # 出力ディレクトリを作成
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # ステップ1: 画像の読み込み
    print("Step 1: Assets画像の読み込み...")
    start_time = time.time()
    
    try:
        image1, image1_full_path = load_image_from_assets(image1_path)
        image2, image2_full_path = load_image_from_assets(image2_path)
        loading_time = time.time() - start_time
        print(f"  ✓ 画像読み込み完了 {loading_time:.2f} 秒")
        
        # 入力画像表示
        display_input_images(image1, image2, image1_full_path, image2_full_path)
        
    except Exception as e:
        print(f"  ✗ 画像読み込みエラー: {e}")
        return False
    
    # ステップ2: JamMa-FDAFT初期化
    print("\nStep 2: JamMa-FDAFT モデル初期化...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  使用デバイス: {device}")
    
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
        # 画像の前処理とデータ準備
        image0, image1_tensor, mask0, mask1, scale0, scale1, prepad_size0, prepad_size1 = prepare_images_for_matching(
            image1_full_path, image2_full_path
        )
        
        # データ辞書を作成（demo/demo.pyと同じ形式）
        data = {
            'imagec_0': image0.to(device),
            'imagec_1': image1_tensor.to(device),
        }
        
        # マスクがある場合は追加
        if mask0 is not None:
            data['mask0'] = mask0.to(device)
        if mask1 is not None:
            data['mask1'] = mask1.to(device)
        
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
    
    # ステップ5: 結果可視化と保存
    print(f"\nStep 5: 結果可視化と保存...")
    try:
        if num_matches > 0:
            # マッチング可視化を複数のスタイルで作成
            print("  信頼度ベース可視化を作成中...")
            make_confidence_figure(data, path=os.path.join(output_dir, 'confidence_matches.png'), dpi=300, topk=4000)
            
            print("  評価ベース可視化を作成中...")
            make_evaluation_figure_wheel(data, path=os.path.join(output_dir, 'evaluation_matches.png'), topk=4000)
            
            print(f"  ✓ 可視化完了 - 結果は {output_dir} に保存されました")
            
            # 簡単な統計情報を保存
            stats_file = os.path.join(output_dir, 'matching_stats.txt')
            with open(stats_file, 'w') as f:
                f.write(f"JamMa-FDAFT Matching Results\n")
                f.write(f"============================\n\n")
                f.write(f"Image 1: {os.path.basename(image1_full_path)}\n")
                f.write(f"Image 2: {os.path.basename(image2_full_path)}\n")
                f.write(f"Processing Time: {processing_time:.2f} seconds\n")
                f.write(f"Coarse Matches: {coarse_matches}\n")
                f.write(f"Fine Matches: {num_matches}\n")
                f.write(f"Device: {device}\n")
            
            print(f"  統計情報を {stats_file} に保存しました")
            
        else:
            print("  ⚠ 可視化用マッチなし")
            
    except Exception as e:
        print(f"  ✗ 可視化エラー: {e}")
    
    # 最終まとめ
    print("\n" + "="*60)
    print("JAMMA-FDAFT 実演まとめ")
    print("="*60)
    print(f"使用画像:")
    print(f"  - {os.path.basename(image1_full_path)}")
    print(f"  - {os.path.basename(image2_full_path)}")
    print(f"処理時間: {processing_time:.2f} 秒")
    print(f"最終マッチ数: {num_matches}")
    print(f"出力ディレクトリ: {output_dir}")
    
    if num_matches >= 8:
        print("✓ 成功: JamMa-FDAFTが画像のマッチングに成功!")
        print("  統合パイプラインが実証:")
        print("  - FDAFT: 堅牢な特徴抽出")
        print("  - JamMa学習済み: 効率的な長距離特徴相互作用")
        print("  - C2F マッチング: 階層的マッチングとサブピクセル精細化")
    else:
        print("⚠ 限定的成功: 少数のマッチのみ検出")
        print("  考えられる原因:")
        print("  - 画像間の視点変化が大きい")
        print("  - テクスチャが少ない")
        print("  - 照明条件の違い")
    
    print(f"\n次のステップ:")
    print(f"  - より多くの画像ペアでテスト")
    print(f"  - 実際のデータセットでの訓練")
    print(f"  - パラメータの調整")
    
    return True


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='JamMa-FDAFT Image Matching Demo with Assets Images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--image1', type=str, 
        default='figs/345822933_b5fb7b6feb_o.jpg',
        help='Path to the first image (relative to assets/ or absolute path)'
    )
    parser.add_argument(
        '--image2', type=str, 
        default='figs/479605349_8aa68e066d_o.jpg',
        help='Path to the second image (relative to assets/ or absolute path)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='output/',
        help='Directory to save output visualizations'
    )
    
    args = parser.parse_args()
    
    print("JamMa-FDAFT Assets Image Demo")
    print("============================")
    print(f"Image 1: {args.image1}")
    print(f"Image 2: {args.image2}")
    print(f"Output: {args.output_dir}")
    print()
    
    try:
        # matplotlib設定
        try:
            import matplotlib
            matplotlib.use('TkAgg')  # インタラクティブ表示
        except:
            matplotlib.use('Agg')    # 非対話的表示
            print("注意: 非対話型matplotlibバックエンドを使用")
        
        success = demonstrate_jamma_fdaft_with_assets(
            args.image1, 
            args.image2, 
            args.output_dir
        )
        
        if success:
            print(f"\n🎉 JamMa-FDAFT デモが正常に完了しました!")
            print(f"結果は {args.output_dir} ディレクトリを確認してください。")
        else:
            print(f"\n❌ デモが失敗しました。上記のエラーメッセージを確認してください。")
            
    except KeyboardInterrupt:
        print(f"\n\nデモがユーザーによって中断されました。")
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()