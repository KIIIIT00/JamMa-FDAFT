"""
Structured Forests ECM (Edge Confidence Map)

事前学習済みStructured Forestsモデルを使用してECMを計算
OpenCVのximgprocモジュールを活用
"""

import cv2
import numpy as np
import os
import urllib.request
import gzip
from typing import Optional
import warnings


class StructuredForestsECM:
    """
    Structured Forests を使用したEdge Confidence Map計算
    
    事前学習済みモデルを自動ダウンロードして使用
    惑星画像向けに後処理を最適化
    """
    
    def __init__(self, 
                 model_path: str = "assets/structured_forests_model.yml",
                 auto_download: bool = True,
                 planetary_optimization: bool = True):
        """
        Initialize Structured Forests ECM
        
        Args:
            model_path: 事前学習済みモデルのパス
            auto_download: モデルが見つからない場合自動ダウンロード
            planetary_optimization: 惑星画像向け最適化を有効化
        """
        self.model_path = model_path
        self.auto_download = auto_download
        self.planetary_optimization = planetary_optimization
        self.edge_detector = None
        
        # モデルディレクトリを作成
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Structured Forestsエッジ検出器を初期化
        self._initialize_edge_detector()
    
    def _initialize_edge_detector(self):
        """
        Structured Forestsエッジ検出器を初期化
        """
        try:
            # OpenCV ximgproc が利用可能かチェック
            if not hasattr(cv2, 'ximgproc'):
                raise ImportError("OpenCV ximgproc module not available")
            
            # モデルファイルの存在確認
            if not os.path.exists(self.model_path):
                if self.auto_download:
                    print(f"Model not found at {self.model_path}. Downloading...")
                    self._download_model()
                else:
                    raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # エッジ検出器を作成
            self.edge_detector = cv2.ximgproc.createStructuredEdgeDetection(self.model_path)
            print("✓ Structured Forests model loaded successfully")
            
        except Exception as e:
            print(f"Warning: Could not initialize Structured Forests ({e})")
            print("Falling back to simplified edge detection")
            self.edge_detector = None
    
    def _download_model(self):
        """
        事前学習済みStructured Forestsモデルをダウンロード
        """
        # OpenCVの公式モデルURL
        model_urls = [
            "https://github.com/opencv/opencv_extra/raw/master/testdata/cv/ximgproc/model.yml.gz",
            "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/cv/ximgproc/model.yml.gz"
        ]
        
        success = False
        for url in model_urls:
            try:
                print(f"Downloading from: {url}")
                
                # 圧縮ファイルをダウンロード
                compressed_path = self.model_path + ".gz"
                urllib.request.urlretrieve(url, compressed_path)
                
                # 解凍
                with gzip.open(compressed_path, 'rb') as f_in:
                    with open(self.model_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                
                # 圧縮ファイルを削除
                os.remove(compressed_path)
                
                print("✓ Model downloaded and extracted successfully")
                success = True
                break
                
            except Exception as e:
                print(f"Failed to download from {url}: {e}")
                continue
        
        if not success:
            raise RuntimeError("Failed to download Structured Forests model from all URLs")
    
    def compute_ecm(self, image: np.ndarray) -> np.ndarray:
        """
        Edge Confidence Mapを計算
        
        Args:
            image: 入力画像 (grayscale or RGB)
            
        Returns:
            Edge confidence map [0, 1]
        """
        if self.edge_detector is not None:
            return self._compute_structured_forests_ecm(image)
        else:
            return self._compute_fallback_ecm(image)
    
    def _compute_structured_forests_ecm(self, image: np.ndarray) -> np.ndarray:
        """
        Structured Forestsを使用してECMを計算
        """
        # 画像の前処理
        processed_image = self._preprocess_image(image)
        
        try:
            # エッジ検出
            edges = self.edge_detector.detectEdges(processed_image)
            
            # オリエンテーションマップ計算
            orientation_map = self.edge_detector.computeOrientation(edges)
            
            # Non-Maximum Suppression
            edges_nms = self.edge_detector.edgesNms(edges, orientation_map)
            
            # 惑星画像向け後処理
            if self.planetary_optimization:
                ecm = self._postprocess_for_planetary(edges_nms, image)
            else:
                ecm = edges_nms
            
            # [0, 1]範囲に正規化
            ecm = np.clip(ecm, 0, 1)
            
            return ecm
            
        except Exception as e:
            print(f"Warning: Structured Forests processing failed ({e})")
            return self._compute_fallback_ecm(image)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Structured Forests用に画像を前処理
        """
        # グレースケールからBGRに変換（Structured Forestsは3チャンネル入力が必要）
        if len(image.shape) == 2:
            # Grayscale → BGR
            bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3:
            if image.shape[2] == 3:
                # RGB → BGR
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                bgr_image = image
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        # [0, 1]の float32に正規化
        bgr_image = bgr_image.astype(np.float32) / 255.0
        
        return bgr_image
    
    def _postprocess_for_planetary(self, edges: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """
        惑星画像向けの後処理
        
        惑星画像の特徴（弱テクスチャ、クレーター構造、照明変化）に最適化
        """
        # グレースケール変換
        if len(original_image.shape) == 3:
            gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = original_image.copy()
        
        gray = gray.astype(np.float32) / 255.0
        
        # 1. 低コントラスト領域のノイズ抑制
        contrast_mask = self._compute_local_contrast(gray)
        edges_enhanced = edges * contrast_mask
        
        # 2. クレーター様の円形構造の強調
        circular_enhancement = self._detect_circular_structures(gray)
        edges_enhanced = edges_enhanced + 0.3 * circular_enhancement * edges
        
        # 3. 非常に滑らかな領域（空/宇宙）のエッジ抑制
        texture_mask = self._compute_texture_mask(gray)
        edges_enhanced = edges_enhanced * texture_mask
        
        # 4. 適応的エッジ強化
        edges_enhanced = self._adaptive_edge_enhancement(edges_enhanced, gray)
        
        # 5. バイラテラルフィルタでノイズ除去
        edges_enhanced = cv2.bilateralFilter(
            edges_enhanced.astype(np.float32), 
            d=5, 
            sigmaColor=0.1, 
            sigmaSpace=5
        )
        
        return edges_enhanced
    
    def _compute_local_contrast(self, image: np.ndarray, window_size: int = 15) -> np.ndarray:
        """
        局所コントラストを計算してノイズ抑制用マスクを作成
        """
        kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        
        # 局所平均
        local_mean = cv2.filter2D(image, -1, kernel)
        
        # 局所分散
        local_var = cv2.filter2D(image * image, -1, kernel) - local_mean * local_mean
        local_std = np.sqrt(local_var + 1e-8)
        
        # コントラストマスク（高コントラスト領域で高い値）
        contrast_mask = np.tanh(local_std * 10)
        
        return contrast_mask
    
    def _detect_circular_structures(self, image: np.ndarray) -> np.ndarray:
        """
        クレーター様の円形構造を検出
        """
        # ハフ変換用に uint8 に変換
        image_uint8 = (image * 255).astype(np.uint8)
        
        # ノイズ除去
        image_filtered = cv2.medianBlur(image_uint8, 5)
        
        # 円検出
        try:
            circles = cv2.HoughCircles(
                image_filtered,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=30,
                param1=50,
                param2=30,
                minRadius=10,
                maxRadius=100
            )
        except:
            circles = None
        
        # 円形構造マップ作成
        circular_map = np.zeros_like(image)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # 円形マスク作成
                Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
                dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)
                
                # 円周辺のエッジ領域を強調
                circle_edge_mask = np.logical_and(
                    dist_from_center >= r-5, 
                    dist_from_center <= r+5
                )
                circular_map[circle_edge_mask] = 1.0
        
        return circular_map
    
    def _compute_texture_mask(self, image: np.ndarray) -> np.ndarray:
        """
        テクスチャマスクを計算して滑らかな領域のエッジを抑制
        """
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        local_mean = cv2.filter2D(image, -1, kernel)
        local_var = cv2.filter2D(image * image, -1, kernel) - local_mean * local_mean
        local_std = np.sqrt(local_var + 1e-8)
        
        # テクスチャマスク（非常に滑らかな領域を抑制）
        texture_threshold = 0.02
        texture_mask = np.tanh(local_std / texture_threshold)
        
        return texture_mask
    
    def _adaptive_edge_enhancement(self, edges: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        局所画像特性に基づく適応的エッジ強化
        """
        # 画像勾配を計算
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # 勾配強度を正規化
        grad_mag_norm = grad_mag / (grad_mag.max() + 1e-8)
        
        # 勾配が強い領域でエッジを強化
        enhancement_factor = 1.0 + 0.5 * grad_mag_norm
        edges_enhanced = edges * enhancement_factor
        
        return edges_enhanced
    
    def _compute_fallback_ecm(self, image: np.ndarray) -> np.ndarray:
        """
        Structured Forestsが利用できない場合のフォールバック実装
        """
        print("Using fallback edge detection method")
        
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # ガウシアンブラー
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        
        # Cannyエッジ検出
        edges = cv2.Canny(blurred, 50, 150)
        
        # [0, 1]範囲に正規化
        edges = edges.astype(np.float32) / 255.0
        
        # 惑星画像向け後処理を適用
        if self.planetary_optimization:
            edges = self._postprocess_for_planetary(edges, image)
        
        return edges
    
    def is_model_available(self) -> bool:
        """
        Structured Forestsモデルが利用可能かチェック
        """
        return self.edge_detector is not None
    
    def get_model_info(self) -> dict:
        """
        モデル情報を取得
        """
        return {
            'model_path': self.model_path,
            'model_available': self.is_model_available(),
            'planetary_optimization': self.planetary_optimization,
            'opencv_ximgproc_available': hasattr(cv2, 'ximgproc')
        }