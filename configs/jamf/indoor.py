"""
JamMa-FDAFT 室内環境用設定

室内環境の特徴（幾何学的構造、テクスチャ豊富、照明安定）に最適化
"""

from .base import JamFConfig, FDAFTConfig, JointMambaConfig, MatchingConfig, TrainingConfig, DataConfig


def get_indoor_config() -> JamFConfig:
    """室内環境用設定を取得"""
    
    # FDAFT設定 - 室内環境用最適化
    fdaft_config = FDAFTConfig(
        num_layers=3,  # 標準層数
        sigma_0=1.0,   # 標準初期スケール
        descriptor_radius=48,  # 標準記述子半径
        max_keypoints=1200,    # 中程度の特徴点数
        
        # 室内環境用パラメータ
        phase_threshold=0.1,       # 高い閾値で幾何学的構造重視
        ecm_enhancement=1.0,       # 標準ECM強化
        contrast_enhancement=False, # コントラスト強化無効
        
        structured_forests_model_path="assets/structured_forests_model.yml"
    )
    
    # Joint Mamba設定 - 標準設定
    joint_mamba_config = JointMambaConfig(
        input_dim=256,     # 標準特徴次元
        hidden_dim=256,
        num_layers=4,      # 標準Mambaレイヤー
        num_heads=8,
        skip_steps=2,      # 効率的スキャン
        
        # SSMパラメータ
        d_state=16,        # 標準状態次元
        d_conv=4,          # 標準畳み込みカーネル
        expand=2           # 標準拡張係数
    )
    
    # マッチング設定 - バランス型
    matching_config = MatchingConfig(
        coarse_resolution=8,      # 標準粗いマッチング
        fine_resolution=2,        # 標準細かいマッチング
        temperature=0.1,          # 標準温度
        coarse_threshold=0.2,     # 標準閾値
        fine_window_size=5        # 標準ウィンドウ
    )
    
    # 学習設定 - 室内環境用調整
    training_config = TrainingConfig(
        max_epochs=100,           # 標準学習期間
        learning_rate=1e-4,       # 標準学習率
        weight_decay=1e-5,        # 標準正則化
        scheduler="cosine",
        warmup_epochs=5,
        
        # 損失重み
        coarse_loss_weight=1.0,
        fine_loss_weight=1.0,     # バランス型
        confidence_loss_weight=0.1,
        
        val_every_n_epochs=5,
        save_every_n_epochs=10,
        early_stopping_patience=20
    )
    
    # データ設定 - 室内環境用
    data_config = DataConfig(
        batch_size=4,              # 標準バッチサイズ
        num_workers=4,
        image_size=(480, 640),     # 標準解像度
        augmentation=True,
        
        dataset_name="indoor",
        data_root="data/indoor/",
        split="train"
    )
    
    return JamFConfig(
        mode="indoor",
        device="cuda",
        
        fdaft=fdaft_config,
        joint_mamba=joint_mamba_config,
        matching=matching_config,
        training=training_config,
        data=data_config,
        
        experiment_name="jamf_indoor",
        output_dir="outputs/indoor/",
        log_dir="logs/indoor/",
        checkpoint_dir="checkpoints/indoor/",
        
        seed=42,
        deterministic=True
    )


def get_nyu_config() -> JamFConfig:
    """NYU Depth V2データセット用設定"""
    config = get_indoor_config()
    
    # NYU特有の調整
    config.fdaft.phase_threshold = 0.12     # やや高い閾値
    config.fdaft.max_keypoints = 1500       # より多くの特徴点
    
    config.joint_mamba.num_layers = 5       # やや深いMamba
    config.matching.temperature = 0.08      # やや低い温度
    
    # NYU用データ設定
    config.data.dataset_name = "nyu"
    config.data.image_size = (480, 640)
    config.data.data_root = "data/nyu/"
    
    config.experiment_name = "jamf_nyu"
    
    return config


def get_scannet_config() -> JamFConfig:
    """ScanNetデータセット用設定"""
    config = get_indoor_config()
    
    # ScanNet特有の調整（より大きなシーン）
    config.fdaft.num_layers = 4             # より多層
    config.fdaft.max_keypoints = 2000       # 最多特徴点
    config.fdaft.phase_threshold = 0.08     # 低い閾値
    
    # より強力なMamba処理
    config.joint_mamba.input_dim = 320
    config.joint_mamba.hidden_dim = 320
    config.joint_mamba.num_layers = 6
    
    # より精密なマッチング
    config.matching.coarse_resolution = 6
    config.matching.fine_window_size = 7
    
    # ScanNet用データ設定
    config.data.dataset_name = "scannet"
    config.data.batch_size = 2              # メモリ制約
    config.data.image_size = (640, 840)     # より高解像度
    config.data.data_root = "data/scannet/"
    
    config.experiment_name = "jamf_scannet"
    
    return config


def get_7scenes_config() -> JamFConfig:
    """7-Scenesデータセット用設定"""
    config = get_indoor_config()
    
    # 7-Scenes特有の調整（テクスチャ多様性）
    config.fdaft.phase_threshold = 0.09     # 中程度の閾値
    config.fdaft.ecm_enhancement = 1.1      # 軽度のECM強化
    config.fdaft.max_keypoints = 1000       # 標準特徴点数
    
    # 適度なMamba処理
    config.joint_mamba.num_layers = 4
    config.joint_mamba.skip_steps = 2
    
    # 7-Scenes用データ設定
    config.data.dataset_name = "7scenes"
    config.data.image_size = (480, 640)
    config.data.data_root = "data/7scenes/"
    
    config.experiment_name = "jamf_7scenes"
    
    return config


def get_tum_config() -> JamFConfig:
    """TUM RGB-Dデータセット用設定"""
    config = get_indoor_config()
    
    # TUM特有の調整（動的環境）
    config.fdaft.phase_threshold = 0.11     # やや高い閾値
    config.fdaft.max_keypoints = 800        # 少なめの特徴点
    
    # 動的環境対応
    config.matching.coarse_threshold = 0.15  # より低い閾値
    config.matching.temperature = 0.12       # やや高い温度
    
    # TUM用データ設定
    config.data.dataset_name = "tum"
    config.data.image_size = (480, 640)
    config.data.data_root = "data/tum/"
    config.data.augmentation = False         # TUMは実データなのでaugmentation控えめ
    
    config.experiment_name = "jamf_tum"
    
    return config


# 設定関数マッピング
INDOOR_CONFIGS = {
    "general": get_indoor_config,
    "nyu": get_nyu_config,
    "scannet": get_scannet_config,
    "7scenes": get_7scenes_config,
    "tum": get_tum_config
}


def get_config(variant: str = "general") -> JamFConfig:
    """指定された室内環境の設定を取得"""
    if variant not in INDOOR_CONFIGS:
        raise ValueError(f"Unknown indoor variant: {variant}. Available: {list(INDOOR_CONFIGS.keys())}")
    
    return INDOOR_CONFIGS[variant]()