"""
JamMa-FDAFT 惑星画像用設定

惑星画像の特徴（照明変化、弱テクスチャ、クレーター構造）に最適化
"""

from .base import JamFConfig, FDAFTConfig, JointMambaConfig, MatchingConfig, TrainingConfig, DataConfig


def get_planetary_config() -> JamFConfig:
    """惑星画像用設定を取得"""
    
    # FDAFT設定 - 惑星画像用最適化
    fdaft_config = FDAFTConfig(
        num_layers=4,  # より多層のスケール空間
        sigma_0=0.8,   # 初期スケールを小さく
        descriptor_radius=52,  # 記述子半径を拡大
        max_keypoints=1500,    # より多くの特徴点
        
        # 惑星画像用パラメータ
        phase_threshold=0.03,      # 低い閾値でより多くの位相特徴を検出
        ecm_enhancement=1.5,       # ECM強化係数を高く
        contrast_enhancement=True,  # コントラスト強化を有効
        
        structured_forests_model_path="assets/structured_forests_model.yml"
    )
    
    # Joint Mamba設定 - 高解像度対応
    joint_mamba_config = JointMambaConfig(
        input_dim=320,     # より大きな特徴次元
        hidden_dim=320,
        num_layers=6,      # より深いMambaレイヤー
        num_heads=10,
        skip_steps=1,      # より細かいスキャン
        
        # SSMパラメータ調整
        d_state=20,        # より大きな状態次元
        d_conv=5,          # より大きな畳み込みカーネル
        expand=3           # より大きな拡張係数
    )
    
    # マッチング設定 - 精密マッチング
    matching_config = MatchingConfig(
        coarse_resolution=6,      # より細かい粗いマッチング
        fine_resolution=1,        # 最高解像度の細かいマッチング
        temperature=0.05,         # より低い温度（よりシャープな分布）
        coarse_threshold=0.15,    # より低い閾値
        fine_window_size=7        # より大きなウィンドウ
    )
    
    # 学習設定 - 惑星画像用調整
    training_config = TrainingConfig(
        max_epochs=150,           # より長い学習
        learning_rate=5e-5,       # より小さな学習率
        weight_decay=2e-5,        # より強い正則化
        scheduler="cosine",
        warmup_epochs=10,
        
        # 損失重み調整
        coarse_loss_weight=0.8,
        fine_loss_weight=1.2,     # 細かいマッチングを重視
        confidence_loss_weight=0.2,
        
        val_every_n_epochs=3,
        save_every_n_epochs=5,
        early_stopping_patience=25
    )
    
    # データ設定 - 惑星画像用
    data_config = DataConfig(
        batch_size=2,              # GPUメモリ考慮で小さく
        num_workers=6,
        image_size=(480, 640),     # 惑星画像の標準サイズ
        augmentation=True,
        
        dataset_name="planetary",
        data_root="data/planetary/",
        split="train"
    )
    
    return JamFConfig(
        mode="planetary",
        device="cuda",
        
        fdaft=fdaft_config,
        joint_mamba=joint_mamba_config,
        matching=matching_config,
        training=training_config,
        data=data_config,
        
        experiment_name="jamf_planetary",
        output_dir="outputs/planetary/",
        log_dir="logs/planetary/",
        checkpoint_dir="checkpoints/planetary/",
        
        seed=2024,
        deterministic=True
    )


def get_mars_config() -> JamFConfig:
    """火星画像専用設定"""
    config = get_planetary_config()
    
    # 火星画像特有の調整
    config.fdaft.phase_threshold = 0.025     # より低い閾値
    config.fdaft.ecm_enhancement = 1.8       # より強いECM強化
    config.fdaft.max_keypoints = 2000        # より多くの特徴点
    
    config.joint_mamba.skip_steps = 1        # 最密スキャン
    config.matching.temperature = 0.03       # より低い温度
    
    config.experiment_name = "jamf_mars"
    
    return config


def get_lunar_config() -> JamFConfig:
    """月面画像専用設定"""
    config = get_planetary_config()
    
    # 月面画像特有の調整（より強いコントラスト変化）
    config.fdaft.phase_threshold = 0.04      # 中程度の閾値
    config.fdaft.ecm_enhancement = 2.0       # 最強のECM強化
    config.fdaft.contrast_enhancement = True
    
    # より強力なMamba処理
    config.joint_mamba.num_layers = 8
    config.joint_mamba.d_state = 24
    
    config.experiment_name = "jamf_lunar"
    
    return config


def get_asteroid_config() -> JamFConfig:
    """小惑星画像専用設定"""
    config = get_planetary_config()
    
    # 小惑星画像特有の調整（非常に弱いテクスチャ）
    config.fdaft.phase_threshold = 0.02      # 最低閾値
    config.fdaft.ecm_enhancement = 2.2       # 最強のECM強化
    config.fdaft.num_layers = 5              # 最多層のスケール空間
    
    # 最大の特徴点数
    config.fdaft.max_keypoints = 2500
    
    # より細かいマッチング
    config.matching.fine_window_size = 9
    config.matching.coarse_threshold = 0.1
    
    config.experiment_name = "jamf_asteroid"
    
    return config


# 設定関数マッピング
PLANETARY_CONFIGS = {
    "general": get_planetary_config,
    "mars": get_mars_config,
    "lunar": get_lunar_config,
    "asteroid": get_asteroid_config
}


def get_config(variant: str = "general") -> JamFConfig:
    """指定された惑星環境の設定を取得"""
    if variant not in PLANETARY_CONFIGS:
        raise ValueError(f"Unknown planetary variant: {variant}. Available: {list(PLANETARY_CONFIGS.keys())}")
    
    return PLANETARY_CONFIGS[variant]()