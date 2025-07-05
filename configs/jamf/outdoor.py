"""
JamMa-FDAFT 屋外環境用設定

屋外環境の特徴（照明変化、天候変化、スケール変化、遠景）に最適化
"""

from .base import JamFConfig, FDAFTConfig, JointMambaConfig, MatchingConfig, TrainingConfig, DataConfig


def get_outdoor_config() -> JamFConfig:
    """屋外環境用設定を取得"""
    
    # FDAFT設定 - 屋外環境用最適化
    fdaft_config = FDAFTConfig(
        num_layers=3,  # 標準層数
        sigma_0=1.2,   # やや大きな初期スケール（遠景対応）
        descriptor_radius=50,  # やや大きな記述子半径
        max_keypoints=1000,    # 標準特徴点数
        
        # 屋外環境用パラメータ
        phase_threshold=0.08,      # バランス型閾値
        ecm_enhancement=1.1,       # 軽度のECM強化
        contrast_enhancement=False, # 基本的に無効（天候で調整）
        
        structured_forests_model_path="assets/structured_forests_model.yml"
    )
    
    # Joint Mamba設定 - 屋外用調整
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
    
    # マッチング設定 - スケール変化対応
    matching_config = MatchingConfig(
        coarse_resolution=8,      # 標準粗いマッチング
        fine_resolution=2,        # 標準細かいマッチング
        temperature=0.1,          # 標準温度
        coarse_threshold=0.18,    # やや低い閾値（照明変化対応）
        fine_window_size=6        # やや大きなウィンドウ
    )
    
    # 学習設定 - 屋外環境用調整
    training_config = TrainingConfig(
        max_epochs=120,           # やや長い学習期間
        learning_rate=8e-5,       # やや小さな学習率
        weight_decay=1.5e-5,      # やや強い正則化
        scheduler="cosine",
        warmup_epochs=8,
        
        # 損失重み
        coarse_loss_weight=1.0,
        fine_loss_weight=0.9,     # 粗いマッチングをやや重視
        confidence_loss_weight=0.15,
        
        val_every_n_epochs=4,
        save_every_n_epochs=8,
        early_stopping_patience=25
    )
    
    # データ設定 - 屋外環境用
    data_config = DataConfig(
        batch_size=3,              # GPUメモリ考慮
        num_workers=6,
        image_size=(480, 640),     # 標準解像度
        augmentation=True,
        
        dataset_name="outdoor",
        data_root="data/outdoor/",
        split="train"
    )
    
    return JamFConfig(
        mode="outdoor",
        device="cuda",
        
        fdaft=fdaft_config,
        joint_mamba=joint_mamba_config,
        matching=matching_config,
        training=training_config,
        data=data_config,
        
        experiment_name="jamf_outdoor",
        output_dir="outputs/outdoor/",
        log_dir="logs/outdoor/",
        checkpoint_dir="checkpoints/outdoor/",
        
        seed=2023,
        deterministic=True
    )


def get_megadepth_config() -> JamFConfig:
    """MegaDepthデータセット用設定"""
    config = get_outdoor_config()
    
    # MegaDepth特有の調整（大規模シーン、大きなスケール変化）
    config.fdaft.num_layers = 4             # より多層のスケール空間
    config.fdaft.sigma_0 = 1.4              # より大きな初期スケール
    config.fdaft.descriptor_radius = 56     # より大きな記述子半径
    config.fdaft.max_keypoints = 1500       # より多くの特徴点
    config.fdaft.phase_threshold = 0.06     # より低い閾値
    
    # より強力なMamba処理
    config.joint_mamba.input_dim = 320
    config.joint_mamba.hidden_dim = 320
    config.joint_mamba.num_layers = 5
    config.joint_mamba.d_state = 20
    
    # スケール変化対応マッチング
    config.matching.coarse_resolution = 6
    config.matching.fine_window_size = 7
    config.matching.coarse_threshold = 0.15
    
    # MegaDepth用データ設定
    config.data.dataset_name = "megadepth"
    config.data.batch_size = 2              # 大画像なのでバッチサイズ小
    config.data.image_size = (640, 832)     # より高解像度
    config.data.data_root = "data/megadepth/"
    
    config.experiment_name = "jamf_megadepth"
    
    return config


def get_hpatches_config() -> JamFConfig:
    """HPatchesデータセット用設定"""
    config = get_outdoor_config()
    
    # HPatches特有の調整（viewpoint + illumination変化）
    config.fdaft.phase_threshold = 0.09     # 中程度の閾値
    config.fdaft.ecm_enhancement = 1.2      # ECM強化
    config.fdaft.max_keypoints = 1200       # 中程度の特徴点数
    
    # Viewpoint変化対応
    config.matching.temperature = 0.08      # より低い温度
    config.matching.coarse_threshold = 0.16
    
    # HPatches用データ設定
    config.data.dataset_name = "hpatches"
    config.data.image_size = (480, 640)
    config.data.data_root = "data/hpatches/"
    
    config.experiment_name = "jamf_hpatches"
    
    return config


def get_aachen_config() -> JamFConfig:
    """Aachen Day-Nightデータセット用設定"""
    config = get_outdoor_config()
    
    # Aachen特有の調整（昼夜照明変化）
    config.fdaft.phase_threshold = 0.05     # 低い閾値（照明変化対応）
    config.fdaft.ecm_enhancement = 1.4      # 強いECM強化
    config.fdaft.contrast_enhancement = True # コントラスト強化有効
    config.fdaft.max_keypoints = 1800       # 多くの特徴点
    
    # 照明変化対応Mamba
    config.joint_mamba.num_layers = 6       # より深いMamba
    config.joint_mamba.d_state = 18
    
    # 照明変化対応マッチング
    config.matching.temperature = 0.06      # より低い温度
    config.matching.coarse_threshold = 0.12  # より低い閾値
    
    # Aachen用データ設定
    config.data.dataset_name = "aachen"
    config.data.data_root = "data/aachen/"
    
    config.experiment_name = "jamf_aachen"
    
    return config


def get_robotcar_config() -> JamFConfig:
    """Oxford RobotCarデータセット用設定"""
    config = get_outdoor_config()
    
    # RobotCar特有の調整（季節・天候変化）
    config.fdaft.phase_threshold = 0.07     # 中低閾値
    config.fdaft.ecm_enhancement = 1.3      # 強めのECM強化
    config.fdaft.num_layers = 4             # 多層スケール空間
    config.fdaft.max_keypoints = 1600       # 多めの特徴点
    
    # 季節変化対応
    config.joint_mamba.input_dim = 288
    config.joint_mamba.hidden_dim = 288
    config.joint_mamba.num_layers = 5
    
    # RobotCar用データ設定
    config.data.dataset_name = "robotcar"
    config.data.data_root = "data/robotcar/"
    config.data.augmentation = True         # 季節変化のaugmentation
    
    config.experiment_name = "jamf_robotcar"
    
    return config


def get_cmu_seasons_config() -> JamFConfig:
    """CMU Seasonsデータセット用設定"""
    config = get_outdoor_config()
    
    # CMU Seasons特有の調整（極端な季節変化）
    config.fdaft.phase_threshold = 0.04     # 最低閾値
    config.fdaft.ecm_enhancement = 1.6      # 最強ECM強化
    config.fdaft.contrast_enhancement = True
    config.fdaft.num_layers = 5             # 最多層スケール空間
    config.fdaft.max_keypoints = 2000       # 最多特徴点
    
    # 極端変化対応Mamba
    config.joint_mamba.input_dim = 384
    config.joint_mamba.hidden_dim = 384
    config.joint_mamba.num_layers = 8       # 最深Mamba
    config.joint_mamba.d_state = 24
    config.joint_mamba.skip_steps = 1       # 最密スキャン
    
    # 極端変化対応マッチング
    config.matching.coarse_resolution = 4   # 最細粗マッチング
    config.matching.temperature = 0.04      # 最低温度
    config.matching.coarse_threshold = 0.08  # 最低閾値
    config.matching.fine_window_size = 9    # 最大ウィンドウ
    
    # CMU用データ設定
    config.data.dataset_name = "cmu_seasons"
    config.data.batch_size = 1              # 最難設定なので小バッチ
    config.data.data_root = "data/cmu_seasons/"
    
    # より長い学習期間
    config.training.max_epochs = 200
    config.training.learning_rate = 3e-5    # より小さな学習率
    
    config.experiment_name = "jamf_cmu_seasons"
    
    return config


def get_extended_cmu_config() -> JamFConfig:
    """Extended CMU Seasonsデータセット用設定"""
    config = get_cmu_seasons_config()
    
    # Extended CMU用の追加調整
    config.fdaft.phase_threshold = 0.03     # さらに低い閾値
    config.fdaft.max_keypoints = 2500       # さらに多くの特徴点
    
    config.joint_mamba.num_layers = 10      # さらに深いMamba
    
    config.data.dataset_name = "extended_cmu"
    config.data.data_root = "data/extended_cmu/"
    config.experiment_name = "jamf_extended_cmu"
    
    return config


# 設定関数マッピング
OUTDOOR_CONFIGS = {
    "general": get_outdoor_config,
    "megadepth": get_megadepth_config,
    "hpatches": get_hpatches_config,
    "aachen": get_aachen_config,
    "robotcar": get_robotcar_config,
    "cmu_seasons": get_cmu_seasons_config,
    "extended_cmu": get_extended_cmu_config
}


def get_config(variant: str = "general") -> JamFConfig:
    """指定された屋外環境の設定を取得"""
    if variant not in OUTDOOR_CONFIGS:
        raise ValueError(f"Unknown outdoor variant: {variant}. Available: {list(OUTDOOR_CONFIGS.keys())}")
    
    return OUTDOOR_CONFIGS[variant]()