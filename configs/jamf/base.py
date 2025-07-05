"""
JamMa-FDAFT 基本設定
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class FDAFTConfig:
    """FDAFT Encoder設定"""
    num_layers: int = 3
    sigma_0: float = 1.0
    descriptor_radius: int = 48
    max_keypoints: int = 1000
    structured_forests_model_path: str = "assets/structured_forests_model.yml"
    
    # モード別パラメータ
    phase_threshold: float = 0.05
    ecm_enhancement: float = 1.2
    contrast_enhancement: bool = True


@dataclass
class JointMambaConfig:
    """Joint Mamba設定"""
    input_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    skip_steps: int = 2
    
    # SSMパラメータ
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2


@dataclass
class MatchingConfig:
    """マッチング設定"""
    coarse_resolution: int = 8
    fine_resolution: int = 2
    temperature: float = 0.1
    coarse_threshold: float = 0.2
    fine_window_size: int = 5


@dataclass
class DataConfig:
    """データセット設定"""
    batch_size: int = 4
    num_workers: int = 4
    image_size: tuple = (480, 640)
    augmentation: bool = True
    
    # データセット固有設定
    dataset_name: str = "megadepth"
    data_root: str = "data/"
    split: str = "train"


@dataclass
class TrainingConfig:
    """学習設定"""
    max_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: str = "cosine"  # cosine, step, exponential
    warmup_epochs: int = 5
    
    # 損失関数重み
    coarse_loss_weight: float = 1.0
    fine_loss_weight: float = 1.0
    confidence_loss_weight: float = 0.1
    
    # 検証・保存
    val_every_n_epochs: int = 5
    save_every_n_epochs: int = 10
    early_stopping_patience: int = 20


@dataclass
class OptimizationConfig:
    """最適化設定"""
    optimizer: str = "adamw"  # adamw, adam, sgd
    gradient_clip_norm: Optional[float] = 1.0
    accumulate_grad_batches: int = 1
    
    # 学習率スケジューラ
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5
    min_lr: float = 1e-7


@dataclass
class JamFConfig:
    """JamMa-FDAFT統合設定"""
    # モデル設定
    mode: str = "planetary"  # planetary, indoor, outdoor
    device: str = "cuda"
    
    # コンポーネント設定
    fdaft: FDAFTConfig = None
    joint_mamba: JointMambaConfig = None
    matching: MatchingConfig = None
    
    # 学習・データ設定
    training: TrainingConfig = None
    optimization: OptimizationConfig = None
    data: DataConfig = None
    
    # ログ・出力設定
    output_dir: str = "outputs/"
    log_dir: str = "logs/"
    checkpoint_dir: str = "checkpoints/"
    
    # 実験設定
    experiment_name: str = "jamf_base"
    seed: int = 42
    deterministic: bool = True
    
    def __post_init__(self):
        """デフォルト値の設定"""
        if self.fdaft is None:
            self.fdaft = FDAFTConfig()
        if self.joint_mamba is None:
            self.joint_mamba = JointMambaConfig()
        if self.matching is None:
            self.matching = MatchingConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.optimization is None:
            self.optimization = OptimizationConfig()
        if self.data is None:
            self.data = DataConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        import dataclasses
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'JamFConfig':
        """辞書から設定を復元"""
        # ネストした設定の復元
        if 'fdaft' in config_dict and isinstance(config_dict['fdaft'], dict):
            config_dict['fdaft'] = FDAFTConfig(**config_dict['fdaft'])
        if 'joint_mamba' in config_dict and isinstance(config_dict['joint_mamba'], dict):
            config_dict['joint_mamba'] = JointMambaConfig(**config_dict['joint_mamba'])
        if 'matching' in config_dict and isinstance(config_dict['matching'], dict):
            config_dict['matching'] = MatchingConfig(**config_dict['matching'])
        if 'training' in config_dict and isinstance(config_dict['training'], dict):
            config_dict['training'] = TrainingConfig(**config_dict['training'])
        if 'optimization' in config_dict and isinstance(config_dict['optimization'], dict):
            config_dict['optimization'] = OptimizationConfig(**config_dict['optimization'])
        if 'data' in config_dict and isinstance(config_dict['data'], dict):
            config_dict['data'] = DataConfig(**config_dict['data'])
        
        return cls(**config_dict)


def get_base_config() -> JamFConfig:
    """基本設定を取得"""
    return JamFConfig(
        experiment_name="jamf_base",
        mode="planetary"
    )