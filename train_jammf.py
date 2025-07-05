"""
Enhanced Training script for JamMa-FDAFT

This script provides comprehensive training for the integrated JamMa-FDAFT model
with support for planetary remote sensing datasets and FDAFT-specific optimizations.

Usage:
    python train_jammaf.py configs/data/megadepth_trainval_832.py configs/jamma_fdaft/outdoor/final.py --exp_name jamma_fdaft_outdoor
    python train_jammaf.py configs/data/scannet_trainval.py configs/jamma_fdaft/indoor/final.py --exp_name jamma_fdaft_indoor
"""

import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
import torch

from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_jamma_fdaft import PL_JamMaFDAFT

loguru_logger = get_rank_zero_only_logger(loguru_logger)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="JamMa-FDAFT Training Script for Planetary Remote Sensing"
    )
    
    # Core configuration
    parser.add_argument('data_cfg_path', type=str, help='data config path')
    parser.add_argument('main_cfg_path', type=str, help='main config path')
    parser.add_argument('--exp_name', type=str, default='jamma_fdaft_exp')
    parser.add_argument('--dump_dir', type=str, default=None)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=lambda x: bool(strtobool(x)),
                       nargs='?', default=True)
    
    # Model parameters
    parser.add_argument('--ckpt_path', type=str, default=None,
                       help='pretrained checkpoint path')
    parser.add_argument('--disable_ckpt', action='store_true')
    parser.add_argument('--profiler_name', type=str, default=None)
    parser.add_argument('--parallel_load_data', action='store_true')
    
    # FDAFT-specific parameters
    parser.add_argument('--fdaft_num_layers', type=int, default=3,
                       help='Number of FDAFT scale space layers')
    parser.add_argument('--fdaft_sigma', type=float, default=1.0,
                       help='Initial scale parameter for FDAFT')
    parser.add_argument('--fdaft_max_keypoints', type=int, default=2000,
                       help='Maximum keypoints for FDAFT detection')
    parser.add_argument('--use_structured_forests', action='store_true',
                       help='Use structured forests for edge detection')
    parser.add_argument('--structured_forests_path', type=str, 
                       default='assets/structured_forests/model.yml',
                       help='Path to structured forests model')
    
    # Training optimizations
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience epochs')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training')
    
    # Learning rate and scheduling
    parser.add_argument('--lr_finder', action='store_true',
                       help='Run learning rate finder')
    parser.add_argument('--auto_lr_find', action='store_true',
                       help='Auto find learning rate')
    
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def setup_callbacks(args, config, logger):
    """Setup training callbacks"""
    callbacks = []
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Model checkpoint
    if not args.disable_ckpt:
        ckpt_dir = Path(logger.log_dir) / 'checkpoints'
        ckpt_callback = ModelCheckpoint(
            monitor='auc@10',
            verbose=True,
            save_top_k=3,
            mode='max',
            save_last=True,
            dirpath=str(ckpt_dir),
            filename='{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}',
            save_weights_only=False,
            auto_insert_metric_name=False
        )
        callbacks.append(ckpt_callback)
    
    # Early stopping for planetary datasets (optional)
    if args.early_stopping_patience > 0:
        early_stop_callback = EarlyStopping(
            monitor='auc@10',
            min_delta=0.001,
            patience=args.early_stopping_patience,
            verbose=True,
            mode='max'
        )
        callbacks.append(early_stop_callback)
    
    return callbacks


def update_config_with_args(config, args):
    """Update configuration with command line arguments"""
    # FDAFT-specific configuration updates
    if hasattr(config, 'FDAFT'):
        config.FDAFT.NUM_LAYERS = args.fdaft_num_layers
        config.FDAFT.SIGMA_0 = args.fdaft_sigma
        config.FDAFT.MAX_KEYPOINTS = args.fdaft_max_keypoints
        config.FDAFT.USE_STRUCTURED_FORESTS = args.use_structured_forests
        if args.structured_forests_path:
            config.FDAFT.STRUCTURED_FORESTS_MODEL = args.structured_forests_path
    
    # Training optimizations
    if args.gradient_clip_val:
        config.TRAINER.GRADIENT_CLIPPING = args.gradient_clip_val
    
    # Mixed precision
    if args.mixed_precision:
        config.JAMMA.MP = True
    
    return config


def print_model_summary(model, config):
    """Print comprehensive model summary"""
    print("\n" + "="*80)
    print("JAMMA-FDAFT MODEL SUMMARY")
    print("="*80)
    
    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Architecture components
    print(f"\nArchitecture Components:")
    print(f"├── FDAFT Encoder: {sum(p.numel() for p in model.backbone.parameters()):,} params")
    print(f"├── Joint Mamba: {sum(p.numel() for p in model.matcher.joint_mamba.parameters()):,} params")
    print(f"├── Coarse Matching: {sum(p.numel() for p in model.matcher.coarse_matching.parameters()):,} params")
    print(f"└── Fine Matching: {sum(p.numel() for p in model.matcher.fine_matching.parameters()):,} params")
    
    # Configuration
    print(f"\nConfiguration:")
    print(f"├── FDAFT Layers: {config.FDAFT.NUM_LAYERS}")
    print(f"├── FDAFT Sigma: {config.FDAFT.SIGMA_0}")
    print(f"├── Max Keypoints: {config.FDAFT.MAX_KEYPOINTS}")
    print(f"├── Structured Forests: {config.FDAFT.USE_STRUCTURED_FORESTS}")
    print(f"├── Coarse D-Model: {config.JAMMA.COARSE.D_MODEL}")
    print(f"├── Fine D-Model: {config.JAMMA.FINE.D_MODEL}")
    print(f"└── Resolution: {config.JAMMA.RESOLUTION}")
    
    print("="*80)


def run_lr_finder(trainer, model, data_module):
    """Run learning rate finder"""
    print("\n" + "="*60)
    print("LEARNING RATE FINDER")
    print("="*60)
    
    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(model, datamodule=data_module)
    
    # Plot results
    fig = lr_finder.plot(suggest=True)
    fig.savefig('lr_finder_plot.png')
    
    # Get suggestion
    suggested_lr = lr_finder.suggestion()
    print(f"Suggested learning rate: {suggested_lr}")
    
    # Update model learning rate
    model.learning_rate = suggested_lr
    
    print("="*60)


def main():
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # Initialize configuration
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    
    # Update config with command line arguments
    config = update_config_with_args(config, args)
    
    pl.seed_everything(config.TRAINER.SEED)
    
    # Setup distributed training
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)
    
    loguru_logger.info(f"JamMa-FDAFT training setup:")
    loguru_logger.info(f"  GPUs: {_n_gpus}, Nodes: {args.num_nodes}")
    loguru_logger.info(f"  World size: {config.TRAINER.WORLD_SIZE}")
    loguru_logger.info(f"  Batch size: {args.batch_size} per GPU, {config.TRAINER.TRUE_BATCH_SIZE} total")
    loguru_logger.info(f"  Learning rate: {config.TRAINER.TRUE_LR}")
    
    # Initialize JamMa-FDAFT model
    profiler = build_profiler(args.profiler_name)
    model = PL_JamMaFDAFT(
        config, 
        pretrained_ckpt=args.ckpt_path, 
        profiler=profiler, 
        dump_dir=args.dump_dir
    )
    loguru_logger.info(f"JamMa-FDAFT LightningModule initialized!")
    
    # Print model summary
    print_model_summary(model, config)
    
    # Data module
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"JamMa-FDAFT DataModule initialized!")
    
    # Logger and callbacks
    logger = TensorBoardLogger(
        save_dir='jamma_fdaft_logs/', 
        name=args.exp_name, 
        default_hp_metric=False
    )
    callbacks = setup_callbacks(args, config, logger)
    
    # Trainer setup
    trainer_kwargs = {
        'plugins': DDPPlugin(
            find_unused_parameters=False,
            num_nodes=args.num_nodes,
            sync_batchnorm=config.TRAINER.WORLD_SIZE > 0
        ),
        'gradient_clip_val': config.TRAINER.get('GRADIENT_CLIPPING', 1.0),
        'callbacks': callbacks,
        'logger': logger,
        'sync_batchnorm': config.TRAINER.WORLD_SIZE > 0,
        'replace_sampler_ddp': False,
        'reload_dataloaders_every_epoch': False,
        'weights_summary': 'full',
        'profiler': profiler,
        'precision': 16 if args.mixed_precision else 32,
        'auto_lr_find': args.auto_lr_find,
        'benchmark': True,
        'deterministic': False,  # Set to True for reproducible results (slower)
    }
    
    # Add trainer args
    trainer = pl.Trainer.from_argparse_args(args, **trainer_kwargs)
    
    loguru_logger.info(f"JamMa-FDAFT Trainer initialized!")
    
    # Learning rate finder
    if args.lr_finder:
        run_lr_finder(trainer, model, data_module)
    
    # Training
    loguru_logger.info(f"Starting JamMa-FDAFT training!")
    loguru_logger.info(f"Experiment: {args.exp_name}")
    loguru_logger.info(f"Log directory: {logger.log_dir}")
    
    # Print training start info
    print("\n" + "="*80)
    print("STARTING JAMMA-FDAFT TRAINING")
    print("="*80)
    print(f"Experiment: {args.exp_name}")
    print(f"Dataset: {config.DATASET.TRAINVAL_DATA_SOURCE}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Architecture: Input → FDAFT Encoder → Joint Mamba → C2F Matching")
    print("="*80)
    
    try:
        trainer.fit(model, datamodule=data_module)
        
        # Training completed successfully
        print("\n" + "="*80)
        print("JAMMA-FDAFT TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Print best metrics
        if not args.disable_ckpt:
            best_model_path = callbacks[1].best_model_path if len(callbacks) > 1 else None
            if best_model_path:
                print(f"Best model saved at: {best_model_path}")
                print(f"Best AUC@10: {callbacks[1].best_model_score:.4f}")
        
        loguru_logger.info("JamMa-FDAFT training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("TRAINING INTERRUPTED BY USER")
        print("="*80)
        loguru_logger.warning("Training interrupted by user")
        
    except Exception as e:
        print("\n" + "="*80)
        print("TRAINING FAILED")
        print("="*80)
        print(f"Error: {e}")
        loguru_logger.error(f"Training failed: {e}")
        raise
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()