"""
Training script for JamMa-FDAFT

Usage:
    python train.py configs/data/megadepth_trainval_832.py configs/jamma_fdaft/outdoor/final.py --exp_name jamma_fdaft_outdoor
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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_jamma_fdaft import PL_JamMaFDAFT

loguru_logger = get_rank_zero_only_logger(loguru_logger)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_cfg_path', type=str, help='data config path')
    parser.add_argument('main_cfg_path', type=str, help='main config path')
    parser.add_argument('--exp_name', type=str, default='jamma_fdaft_exp')
    parser.add_argument('--dump_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=lambda x: bool(strtobool(x)),
                       nargs='?', default=True)
    parser.add_argument('--ckpt_path', type=str, default=None,
                       help='pretrained checkpoint path')
    parser.add_argument('--disable_ckpt', action='store_true')
    parser.add_argument('--profiler_name', type=str, default=None)
    parser.add_argument('--parallel_load_data', action='store_true')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # Initialize configuration
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)
    
    # Setup distributed training
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)
    
    # Initialize JamMa-FDAFT model
    profiler = build_profiler(args.profiler_name)
    model = PL_JamMaFDAFT(config, pretrained_ckpt=args.ckpt_path, profiler=profiler, dump_dir=args.dump_dir)
    loguru_logger.info(f"JamMa-FDAFT LightningModule initialized!")
    
    # Data module
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"JamMa-FDAFT DataModule initialized!")
    
    # Logger and callbacks
    logger = TensorBoardLogger(save_dir='jamma_fdaft_logs/', name=args.exp_name, default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / 'checkpoints'

    ckpt_callback = ModelCheckpoint(
        monitor='auc@10', verbose=True, save_top_k=3, mode='max',
        save_last=True, dirpath=str(ckpt_dir),
        filename='{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)
    
    # Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        plugins=DDPPlugin(find_unused_parameters=False,
                          num_nodes=args.num_nodes,
                          sync_batchnorm=config.TRAINER.WORLD_SIZE > 0),
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
        replace_sampler_ddp=False,
        reload_dataloaders_every_epoch=False,
        weights_summary='full',
        profiler=profiler,
    )

    loguru_logger.info(f"JamMa-FDAFT Trainer initialized!")
    loguru_logger.info(f"Starting JamMa-FDAFT training!")
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()