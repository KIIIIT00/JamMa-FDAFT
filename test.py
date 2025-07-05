"""
Testing script for JamMa-FDAFT

Usage:
    python test.py configs/data/megadepth_test_1500.py configs/jamma_fdaft/outdoor/test.py --ckpt_path weight/jamma_fdaft_weight.ckpt
"""

import argparse
import pprint
import pytorch_lightning as pl
from loguru import logger as loguru_logger

from src.config.default import get_cfg_defaults
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_jamma_fdaft import PL_JamMaFDAFT
from src.utils.profiler import build_profiler


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_cfg_path', type=str, help='data config path')
    parser.add_argument('main_cfg_path', type=str, help='main config path')
    parser.add_argument('--ckpt_path', type=str, default="weights/jamma_fdaft.ckpt")
    parser.add_argument('--dump_dir', type=str, default=None)
    parser.add_argument('--profiler_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--thr', type=float, default=None)

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pprint.pprint(vars(args))

    # Initialize configuration
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)

    # Tune matching threshold if specified
    if args.thr is not None:
        config.JAMMA.MATCH_COARSE.THR = args.thr

    loguru_logger.info(f"JamMa-FDAFT test configuration initialized!")

    # Model and data
    profiler = build_profiler(args.profiler_name)
    model = PL_JamMaFDAFT(config, pretrained_ckpt=args.ckpt_path, profiler=profiler, dump_dir=args.dump_dir)
    loguru_logger.info(f"JamMa-FDAFT model loaded!")

    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"JamMa-FDAFT DataModule initialized!")

    # Trainer
    trainer = pl.Trainer.from_argparse_args(args, replace_sampler_ddp=False, logger=False)

    loguru_logger.info(f"Starting JamMa-FDAFT testing!")
    trainer.test(model, datamodule=data_module, verbose=False)