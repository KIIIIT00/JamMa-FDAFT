"""
Enhanced Testing script for JamMa-FDAFT

This script provides comprehensive testing for the integrated JamMa-FDAFT model
with support for planetary remote sensing datasets and detailed performance analysis.

Usage:
    python test_jammf.py configs/data/megadepth_test_1500.py configs/jamma_fdaft/outdoor/test.py --ckpt_path weight/jamma_fdaft_weight.ckpt
    python test_jammf.py configs/data/scannet_test_1500.py configs/jamma_fdaft/indoor/test.py --ckpt_path weight/jamma_fdaft_weight.ckpt
"""

import argparse
import pprint
import time
from pathlib import Path
import pytorch_lightning as pl
from loguru import logger as loguru_logger
import torch
import numpy as np

from src.config.default import get_cfg_defaults
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_jamma_fdaft import PL_JamMaFDAFT
from src.utils.profiler import build_profiler


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="JamMa-FDAFT Testing Script for Planetary Remote Sensing"
    )
    
    # Core configuration
    parser.add_argument('data_cfg_path', type=str, help='data config path')
    parser.add_argument('main_cfg_path', type=str, help='main config path')
    parser.add_argument('--ckpt_path', type=str, default="weights/jamma_fdaft.ckpt")
    parser.add_argument('--dump_dir', type=str, default=None)
    parser.add_argument('--profiler_name', type=str, default=None)
    
    # Testing parameters
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--thr', type=float, default=None, help='matching threshold override')
    
    # FDAFT-specific testing parameters
    parser.add_argument('--fdaft_eval_mode', type=str, default='standard',
                       choices=['standard', 'comprehensive', 'fast'],
                       help='FDAFT evaluation mode')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save matching visualizations')
    parser.add_argument('--visualization_dir', type=str, default='test_visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--eval_structured_forests', action='store_true',
                       help='Evaluate with/without structured forests')
    
    # Performance analysis
    parser.add_argument('--detailed_analysis', action='store_true',
                       help='Perform detailed performance analysis')
    parser.add_argument('--save_features', action='store_true',
                       help='Save extracted FDAFT features for analysis')
    parser.add_argument('--benchmark_mode', action='store_true',
                       help='Run in benchmark mode for speed testing')
    
    # Output options
    parser.add_argument('--save_results', action='store_true',
                       help='Save detailed results to file')
    parser.add_argument('--results_file', type=str, default='jamma_fdaft_results.json',
                       help='Results file name')
    
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def setup_test_environment(args):
    """Setup testing environment and directories"""
    # Create output directories
    if args.save_visualizations:
        Path(args.visualization_dir).mkdir(parents=True, exist_ok=True)
        loguru_logger.info(f"Visualization directory: {args.visualization_dir}")
    
    if args.dump_dir:
        Path(args.dump_dir).mkdir(parents=True, exist_ok=True)
        loguru_logger.info(f"Dump directory: {args.dump_dir}")
    
    # Setup logging for testing
    log_file = f"jamma_fdaft_test_{time.strftime('%Y%m%d_%H%M%S')}.log"
    loguru_logger.add(log_file, rotation="500 MB")
    loguru_logger.info(f"Test log file: {log_file}")


def print_test_summary(args, config):
    """Print comprehensive test setup summary"""
    print("\n" + "="*80)
    print("JAMMA-FDAFT TESTING CONFIGURATION")
    print("="*80)
    
    print(f"Model checkpoint: {args.ckpt_path}")
    print(f"Data config: {args.data_cfg_path}")
    print(f"Main config: {args.main_cfg_path}")
    print(f"Dataset: {config.DATASET.TEST_DATA_SOURCE}")
    
    print(f"\nFDAFT Configuration:")
    print(f"├── Evaluation mode: {args.fdaft_eval_mode}")
    print(f"├── Structured forests eval: {args.eval_structured_forests}")
    print(f"├── Save visualizations: {args.save_visualizations}")
    print(f"├── Detailed analysis: {args.detailed_analysis}")
    print(f"└── Benchmark mode: {args.benchmark_mode}")
    
    print(f"\nTesting Parameters:")
    print(f"├── Batch size: {args.batch_size}")
    print(f"├── Num workers: {args.num_workers}")
    print(f"├── Matching threshold: {args.thr if args.thr else 'default'}")
    print(f"└── Profiler: {args.profiler_name if args.profiler_name else 'disabled'}")
    
    print("="*80)


def run_performance_benchmark(model, data_module, trainer):
    """Run performance benchmark for JamMa-FDAFT"""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    # Warm up
    loguru_logger.info("Warming up model...")
    with torch.no_grad():
        dummy_batch = next(iter(data_module.test_dataloader()))
        for key in dummy_batch:
            if isinstance(dummy_batch[key], torch.Tensor):
                dummy_batch[key] = dummy_batch[key][:1]  # Single sample
        
        # Run a few warm-up iterations
        for _ in range(3):
            model.backbone(dummy_batch)
            model.matcher(dummy_batch, mode='test')
    
    # Benchmark timing
    loguru_logger.info("Running benchmark...")
    times = []
    torch.cuda.synchronize()
    
    for i in range(10):
        start_time = time.time()
        
        with torch.no_grad():
            model.backbone(dummy_batch)
            model.matcher(dummy_batch, mode='test')
        
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    
    # Statistics
    times = np.array(times[2:])  # Remove first 2 runs
    mean_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    
    print(f"Inference time statistics (ms):")
    print(f"├── Mean: {mean_time:.2f} ± {std_time:.2f}")
    print(f"├── Min: {min_time:.2f}")
    print(f"├── Max: {max_time:.2f}")
    print(f"└── FPS: {1000/mean_time:.1f}")
    
    print("="*60)
    
    return {
        'mean_time_ms': mean_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'fps': 1000/mean_time
    }


def analyze_fdaft_components(model, sample_batch):
    """Analyze FDAFT components performance"""
    print("\n" + "="*60)
    print("FDAFT COMPONENTS ANALYSIS")
    print("="*60)
    
    with torch.no_grad():
        # Extract FDAFT features
        model.backbone(sample_batch)
        
        # Analyze feature quality
        feat_8_0 = sample_batch['feat_8_0']
        feat_4_0 = sample_batch['feat_4_0']
        
        # Feature statistics
        feat_8_stats = {
            'shape': feat_8_0.shape,
            'mean': feat_8_0.mean().item(),
            'std': feat_8_0.std().item(),
            'min': feat_8_0.min().item(),
            'max': feat_8_0.max().item(),
            'sparsity': (feat_8_0 == 0).float().mean().item()
        }
        
        feat_4_stats = {
            'shape': feat_4_0.shape,
            'mean': feat_4_0.mean().item(),
            'std': feat_4_0.std().item(),
            'min': feat_4_0.min().item(),
            'max': feat_4_0.max().item(),
            'sparsity': (feat_4_0 == 0).float().mean().item()
        }
        
        print("Feature 8x (Coarse):")
        print(f"├── Shape: {feat_8_stats['shape']}")
        print(f"├── Mean: {feat_8_stats['mean']:.4f}")
        print(f"├── Std: {feat_8_stats['std']:.4f}")
        print(f"├── Range: [{feat_8_stats['min']:.4f}, {feat_8_stats['max']:.4f}]")
        print(f"└── Sparsity: {feat_8_stats['sparsity']:.4f}")
        
        print("\nFeature 4x (Fine):")
        print(f"├── Shape: {feat_4_stats['shape']}")
        print(f"├── Mean: {feat_4_stats['mean']:.4f}")
        print(f"├── Std: {feat_4_stats['std']:.4f}")
        print(f"├── Range: [{feat_4_stats['min']:.4f}, {feat_4_stats['max']:.4f}]")
        print(f"└── Sparsity: {feat_4_stats['sparsity']:.4f}")
    
    print("="*60)
    
    return {
        'feat_8_stats': feat_8_stats,
        'feat_4_stats': feat_4_stats
    }


def save_test_results(results, args):
    """Save test results to file"""
    if not args.save_results:
        return
    
    import json
    from datetime import datetime
    
    # Prepare results dictionary
    results_dict = {
        'timestamp': datetime.now().isoformat(),
        'model': 'JamMa-FDAFT',
        'test_config': {
            'data_config': args.data_cfg_path,
            'main_config': args.main_cfg_path,
            'checkpoint': args.ckpt_path,
            'evaluation_mode': args.fdaft_eval_mode,
            'threshold': args.thr,
        },
        'results': results
    }
    
    # Save to file
    results_file = Path(args.results_file)
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    loguru_logger.info(f"Results saved to: {results_file}")
    print(f"Results saved to: {results_file}")


def main():
    args = parse_args()
    
    # Setup test environment
    setup_test_environment(args)
    
    # Print arguments
    rank_zero_only = lambda x: x  # Simplified for testing
    rank_zero_only(pprint.pprint)(vars(args))

    # Initialize configuration
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)

    # Tune matching threshold if specified
    if args.thr is not None:
        config.JAMMA.MATCH_COARSE.THR = args.thr
        loguru_logger.info(f"Matching threshold set to: {args.thr}")

    # Print test summary
    print_test_summary(args, config)
    
    # Model and data
    profiler = build_profiler(args.profiler_name)
    model = PL_JamMaFDAFT(
        config, 
        pretrained_ckpt=args.ckpt_path, 
        profiler=profiler, 
        dump_dir=args.dump_dir
    )
    loguru_logger.info(f"JamMa-FDAFT model loaded from: {args.ckpt_path}")

    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"DataModule initialized for: {config.DATASET.TEST_DATA_SOURCE}")

    # Trainer
    trainer = pl.Trainer.from_argparse_args(
        args, 
        replace_sampler_ddp=False, 
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True
    )

    # Run tests
    print("\n" + "="*80)
    print("STARTING JAMMA-FDAFT TESTING")
    print("="*80)
    
    results = {}
    
    # Performance benchmark
    if args.benchmark_mode:
        try:
            benchmark_results = run_performance_benchmark(model, data_module, trainer)
            results['benchmark'] = benchmark_results
        except Exception as e:
            loguru_logger.error(f"Benchmark failed: {e}")
            results['benchmark'] = {'error': str(e)}
    
    # Component analysis
    if args.detailed_analysis:
        try:
            sample_batch = next(iter(data_module.test_dataloader()))
            component_results = analyze_fdaft_components(model, sample_batch)
            results['component_analysis'] = component_results
        except Exception as e:
            loguru_logger.error(f"Component analysis failed: {e}")
            results['component_analysis'] = {'error': str(e)}
    
    # Main testing
    loguru_logger.info(f"Starting main testing on {config.DATASET.TEST_DATA_SOURCE}")
    
    start_time = time.time()
    test_results = trainer.test(model, datamodule=data_module, verbose=True)
    end_time = time.time()
    
    # Process results
    total_time = end_time - start_time
    results['main_test'] = {
        'total_time_seconds': total_time,
        'test_results': test_results
    }
    
    # Final summary
    print("\n" + "="*80)
    print("JAMMA-FDAFT TESTING COMPLETED")
    print("="*80)
    
    print(f"Total testing time: {total_time:.2f} seconds")
    
    if args.benchmark_mode and 'benchmark' in results:
        bench = results['benchmark']
        if 'mean_time_ms' in bench:
            print(f"Average inference time: {bench['mean_time_ms']:.2f} ms")
            print(f"Throughput: {bench['fps']:.1f} FPS")
    
    # Print profiler summary if available
    if hasattr(model, 'profiler') and hasattr(model.profiler, 'summary'):
        print("\nProfiler Summary:")
        print(model.profiler.summary())
    
    # Save results
    save_test_results(results, args)
    
    loguru_logger.info("JamMa-FDAFT testing completed successfully!")
    
    print("\n" + "="*80)
    print("JamMa-FDAFT: FDAFT features + Joint Mamba + C2F matching")
    print("Optimized for planetary remote sensing image matching")
    print("="*80)


if __name__ == '__main__':
    main()