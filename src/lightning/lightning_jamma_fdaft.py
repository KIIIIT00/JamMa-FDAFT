"""
PyTorch Lightning Module for JamMa-FDAFT

Integrates FDAFT feature extraction with JamMa's Joint Mamba and C2F matching
in a unified training and inference pipeline.

Key Features:
- FDAFT backbone for robust planetary image feature extraction
- Joint Mamba (JEGO) for efficient long-range feature interaction
- Hierarchical coarse-to-fine matching with sub-pixel refinement
- Compatible with JamMa's existing training infrastructure
"""

from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from ..jamma_fdaft.jamma_fdaft import JamMaFDAFT
from ..jamma_fdaft.backbone_fdaft import FDAFTEncoder
from ..jamma.utils.supervision import compute_supervision_fine, compute_supervision_coarse
from ..losses.loss import Loss
from ..optimizers import build_optimizer, build_scheduler
from ..utils.metrics import (
    compute_f1,
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics_train_val, aggregate_metrics_test
)
from ..utils.comm import gather, all_gather
from ..utils.misc import lower_config, flattenList
from ..utils.profiler import PassThroughProfiler
from ..utils.plotting import make_matching_figures


class PL_JamMaFDAFT(pl.LightningModule):
    """
    PyTorch Lightning module for JamMa-FDAFT training and inference
    
    Combines FDAFT's robust feature extraction with JamMa's efficient matching pipeline
    """
    
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        super().__init__()
        
        # Configuration and setup
        self.config = config
        _config = lower_config(self.config)
        self.JAMMA_FDAFT_cfg = lower_config(_config['jamma'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)
        
        # Visualization setup
        self.viz_path = Path('visualization')
        self.viz_path.mkdir(parents=True, exist_ok=True)
        
        # Model components
        self.backbone = FDAFTEncoder(
            num_layers=_config.get('fdaft', {}).get('num_layers', 3),
            sigma_0=_config.get('fdaft', {}).get('sigma_0', 1.0),
            use_structured_forests=_config.get('fdaft', {}).get('use_structured_forests', True),
            max_keypoints=_config.get('fdaft', {}).get('max_keypoints', 2000),
            nms_radius=_config.get('fdaft', {}).get('nms_radius', 5)
        )
        
        self.matcher = JamMaFDAFT(config=_config['jamma'], profiler=profiler)
        self.loss = Loss(_config)
        
        # Load pretrained weights if specified
        if pretrained_ckpt == 'official':
            self._load_official_weights()
        elif pretrained_ckpt:
            self._load_pretrained_weights(pretrained_ckpt)
            
        # Testing setup
        self.dump_dir = dump_dir
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.total_ms = 0
        
        # Model info
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'JamMa-FDAFT model parameters: {n_parameters / 1e6:.2f}M')
        self._print_model_info()
    
    def _print_model_info(self):
        """Print model architecture information"""
        print("\n" + "="*60)
        print("JAMMA-FDAFT MODEL ARCHITECTURE")
        print("="*60)
        
        # Backbone info
        fdaft_info = self.backbone.get_fdaft_info()
        print(f"Backbone: {fdaft_info['encoder_type']}")
        print(f"  Scale Space Layers: {fdaft_info['scale_space_layers']}")
        print(f"  Structured Forests: {fdaft_info['structured_forests']}")
        print(f"  Output Features: {fdaft_info['output_features']['feat_8']}, {fdaft_info['output_features']['feat_4']}")
        
        # Matcher info
        matcher_info = self.matcher.get_model_info()
        print(f"Matcher: {matcher_info['model_name']}")
        print(f"  Architecture: {matcher_info['architecture']}")
        print(f"  Coarse D-Model: {matcher_info['parameters']['coarse_d_model']}")
        print(f"  Fine D-Model: {matcher_info['parameters']['fine_d_model']}")
        print("="*60)
    
    def _load_official_weights(self):
        """Load official JamMa-FDAFT weights (when available)"""
        try:
            # This would load official weights when they become available
            print("Official JamMa-FDAFT weights not yet available.")
            print("Training from scratch with FDAFT initialization.")
        except Exception as e:
            logger.warning(f"Could not load official weights: {e}")
    
    def _load_pretrained_weights(self, pretrained_ckpt):
        """Load pretrained weights from checkpoint"""
        try:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            
            # Filter out backbone weights if loading JamMa weights
            # (since we're replacing ConvNext with FDAFT)
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if 'backbone' not in key:  # Skip original backbone weights
                    filtered_state_dict[key] = value
            
            self.load_state_dict(filtered_state_dict, strict=False)
            logger.info(f"Loaded compatible weights from: {pretrained_ckpt}")
            
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")

    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                      optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        """Custom optimizer step with learning rate warm-up"""
        warmup_step = self.config.TRAINER.WARMUP_STEP
        
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                     (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                     abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _train_inference(self, batch):
        """Training inference pipeline"""
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("FDAFT Backbone"):
            with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
                self.backbone(batch)

        with self.profiler.profile("JamMa-FDAFT Matcher"):
            self.matcher(batch, mode='train')

        with self.profiler.profile("Compute fine supervision"):
            with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
                compute_supervision_fine(batch, self.config)

        with self.profiler.profile("Compute losses"):
            with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
                self.loss(batch)

    def _val_inference(self, batch):
        """Validation inference pipeline"""
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("FDAFT Backbone"):
            with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
                self.backbone(batch)

        with self.profiler.profile("JamMa-FDAFT Matcher"):
            self.matcher(batch, mode='val')

        with self.profiler.profile("Compute fine supervision"):
            with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
                compute_supervision_fine(batch, self.config)

        with self.profiler.profile("Compute losses"):
            with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
                self.loss(batch)

    def _compute_metrics_val(self, batch):
        """Compute validation metrics"""
        with self.profiler.profile("Compute metrics"):
            compute_f1(batch)
            compute_symmetrical_epipolar_errors(batch)
            compute_pose_errors(batch, self.config)

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['imagec_0'].size(0)
            metrics = {
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                'precision': batch['precision'],
                'recall': batch['recall'],
                'f1_score': batch['f1_score'],
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                'inliers': batch['inliers']
            }
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names

    def _compute_metrics(self, batch):
        """Compute test metrics"""
        with self.profiler.profile("Compute metrics"):
            compute_symmetrical_epipolar_errors(batch)
            compute_pose_errors(batch, self.config)

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['imagec_0'].size(0)
            metrics = {
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                'inliers': batch['inliers']
            }
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names

    def training_step(self, batch, batch_idx):
        """Training step"""
        self._train_inference(batch)
        
        # Logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            self.logger.experiment.add_scalar(f'train_loss', batch['loss'], self.global_step)

        return {'loss': batch['loss']}

    def training_epoch_end(self, outputs):
        """Training epoch end"""
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                'train/avg_loss_on_epoch', avg_loss,
                global_step=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        self._val_inference(batch)
        ret_dict, _ = self._compute_metrics_val(batch)
        ret_dict['metrics'] = {**ret_dict['metrics'], 'max_matches': [batch['num_candidates_max']]}
        
        figures = {self.config.TRAINER.PLOT_MODE: []}

        return {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
        }

    def validation_epoch_end(self, outputs):
        """Validation epoch end"""
        # Handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)

        for valset_idx, outputs in enumerate(multi_outputs):
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
                cur_epoch = -1

            # Loss scalars
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            # Validation metrics
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            
            val_metrics_4tb = aggregate_metrics_train_val(metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config)
            for thr in [5, 10, 20]:
                multi_val_metrics[f'auc@{thr}'].append(val_metrics_4tb[f'auc@{thr}'])

            # Tensorboard logging
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)

            plt.close('all')

        for thr in [5, 10, 20]:
            self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])))

    def test_step(self, batch, batch_idx):
        """Test step"""
        with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
            self.start_event.record()
            
            with self.profiler.profile("FDAFT Backbone"):
                self.backbone(batch)

            with self.profiler.profile("JamMa-FDAFT Matcher"):
                self.matcher(batch, mode='test')

            self.end_event.record()
            torch.cuda.synchronize()
            self.total_ms += self.start_event.elapsed_time(self.end_event)
            batch['runtime'] = self.start_event.elapsed_time(self.end_event)

        ret_dict, rel_pair_names = self._compute_metrics(batch)

        # Optional result dumping
        if self.dump_dir is not None:
            with self.profiler.profile("dump_results"):
                bs = batch['imagec_0'].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch['m_bids'] == b_id
                    epi_errs = batch['epi_errs'][mask].cpu().numpy()
                    correct_mask = epi_errs < 1e-4
                    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
                    n_correct = np.sum(correct_mask)
                    
                    item.update({
                        'precision': precision,
                        'n_correct': n_correct,
                        'runtime': batch['runtime']
                    })
                    
                    for key in ['R_errs', 't_errs']:
                        item[key] = batch[key][b_id][0]
                    
                    dumps.append(item)
                ret_dict['dumps'] = dumps
                
        return ret_dict

    def test_epoch_end(self, outputs):
        """Test epoch end"""
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o['dumps'] for o in outputs])
            dumps = flattenList(gather(_dumps))
            logger.info(f'Results will be saved to: {self.dump_dir}')

        if self.trainer.global_rank == 0:
            print("\n" + "="*60)
            print("JAMMA-FDAFT TEST RESULTS")
            print("="*60)
            print(self.profiler.summary())
            
            val_metrics_4tb = aggregate_metrics_test(metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))
            print(f'Average matching time: {self.total_ms / len(outputs):.2f} ms')
            
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / 'JamMa_FDAFT_results', dumps)
                
            print("="*60)