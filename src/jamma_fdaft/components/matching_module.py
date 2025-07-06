"""
JamMa-FDAFT Matching Module

This module implements the matching components for JamMa-FDAFT,
including coarse matching and fine matching optimized for FDAFT features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from loguru import logger
import numpy as np

INF = 1e9


def mask_border(m, b: int, v):
    """Mask borders with value"""
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch"""
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


def generate_random_mask(n, num_true):
    """Create random mask for training"""
    mask = torch.zeros(n, dtype=torch.bool)
    indices = torch.randperm(n)[:num_true]
    mask[indices] = True
    return mask


class FDAFTCoarseMatching(nn.Module):
    """
    Coarse matching module optimized for FDAFT features
    """
    
    def __init__(self, config, profiler):
        super().__init__()
        self.config = config
        self.profiler = profiler
        
        # Configuration
        d_model = 256  # Match FDAFT output dimension
        self.thr = config.THR
        self.use_sm = config['use_sm']
        self.inference = config['inference']
        self.border_rm = config['border_rm']
        self.temperature = config['dsmax_temperature']
        
        # FDAFT feature projection
        self.fdaft_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Final projection for matching
        self.final_proj = nn.Linear(d_model, d_model, bias=True)
        
        # Adaptive temperature learning for FDAFT features
        self.adaptive_temperature = nn.Parameter(torch.tensor(self.temperature))

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        # Project FDAFT features
        feat_c0 = self.fdaft_proj(feat_c0)
        feat_c1 = self.fdaft_proj(feat_c1)
        
        # Final projection
        feat_c0 = self.final_proj(feat_c0)
        feat_c1 = self.final_proj(feat_c1)

        # Normalize features
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1] ** .5,
                               [feat_c0, feat_c1])

        # Compute similarity matrix with adaptive temperature
        sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                  feat_c1) / self.adaptive_temperature
        
        if mask_c0 is not None:
            sim_matrix.masked_fill_(
                ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                -INF)
        
        if self.inference:
            # Inference mode
            data.update(**self.get_coarse_match_inference(sim_matrix, data))
        else:
            # Training mode
            conf_matrix_0_to_1 = F.softmax(sim_matrix, 2)
            conf_matrix_1_to_0 = F.softmax(sim_matrix, 1)
            data.update({
                'conf_matrix_0_to_1': conf_matrix_0_to_1,
                'conf_matrix_1_to_0': conf_matrix_1_to_0
            })
            data.update(**self.get_coarse_match_training(conf_matrix_0_to_1, conf_matrix_1_to_0, data))

    @torch.no_grad()
    def get_coarse_match_training(self, conf_matrix_0_to_1, conf_matrix_1_to_0, data):
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        _device = conf_matrix_0_to_1.device

        # Confidence thresholding with FDAFT-optimized strategy
        # Lower threshold for planetary images with weak textures
        adaptive_thr = self.thr * 0.8  # Reduced threshold for FDAFT
        
        mask = torch.logical_or(
            (conf_matrix_0_to_1 > adaptive_thr) * (conf_matrix_0_to_1 == conf_matrix_0_to_1.max(dim=2, keepdim=True)[0]),
            (conf_matrix_1_to_0 > adaptive_thr) * (conf_matrix_1_to_0 == conf_matrix_1_to_0.max(dim=1, keepdim=True)[0]))

        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                         **axes_lengths)
        
        if 'mask0' not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False,
                                     data['mask0'], data['mask1'])
        
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                         **axes_lengths)

        # Find all valid coarse matches
        b_ids, i_ids, j_ids = mask.nonzero(as_tuple=True)
        mconf = torch.maximum(conf_matrix_0_to_1[b_ids, i_ids, j_ids], 
                             conf_matrix_1_to_0[b_ids, i_ids, j_ids])

        # Training sample selection for FDAFT
        if self.training:
            if 'mask0' not in data:
                num_candidates_max = mask.size(0) * max(mask.size(1), mask.size(2))
            else:
                num_candidates_max = compute_max_candidates(data['mask0'], data['mask1'])
            
            # Increase training percentage for FDAFT to handle weak features
            train_coarse_percent = self.config.get('train_coarse_percent', 0.3) * 1.2
            num_matches_train = int(num_candidates_max * train_coarse_percent)
            num_matches_pred = len(b_ids)
            train_pad_num_gt_min = self.config.get('train_pad_num_gt_min', 20)
            
            assert train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

            # Select prediction indices
            if num_matches_pred <= num_matches_train - train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,
                    (num_matches_train - train_pad_num_gt_min,),
                    device=_device)

            # GT padding indices
            gt_pad_indices = torch.randint(
                len(data['spv_b_ids']),
                (max(num_matches_train - num_matches_pred, train_pad_num_gt_min),),
                device=_device)
            
            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]], dim=0),
                *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']],
                     [j_ids, data['spv_j_ids']], [mconf, mconf_gt]))

        # Prepare coarse matches
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # Update with matches in original image resolution
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], torch.div(i_ids, data['hw0_c'][1], rounding_mode='trunc')],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], torch.div(j_ids, data['hw1_c'][1], rounding_mode='trunc')],
            dim=1) * scale1

        coarse_matches.update({
            'gt_mask': mconf == 0,
            'm_bids': b_ids[mconf != 0],
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mkpts0_c_train': mkpts0_c,
            'mkpts1_c_train': mkpts1_c,
            'mconf': mconf[mconf != 0]
        })

        return coarse_matches

    @torch.no_grad()
    def get_coarse_match_inference(self, sim_matrix, data):
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }

        # Adaptive threshold for planetary images
        adaptive_thr = self.thr * 0.7  # Lower threshold for inference
        
        # Softmax for 0 to 1
        conf_matrix_ = F.softmax(sim_matrix, 2) if self.use_sm else sim_matrix

        # Confidence thresholding and nearest neighbor
        mask = (conf_matrix_ > adaptive_thr) * (conf_matrix_ == conf_matrix_.max(dim=2, keepdim=True)[0])

        # Softmax for 1 to 0
        conf_matrix_ = F.softmax(sim_matrix, 1) if self.use_sm else sim_matrix

        # Update mask
        mask = torch.logical_or(mask,
                                (conf_matrix_ > adaptive_thr) * (conf_matrix_ == conf_matrix_.max(dim=1, keepdim=True)[0]))

        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                         **axes_lengths)
        
        if 'mask0' not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False,
                                     data['mask0'], data['mask1'])
        
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                         **axes_lengths)

        # Find all valid coarse matches
        b_ids, i_ids, j_ids = mask.nonzero(as_tuple=True)
        mconf = sim_matrix[b_ids, i_ids, j_ids]

        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # Update with matches in original image resolution
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], torch.div(i_ids, data['hw0_c'][1], rounding_mode='trunc')],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], torch.div(j_ids, data['hw1_c'][1], rounding_mode='trunc')],
            dim=1) * scale1

        coarse_matches.update({
            'mconf': mconf,
            'm_bids': b_ids,
            'mkpts0_c': mkpts0_c,
            'mkpts1_c': mkpts1_c,
        })

        return coarse_matches


class FDAFTFineMatching(nn.Module):
    """
    Fine-level matching module optimized for FDAFT features
    """

    def __init__(self, config, profiler):
        super().__init__()
        self.config = config
        self.profiler = profiler
        
        # Configuration
        self.temperature = config.FINE.DSMAX_TEMPERATURE
        self.W_f = config.FINE_WINDOW_SIZE
        self.inference = config.FINE.INFERENCE
        self.fine_thr = config.FINE.THR * 0.8
        
        dim_f = 64
        
        # FDAFT-specific fine matching components
        self.fdaft_fine_proj = nn.Sequential(
            nn.Linear(dim_f, dim_f),
            nn.LayerNorm(dim_f),
            nn.GELU()
        )
        
        self.fine_proj = nn.Linear(dim_f, dim_f, bias=False)
        
        # Enhanced sub-pixel refinement for planetary images
        self.subpixel_mlp = nn.Sequential(
            nn.Linear(2 * dim_f, 2 * dim_f, bias=False),
            nn.LayerNorm(2 * dim_f),
            nn.ReLU(),
            nn.Linear(2 * dim_f, dim_f, bias=False),
            nn.ReLU(),
            nn.Linear(dim_f, 4, bias=False)
        )
        
        self.fine_spv_max = 500

    def forward(self, feat_f0_unfold, feat_f1_unfold, data):
        M, WW, C = feat_f0_unfold.shape
        W_f = self.W_f

        # Handle no matches case
        if M == 0:
            assert self.training == False, "M is always >0, when training"
            logger.warning('No matches found in coarse-level.')
            self._handle_no_matches(data, feat_f0_unfold.device)
            return

        # FDAFT feature projection
        feat_f0_unfold = self.fdaft_fine_proj(feat_f0_unfold)
        feat_f1_unfold = self.fdaft_fine_proj(feat_f1_unfold)
        
        # Fine projection
        feat_f0 = self.fine_proj(feat_f0_unfold)
        feat_f1 = self.fine_proj(feat_f1_unfold)

        # Normalize
        feat_f0, feat_f1 = map(lambda feat: feat / feat.shape[-1] ** .5,
                               [feat_f0, feat_f1])
        
        sim_matrix = torch.einsum("nlc,nsc->nls", feat_f0,
                                  feat_f1) / self.temperature

        conf_matrix_fine = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        # Get fine and sub-pixel matches
        data.update(**self.get_fine_sub_match(conf_matrix_fine, feat_f0_unfold, feat_f1_unfold, data))

    def _handle_no_matches(self, data, device):
        """Handle case when no coarse matches are found"""
        W_f = self.W_f
        if self.inference:
            data.update({
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
                'mconf_f': torch.zeros(0, device=device),
            })
        else:
            data.update({
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
                'mconf_f': torch.zeros(0, device=device),
                'mkpts0_f_train': data['mkpts0_c_train'],
                'mkpts1_f_train': data['mkpts1_c_train'],
                'conf_matrix_fine': torch.zeros(1, W_f * W_f, W_f * W_f, device=device),
                'b_ids_fine': torch.zeros(0, device=device),
                'i_ids_fine': torch.zeros(0, device=device),
                'j_ids_fine': torch.zeros(0, device=device),
            })

    def get_fine_sub_match(self, conf_matrix_fine, feat_f0_unfold, feat_f1_unfold, data):
        with torch.no_grad():
            W_f = self.W_f

            # FDAFT-optimized confidence thresholding
            mask = conf_matrix_fine > self.fine_thr

            if mask.sum() == 0:
                mask[0, 0, 0] = 1
                conf_matrix_fine[0, 0, 0] = 1

            # Match only the highest confidence
            mask = mask * (conf_matrix_fine == conf_matrix_fine.amax(dim=[1, 2], keepdim=True))

            # Find all valid fine matches
            mask_v, all_j_ids = mask.max(dim=2)
            b_ids, i_ids = torch.where(mask_v)
            j_ids = all_j_ids[b_ids, i_ids]
            mconf = conf_matrix_fine[b_ids, i_ids, j_ids]

            # Update with matches in original image resolution
            b_ids_c, i_ids_c, j_ids_c = data['b_ids'], data['i_ids'], data['j_ids']

            # Scale factors
            scale_f_c = data['hw0_f'][0] // data['hw0_c'][0]

            # Coarse level matches scaled to fine-level
            mkpts0_c_scaled_to_f = torch.stack(
                [i_ids_c % data['hw0_c'][1], torch.div(i_ids_c, data['hw0_c'][1], rounding_mode='trunc')],
                dim=1) * scale_f_c

            mkpts1_c_scaled_to_f = torch.stack(
                [j_ids_c % data['hw1_c'][1], torch.div(j_ids_c, data['hw1_c'][1], rounding_mode='trunc')],
                dim=1) * scale_f_c

            # Updated batch IDs
            updated_b_ids = b_ids_c[b_ids]

            # Scales (image res / fine level)
            scale = data['hw0_i'][0] / data['hw0_f'][0]
            scale0 = scale * data['scale0'][updated_b_ids] if 'scale0' in data else scale
            scale1 = scale * data['scale1'][updated_b_ids] if 'scale1' in data else scale

            # Fine-level discrete matches on window coordinates
            mkpts0_f_window = torch.stack(
                [i_ids % W_f, torch.div(i_ids, W_f, rounding_mode='trunc')],
                dim=1)

            mkpts1_f_window = torch.stack(
                [j_ids % W_f, torch.div(j_ids, W_f, rounding_mode='trunc')],
                dim=1)

        # Enhanced sub-pixel refinement for FDAFT
        combined_features = torch.cat([feat_f0_unfold[b_ids, i_ids], feat_f1_unfold[b_ids, j_ids]], dim=-1)
        sub_ref = self.subpixel_mlp(combined_features)
        sub_ref0, sub_ref1 = torch.chunk(sub_ref, 2, dim=1)
        
        # Apply tanh for bounded refinement
        sub_ref0 = torch.tanh(sub_ref0) * 0.5
        sub_ref1 = torch.tanh(sub_ref1) * 0.5

        pad = 0 if W_f % 2 == 0 else W_f // 2
        
        # Final sub-pixel matches
        mkpts0_f1 = (mkpts0_f_window + mkpts0_c_scaled_to_f[b_ids] - pad) * scale0
        mkpts1_f1 = (mkpts1_f_window + mkpts1_c_scaled_to_f[b_ids] - pad) * scale1
        mkpts0_f_train = mkpts0_f1 + sub_ref0 * scale0
        mkpts1_f_train = mkpts1_f1 + sub_ref1 * scale1
        mkpts0_f = mkpts0_f_train.clone().detach()
        mkpts1_f = mkpts1_f_train.clone().detach()

        # Return matches
        sub_pixel_matches = {
            'm_bids': b_ids_c[b_ids[mconf != 0]],
            'mkpts0_f1': mkpts0_f1[mconf != 0],
            'mkpts1_f1': mkpts1_f1[mconf != 0],
            'mkpts0_f': mkpts0_f[mconf != 0],
            'mkpts1_f': mkpts1_f[mconf != 0],
            'mconf_f': mconf[mconf != 0]
        }

        # Training data
        if not self.inference:
            if self.fine_spv_max is None or self.fine_spv_max > len(data['b_ids']):
                sub_pixel_matches.update({
                    'mkpts0_f_train': mkpts0_f_train,
                    'mkpts1_f_train': mkpts1_f_train,
                    'b_ids_fine': data['b_ids'],
                    'i_ids_fine': data['i_ids'],
                    'j_ids_fine': data['j_ids'],
                    'conf_matrix_fine': conf_matrix_fine
                })
            else:
                train_mask = generate_random_mask(len(data['b_ids']), self.fine_spv_max)
                sub_pixel_matches.update({
                    'mkpts0_f_train': mkpts0_f_train,
                    'mkpts1_f_train': mkpts1_f_train,
                    'b_ids_fine': data['b_ids'][train_mask],
                    'i_ids_fine': data['i_ids'][train_mask],
                    'j_ids_fine': data['j_ids'][train_mask],
                    'conf_matrix_fine': conf_matrix_fine[train_mask]
                })

        return sub_pixel_matches


# Update the import aliases for consistency with existing code
CoarseMatching = FDAFTCoarseMatching
FineSubMatching = FDAFTFineMatching