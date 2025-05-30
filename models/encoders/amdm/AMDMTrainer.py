"""
Training loop and utilities for ArmatureMDM.
"""
import os
import math
import logging
from typing import Dict, Any, Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from AMDM import ArmatureMDM
from sentence_transformers import SentenceTransformer

from diffusion import generate_motion_mdm_style, GaussianDiffusionSamplerUtil
from utils.diffusion_utils import create_motion_animation, T2M_KINEMATIC_CHAIN, plot_detailed_training_history, sanitize_filename


logger = logging.getLogger(__name__)

def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """
    Takes the mean over all non-batch dimensions.
    Equivalent to diffusion.nn.mean_flat from MDM.

    :param torch.Tensor tensor: The input tensor.
    :return: torch.Tensor: The mean over non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps: int, scale_betas: float = 1.0) -> np.ndarray:
    """
    Returns a beta schedule based on the specified name and number of diffusion timesteps.
    This is a simplified version of diffusion.get_named_beta_schedule from MDM.
    :param str schedule_name: The name of the beta schedule (e.g., "linear", "cosine").
    :param int num_diffusion_timesteps: The number of diffusion timesteps.
    :param float scale_betas: A scaling factor for the beta values.
    :return: numpy.ndarray: A 1D numpy array of beta values for the specified schedule.
    """
    if schedule_name == "linear":
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        def betas_for_alpha_bar(num_diffusion_timesteps_inner, alpha_bar_fn, max_beta=0.999): #
            betas_out = []
            for i in range(num_diffusion_timesteps_inner):
                t1 = i / num_diffusion_timesteps_inner
                t2 = (i + 1) / num_diffusion_timesteps_inner
                betas_out.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
            return np.array(betas_out)

        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2, #
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

class UniformSampler:
    """
    A simple schedule sampler that samples timesteps uniformly.
    Similar to diffusion.resample.UniformSampler from MDM.

    :param diffusion_process_obj: An object that has a 'num_timesteps' attribute.
    """
    def __init__(self, diffusion_process_obj: Any):
        self.diffusion = diffusion_process_obj
        self._weights = np.ones([getattr(self.diffusion, 'num_timesteps', 1000)])

    def weights(self) -> np.ndarray:
        """
        Returns the weights for each diffusion timestep. For uniform sampling, these are all ones.
        :return: numpy.ndarray: Array of weights.
        """
        return self._weights

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Importance-samples timesteps for a batch.

        :param int batch_size: The number of timesteps to sample.
        :param torch.device device: The torch device to create tensors on.
        :return: tuple[torch.Tensor, torch.Tensor]:
                 - timesteps: A tensor of timestep indices.
                 - weights: A tensor of weights to scale the resulting losses (inverse probability).
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np]) # For unbiased loss
        weights_pt = torch.from_numpy(weights_np).float().to(device)
        return indices, weights_pt


class GaussianDiffusionTrainerUtil:
    """
    Helper class to store and provide diffusion schedule parameters (betas, alphas, etc.)
    and the q_sample method, needed for the training loop.
    Inspired by diffusion.gaussian_diffusion.GaussianDiffusion from MDM.
    """
    def __init__(self, betas: np.ndarray):
        """
        Initializes the diffusion utility with a precomputed beta schedule.

        :param numpy.ndarray betas: A 1D numpy array of beta values for each diffusion timestep.
        """
        self.betas = betas
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        logger.info(f"GaussianDiffusionTrainerUtil initialized with {self.num_timesteps} timesteps.")

    def _extract_into_tensor(self, arr: np.ndarray, timesteps: torch.Tensor, broadcast_shape: tuple) -> torch.Tensor:
        """
        Extracts values from a 1-D numpy array for a batch of indices and broadcasts to a target shape.
        Helper function from MDM's diffusion.gaussian_diffusion.
        """
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps.long()].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuses the data for a given number of diffusion steps. q(x_t | x_0).
        From MDM's diffusion.gaussian_diffusion.GaussianDiffusion.

        :param torch.Tensor x_start: The initial clean data batch, shape [bs, num_frames, features].
        :param torch.Tensor t: A 1D tensor of timesteps (batch_size,).
        :param Optional[torch.Tensor] noise: Optional noise tensor. If None, generated randomly.
        :return: torch.Tensor: The noised version of x_start at timesteps t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )


class ADMTrainer:
    """
    Trainer class for the ArmatureMDM model, aligned with MDM's training loop concepts.
    Handles the training loop, loss calculations (including armature bone masking),
    model saving, evaluation, and optional sample generation during training.
    """
    def __init__(self,
                 config: Dict[str, Any],
                 model: ArmatureMDM,
                 diffusion_util: GaussianDiffusionTrainerUtil,
                 train_loader: DataLoader,
                 get_bone_mask_fn: Callable[..., torch.Tensor],
                 model_save_dir: Path, # Expects the full Path object now
                 armature_config_data: Optional[Dict[str, Any]] = None,
                 val_loader: Optional[DataLoader] = None,
                 lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
                ):
        """
        Initializes the ArmatureMDMTrainerRevised.

        :param Dict[str, Any] config: The full configuration dictionary.
        :param ArmatureMDM model: The ArmatureMDM model instance to be trained.
        :param GaussianDiffusionTrainerUtil diffusion_util: Utility object for diffusion parameters and q_sample.
        :param DataLoader train_loader: DataLoader for the training data.
        :param Callable get_bone_mask_fn: Function to generate a bone mask based on armature IDs.
        :param Path model_save_dir: The full Path object to the directory where models and logs for this run will be saved.
        :param Optional[Dict[str, Any]] armature_config_data: Configuration data for armatures (e.g., bone definitions).
        :param Optional[DataLoader] val_loader: DataLoader for the validation data.
        :param Optional[torch.optim.lr_scheduler._LRScheduler] lr_scheduler: Learning rate scheduler.
        """
        self.args = config
        self.model = model
        self.diffusion_util = diffusion_util
        self.data_loader = train_loader
        self.val_data_loader = val_loader
        self.get_bone_mask_fn = get_bone_mask_fn
        self.armature_config_data = armature_config_data
        self.lr_scheduler = lr_scheduler

        train_cfg = self.args.get('training_hyperparameters', {})
        paths_cfg = self.args.get('paths', {})
        model_cfg = self.args.get('model_hyperparameters', {})
        diffusion_cfg = self.args.get('diffusion_hyperparameters', {})

        self.training_history: Dict[str, List[float]] = {} 

        self.device = torch.device(train_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)

        self.lr = train_cfg.get('learning_rate', 1e-4)
        self.weight_decay = train_cfg.get('weight_decay', 0.0)
        self.adam_beta1 = train_cfg.get('adam_beta1', 0.9)
        self.adam_beta2 = train_cfg.get('adam_beta2', 0.999)
        self.opt = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
            betas=(self.adam_beta1, self.adam_beta2)
        )
        if self.lr_scheduler is not None: # If scheduler was created with a temp optimizer
            self.lr_scheduler.optimizer = self.opt

        self.num_epochs = train_cfg.get('num_epochs', 300)
        self.cfg_drop_prob_trainer = train_cfg.get('cfg_drop_prob_trainer', 0.15)
        
        self.model_save_dir = model_save_dir # This is now a Path object
        self.model_filename_base = paths_cfg.get('model_filename', 'armature_mdm_checkpoint.pth').replace('.pth', '')
        self.best_model_save_path = self.model_save_dir / f"{self.model_filename_base}_best.pth"


        self.current_epoch = 0
        self.completed_steps = 0

        self.schedule_sampler = UniformSampler(self.diffusion_util)
        self.main_loss_type = train_cfg.get('main_loss_type_trainer', "mse")
        
        self._initialize_loss_weighting()
        self._initialize_aux_loss_params()
        self._initialize_early_stopping()
        self._initialize_sample_generation_params(train_cfg, model_cfg, diffusion_cfg)

        logger.info("ArmatureMDMTrainerRevised initialized.")
        # Log key parameters

    def _initialize_loss_weighting(self):
        """Initializes parameters for timestep loss weighting, using the configured noise schedule."""
        main_x0_loss_cfg = self.args.get('main_x0_loss_config', {})
        weighting_cfg = main_x0_loss_cfg.get('timestep_weighting', {})
        self.loss_weighting_scheme = weighting_cfg.get('scheme', 'none')
        self.min_snr_gamma_value = weighting_cfg.get('min_snr_gamma_value', 5.0)

        if self.loss_weighting_scheme != "none":
            diffusion_hyperparams = self.args.get('diffusion_hyperparameters', {})
            
            noise_schedule_name = diffusion_hyperparams.get('noise_schedule_mdm', 'cosine')
            num_timesteps = diffusion_hyperparams.get('num_diffusion_timesteps', 1000)
            
            betas_np = get_named_beta_schedule( 
                schedule_name=noise_schedule_name,
                num_diffusion_timesteps=num_timesteps,
            )
            betas_torch = torch.from_numpy(betas_np).float().to(self.device)
            
            alphas = 1.0 - betas_torch
            self.alphas_cumprod_for_loss_weighting = torch.cumprod(alphas, axis=0)
            
            if self.loss_weighting_scheme in ["snr_plus_one", "min_snr_gamma"]:
                if hasattr(self, 'alphas_cumprod_for_loss_weighting'): 
                    self.snr_for_loss_weighting = self.alphas_cumprod_for_loss_weighting / \
                                                 (1.0 - self.alphas_cumprod_for_loss_weighting + 1e-8)
                else: 
                    logger.warning("alphas_cumprod_for_loss_weighting not initialized for SNR calc. Disabling weighting.")
                    self.loss_weighting_scheme = "none" 
        logger.info(f"Main loss type (trainer): {self.main_loss_type}, Timestep weighting: {self.loss_weighting_scheme}")


    def _initialize_aux_loss_params(self):
        """Initializes parameters for auxiliary kinematic and geometric losses."""
        kin_cfg = self.args.get('kinematic_losses', {})
        self.use_kinematic_losses = kin_cfg.get('use_kinematic_losses', False)
        self.lambda_kin_vel = kin_cfg.get('velocity_loss_weight', 0.0)
        self.lambda_kin_accel = kin_cfg.get('acceleration_loss_weight', 0.0)
        self.kinematic_loss_type = kin_cfg.get('kinematic_loss_type', 'l1')

        geom_cfg = self.args.get('mdm_geometric_losses', {})
        self.use_mdm_geometric_losses = geom_cfg.get('use_mdm_geometric_losses', False)
        self.lambda_geom_pos = geom_cfg.get('lambda_pos', 0.0)
        self.lambda_geom_vel = geom_cfg.get('lambda_vel', 0.1)
        self.lambda_geom_foot = geom_cfg.get('lambda_foot', 0.0)
        self.foot_joint_indices = geom_cfg.get('foot_joint_indices', [])
        
        model_hyperparams = self.args.get('model_hyperparameters', {})
        self.num_joints_for_geom = model_hyperparams.get('num_joints_for_geom', 22)
        self.features_per_joint_for_geom = model_hyperparams.get('features_per_joint_for_geom', 3)
        logger.info(f"Kinematic losses: Use={self.use_kinematic_losses}, Vel_w={self.lambda_kin_vel}, Accel_w={self.lambda_kin_accel}")
        logger.info(f"Geometric losses: Use={self.use_mdm_geometric_losses}, Pos_w={self.lambda_geom_pos}, Vel_w={self.lambda_geom_vel}, Foot_w={self.lambda_geom_foot}")

    def _initialize_early_stopping(self):
        """Initializes parameters for early stopping."""
        early_stop_cfg = self.args.get('early_stopping', {})
        self.early_stopping_patience = early_stop_cfg.get('early_stopping_patience', 30)
        self.early_stopping_min_delta = early_stop_cfg.get('early_stopping_min_delta', 0.0001)
        self._early_stopping_counter = 0
        self._best_val_loss = float('inf')
        logger.info(f"Early stopping: Patience={self.early_stopping_patience}, Min delta={self.early_stopping_min_delta}")

    def _initialize_sample_generation_params(self, train_cfg, model_cfg, diffusion_cfg):
        """Initializes parameters for sample generation during training."""
        self.generate_sample_every_n_epochs = train_cfg.get('generate_sample_every_n_epochs', 0)

        self.sample_generation_use_best_model = train_cfg.get('sample_generation_use_best_model', False)

        self.sbert_processor_for_sampling = None
        self.diffusion_sampler_for_sampling = None

        if self.generate_sample_every_n_epochs > 0:
            self.sample_generation_prompt = train_cfg.get('sample_generation_prompt', "a person walks")
            self.sample_generation_armature_id = train_cfg.get('sample_generation_armature_id', 1)
            self.sample_generation_num_frames = train_cfg.get('sample_generation_num_frames', 100)
            self.sample_generation_cfg_scale = train_cfg.get('sample_generation_cfg_scale', 2.5)
            self.sample_generation_const_noise = train_cfg.get('sample_generation_const_noise', False)
            self.sample_render_fps = self.args.get('generation_params',{}).get('render_fps', 30)
            
            # Output directory for samples is within the run's model_save_dir
            self.sample_generation_output_dir = self.model_save_dir / "training_samples"
            self.sample_generation_output_dir.mkdir(parents=True, exist_ok=True)

            sbert_model_name_cfg = model_cfg.get('sbert_model_name')
            if sbert_model_name_cfg:
                logger.info(f"Initializing SBERT model ('{sbert_model_name_cfg}') for sample generation...")
                try:
                    self.sbert_processor_for_sampling = SentenceTransformer(sbert_model_name_cfg, device=self.device)
                except Exception as e:
                    logger.error(f"Failed to load SBERT for sampling: {e}. Disabling sample generation.")
                    self.generate_sample_every_n_epochs = 0
            else:
                logger.warning("SBERT model name not in config for sample generation. Disabling it.")
                self.generate_sample_every_n_epochs = 0


            if self.generate_sample_every_n_epochs > 0:
                betas_np_sample_gen = get_named_beta_schedule(
                    schedule_name=diffusion_cfg.get('noise_schedule_mdm', 'linear'),
                    num_diffusion_timesteps=diffusion_cfg.get('num_diffusion_timesteps', 1000)
                )
                self.diffusion_sampler_for_sampling = GaussianDiffusionSamplerUtil(
                    betas=betas_np_sample_gen,
                    model_mean_type=diffusion_cfg.get('model_mean_type_mdm', 'START_X'),
                    model_var_type=diffusion_cfg.get('model_var_type_mdm', 'FIXED_SMALL')
                )
                status_msg = f"ENABLED every {self.generate_sample_every_n_epochs} epochs"

        if self.generate_sample_every_n_epochs > 0:
            status_msg += f" (using {'BEST' if self.sample_generation_use_best_model else 'CURRENT'} model)"
        else:
            status_msg = "DISABLED"
        logger.info(f"Sample generation during training: {status_msg}.")


    def _get_timestep_loss_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """ Returns the loss weights for each timestep based on the configured scheme. """
        if self.loss_weighting_scheme == "none" or not hasattr(self, 'alphas_cumprod_for_loss_weighting'):
            return torch.ones_like(timesteps, dtype=torch.float32, device=self.device)
        long_timesteps = timesteps.long()
        if self.loss_weighting_scheme == "snr_plus_one":
            selected_snr = self.snr_for_loss_weighting[long_timesteps]
            weights = selected_snr + 1.0
        elif self.loss_weighting_scheme == "min_snr_gamma":
            selected_snr = self.snr_for_loss_weighting[long_timesteps]
            weights = torch.minimum(selected_snr, torch.tensor(self.min_snr_gamma_value, device=self.device))
        else:
            logger.warning(f"Unknown weighting scheme '{self.loss_weighting_scheme}'. Defaulting to ones.")
            return torch.ones_like(timesteps, dtype=torch.float32, device=self.device)
        normalized_weights = weights / (weights.mean() + 1e-8)
        return normalized_weights


    def _get_combined_mask(self, temporal_mask: Optional[torch.Tensor], bone_mask: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """ Combines temporal and bone masks to create a final mask for loss calculations. """
        bs, nframes_current, nfeatures = target_shape[0], target_shape[1], target_shape[2]

        if temporal_mask is not None:
            # Ensure temporal_mask is boolean and True means valid
            processed_temporal_mask = temporal_mask.squeeze(1).squeeze(1) # -> [bs, nframes_orig_tempo]
            # Slice or pad temporal_mask to match nframes_current
            if processed_temporal_mask.shape[1] > nframes_current:
                processed_temporal_mask = processed_temporal_mask[:, :nframes_current]
            elif processed_temporal_mask.shape[1] < nframes_current:
                padding = torch.zeros(bs, nframes_current - processed_temporal_mask.shape[1],
                                      dtype=torch.bool, device=processed_temporal_mask.device)
                processed_temporal_mask = torch.cat((processed_temporal_mask, padding), dim=1)
            processed_temporal_mask = processed_temporal_mask.unsqueeze(-1) # -> [bs, nframes_current, 1]
        else:
            processed_temporal_mask = torch.ones(bs, nframes_current, 1, device=bone_mask.device, dtype=torch.bool)
        
        # Ensure bone_mask is boolean and True means active, and slice/pad frames
        processed_bone_mask = bone_mask.bool() # Assuming bone_mask comes as float 0/1
        if processed_bone_mask.shape[1] > nframes_current:
            processed_bone_mask = processed_bone_mask[:, :nframes_current, :]
        elif processed_bone_mask.shape[1] < nframes_current:
            padding = torch.zeros(bs, nframes_current - processed_bone_mask.shape[1], nfeatures,
                                  dtype=torch.bool, device=bone_mask.device)
            processed_bone_mask = torch.cat((processed_bone_mask, padding), dim=1)
        
        # Combine: True if both temporal is valid AND bone is active
        combined_mask = processed_temporal_mask & processed_bone_mask # Element-wise AND
        return combined_mask.float() # Return as float (0.0 or 1.0) for multiplication

    def _calculate_masked_loss(self, prediction: torch.Tensor, target: torch.Tensor,
                               combined_mask: torch.Tensor, loss_type: str,
                               sample_timestep_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Calculates the masked loss between prediction and target tensors. """
        if loss_type.lower() == "mse":
            element_wise_loss = F.mse_loss(prediction, target, reduction='none')
        elif loss_type.lower() == "l1":
            element_wise_loss = F.l1_loss(prediction, target, reduction='none')
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'mse' or 'l1'.")
        
        masked_element_wise_loss = element_wise_loss * combined_mask
        sum_loss_per_sample = masked_element_wise_loss.sum(dim=list(range(1, masked_element_wise_loss.ndim)))
        num_active_elements_per_sample = combined_mask.sum(dim=list(range(1, combined_mask.ndim)))
        num_active_safe = torch.clamp(num_active_elements_per_sample, min=1e-8)
        mean_loss_per_sample = sum_loss_per_sample / num_active_safe
        mean_loss_per_sample[num_active_elements_per_sample == 0] = 0.0

        if sample_timestep_weights is not None:
            weighted_loss_per_sample = mean_loss_per_sample * sample_timestep_weights
        else:
            weighted_loss_per_sample = mean_loss_per_sample
        return weighted_loss_per_sample.mean()

    def _get_derivatives(self, motion_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Computes the velocity and acceleration of a motion sequence. """
        vel = torch.empty(motion_sequence.shape[0], 0, motion_sequence.shape[2], device=motion_sequence.device, dtype=motion_sequence.dtype)
        accel = torch.empty(motion_sequence.shape[0], 0, motion_sequence.shape[2], device=motion_sequence.device, dtype=motion_sequence.dtype)
        if motion_sequence.shape[1] >= 2:
            vel = motion_sequence[:, 1:] - motion_sequence[:, :-1]
            if vel.shape[1] >= 2:
                accel = vel[:, 1:] - vel[:, :-1]
        return vel, accel

    def _compute_losses_dict(self, x_start_gt: torch.Tensor, predicted_x0: torch.Tensor,
                                t: torch.Tensor, y_cond: Dict[str, Any], bone_mask: torch.Tensor
                                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """ 
        Computes the losses for the given ground truth and predicted tensors. 
        Returns a tuple of total loss tensor and a dictionary of individual loss terms.
        :param x_start_gt: Ground truth motion tensor, shape [bs, nframes, nfeatures].
        :param predicted_x0: Predicted motion tensor from the model, shape [bs, nframes, nfeatures].
        :param t: Tensor of timesteps, shape [bs].
        :param y_cond: Dictionary containing conditioning information (e.g., text embeddings, armature IDs).
        :param bone_mask: Bone mask tensor, shape [bs, nframes, nfeatures].
        :return: Tuple[torch.Tensor, Dict[str, float]]: Total loss tensor and a dictionary of loss terms.
        """
        terms = {}
        current_total_loss = torch.tensor(0.0, device=self.device)
        temporal_mask = y_cond.get('mask') # Expected: [bs, 1, 1, nframes_gt], True for VALID

        combined_mask_x0 = self._get_combined_mask(temporal_mask, bone_mask, x_start_gt.shape)
        timestep_weights = self._get_timestep_loss_weights(t)

        main_loss_val = self._calculate_masked_loss(
            predicted_x0, x_start_gt, combined_mask_x0, self.main_loss_type,
            sample_timestep_weights=timestep_weights )
        terms[f"main_loss/{self.main_loss_type}"] = main_loss_val.item()
        current_total_loss += main_loss_val
        if timestep_weights is not None: # Assuming timestep_weights is always a tensor if not None
            terms["debug/avg_timestep_weight"] = timestep_weights.mean().item()

        pred_vel, pred_accel = self._get_derivatives(predicted_x0)
        target_vel, target_accel = self._get_derivatives(x_start_gt)

        if self.use_kinematic_losses:
            if self.lambda_kin_vel > 0 and pred_vel.shape[1] > 0: # Check if velocity frames exist
                mask_vel_temp = temporal_mask[..., :pred_vel.shape[1]] if temporal_mask is not None else None
                bone_mask_vel = bone_mask[:, :pred_vel.shape[1], :] if bone_mask.shape[1] > pred_vel.shape[1] else bone_mask[:, :-1, :]
                combined_mask_kin_vel = self._get_combined_mask(mask_vel_temp, bone_mask_vel, target_vel.shape)
                loss_val = self._calculate_masked_loss(pred_vel, target_vel, combined_mask_kin_vel, self.kinematic_loss_type)
                terms["kinematic/velocity_loss"] = loss_val.item()
                current_total_loss += self.lambda_kin_vel * loss_val

            if self.lambda_kin_accel > 0 and pred_accel.shape[1] > 0: # Check if acceleration frames exist
                mask_accel_temp = temporal_mask[..., :pred_accel.shape[1]] if temporal_mask is not None else None
                bone_mask_accel = bone_mask[:, :pred_accel.shape[1], :] if bone_mask.shape[1] > pred_accel.shape[1] else bone_mask[:, :-2, :]
                combined_mask_kin_accel = self._get_combined_mask(mask_accel_temp, bone_mask_accel, target_accel.shape)
                loss_val = self._calculate_masked_loss(pred_accel, target_accel, combined_mask_kin_accel, self.kinematic_loss_type)
                terms["kinematic/acceleration_loss"] = loss_val.item()
                current_total_loss += self.lambda_kin_accel * loss_val

        if self.use_mdm_geometric_losses:
            bs_geom, nframes_geom, _ = predicted_x0.shape # Or x_start_gt.shape

            if self.lambda_geom_pos > 0:
                loss_val = self._calculate_masked_loss(predicted_x0, x_start_gt, combined_mask_x0, "mse") # MDM uses MSE for geometric pos
                terms["geometric/position_loss"] = loss_val.item()
                current_total_loss += self.lambda_geom_pos * loss_val
            
            if self.lambda_geom_vel > 0 and pred_vel.shape[1] > 0:
                mask_vel_temp_geom = temporal_mask[..., :pred_vel.shape[1]] if temporal_mask is not None else None
                bone_mask_vel_geom = bone_mask[:, :pred_vel.shape[1], :]
                combined_mask_geom_vel = self._get_combined_mask(mask_vel_temp_geom, bone_mask_vel_geom, target_vel.shape)
                loss_val = self._calculate_masked_loss(pred_vel, target_vel, combined_mask_geom_vel, "mse") # MDM uses MSE for geometric vel
                terms["geometric/velocity_loss"] = loss_val.item()
                current_total_loss += self.lambda_geom_vel * loss_val

            if self.lambda_geom_foot > 0 and pred_vel.shape[1] > 0:
                pred_x0_reshaped_geom = predicted_x0.reshape(bs_geom, nframes_geom, self.num_joints_for_geom, self.features_per_joint_for_geom)
                
                left_foot_idx = self.foot_joint_indices[0]
                right_foot_idx = self.foot_joint_indices[1]
                
                # Velocities of predicted foot joints
                pred_vel_reshaped_geom = pred_vel.reshape(bs_geom, nframes_geom - 1, self.num_joints_for_geom, self.features_per_joint_for_geom)
                pred_left_foot_vel = pred_vel_reshaped_geom[:, :, left_foot_idx, :]  # [bs, nframes-1, 3 (xyz)]
                pred_right_foot_vel = pred_vel_reshaped_geom[:, :, right_foot_idx, :] # [bs, nframes-1, 3 (xyz)]

                foot_contact_gt = y_cond.get("foot_contact_ground_truth") # Expected: [bs, nframes, 2] (L, R) or similar

                if foot_contact_gt is not None:
                    # Ensure foot_contact_gt is [bs, nframes_vel, 2] and then unsqueeze
                    contact_mask_left = foot_contact_gt[:, :nframes_geom-1, 0].unsqueeze(-1).float()  # [bs, nframes-1, 1]
                    contact_mask_right = foot_contact_gt[:, :nframes_geom-1, 1].unsqueeze(-1).float() # [bs, nframes-1, 1]
                else:
                    # Heuristic: foot is in contact if its Y position in ground truth is low
                    gt_x0_reshaped_geom = x_start_gt.reshape(bs_geom, nframes_geom, self.num_joints_for_geom, self.features_per_joint_for_geom)
                    gt_left_foot_y_t = gt_x0_reshaped_geom[:, :nframes_geom-1, left_foot_idx, 1] # Y-coord at frame t for vel between t, t+1
                    gt_right_foot_y_t = gt_x0_reshaped_geom[:, :nframes_geom-1, right_foot_idx, 1]

                    height_threshold = 0.05 # Example: 5cm if data is in meters; tune this
                    contact_mask_left = (torch.abs(gt_left_foot_y_t) < height_threshold).float().unsqueeze(-1)
                    contact_mask_right = (torch.abs(gt_right_foot_y_t) < height_threshold).float().unsqueeze(-1)
                    # logger.debug("Foot contact GT not provided, using Y-height heuristic for L_foot loss.")

                target_foot_vel_when_contact = torch.zeros_like(pred_left_foot_vel, device=self.device)

                loss_foot_left = F.mse_loss(pred_left_foot_vel * contact_mask_left,
                                            target_foot_vel_when_contact * contact_mask_left,
                                            reduction='none')
                loss_foot_right = F.mse_loss(pred_right_foot_vel * contact_mask_right,
                                                target_foot_vel_when_contact * contact_mask_right,
                                                reduction='none')
                
                # Apply temporal mask if it exists
                mask_vel_temp_geom_foot = None
                if temporal_mask is not None:
                    mask_vel_temp_geom_foot = temporal_mask[..., :pred_vel.shape[1]] # Up to nframes-1
                    valid_temporal_mask_for_vel = mask_vel_temp_geom_foot.squeeze(1).squeeze(1).unsqueeze(-1).float() # -> [bs, nframes-1, 1]
                    loss_foot_left = loss_foot_left * valid_temporal_mask_for_vel
                    loss_foot_right = loss_foot_right * valid_temporal_mask_for_vel
                        
                    # Denominator for normalization: count elements where contact AND temporal mask are active
                    num_active_left = (contact_mask_left * valid_temporal_mask_for_vel).sum() + 1e-8
                    num_active_right = (contact_mask_right * valid_temporal_mask_for_vel).sum() + 1e-8
                else:
                    num_active_left = contact_mask_left.sum() + 1e-8
                    num_active_right = contact_mask_right.sum() + 1e-8

                # Sum losses and normalize by active elements
                loss_foot_val = (loss_foot_left.sum() / num_active_left + loss_foot_right.sum() / num_active_right) / 2.0
                
                if not torch.isnan(loss_foot_val) and not torch.isinf(loss_foot_val):
                    terms["geometric/foot_contact_loss"] = loss_foot_val.item()
                    current_total_loss += self.lambda_geom_foot * loss_foot_val
                else:
                    terms["geometric/foot_contact_loss"] = 0.0 # Or some other indicator of failure for this batch item
                    logger.warning(f"NaN or Inf in foot_contact_loss, skipping for this batch. L:{num_active_left.item()}, R:{num_active_right.item()}")


        terms["total_loss"] = current_total_loss.item()
        return current_total_loss, terms


    def run_step(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Performs a single training step (forward pass, loss calculation, backward pass, optimizer step).
        Corresponds to MDM's TrainLoop.run_step.

        :param Dict[str, Any] batch_data: The batch data from the DataLoader.
        :return: Dict[str, float]: A dictionary of loss components for logging.
        """
        self.model.train()
        self.opt.zero_grad()

        x_start_gt = batch_data["target_x0"].to(self.device)
        
        y_cond = {
            'text_embeddings_batch': batch_data["text_embeddings"].to(self.device),
            'armature_class_ids': batch_data["armature_class_ids"].to(self.device),
            # Prepare motion_padding_mask for y_cond['mask']
            # Expected: [bs, 1, 1, nframes], True for VALID
            'mask': batch_data.get('motion_padding_mask_for_loss').to(self.device) if batch_data.get('motion_padding_mask_for_loss') is not None else None,
            'foot_contact_ground_truth': batch_data.get("foot_contact_ground_truth").to(self.device) if batch_data.get("foot_contact_ground_truth") is not None else None,
            'uncond': False, 'uncond_text': False, 'uncond_armature': False # Defaults for CFG
        }
        
        # Apply CFG dropout based on training settings
        if self.cfg_drop_prob_trainer > 0: # Check if CFG is active for training
            if torch.rand(1).item() < self.cfg_drop_prob_trainer:
                y_cond['uncond'] = True # Global uncond flag for _mask_cond in model
                y_cond['uncond_text'] = True
                y_cond['uncond_armature'] = True

        t, _ = self.schedule_sampler.sample(x_start_gt.shape[0], self.device)
        noise = torch.randn_like(x_start_gt)
        x_t = self.diffusion_util.q_sample(x_start_gt, t, noise=noise)

        predicted_x0 = self.model(x_t, t, y_cond)

        bs, nframes, nfeatures = x_start_gt.shape
        bone_mask = self.get_bone_mask_fn(
            y_cond['armature_class_ids'], nfeatures, nframes, str(self.device),
            armature_config_data=self.armature_config_data
        )
        
        total_loss_tensor, losses_dict_float = self._compute_losses_dict(
            x_start_gt, predicted_x0, t, y_cond, bone_mask
        )
        
        total_loss_tensor.backward()
        self.opt.step()
        
        return losses_dict_float


    def run_loop(self):
        """
        Main training loop that iterates over epochs and batches.
        Corresponds to MDM's TrainLoop.run_loop.
        """
        logger.info(f"Starting training for {self.num_epochs} epochs on device {self.device}.")
        training_history = {'train_loss_total': [], 'val_loss_total': []} # Store total loss per epoch

        for epoch_idx in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch_idx
            logger.info(f"--- Epoch {self.current_epoch + 1}/{self.num_epochs} ---")
            
            self.model.train() # Ensure model is in training mode for this epoch
            epoch_loss_components_sum: Dict[str, float] = {}
            num_batches_in_epoch = len(self.data_loader)

            if num_batches_in_epoch == 0:
                logger.warning(f"Epoch {self.current_epoch + 1}: Training DataLoader is empty. Skipping epoch.")
                continue

            for batch_idx, batch_data in enumerate(tqdm(self.data_loader, desc=f"Training Epoch {self.current_epoch + 1}")):
                if batch_data is None:
                    logger.warning(f"Epoch {self.current_epoch + 1}, Batch {batch_idx}: Skipping None batch.")
                    continue
                
                losses_batch = self.run_step(batch_data)
                self.completed_steps += 1

                for key, val in losses_batch.items():
                    epoch_loss_components_sum[key] = epoch_loss_components_sum.get(key, 0.0) + val
            
            # Log average losses for the epoch
            if num_batches_in_epoch > 0:
                avg_epoch_losses = {k: v / num_batches_in_epoch for k, v in epoch_loss_components_sum.items()}
                log_train_summary = f"Epoch {self.current_epoch + 1} Training Summary: " + \
                                    ", ".join([f"Avg {k}: {v:.4f}" for k,v in avg_epoch_losses.items()])
                logger.info(log_train_summary)
                training_history['train_loss_total'].append(avg_epoch_losses.get("total_loss", float('nan')))
                
                # Store each loss component in training history
                for loss_name, avg_value in avg_epoch_losses.items():
                    history_key_train = f'train_{sanitize_filename(loss_name)}'
                    if history_key_train not in self.training_history:
                        self.training_history[history_key_train] = []
                    self.training_history[history_key_train].append(avg_value)


            # Validation phase
            avg_val_total_loss = float('inf')
            if self.val_data_loader is not None:
                avg_val_total_loss, avg_val_loss_components_epoch = self.evaluate_epoch() # evaluate_epoch will log its details
                training_history['val_loss_total'].append(avg_val_total_loss)

                # Store validation losses in training history
                if avg_val_loss_components_epoch: 
                    for loss_name, avg_value in avg_val_loss_components_epoch.items():
                        history_key_val = f'val_{sanitize_filename(loss_name)}'
                        if history_key_val not in self.training_history:
                            self.training_history[history_key_val] = []
                        self.training_history[history_key_val].append(avg_value)

                # Reduce LR if using ReduceLROnPlateau scheduler
                if self.lr_scheduler and isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(avg_val_total_loss)
                
                if avg_val_total_loss < self._best_val_loss - self.early_stopping_min_delta:
                    self._best_val_loss = avg_val_total_loss
                    self._early_stopping_counter = 0
                    logger.info(f"New best validation loss: {self._best_val_loss:.4f}. Saving model...")
                    self.save_checkpoint(suffix="best")
                else:
                    self._early_stopping_counter += 1
                    logger.info(f"Validation loss did not improve significantly. Early stopping counter: {self._early_stopping_counter}/{self.early_stopping_patience}")
                    if self._early_stopping_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping triggered at epoch {self.current_epoch + 1}.")
                        break # Exit training loop
            
            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and self.lr_scheduler is not None:
                self.lr_scheduler.step() # For schedulers like StepLR

            # Periodic checkpoint saving (based on epochs)
            save_interval = self.args.get('training_hyperparameters',{}).get('save_interval_epochs', 10)
            if (self.current_epoch + 1) % save_interval == 0 or (self.current_epoch + 1) == self.num_epochs:
                self.save_checkpoint(suffix=f"epoch_{self.current_epoch+1}")
            
            # Optional: Generate and save a sample animation
            if self.generate_sample_every_n_epochs > 0 and \
               (self.current_epoch + 1) % self.generate_sample_every_n_epochs == 0:
                self._run_sample_generation(epoch_num=self.current_epoch + 1)

            # Final save plot of training history
            plot_detailed_training_history(
                self.training_history,
                self.model_save_dir,
                self.args.get('run_name', 'run'),
                self.current_epoch + 1
            )

        self.save_checkpoint(suffix="final")
        logger.info(f"Training finished after {self.current_epoch + 1} epochs / {self.completed_steps} steps.")
        return training_history


    def evaluate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Evaluates the model for one epoch on the validation set.

        :return: Tuple[float, Dict[str, float]]: Average total validation loss and dictionary of averaged component losses.
        """
        self.model.eval()
        total_val_loss_sum = 0.0
        val_loss_components_sum: Dict[str, float] = {}
        num_val_batches = len(self.val_data_loader)

        if num_val_batches == 0:
            logger.warning("Validation DataLoader is empty. Skipping evaluation.")
            return float('inf'), {}

        with torch.no_grad():
            for batch_data in tqdm(self.val_data_loader, desc=f"Validating Epoch {self.current_epoch + 1}"):
                if batch_data is None: 
                    logger.warning(f"Epoch {self.current_epoch + 1} Validation: Skipping None batch.")
                    num_val_batches = max(1, num_val_batches -1) # Adjust count if a batch is skipped
                    continue

                x_start_gt = batch_data["target_x0"].to(self.device)
                y_cond = {
                    'text_embeddings_batch': batch_data["text_embeddings"].to(self.device),
                    'armature_class_ids': batch_data["armature_class_ids"].to(self.device),
                    'mask': batch_data.get('motion_padding_mask_for_loss').to(self.device) if batch_data.get('motion_padding_mask_for_loss') is not None else None,
                    'foot_contact_ground_truth': batch_data.get("foot_contact_ground_truth").to(self.device) if batch_data.get("foot_contact_ground_truth") is not None else None,
                    'uncond': False, 'uncond_text': False, 'uncond_armature': False
                }
                t, _ = self.schedule_sampler.sample(x_start_gt.shape[0], self.device)
                
                predicted_x0 = self.model(self.diffusion_util.q_sample(x_start_gt, t, noise=torch.randn_like(x_start_gt)), t, y_cond)

                bs, nframes, nfeatures = x_start_gt.shape
                bone_mask = self.get_bone_mask_fn(
                    y_cond['armature_class_ids'], nfeatures, nframes, str(self.device),
                    armature_config_data=self.armature_config_data )
                
                _, losses_dict_val_batch = self._compute_losses_dict(
                    x_start_gt, predicted_x0, t, y_cond, bone_mask )

                total_val_loss_sum += losses_dict_val_batch.get("total_loss", 0.0)
                for key, val in losses_dict_val_batch.items():
                    val_loss_components_sum[key] = val_loss_components_sum.get(key, 0.0) + val
        
        avg_val_total_loss = total_val_loss_sum / num_val_batches if num_val_batches > 0 else float('inf')
        avg_val_loss_components = {k: v / num_val_batches if num_val_batches > 0 else 0.0 for k, v in val_loss_components_sum.items()}
        
        log_val_summary = f"Epoch {self.current_epoch + 1} Validation Summary: " + \
                          ", ".join([f"Avg {k}: {v:.4f}" for k,v in avg_val_loss_components.items()])
        logger.info(log_val_summary)
        
        return avg_val_total_loss, avg_val_loss_components


    def save_checkpoint(self, suffix: str):
        """
        Saves a model checkpoint.

        :param str suffix: Suffix for the checkpoint filename (e.g., "best", "final", "epoch_100").
        """
        filename = f"{self.model_filename_base}_{suffix}.pth"
        save_path = self.model_save_dir / filename # model_save_dir is now Path
        
        logger.info(f"Saving model checkpoint to {save_path} (Epoch: {self.current_epoch+1}, Total Steps: {self.completed_steps})")

        model_state_dict_to_save = self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict()

        state_to_save = {
            'epoch': self.current_epoch,
            'completed_steps': self.completed_steps,
            'model_state_dict': model_state_dict_to_save,
            'optimizer_state_dict': self.opt.state_dict(),
            'config': self.args,
            '_best_val_loss': self._best_val_loss,
            '_early_stopping_counter': self._early_stopping_counter
        }
        if self.lr_scheduler is not None:
            state_to_save['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        try:
            torch.save(state_to_save, str(save_path)) # torch.save needs string path
            logger.info(f"Checkpoint saved successfully: {save_path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint {save_path}: {e}")

    @torch.no_grad()
    def _run_sample_generation(self, epoch_num: int):
        """Generates and saves a sample animation during training."""
        if not (self.generate_sample_every_n_epochs > 0 and \
                self.sbert_processor_for_sampling is not None and \
                self.diffusion_sampler_for_sampling is not None):
            if self.generate_sample_every_n_epochs > 0:
                logger.warning("Sample generation skipped due to missing SBERT processor or diffusion sampler for sampling.")
            return

        logger.info(f"Generating inspection sample for epoch {epoch_num} for text '{self.sample_generation_prompt}' "
                    f"and armature ID {self.sample_generation_armature_id} "
                    f"(using {'BEST' if self.sample_generation_use_best_model else 'CURRENT'} model)...")
                    
        model_to_use_for_sampling: ArmatureMDM
        checkpoint_to_load_path: Optional[Path] = None

        if self.sample_generation_use_best_model and self.best_model_save_path.exists():
            checkpoint_to_load_path = self.best_model_save_path
            logger.info(f"Loading BEST model state from: {checkpoint_to_load_path}")
        else:
            if self.sample_generation_use_best_model and not self.best_model_save_path.exists():
                logger.warning(f"BEST model for sampling requested, but {self.best_model_save_path} not found. "
                               "Falling back to CURRENT model state.")
            # Use the current model state for sampling
            model_to_use_for_sampling = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model


        if checkpoint_to_load_path:
            try:
                # Load the checkpoint with weights_only=True to avoid loading optimizer state
                checkpoint = torch.load(str(checkpoint_to_load_path), map_location=self.device, weights_only=True)
                
                config_from_ckpt = checkpoint.get('config', self.args)
                model_cfg_sample = config_from_ckpt.get('model_hyperparameters', {})
                
                if "flat" in model_cfg_sample.get('data_rep', ""):
                    sample_num_motion_features = model_cfg_sample.get('num_motion_features_actual')
                else: 
                    sample_num_motion_features = model_cfg_sample.get('njoints') * model_cfg_sample.get('nfeats_per_joint')

                # Initialize the model with the loaded configuration
                model_to_use_for_sampling = ArmatureMDM(
                    data_rep=model_cfg_sample.get('data_rep'),
                    njoints=model_cfg_sample.get('njoints'),
                    nfeats_per_joint=model_cfg_sample.get('nfeats_per_joint'),
                    num_motion_features=sample_num_motion_features,
                    latent_dim=model_cfg_sample.get('latent_dim'),
                    ff_size=model_cfg_sample.get('ff_size'),
                    num_layers=model_cfg_sample.get('num_layers'),
                    num_heads=model_cfg_sample.get('num_heads'),
                    dropout=model_cfg_sample.get('dropout'),
                    activation=model_cfg_sample.get('activation', 'gelu'),
                    arch=model_cfg_sample.get('arch', 'trans_enc'),
                    conditioning_integration_mode=model_cfg_sample.get('conditioning_integration_mode', "mlp"),
                    conditioning_transformer_config=model_cfg_sample.get('conditioning_transformer_config', None),
                    armature_integration_policy=model_cfg_sample.get('armature_integration_policy', "add_refined"),
                    sbert_embedding_dim=model_cfg_sample.get('sbert_embedding_dim'),
                    max_armature_classes=model_cfg_sample.get('max_armature_classes'),
                    armature_embedding_dim=model_cfg_sample.get('armature_embedding_dim'),
                    armature_mlp_hidden_dims=model_cfg_sample.get('armature_mlp_hidden_dims'),
                    max_seq_len_pos_enc=model_cfg_sample.get('max_seq_len_pos_enc'),
                    text_cond_mask_prob=model_cfg_sample.get('text_cond_mask_prob', 0.1),
                    armature_cond_mask_prob=model_cfg_sample.get('armature_cond_mask_prob', 0.1),
                    batch_first_transformer=model_cfg_sample.get('batch_first_transformer', False),
                ).to(self.device)

                model_to_use_for_sampling.load_state_dict(checkpoint['model_state_dict'])

                logger.info(f"Successfully loaded BEST model state for sample generation.")
            except Exception as e:
                logger.error(f"Failed to load BEST model state from {checkpoint_to_load_path}: {e}. "
                               "Falling back to CURRENT model state for sample generation.")
                model_to_use_for_sampling = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

        current_model_training_state = model_to_use_for_sampling.training
        model_to_use_for_sampling.eval()

        try:
            text_emb = self.sbert_processor_for_sampling.encode(
                self.sample_generation_prompt, convert_to_tensor=True).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"Error encoding sample text with SBERT: {e}. Skipping sample generation this epoch.")
            model_to_use_for_sampling.train(current_model_training_state)
            return

        y_conditions_sample = {
            'text_embeddings_batch': text_emb,
            'armature_class_ids': torch.tensor([self.sample_generation_armature_id], dtype=torch.long).to(self.device),
            'mask': None, 
            'cfg_scale': self.sample_generation_cfg_scale,
            'uncond': False, 'uncond_text': False, 'uncond_armature': False
        }

        try:
            generated_motion_tensor = generate_motion_mdm_style(
                armature_mdm_model=model_to_use_for_sampling,
                diffusion_sampler_util=self.diffusion_sampler_for_sampling,
                y_conditions=y_conditions_sample,
                num_frames=self.sample_generation_num_frames,
                device=str(self.device),
                clip_denoised=True,
                progress=False,
                const_noise=self.sample_generation_const_noise
            )
        except Exception as e:
            logger.error(f"Error during generate_motion_mdm_style for sample: {e}", exc_info=True)
            model_to_use_for_sampling.train(current_model_training_state)
            return

        if generated_motion_tensor is not None:
            motion_np = generated_motion_tensor.squeeze(0).cpu().numpy()
            
            if hasattr(self.data_loader.dataset, 'dataset_mean') and hasattr(self.data_loader.dataset, 'dataset_std'):
                mean_for_denorm = self.data_loader.dataset.dataset_mean
                std_for_denorm = np.where(self.data_loader.dataset.dataset_std == 0, 1e-8, self.data_loader.dataset.dataset_std)
                motion_np_denormalized = motion_np * std_for_denorm + mean_for_denorm
            else:
                logger.warning("Mean/std for de-normalization not found in dataset for sample. Animating normalized data.")
                motion_np_denormalized = motion_np # Use raw motion if no mean/std available

            # Check if model has attributes for joints/features
            num_j_viz = getattr(model_to_use_for_sampling, 'njoints', self.num_joints_for_geom)
            feat_p_j_viz = getattr(model_to_use_for_sampling, 'nfeats', self.features_per_joint_for_geom)

            sanitized_prompt_fn = sanitize_filename(self.sample_generation_prompt, max_len=40)
            model_type_suffix = "_best" if self.sample_generation_use_best_model and checkpoint_to_load_path else "_curr"
            animation_filename = f"epoch{epoch_num}_arm{self.sample_generation_armature_id}{model_type_suffix}_{sanitized_prompt_fn}.gif"
            animation_output_path = self.sample_generation_output_dir / animation_filename
            
            # Check if motion_np_denormalized has the expected shape
            expected_flat_features = num_j_viz * feat_p_j_viz
            if motion_np_denormalized.shape[0] == self.sample_generation_num_frames and \
               motion_np_denormalized.shape[1] == expected_flat_features:
                motion_reshaped = motion_np_denormalized.reshape(self.sample_generation_num_frames, num_j_viz, feat_p_j_viz)
                try:
                    create_motion_animation(
                        motion_data_frames=motion_reshaped,
                        kinematic_chain=T2M_KINEMATIC_CHAIN,
                        output_filename=str(animation_output_path),
                        fps=self.sample_render_fps
                    )
                    logger.info(f"Saved sample animation to {animation_output_path}")
                except Exception as e:
                    logger.error(f"Failed to create sample animation for epoch {epoch_num}: {e}", exc_info=True)
            else:
                 logger.warning(f"Sample shape mismatch for animation: motion_np_denormalized shape {motion_np_denormalized.shape}, "
                                f"expected frames {self.sample_generation_num_frames}, "
                                f"expected flat features {expected_flat_features} (for {num_j_viz}j x {feat_p_j_viz}f). Skipping animation.")

        model_to_use_for_sampling.train(current_model_training_state) # Restore original training state
        if model_to_use_for_sampling is not self.model and hasattr(self.model, 'train'): # Check if the trained model has a train method
             self.model.train(current_model_training_state if current_model_training_state is not None else True)

        logger.info(f"Inspection sample generation attempt complete for epoch {epoch_num}.")