"""
Handles the diffusion sampling process for generating motion.
Includes utilities for managing diffusion schedule parameters during sampling
and the main generation loop that performs denoising.
"""
import logging
from typing import Any, Dict, Optional, Callable, Tuple

import numpy as np
import torch
from tqdm import tqdm
from AMDM import ArmatureMDM
from torch import nn

logger = logging.getLogger(__name__)


class GaussianDiffusionSamplerUtil:
    """
    Utility class to manage diffusion parameters and perform steps of the
    reverse diffusion process (sampling), inspired by MDM's GaussianDiffusion.
    This class assumes the model predicts x_0 (model_mean_type = "START_X").
    """
    def __init__(self,
                 betas: np.ndarray,
                 model_mean_type: str,
                 model_var_type: str):
        """
        Initializes the GaussianDiffusionSamplerUtil.

        :param numpy.ndarray betas: A 1D numpy array of beta values for each diffusion timestep.
        :param str model_mean_type: Specifies what the model predicts (e.g., "START_X", "EPSILON").
                                    This implementation primarily supports "START_X".
        :param str model_var_type: Specifies the type of variance used during sampling
                                   (e.g., "FIXED_SMALL", "FIXED_LARGE").
        """
        self.betas = betas.astype(np.float64)
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # For q_posterior_mean_variance (coefficients for q(x_{t-1} | x_t, x_0))
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod + 1e-12)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod + 1e-12)
        )

        # For p_mean_variance (variance of p(x_{t-1} | x_t))
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod + 1e-12)
        )
        # Clipped log variance to prevent log(0)
        self.posterior_log_variance_clipped = np.log(
            np.maximum(self.posterior_variance, 1e-20) # Clip variance at 1e-20 before log
        )
        
        # For _predict_xstart_from_eps (if model predicted epsilon)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)


        self.model_mean_type = self._parse_model_mean_type(model_mean_type)
        self.model_var_type = self._parse_model_var_type(model_var_type)
        
        logger.info(f"GaussianDiffusionSamplerUtil initialized for {self.num_timesteps} timesteps. "
                    f"ModelMeanType: {self.model_mean_type}, ModelVarType: {self.model_var_type}")

    def _parse_model_mean_type(self, mean_type_str: str) -> str:
        """Parses and validates the model mean type string."""
        valid_types = ["START_X", "EPSILON"]
        parsed_type = mean_type_str.upper()
        if parsed_type not in valid_types:
            raise ValueError(f"Unknown model_mean_type: {mean_type_str}. Supported: {valid_types}")
        if parsed_type == "EPSILON":
            logger.warning("This sampler implementation is primarily for START_X prediction. "
                           "Ensure model_func output is handled accordingly if EPSILON is used.")
        return parsed_type

    def _parse_model_var_type(self, var_type_str: str) -> str:
        """Parses and validates the model variance type string."""
        valid_types = ["FIXED_SMALL", "FIXED_LARGE"] # Add "LEARNED", "LEARNED_RANGE" if supported
        parsed_type = var_type_str.upper()
        if parsed_type not in valid_types:
            raise ValueError(f"Unknown model_var_type: {var_type_str}. Supported: {valid_types}")
        return parsed_type

    def _extract_into_tensor(self, arr_numpy: np.ndarray, timesteps: torch.Tensor, broadcast_shape: tuple) -> torch.Tensor:
        """
        Extracts values from a 1-D numpy array for a batch of timesteps and broadcasts to a target shape.
        """
        res = torch.from_numpy(arr_numpy).to(device=timesteps.device)[timesteps.long()].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def _predict_xstart_from_epsilon(self, x_t: torch.Tensor, t: torch.Tensor, epsilon_pred: torch.Tensor) -> torch.Tensor:
        """
        Predicts x_0 from x_t and predicted epsilon, using the DDPM formula.
        x_0 = (x_t - sqrt(1-alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)
        """
        assert x_t.shape == epsilon_pred.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * epsilon_pred
        )

    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
                                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the mean and variance of the posterior distribution q(x_{t-1} | x_t, x_0).

        :param torch.Tensor x_start: The predicted or ground truth x_0, shape [bs, ...].
        :param torch.Tensor x_t: The noised data at timestep t, shape [bs, ...].
        :param torch.Tensor t: A 1D tensor of timesteps, shape [bs].
        :return: tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                 - posterior_mean: Mean of q(x_{t-1} | x_t, x_0).
                 - posterior_variance: Variance of q(x_{t-1} | x_t, x_0).
                 - posterior_log_variance_clipped: Clipped log of posterior_variance.
        """
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model_fn: Callable, x_t: torch.Tensor, t: torch.Tensor,
                        model_kwargs: Optional[Dict[str, Any]] = None,
                        clip_denoised: bool = True) -> Dict[str, torch.Tensor]:
        """
        Calculates the mean and variance of the reverse process p(x_{t-1} | x_t)
        and the model's prediction of x_0.

        :param Callable model_fn: The model's forward function (takes x_t, t, model_kwargs).
        :param torch.Tensor x_t: The current noised sample x_t.
        :param torch.Tensor t: The current timestep t.
        :param Optional[Dict[str, Any]] model_kwargs: Additional arguments for the model_fn.
        :param bool clip_denoised: If True, clips the predicted x_start to [-1, 1].
        :return: Dict[str, torch.Tensor]: A dictionary containing "mean", "variance",
                                          "log_variance", and "pred_xstart".
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        # Model predicts x0 directly in this setup
        pred_xstart = model_fn(x_t, t, model_kwargs)

        if self.model_mean_type == "EPSILON":
            # If model was trained to predict epsilon, convert its output to x0
            pred_xstart = self._predict_xstart_from_epsilon(x_t, t, epsilon_pred=pred_xstart)
            logger.debug("p_mean_variance: Converted model's epsilon prediction to x_start.")
        elif self.model_mean_type != "START_X":
            raise NotImplementedError(f"model_mean_type '{self.model_mean_type}' not supported for sampling in p_mean_variance.")

        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1.0, 1.0) # MDM clips x_start

        # Calculate the mean of q(x_{t-1} | x_t, pred_xstart)
        model_mean, _, model_log_variance_from_q = self.q_posterior_mean_variance(pred_xstart, x_t, t)

        # Determine variance for p(x_{t-1} | x_t) based on model_var_type
        if self.model_var_type == "FIXED_SMALL":
            # Use the posterior variance of q(x_{t-1} | x_t, x_0) but with fixed small log variance
            model_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
            model_log_variance = self._extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        elif self.model_var_type == "FIXED_LARGE":
            # Use beta_t as variance (approximation used in some DDPM variants)
            model_variance = self._extract_into_tensor(np.append(self.posterior_variance[1], self.betas[1:]), t, x_t.shape)
            model_log_variance = self._extract_into_tensor(np.log(np.maximum(np.append(self.posterior_variance[1], self.betas[1:]), 1e-20)), t, x_t.shape)
        else:
            # Add "LEARNED" or "LEARNED_RANGE" if model outputs variance
            raise NotImplementedError(f"Variance type '{self.model_var_type}' not implemented for sampling.")

        return {
            "mean": model_mean, # Mean of p(x_{t-1} | x_t)
            "variance": model_variance, # Variance of p(x_{t-1} | x_t)
            "log_variance": model_log_variance, # Log variance of p(x_{t-1} | x_t)
            "pred_xstart": pred_xstart, # Model's (clipped) prediction of x_0
        }

    @torch.no_grad()
    def p_sample_loop(self,
                      model_fn: Callable,
                      shape: tuple,
                      model_kwargs: Optional[Dict[str, Any]] = None,
                      device: Optional[torch.device] = None,
                      clip_denoised: bool = True,
                      progress: bool = False,
                      const_noise: bool = False,
                      custom_initial_noise: Optional[torch.Tensor] = None
                     ) -> torch.Tensor:
        """
        Generates samples from the diffusion model using the DDPM sampling loop.

        :param Callable model_fn: The model's forward function.
        :param tuple shape: The shape of the desired output tensor (batch_size, num_frames, features).
        :param Optional[Dict[str, Any]] model_kwargs: Additional arguments for model_fn.
        :param Optional[torch.device] device: The device to perform sampling on. If None, inferred from model.
        :param bool clip_denoised: If True, clips the predicted x_start at each step.
        :param bool progress: If True, displays a tqdm progress bar.
        :param bool const_noise: If True, uses constant noise for non-zero timesteps (simplified).
        :param Optional[torch.Tensor] custom_initial_noise: Optional custom noise to start the process.
        :return: torch.Tensor: The generated sample (predicted x_0).
        """
        if device is None:
            # Try to infer device from model_fn if it's a bound method of an nn.Module
            if hasattr(model_fn, '__self__') and isinstance(model_fn.__self__, nn.Module):
                device = next(model_fn.__self__.parameters()).device
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.warning(f"Device not specified and could not be inferred from model_fn. Defaulting to {device}.")

        current_sample_x_t = custom_initial_noise if custom_initial_noise is not None else torch.randn(*shape, device=device)
        
        fixed_noise_for_steps = None
        if const_noise:
            fixed_noise_for_steps = torch.randn_like(current_sample_x_t)

        indices = list(range(self.num_timesteps))[::-1] # Iterate t from T-1 down to 0
        if progress:
            indices = tqdm(indices, desc="DDPM Sampling Loop")

        for i in indices:
            t_batch = torch.tensor([i] * shape[0], device=device, dtype=torch.long)
            
            p_sample_output = self.p_mean_variance(
                model_fn, current_sample_x_t, t_batch,
                model_kwargs=model_kwargs, clip_denoised=clip_denoised
            )
            
            noise_for_this_step = fixed_noise_for_steps if const_noise and i != 0 else torch.randn_like(current_sample_x_t)
            
            # Mask to ensure no noise is added at the last step (t=0)
            nonzero_mask = (t_batch != 0).float().view(-1, *([1] * (len(current_sample_x_t.shape) - 1)))
            
            current_sample_x_t = (
                p_sample_output["mean"] +
                nonzero_mask * torch.exp(0.5 * p_sample_output["log_variance"]) * noise_for_this_step
            )
            # If i == 0, current_sample_x_t is effectively p_sample_output["mean"],
            # which should be very close to p_sample_output["pred_xstart"].
            # MDM's p_sample_loop returns out["pred_xstart"] if i == 0.
            # For START_X prediction, out["mean"] is derived from out["pred_xstart"], so they are closely related.

        # The final current_sample_x_t after the loop is the model's best estimate of x_0
        # If the last step (t=0) used pred_xstart as mean and zero variance (nonzero_mask=0), this is x0.
        # Let's return the explicit pred_xstart from the last step for clarity.
        final_pred_xstart = p_sample_output["pred_xstart"] if 'p_sample_output' in locals() else current_sample_x_t

        return final_pred_xstart


@torch.no_grad()
def generate_motion_mdm_style(
    armature_mdm_model: ArmatureMDM,
    diffusion_sampler_util: GaussianDiffusionSamplerUtil,
    y_conditions: Dict[str, Any],
    num_frames: int,
    device: str = 'cuda',
    clip_denoised: bool = True,
    progress: bool = True,
    const_noise: bool = False,
    custom_initial_noise: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Generates a motion sample using the ArmatureMDM model and DDPM sampling.
    Handles Classifier-Free Guidance (CFG) if cfg_scale is provided in y_conditions.

    :param ArmatureMDM armature_mdm_model: The trained ArmatureMDM model.
    :param GaussianDiffusionSamplerUtil diffusion_sampler_util: Utility for diffusion sampling.
    :param Dict[str, Any] y_conditions: Dictionary of conditions, including:
        'text_embeddings_batch': Precomputed SBERT embeddings [bs, sbert_dim].
        'armature_class_ids': Armature IDs [bs].
        'cfg_scale' (Optional): Classifier-Free Guidance scale. If > 1.0, CFG is applied.
        'uncond_text' (Optional, for CFG): Force text to be unconditional.
        'uncond_armature' (Optional, for CFG): Force armature to be unconditional.
    :param int num_frames: The number of frames to generate for the motion.
    :param str device: The device to perform generation on ('cuda' or 'cpu').
    :param bool clip_denoised: If True, clips the predicted x_start at each step.
    :param bool progress: If True, displays a tqdm progress bar for the sampling loop.
    :param bool const_noise: If True, uses constant noise for non-zero timesteps (simplified).
    :param Optional[torch.Tensor] custom_initial_noise: Optional custom noise to start the process.
    :return: torch.Tensor: The generated motion tensor (predicted x_0),
                           shape [batch_size, num_frames, num_motion_features].
    """
    armature_mdm_model.eval()
    armature_mdm_model.to(device)

    batch_size = y_conditions['text_embeddings_batch'].shape[0]
    # Assuming model has input_feats attribute correctly set
    num_motion_features = getattr(armature_mdm_model, 'input_feats', 66) # Fallback if not present

    output_shape = (batch_size, num_frames, num_motion_features)

    cfg_scale = y_conditions.get('cfg_scale', 1.0)

    model_fn_for_sampling: Callable
    final_model_kwargs_for_sampling: Dict[str, Any]

    if cfg_scale != 1.0 and cfg_scale > 0: # CFG is active
        logger.info(f"Applying Classifier-Free Guidance with scale: {cfg_scale}")
        # Conditional pass setup
        y_cond_pass = {k: v for k, v in y_conditions.items() if k != 'cfg_scale'}
        y_cond_pass['uncond'] = False # Explicitly conditional
        y_cond_pass['uncond_text'] = False
        y_cond_pass['uncond_armature'] = False

        # Unconditional pass setup
        y_uncond_pass = {k: v for k, v in y_conditions.items() if k != 'cfg_scale'}
        y_uncond_pass['uncond'] = True # Global unconditional flag
        y_uncond_pass['uncond_text'] = True # Force unconditional text
        y_uncond_pass['uncond_armature'] = True # Force unconditional armature

        def cfg_model_fn(x_t_cfg: torch.Tensor, t_cfg: torch.Tensor, y_ignored_kwargs: Dict[str,Any]) -> torch.Tensor:
            # y_ignored_kwargs is not used here because y_cond_pass and y_uncond_pass are captured
            # This matches how MDM's ClassifierFreeSampleModel works by wrapping the model.
            out_cond = armature_mdm_model(x_t_cfg, t_cfg, y_cond_pass)
            out_uncond = armature_mdm_model(x_t_cfg, t_cfg, y_uncond_pass)
            return out_uncond + cfg_scale * (out_cond - out_uncond)
        
        model_fn_for_sampling = cfg_model_fn
        final_model_kwargs_for_sampling = {} # kwargs are handled inside cfg_model_fn
    else: # No CFG or purely conditional generation
        logger.info("Generating with conditional model (CFG scale <= 1.0 or not applied).")
        y_input_direct = {k: v for k, v in y_conditions.items() if k != 'cfg_scale'}
        y_input_direct['uncond'] = False # Default to conditional
        y_input_direct['uncond_text'] = y_conditions.get('uncond_text', False)
        y_input_direct['uncond_armature'] = y_conditions.get('uncond_armature', False)
        
        # Lambda to match the expected signature for model_fn in p_sample_loop
        model_fn_for_sampling = lambda x_t_lambda, t_lambda, y_lambda_cond: armature_mdm_model(x_t_lambda, t_lambda, y_lambda_cond)
        final_model_kwargs_for_sampling = y_input_direct

    generated_motion = diffusion_sampler_util.p_sample_loop(
        model_fn=model_fn_for_sampling,
        shape=output_shape,
        model_kwargs=final_model_kwargs_for_sampling,
        device=torch.device(device),
        clip_denoised=clip_denoised,
        progress=progress,
        const_noise=const_noise,
        custom_initial_noise=custom_initial_noise
    )
    
    logger.info(f"Motion generation complete. Output shape: {generated_motion.shape}")
    return generated_motion