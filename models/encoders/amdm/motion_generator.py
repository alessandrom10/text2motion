import logging
import math
from pathlib import Path
from tqdm import tqdm
import yaml
from typing import List, Optional, Tuple, Union, Dict, Any, Callable
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from torch import nn
from TAMDM import ArmatureMDM
from utils.diffusion_utils import (
    load_armature_config, create_motion_animation
)

logger = logging.getLogger(__name__)

try:
    _motion_generator_file_path = Path(__file__).resolve()
    project_root = _motion_generator_file_path.parents[3]
except NameError:
    project_root = Path(".").resolve()
    logger.warning(f"__file__ not defined for MotionGenerator. Using CWD for project_root: {project_root}. Paths might be incorrect.")
logger.info(f"MotionGenerator inferred project_root: {project_root}")


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


class MotionGenerator:
    """
    A class for generating motion sequences using a trained ArmatureMDM model.
    This class handles loading configurations, initializing the model, and generating motions based on text prompts and armature IDs.
    """

    def __init__(self,
                 config_path: Union[str, Path],
                 model_checkpoint_path: Optional[Union[str, Path]] = None,
                 model_instance: Optional[torch.nn.Module] = None,
                 device_str: Optional[str] = None):
        """
        Initializes the MotionGenerator with the provided configuration and model.
        :param config_path: Path to the YAML configuration file.
        :param model_checkpoint_path: Optional path to a pre-trained model checkpoint.
        :param model_instance: Optional pre-initialized model instance to use instead of loading from checkpoint.
        :param device_str: Optional string to specify the device (e.g., 'cuda', 'cpu'). If None, uses config or auto-detects.
        """
        self.config_path = Path(config_path)
        logger.info(f"MotionGenerator: Initializing with config: {self.config_path}")

        # 1. Load Master Configuration
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"MotionGenerator: Error loading configuration file {self.config_path}: {e}")
            raise

        self.paths_cfg = self.config.get('paths', {})
        self.model_cfg_yaml = self.config.get('model_hyperparameters', {})
        self.diffusion_cfg = self.config.get('diffusion_hyperparameters', {})
        self.dataset_cfg = self.config.get('dataset_parameters', {})
        self.gen_params_cfg = self.config.get('generation_params', {})

        # 2. Setup Device
        if device_str:
            self.device = torch.device(device_str)
        else:
            # Use device from training config as fallback, then auto-detect
            training_cfg = self.config.get('training_hyperparameters', {})
            self.device = torch.device(training_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        logger.info(f"MotionGenerator: Using device: {self.device}")

        # 3. Load Armature Configuration
        armature_config_file_rel = self.paths_cfg.get('armature_config_file', 'config/armature_config.json')
        armature_config_file_abs = project_root / armature_config_file_rel

        self.loaded_armature_config = load_armature_config(str(armature_config_file_abs))
        if self.loaded_armature_config is None:
            raise ValueError(f"MotionGenerator: Could not load armature config from {armature_config_file_abs}")

        # 4. Initialize Beta Schedule
        self.betas_np = get_named_beta_schedule(
            schedule_name=self.diffusion_cfg.get('noise_schedule_mdm', 'cosine'),
            num_diffusion_timesteps=self.diffusion_cfg.get('num_diffusion_timesteps', 1000)
        )

        # 5. Load SBERT Model
        sbert_model_name = self.model_cfg_yaml.get('sbert_model_name', 'sentence-transformers/all-mpnet-base-v2')
        logger.info(f"MotionGenerator: Loading SBERT model '{sbert_model_name}'...")
        try:
            self.sbert_processor = SentenceTransformer(sbert_model_name, device=self.device)
        except Exception as e:
            logger.error(f"MotionGenerator: Failed to load SBERT model: {e}")
            raise
        logger.info("MotionGenerator: SBERT model loaded.")

        # 6. Initialize Diffusion Sampler
        self.diffusion_sampler = GaussianDiffusionSamplerUtil(
            betas=self.betas_np,
            model_mean_type=self.diffusion_cfg.get('model_mean_type_mdm', 'START_X'),
            model_var_type=self.diffusion_cfg.get('model_var_type_mdm', 'FIXED_SMALL')
        )
        logger.info("MotionGenerator: Diffusion sampler initialized.")

        # 7. Load Normalization Stats
        data_root_rel = self.paths_cfg.get('data_root_dir')
        # Ensure data_root_abs is correct whether data_root_rel is absolute or relative
        data_root_abs = Path(data_root_rel) if Path(data_root_rel).is_absolute() else project_root / data_root_rel
        mean_path = data_root_abs / self.dataset_cfg.get('dataset_mean_filename', 'mean.npy')
        std_path = data_root_abs / self.dataset_cfg.get('dataset_std_filename', 'std.npy')
        self._load_normalization_stats(str(mean_path), str(std_path))

        # 8. Model Initialization
        self.generation_model: Optional[ArmatureMDM] = None
        self.model_cfg_loaded = self.model_cfg_yaml # Initialize with config from YAML

        if model_checkpoint_path:
            self.load_model_from_checkpoint(Path(model_checkpoint_path))
        elif model_instance:
            self.set_model_instance(model_instance)
        # Else, the model will need to be loaded/set later

    def _determine_model_features(self, cfg_to_use: Dict[str, Any]) -> Optional[int]:
        data_rep = cfg_to_use.get('data_rep', "")
        if "flat" in data_rep:
            return cfg_to_use.get('num_motion_features_actual')
        else:
            num_j = cfg_to_use.get('njoints')
            num_f = cfg_to_use.get('nfeats_per_joint')
            if num_j is not None and num_f is not None:
                return num_j * num_f
        logger.error("MotionGenerator: Could not determine num_motion_features from model config.")
        return None

    def _instantiate_model_shell(self, model_params_cfg: Dict[str, Any]) -> ArmatureMDM:
        num_motion_features = self._determine_model_features(model_params_cfg)
        if num_motion_features is None:
            raise ValueError("MotionGenerator: Failed to determine num_motion_features for model instantiation.")

        # Instantiate ArmatureMDM using parameters from model_params_cfg
        return ArmatureMDM(
            data_rep=model_params_cfg.get('data_rep'),
            njoints=model_params_cfg.get('njoints'),
            nfeats_per_joint=model_params_cfg.get('nfeats_per_joint'),
            num_motion_features=num_motion_features,
            latent_dim=model_params_cfg.get('latent_dim'),
            ff_size=model_params_cfg.get('ff_size'),
            num_layers=model_params_cfg.get('num_layers'),
            num_heads=model_params_cfg.get('num_heads'),
            dropout=model_params_cfg.get('dropout', 0.1),
            activation=model_params_cfg.get('activation', 'gelu'),
            sbert_embedding_dim=model_params_cfg.get('sbert_embedding_dim'),
            max_armature_classes=model_params_cfg.get('max_armature_classes'),
            armature_embedding_dim=model_params_cfg.get('armature_embedding_dim'),
            armature_mlp_hidden_dims=model_params_cfg.get('armature_mlp_hidden_dims'),
            max_seq_len_pos_enc=model_params_cfg.get('max_seq_len_pos_enc', 5000),
            text_cond_mask_prob=0.0, # No masking for generation
            armature_cond_mask_prob=0.0, # No masking for generation
            arch=model_params_cfg.get('arch', 'trans_enc'),
            batch_first_transformer=model_params_cfg.get('batch_first_transformer', False),
            conditioning_integration_mode=model_params_cfg.get('conditioning_integration_mode', "mlp"),
            armature_integration_policy=model_params_cfg.get('armature_integration_policy', "add_refined"),
            conditioning_transformer_config=model_params_cfg.get('conditioning_transformer_config', {})
        ).to(self.device)


    def load_model_from_checkpoint(self, checkpoint_path: Path):
        """
        Load the model from a checkpoint file.
        :param checkpoint_path: Path to the model checkpoint file.
        """
        logger.info(f"MotionGenerator: Loading model from checkpoint: {checkpoint_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"MotionGenerator: Model checkpoint not found at {checkpoint_path}")

        # Load with weights_only=False to access config saved in checkpoint
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)

        ckpt_full_config = checkpoint.get('config')
        if ckpt_full_config and 'model_hyperparameters' in ckpt_full_config:
            self.model_cfg_loaded = ckpt_full_config['model_hyperparameters']
            logger.info("MotionGenerator: Using model hyperparameters from checkpoint.")
        else:
            logger.warning("MotionGenerator: No model_hyperparameters found in checkpoint's config. Using main YAML for model architecture.")
            self.model_cfg_loaded = self.model_cfg_yaml # Fallback to main YAML config

        self.generation_model = self._instantiate_model_shell(self.model_cfg_loaded)

        state_dict = checkpoint['model_state_dict']
        # Handle models saved with DataParallel
        if any(key.startswith('module.') for key in state_dict.keys()):
            logger.info("MotionGenerator: Loading DataParallel (module.) model state_dict.")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            self.generation_model.load_state_dict(new_state_dict)
        else:
            self.generation_model.load_state_dict(state_dict)

        self.generation_model.eval()
        logger.info(f"MotionGenerator: Model loaded. Parameters: {sum(p.numel() for p in self.generation_model.parameters()):,}")


    def set_model_instance(self, model_instance: torch.nn.Module):
        """
        Set the model instance directly, bypassing checkpoint loading.
        :param model_instance: An instance of ArmatureMDM or DataParallel(ArmatureMDM).
        """
        logger.info("MotionGenerator: Setting provided model instance.")
        # If the model is wrapped in DataParallel, get the underlying module
        actual_model = model_instance.module if isinstance(model_instance, torch.nn.DataParallel) else model_instance
        
        if not isinstance(actual_model, ArmatureMDM): # Check the type of the actual model
            raise TypeError(f"MotionGenerator: Provided model_instance is not of type ArmatureMDM or DataParallel(ArmatureMDM). Got {type(actual_model)}")

        self.generation_model = actual_model.to(self.device)
        self.generation_model.eval()
        # You might want to update self.model_cfg_loaded here if the model has its own internal config
        # e.g., if `actual_model.config_params` existed. For now, consistency is assumed.
        logger.info("MotionGenerator: Model instance set and in eval mode.")


    def _load_normalization_stats(self, mean_path_str: str, std_path_str: str):
        logger.info(f"MotionGenerator: Loading norm stats. Mean: {mean_path_str}, Std: {std_path_str}")
        try:
            self.dataset_mean = np.load(mean_path_str)
            raw_std = np.load(std_path_str)
            self.dataset_std_safe = np.where(raw_std == 0, 1e-8, raw_std) # Avoid division by zero
            logger.info("MotionGenerator: Normalization stats loaded.")
        except FileNotFoundError:
            logger.error(f"MotionGenerator: Mean/Std file not found ({mean_path_str} or {std_path_str}). De-normalization may fail or be incorrect.")
            self.dataset_mean = None
            self.dataset_std_safe = None
            # You might want to raise an exception here if stats are crucial

    @torch.no_grad()
    def _generate_motion_core_loop(self,
                                   y_conditions: Dict[str, Any],
                                   num_frames: int,
                                   clip_denoised: bool = True,
                                   progress: bool = False,
                                   const_noise: bool = False,
                                   custom_initial_noise: Optional[torch.Tensor] = None
                                   ) -> torch.Tensor:
        """Internal method to run the diffusion sampling loop with CFG handling."""
        if self.generation_model is None:
            raise RuntimeError("MotionGenerator: Model not loaded or set before calling _generate_motion_core_loop.")

        self.generation_model.eval() # Ensure model is in eval mode

        batch_size = y_conditions['text_embeddings_batch'].shape[0]
        num_motion_features = self._determine_model_features(self.model_cfg_loaded)
        if num_motion_features is None:
            raise ValueError("MotionGenerator: Cannot determine num_motion_features for generation shape.")
        output_shape = (batch_size, num_frames, num_motion_features)

        cfg_scale = y_conditions.get('cfg_scale', 1.0)
        model_fn_for_sampling: Callable
        
        # This lambda directly calls the model, which should handle y_conditions internally
        # based on its 'uncond', 'uncond_text', 'uncond_armature' flags.
        # The ArmatureMDM model's forward method should be designed to respect these flags.
        wrapped_model_fn = lambda x_t_lambda, t_lambda, y_lambda_cond_dict: self.generation_model(x_t_lambda, t_lambda, y_lambda_cond_dict)

        if cfg_scale != 1.0 and cfg_scale > 0: # CFG is active
            logger.debug(f"Applying CFG with scale: {cfg_scale}")
            y_cond_pass = {k: v for k, v in y_conditions.items() if k != 'cfg_scale'}
            y_cond_pass['uncond'] = False; y_cond_pass['uncond_text'] = False; y_cond_pass['uncond_armature'] = False
            
            y_uncond_pass = {k: v for k, v in y_conditions.items() if k != 'cfg_scale'}
            y_uncond_pass['uncond'] = True; y_uncond_pass['uncond_text'] = True; y_uncond_pass['uncond_armature'] = True
            
            def cfg_intermediate_model_fn(x_t_cfg: torch.Tensor, t_cfg: torch.Tensor, ignored_kwargs: Dict[str,Any]) -> torch.Tensor:
                out_cond = wrapped_model_fn(x_t_cfg, t_cfg, y_cond_pass)
                out_uncond = wrapped_model_fn(x_t_cfg, t_cfg, y_uncond_pass)
                return out_uncond + cfg_scale * (out_cond - out_uncond)
            
            model_fn_for_sampling = cfg_intermediate_model_fn
            kwargs_for_loop = {} # kwargs are handled inside cfg_intermediate_model_fn
        else: # No CFG
            logger.debug("Generating with conditional model (no CFG or CFG scale <= 1.0).")
            y_input_direct = {k: v for k, v in y_conditions.items() if k != 'cfg_scale'}
            # Ensure uncond flags are set based on y_conditions or default to False
            y_input_direct['uncond'] = y_conditions.get('uncond', False) 
            y_input_direct['uncond_text'] = y_conditions.get('uncond_text', False)
            y_input_direct['uncond_armature'] = y_conditions.get('uncond_armature', False)
            model_fn_for_sampling = wrapped_model_fn
            kwargs_for_loop = y_input_direct
            
        generated_motion = self.diffusion_sampler.p_sample_loop(
            model_fn=model_fn_for_sampling,
            shape=output_shape,
            model_kwargs=kwargs_for_loop, 
            device=self.device,
            clip_denoised=clip_denoised,
            progress=progress,
            const_noise=const_noise,
            custom_initial_noise=custom_initial_noise
        )
        logger.debug(f"Motion generation core loop complete. Output shape: {generated_motion.shape}")
        return generated_motion

    def generate_single_motion(self,
                               text_prompt: str,
                               armature_id: Union[str, int],
                               num_frames: int,
                               cfg_scale: float,
                               clip_denoised: bool = True,
                               progress_bar: bool = False,
                               const_noise_for_sampling: bool = False,
                               initial_noise_custom: Optional[torch.Tensor] = None
                               ) -> Optional[np.ndarray]:
        """
        Generate a single motion sequence based on the provided text prompt and armature ID.
        :param text_prompt: The text prompt to condition the motion generation on.
        :param armature_id: The ID of the armature class to use for the motion generation.
                            Can be a string or an integer.
        :param num_frames: The number of frames to generate in the motion sequence.
        :param cfg_scale: The classifier-free guidance scale to apply during generation.
        :param clip_denoised: Whether to clip the denoised output to [-1, 1].
        :param progress_bar: Whether to show a progress bar during generation.
        :param const_noise_for_sampling: If True, uses constant noise for non-zero timesteps.
        :param initial_noise_custom: Optional custom noise tensor to use as the starting point for generation.
        :return: A numpy array of shape (num_frames, num_joints, num_features) representing the generated motion,
                 or None if generation fails.
        """
        if self.generation_model is None:
            logger.error("MotionGenerator: Model not loaded. Cannot generate motion.")
            return None
        try:
            text_embedding = self.sbert_processor.encode(text_prompt, convert_to_tensor=True).to(self.device)
        except Exception as e:
            logger.error(f"MotionGenerator: SBERT encoding error for \"{text_prompt}\": {e}"); return None

        current_armature_id_int = int(armature_id)
        y_conditions = {
            'text_embeddings_batch': text_embedding.unsqueeze(0),
            'armature_class_ids': torch.tensor([current_armature_id_int], dtype=torch.long).to(self.device),
            'mask': None, 'cfg_scale': cfg_scale,
        }
        logger.info(f"MotionGenerator: Generating {num_frames}f for '{text_prompt}', armID {current_armature_id_int}, CFG {cfg_scale}")
        
        generated_motion_tensor = self._generate_motion_core_loop(
            y_conditions=y_conditions,
            num_frames=num_frames,
            clip_denoised=clip_denoised,
            progress=progress_bar,
            const_noise=const_noise_for_sampling,
            custom_initial_noise=initial_noise_custom
        )
        if generated_motion_tensor is None: return None

        motion_np = generated_motion_tensor.squeeze(0).cpu().numpy()
        if self.dataset_mean is not None and self.dataset_std_safe is not None:
            motion_np_denormalized = motion_np * self.dataset_std_safe + self.dataset_mean
        else:
            logger.warning("MotionGenerator: Norm stats N/A. Returning normalized motion.")
            motion_np_denormalized = motion_np
        
        num_j = self.model_cfg_loaded.get('njoints')
        feat_p_j = self.model_cfg_loaded.get('nfeats_per_joint')
        # Fallback to geom specific if general ones are not in model_cfg_loaded
        if num_j is None: num_j = self.model_cfg_loaded.get('num_joints_for_geom')
        if feat_p_j is None: feat_p_j = self.model_cfg_loaded.get('features_per_joint_for_geom')

        if num_j is None or feat_p_j is None:
            logger.error("MotionGenerator: n_joints/n_feats not in loaded model config. Returning unshaped.")
            return motion_np_denormalized
        
        expected_features = num_j * feat_p_j
        if motion_np_denormalized.shape[1] == expected_features:
            return motion_np_denormalized.reshape(num_frames, num_j, feat_p_j)
        else:
            logger.warning(f"MotionGenerator: Output mismatch. Expected {expected_features} feats, got {motion_np_denormalized.shape[1]}. Ret unshaped.")
            return motion_np_denormalized


    def save_motion_as_gif(self,
                            motion_data_frames: np.ndarray, # Expects reshaped motion data
                            output_path_abs: Path,
                            armature_id: int,
                            title: Optional[str] = None,
                            fps: Optional[int] = None):
        """
        Save the motion data as a GIF animation.
        :param motion_data_frames: A numpy array of shape (frames, joints, features) representing the motion.
        :param output_path_abs: The absolute path where the GIF will be saved.
        :param armature_id: The ID of the armature class to use for the kinematic chain.
        :param title: Optional title for the GIF. If None, no title is added.
        :param fps: Optional frames per second for the GIF. If None, uses default from generation params.
        """
        if fps is None:
            fps = self.gen_params_cfg.get('render_fps', 20) # Get from generation params in config

        if motion_data_frames.ndim != 3: # Expected shape (frames, joints, features)
            logger.error(f"MotionGenerator: Motion data for GIF saving has incorrect dimensions ({motion_data_frames.ndim}). GIF not created.")
            return
        
        armature_id_str = str(armature_id)
        armature_definitions = self.loaded_armature_config.get('ARMATURE_DEFINITIONS', {})
        specific_armature_def = armature_definitions.get(armature_id_str)

        if specific_armature_def is None:
            logger.error(f"MotionGenerator: Armature definition for ID '{armature_id_str}' not found in config. Cannot determine kinematic chain.")
            return

        graph_connectivity = specific_armature_def.get('graph_connectivity')
        if graph_connectivity is None or 'edge_index' not in graph_connectivity:
            logger.error(f"MotionGenerator: 'graph_connectivity' or 'edge_index' not found for armature ID '{armature_id_str}'.")
            return
            
        kinematic_chain_for_render: List[List[int]] = graph_connectivity['edge_index']

        logger.info(f"MotionGenerator: Saving animation to {output_path_abs}")
        try:
            output_path_abs.parent.mkdir(parents=True, exist_ok=True)
            create_motion_animation(
                motion_data_frames=motion_data_frames,
                kinematic_chain=kinematic_chain_for_render,
                output_filename=str(output_path_abs),
                fps=fps,
                title=title
            )
            logger.info(f"MotionGenerator: Animation saved successfully.")
        except Exception as e:
            logger.error(f"MotionGenerator: Error creating animation: {e}")

    def generate_and_save_from_metadata_list(self,
                                             generation_requests: List[Dict[str, Any]],
                                             base_output_dir: Path,
                                             cfg_scale: float, # This comes from CLI args for batch script
                                             default_num_frames: Optional[int] = None, # From CLI or config
                                             num_frames_override: Optional[int] = None, # From CLI
                                             render_fps: Optional[int] = None, # From CLI or config
                                             progress_bar: bool = True,
                                             const_noise_for_sampling: bool = False):
        """
        Generates and saves multiple motions based on a list of metadata.
        :param generation_requests: List of dictionaries with keys 'text', 'armature_id', 'original_length', and optionally 'base_filename_part'.
        :param base_output_dir: The directory where the generated motions will be saved.
        :param cfg_scale: The classifier-free guidance scale to apply during generation.
        :param default_num_frames: Default number of frames to generate if not specified in requests.
        :param num_frames_override: If provided, overrides the number of frames for all requests.
        :param render_fps: Frames per second for the rendered GIFs. If None, uses default from generation params.
        :param progress_bar: Whether to show a progress bar during batch generation.
        :param const_noise_for_sampling: If True, uses constant noise for non-zero timesteps.
        """
        if default_num_frames is None: # Get from general generation config if not passed
            default_num_frames = self.gen_params_cfg.get('num_frames_to_generate', 120)
        if render_fps is None: # Get from general generation config if not passed
            render_fps = self.gen_params_cfg.get('render_fps', 20)

        logger.info(f"MotionGenerator: Batch generating {len(generation_requests)} motions to {base_output_dir}...")
        
        iterable_requests = tqdm(generation_requests, desc="Batch Generating Motions") if progress_bar else generation_requests

        for idx, req in enumerate(iterable_requests):
            text_prompt = req.get('text')
            armature_id = req.get('armature_id')
            original_length = req.get('original_length')
            # Use a more robust base filename part, e.g., from a dataset item ID or index
            base_fn_part = req.get('base_filename_part', f"item_{req.get('dataset_idx', idx)}")

            if not text_prompt or armature_id is None:
                logger.warning(f"Skipping request {idx+1} due to missing text or armature_id: {req}")
                continue

            num_frames = num_frames_override if num_frames_override is not None else \
                         (original_length if original_length and original_length > 0 else default_num_frames)

            logger.debug(f"Request {idx+1}/{len(generation_requests)}: Text='{text_prompt}', ArmID={armature_id}, Frames={num_frames}")
            
            # Note: generate_single_motion now includes progress_bar and const_noise arguments
            motion_data = self.generate_single_motion(
                text_prompt=text_prompt,
                armature_id=armature_id,
                num_frames=num_frames,
                cfg_scale=cfg_scale, # Passed from calling script's arguments
                progress_bar=False, # Progress bar is on the outer loop here
                const_noise_for_sampling=const_noise_for_sampling
            )
            if motion_data is not None:
                # Sanitize text_prompt for filename
                text_snippet = "".join(c if c.isalnum() else "_" for c in text_prompt[:30])
                animation_filename = f"gen_{base_fn_part}_arm{str(armature_id)}_cfg{cfg_scale}_{text_snippet}.gif"
                output_path = base_output_dir / animation_filename
                self.save_motion_as_gif(
                    motion_data_frames=motion_data,
                    output_path_abs=output_path,
                    armature_id=armature_id,
                    title=text_prompt,
                    fps=render_fps
                )
            else:
                logger.warning(f"Failed to generate motion for request {idx+1}: Text='{text_prompt}'")
        logger.info(f"MotionGenerator: Batch generation finished for {len(generation_requests)} requests.")
