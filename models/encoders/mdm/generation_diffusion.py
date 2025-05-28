# run_generation_english.py
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ArmatureMDM import ArmatureMDM
from utils.diffusion_utils import T2M_KINEMATIC_CHAIN, create_motion_animation

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_motion_from_trained_model(
    model: ArmatureMDM,
    text_prompt: str,
    sbert_model_name: str,
    armature_id: int,
    num_frames: int,
    model_hyperparams: Dict[str, Any],
    diffusion_hyperparams: Dict[str, Any],
    cfg_scale: float = 2.5,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Generates a single motion sample using a trained ArmatureMDM model.

    :param model: The trained ArmatureMDM model.
    :param text_prompt: The text prompt for conditioning.
    :param sbert_model_name: The name of the SBERT model for text embedding.
    :param armature_id: The armature class ID for generation.
    :param num_frames: Number of frames for the generated motion.
    :param model_hyperparams: Dictionary of model hyperparameters (e.g., 'num_motion_features').
    :param diffusion_hyperparams: Dictionary of diffusion schedule parameters 
                                  (e.g., 'num_diffusion_timesteps', 'beta_start', 'beta_end').
    :param cfg_scale: Classifier-Free Guidance scale.
    :param device: The device to run generation on ('cuda' or 'cpu').
    :return: Tensor of the generated motion sample, shape [1, num_frames, num_motion_features].
    """
    model.eval()
    model.to(device)

    num_motion_features = model_hyperparams.get('num_motion_features', 66)

    try:
        sbert_processor = SentenceTransformer(sbert_model_name, device=device)
        text_embedding = sbert_processor.encode(text_prompt, convert_to_tensor=True).unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"Error loading SBERT model ('{sbert_model_name}') or encoding text: {e}")
        raise # Re-raise the exception to stop execution if SBERT fails

    armature_class_ids_tensor = torch.tensor([armature_id], device=device, dtype=torch.long)

    num_diffusion_timesteps = diffusion_hyperparams.get('num_diffusion_timesteps', 100)
    beta_start = diffusion_hyperparams.get('beta_start', 0.0001)
    beta_end = diffusion_hyperparams.get('beta_end', 0.02)

    betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)
    
    # Coefficients for DDPM reverse process (predicting x0)
    posterior_variance = (1.0 - alphas_cumprod_prev) * betas / (1.0 - alphas_cumprod + 1e-9) # Add epsilon for stability
    posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
    posterior_mean_coef1 = torch.sqrt(alphas_cumprod_prev) * betas / (1.0 - alphas_cumprod + 1e-9)
    posterior_mean_coef2 = torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-9)

    current_xt = torch.randn((1, num_frames, num_motion_features), device=device) 

    for t_idx in tqdm(reversed(range(num_diffusion_timesteps)), desc="Generating Motion Sample", total=num_diffusion_timesteps, leave=True):
        timesteps_batch = torch.full((1,), t_idx, device=device, dtype=torch.long)

        predicted_x0_cond = model(
            current_xt, timesteps_batch, text_embedding, armature_class_ids_tensor,
            uncond_text=False, uncond_armature=False
        )
        predicted_x0_uncond = model(
            current_xt, timesteps_batch, text_embedding, armature_class_ids_tensor,
            uncond_text=True, uncond_armature=True
        )
        final_predicted_x0 = predicted_x0_uncond + cfg_scale * (predicted_x0_cond - predicted_x0_uncond)
        
        # Optional debug log inside the loop (set logger to DEBUG level to see this)
        if t_idx % (num_diffusion_timesteps // 10) == 0 or t_idx < 5 or t_idx == num_diffusion_timesteps -1 :
            temp_pose_data = final_predicted_x0.squeeze(0).cpu().numpy()
            is_pred_static_internal = np.allclose(temp_pose_data[0], temp_pose_data[1:], atol=1e-4) if temp_pose_data.shape[0] > 1 else "N/A (1 frame)"
            std_across_frames = temp_pose_data[:, 0].std() if temp_pose_data.shape[0] > 1 else float('nan')
            logger.debug(f"t={t_idx:03d}, pred_x0 static? {is_pred_static_internal}. Std_frames[0]: {std_across_frames:.6f}. Mean: {temp_pose_data.mean():.4f}")

        if t_idx == 0:
            current_xt = final_predicted_x0
        else:
            current_posterior_mean_coef1 = posterior_mean_coef1[t_idx]
            current_posterior_mean_coef2 = posterior_mean_coef2[t_idx]
            posterior_mean = current_posterior_mean_coef1 * final_predicted_x0 + \
                             current_posterior_mean_coef2 * current_xt
            noise_for_sampling = torch.randn_like(current_xt)
            current_xt = posterior_mean + (0.5 * posterior_log_variance_clipped[t_idx]).exp() * noise_for_sampling
            
    return current_xt


if __name__ == '__main__':
    # --- 1. Load Configuration ---
    # Assumes 'config/diffusion_config.yaml' exists relative to the determined project_root
    CONFIG_PATH = Path(project_root) / 'config' / 'diffusion_config.yaml'
    
    logger.info(f"Attempting to load configuration from: {CONFIG_PATH}")
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {CONFIG_PATH}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading config file {CONFIG_PATH}: {e}")
        sys.exit(1)
    logger.info("Configuration loaded successfully.")

    device = config.get('training_hyperparameters', {}).get('device', 'cuda')
    model_hyperparams_cfg = config.get('model_hyperparameters', {})
    diffusion_hyperparams_cfg = config.get('diffusion_hyperparameters', {})
    paths_cfg = config.get('paths', {})

    # --- 2. Initialize and Load Your Trained Model ---
    logger.info("Initializing ArmatureMDM model...")
    # Ensure all required arguments for ArmatureMDM's __init__ are provided from model_hyperparams_cfg
    try:
        loaded_model = ArmatureMDM(
            num_motion_features=model_hyperparams_cfg['num_motion_features'],
            latent_dim=model_hyperparams_cfg['latent_dim'],
            ff_size=model_hyperparams_cfg['ff_size'],
            num_layers=model_hyperparams_cfg['num_layers'],
            num_heads=model_hyperparams_cfg['num_heads'],
            dropout=model_hyperparams_cfg['dropout'],
            activation=model_hyperparams_cfg.get('activation', 'gelu'), # .get for safety
            sbert_model_name=model_hyperparams_cfg['sbert_model_name'],
            sbert_embedding_dim=model_hyperparams_cfg['sbert_embedding_dim'],
            max_armature_classes=model_hyperparams_cfg['max_armature_classes'],
            armature_embedding_dim=model_hyperparams_cfg['armature_embedding_dim'],
            text_time_conditioning_policy=model_hyperparams_cfg['text_time_cond_policy'],
            armature_integration_policy=model_hyperparams_cfg['armature_integration_policy'],
            timestep_embedder_type=model_hyperparams_cfg['timestep_embedder_type'],
            text_cond_dropout_prob=model_hyperparams_cfg['text_cond_dropout_prob'],
            armature_cond_dropout_prob=model_hyperparams_cfg['armature_cond_dropout_prob'],
            max_seq_len=model_hyperparams_cfg['max_seq_len_pos_enc'],
            use_final_algebraic_refinement_encoder=model_hyperparams_cfg['use_final_algebraic_refinement_encoder'],
            algebraic_refinement_hidden_dim=model_hyperparams_cfg.get('algebraic_refinement_hidden_dim'), # .get if can be None
            timestep_gru_num_layers=model_hyperparams_cfg['timestep_gru_num_layers'],
            timestep_gru_dropout=model_hyperparams_cfg['timestep_gru_dropout']
        )
    except KeyError as e:
        logger.error(f"Missing key in model_hyperparameters from config: {e}. Please check your diffusion_config.yaml.")
        sys.exit(1)
    
    model_save_dir_from_config = paths_cfg.get('model_save_dir', './trained_models_final/')
    # Construct model checkpoint path carefully, considering if it's relative to project_root
    if model_save_dir_from_config.startswith('./') or model_save_dir_from_config.startswith('../'):
        # Path is relative to where config is, or project root if config path is simple
        # Assuming model_save_dir is relative to project_root if it starts with './'
        model_checkpoint_path = project_root / model_save_dir_from_config.lstrip("./") / paths_cfg.get('model_filename', 'armature_mdm_final.pth')
    else: # Assume it's an absolute path or a path that doesn't need project_root
        model_checkpoint_path = Path(model_save_dir_from_config) / paths_cfg.get('model_filename', 'armature_mdm_final.pth')
    
    try:
        logger.info(f"Loading trained model from: {model_checkpoint_path}")
        checkpoint = torch.load(str(model_checkpoint_path), map_location=device, weights_only=True) # Set weights_only=True for safety if from untrusted source
        if 'model_state_dict' in checkpoint:
            loaded_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            loaded_model.load_state_dict(checkpoint)
        logger.info("Successfully loaded trained model.")
    except FileNotFoundError:
        logger.error(f"Model checkpoint not found: {model_checkpoint_path}.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading model checkpoint: {e}")
        sys.exit(1)

    # --- 3. Define Generation Parameters ---
    text_to_generate_for_sample = "a person is jumping excitedly"
    armature_id_for_sample = 1 
    num_frames_to_generate = 100  # Keep consistent with dataset_parameters.max_seq_len_dataset for now
    
    # CFG scale: try 0.0 first for debugging static output, then 1.5, 2.0, 2.5
    cfg_guidance_scale = config.get('generation_params', {}).get('cfg_scale', 2.0) 

    # --- 4. Generate Motion ---
    logger.info(f"Generating motion for text: '{text_to_generate_for_sample}' "
                f"with armature ID: {armature_id_for_sample}, CFG Scale: {cfg_guidance_scale}, "
                f"Frames: {num_frames_to_generate}")
                
    generated_motion_tensor = generate_motion_from_trained_model(
        model=loaded_model,
        text_prompt=text_to_generate_for_sample,
        sbert_model_name=model_hyperparams_cfg['sbert_model_name'],
        armature_id=armature_id_for_sample,
        num_frames=num_frames_to_generate,
        model_hyperparams=model_hyperparams_cfg,
        diffusion_hyperparams=diffusion_hyperparams_cfg,
        cfg_scale=cfg_guidance_scale,
        device=device
    )
    logger.info(f"Generated motion tensor shape: {generated_motion_tensor.shape}")

    # --- Debugging: Inspect the generated motion data ---
    motion_np_raw = generated_motion_tensor.squeeze(0).cpu().numpy()
    logger.info(f"Shape of raw numpy motion: {motion_np_raw.shape}")
    logger.info(f"Raw motion stats: Min={motion_np_raw.min():.6f}, Max={motion_np_raw.max():.6f}, "
                f"Mean={motion_np_raw.mean():.6f}, Std={motion_np_raw.std():.6f}")
    if num_frames_to_generate > 1:
        is_static_check = np.allclose(motion_np_raw[0], motion_np_raw[1:], atol=1e-5)
        logger.info(f"Is raw generated motion static (all frames ~equal to first)? {is_static_check}")
        if is_static_check:
            logger.warning("All raw generated frames are nearly identical. Model might be outputting a static pose.")

    # --- 5. Prepare Data for Animation & Visualize ---
    num_joints_for_viz = model_hyperparams_cfg.get('num_joints_for_geom', 22)
    features_per_joint_viz = model_hyperparams_cfg.get('features_per_joint_for_geom', 3)
    
    if motion_np_raw.shape[1] != num_joints_for_viz * features_per_joint_viz:
        logger.error(f"Generated motion features ({motion_np_raw.shape[1]}) "
                     f"mismatch expected {num_joints_for_viz}j * {features_per_joint_viz}f/j. Cannot reshape.")
        sys.exit(1)
    
    motion_reshaped = motion_np_raw.reshape((num_frames_to_generate, num_joints_for_viz, features_per_joint_viz))
    motion_centered = motion_reshaped - motion_reshaped[:, 0:1, :] 
    
    # UNIT_CONVERSION_FOR_VIZ: Crucial for correct visualization scale.
    # If model outputs values similar to normalized data (-1 to 1, or meter scale), set to None or 1.0.
    # If model outputs large values (e.g., millimeters), use 1.0/1000.0.
    # Given previous raw stats (Min=-0.83, Max=0.51), drastic scaling down might not be needed
    # unless your original animation utility expected very large input values.
    UNIT_CONVERSION_FOR_VIZ = None # Try None or 1.0 first.

    animation_output_filename = f"gen_arm{armature_id_for_sample}_cfg{cfg_guidance_scale}.gif"
    animation_output_path = Path(config.get("paths", {}).get("model_save_dir", ".")) / animation_output_filename
    os.makedirs(animation_output_path.parent, exist_ok=True)


    logger.info(f"Creating animation and saving to: {animation_output_path}")
    anim = create_motion_animation(
        motion_data_frames=motion_centered,
        kinematic_chain=T2M_KINEMATIC_CHAIN, 
        output_filename=str(animation_output_path),
        fps=30,
        unit_conversion_factor=UNIT_CONVERSION_FOR_VIZ,
        y_z_swap=True
    )
    
    logger.info(f"Script finished. Animation saved to {animation_output_path}")