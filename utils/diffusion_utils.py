import json
import logging
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str, run_name: str) -> None:
    """
    Sets up file and console logging for the training run.

    :param log_dir: Directory to save the log file.
    :param run_name: Name of the current run, used for the log filename.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"training_log_{run_name}.txt")

    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

    file_handler = logging.FileHandler(log_file_path, mode='a')  # Append mode
    file_handler.setFormatter(log_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    logger.info(f"Logging setup complete. Log file: {log_file_path}")


def load_armature_config(config_path: str) -> dict:
    """Loads the armature configuration from a JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Successfully loaded armature configuration from: {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Armature configuration file not found: {config_path}. Using default full mask.")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from armature configuration file: {config_path}. Using default full mask.")
    except Exception as e:
        logger.error(f"An unexpected error occurred loading armature config {config_path}: {e}. Using default full mask.")
    return None


def get_noise_schedule(num_diffusion_timesteps: int,
                       beta_start: float = 0.0001,
                       beta_end: float = 0.02,
                       device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Generates a linear beta schedule and corresponding alpha_cumprod values
    as used in DDPMs.

    :param num_diffusion_timesteps: The total number of diffusion steps (T).
    :param beta_start: The starting value for beta at t=0 (or t=1 if 1-indexed).
    :param beta_end: The ending value for beta at t=T.
    :param device: The device to create the tensors on.
    :return: Tensor of alpha_bar_t (cumulative product of alphas), shape [T].
    """
    betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    logger.debug(f"Generated noise schedule with {num_diffusion_timesteps} timesteps on device {device}.")
    return alphas_cumprod


def plot_training_history(history: Dict[str, List[float]],
                            main_val_metric_name: str,
                            model_save_dir: str,
                            run_name: str) -> None:
    """
    Plots training & validation total loss, and a specified main validation metric.

    :param history: Dictionary containing 'train_loss', 'val_loss', 'val_metric' lists.
    :param main_val_metric_name: Name of the main validation metric for plotting title and label.
    :param model_save_dir: Directory to save the plot.
    :param run_name: Name of the current run, used for the plot filename.
    """
    train_losses_epochs = history.get('train_loss', [])
    val_losses_epochs = history.get('val_loss', [])
    val_main_metric_epochs = history.get('val_metric', [])

    if not train_losses_epochs:
        logger.warning("No training loss data provided to plot.")
        return

    num_train_epochs_completed = len(train_losses_epochs)
    epochs_axis_train = range(1, num_train_epochs_completed + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_axis_train, train_losses_epochs, 'bo-', label='Training Loss (Avg Total)')
    if val_losses_epochs:
        epochs_axis_val_loss = range(1, len(val_losses_epochs) + 1)
        plt.plot(epochs_axis_val_loss, val_losses_epochs, 'ro-', label='Validation Loss (Avg Total)')
    plt.title('Training & Validation Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot 2: Main Validation Metric
    if val_main_metric_epochs:
        epochs_axis_val_metric = range(1, len(val_main_metric_epochs) + 1)
        plt.subplot(1, 2, 2)
        plt.plot(epochs_axis_val_metric, val_main_metric_epochs, 'go-', label=f'Validation {main_val_metric_name}')
        plt.title(f'Validation {main_val_metric_name}')
        plt.xlabel('Epochs')
        plt.ylabel(main_val_metric_name)
        plt.legend()
        plt.grid(True)
    else:
        logger.info("No main validation metric data provided for plotting.")


    plt.tight_layout()
    plot_save_path = os.path.join(model_save_dir, f"training_plots_{run_name}.png")
    try:
        plt.savefig(plot_save_path)
        logger.info(f"Training plots saved to {plot_save_path}")
    except Exception as e:
        logger.error(f"Failed to save plots: {e}", exc_info=True)
    # plt.show() # Typically commented out for non-interactive server runs


def get_bone_mask_for_armature(armature_class_ids: torch.Tensor,
                               num_total_motion_features: int,
                               num_frames: int,
                               device: str,
                               armature_config_data: Optional[Dict] = None,
                               features_per_bone_override: Optional[int] = None
                            ) -> torch.Tensor:
    """
    Generates a bone mask based on armature class IDs.
    This is a placeholder and needs to be implemented or adapted
    according to your specific armature definitions and how features map to them.
    Currently, it creates a mask that activates all features for all samples.

    :param armature_class_ids: Tensor of armature class IDs for the batch. Shape: (batch_size,).
    :param num_total_motion_features: Total number of features in the motion data (e.g., joints * coords).
    :param num_frames: Number of frames in the motion sequence.
    :param device: Device to create the mask tensor on (e.g., 'cuda', 'cpu').
    :param armature_config_data: Optional dictionary containing armature configuration data.
    :param features_per_bone_override: Optional override for number of features per bone.
    :return: Bone mask tensor of shape (batch_size, num_frames, num_total_motion_features),
             with 1s for active features and 0s for inactive ones.
    """
    batch_size = armature_class_ids.shape[0]
    mask = torch.zeros((batch_size, num_frames, num_total_motion_features), device=device, dtype=torch.float32)

    if armature_config_data is None:
        logger.warning("Armature config data not provided to get_bone_mask_for_armature. Defaulting to activate all features.")
        mask.fill_(1.0)
        return mask

    features_per_bone = features_per_bone_override if features_per_bone_override is not None else armature_config_data.get("FEATURES_PER_BONE", 3)
    armature_definitions = armature_config_data.get("ARMATURE_DEFINITIONS", {})
    default_num_bones = armature_config_data.get("DEFAULT_FALLBACK_NUM_ACTIVE_BONES") # Can be None

    if num_total_motion_features % features_per_bone != 0:
        logger.error(f"num_total_motion_features ({num_total_motion_features}) is not divisible by "
                       f"FEATURES_PER_BONE ({features_per_bone}). Masking logic might be incorrect. Activating all features.")
        mask.fill_(1.0)
        return mask

    for i in range(batch_size):
        current_armature_id_int = armature_class_ids[i].item()
        current_armature_id_str = str(current_armature_id_int) # JSON keys are strings

        num_active_bones = None
        if current_armature_id_str in armature_definitions:
            num_active_bones = armature_definitions[current_armature_id_str].get("num_active_bones")
        
        if num_active_bones is None: # ID not found or 'num_active_bones' key missing
            if default_num_bones is not None:
                num_active_bones = default_num_bones
                logger.warning(f"Sample {i}: Armature ID {current_armature_id_int} not in ARMATURE_DEFINITIONS "
                               f"or 'num_active_bones' missing. Using default fallback of {default_num_bones} bones.")
            else:
                logger.warning(f"Sample {i}: Armature ID {current_armature_id_int} not in ARMATURE_DEFINITIONS "
                               f"and no default fallback. Activating all features for this sample.")
                mask[i, :, :] = 1.0
                continue # Move to next sample in batch

        num_active_features = num_active_bones * features_per_bone

        if num_active_features > num_total_motion_features:
            logger.warning(f"Sample {i}: For armature ID {current_armature_id_int}, calculated num_active_features "
                           f"({num_active_features}) exceeds num_total_motion_features ({num_total_motion_features}). "
                           f"Clamping to num_total_motion_features.")
            num_active_features = num_total_motion_features
        elif num_active_features < 0:
            logger.warning(f"Sample {i}: For armature ID {current_armature_id_int}, calculated num_active_features "
                           f"is negative ({num_active_features}). Setting to 0.")
            num_active_features = 0
        
        if num_active_features > 0:
            mask[i, :, :num_active_features] = 1.0
        
        logger.debug(f"Sample {i}: Armature ID {current_armature_id_int} - Activating first {num_active_features} features "
                     f"(for {num_active_bones} bones).")
            
    return mask

def default_uniform_timestep_sampler(num_diffusion_timesteps: int) -> int:
    """
    Samples a timestep uniformly at random.
    :param num_diffusion_timesteps: The total number of diffusion timesteps (T).
    :return: A randomly sampled integer timestep t (from 0 to T-1).
    """
    return torch.randint(0, num_diffusion_timesteps, (1,)).item()

def default_gaussian_noise_fn(target_x0_shape: tuple, device: torch.device) -> torch.Tensor:
    """
    Generates standard Gaussian noise with the given shape and device.
    :param target_x0_shape: The shape of the target_x0 tensor (e.g., (num_frames, num_features)).
    :param device: The device to create the noise tensor on.
    :return: A noise tensor epsilon.
    """
    return torch.randn(target_x0_shape, device=device)

def default_ddpm_noising_fn(target_x0: torch.Tensor,
                             epsilon: torch.Tensor,
                             t: int,
                             alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """
    Applies noise to target_x0 according to the DDPM forward process formula.
    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * epsilon
    :param target_x0: The clean data tensor.
    :param epsilon: The noise tensor.
    :param t: The current timestep.
    :param alphas_cumprod: Tensor of cumulative products of alphas (should be on the same device as t is indexed on, usually CPU).
    :return: The noised data tensor x_t.
    """
    sqrt_alpha_bar_t = torch.sqrt(alphas_cumprod[t])
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alphas_cumprod[t])
    
    return sqrt_alpha_bar_t * target_x0 + sqrt_one_minus_alpha_bar_t * epsilon


def generate_and_save_sbert_embeddings(
    annotations_csv_path: str,      # Es. "./data/annotations_train.csv"
    sbert_model_name: str,          # Es. 'all-mpnet-base-v2'
    output_embeddings_path: str,    # Es. "./data/sbert_embeddings_train.pt"
    text_column_name: str = 'text',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Reads text prompts from an annotations CSV, computes their SBERT embeddings,
    and saves them to a file as a dictionary mapping original text to embedding.
    :param annotations_csv_path: Path to the CSV file containing text prompts.
    :param sbert_model_name: Name of the SBERT model to use (e.g., 'all-mpnet-base-v2').
    :param output_embeddings_path: Path to save the computed embeddings (as a .pt file).
    :param text_column_name: Name of the column in the CSV containing the text prompts.
    :param device: Device to load the SBERT model onto ('cuda' or 'cpu').
    :return: None
    """
    logger.info(f"Starting SBERT embedding pre-computation...")
    logger.info(f"Loading annotations from: {annotations_csv_path}")
    try:
        df = pd.read_csv(annotations_csv_path)
        if text_column_name not in df.columns:
            logger.error(f"Text column '{text_column_name}' not found in {annotations_csv_path}. Available columns: {df.columns.tolist()}")
            return
        
        # Extract unique text prompts from the specified column
        unique_texts = df[text_column_name].astype(str).unique().tolist()
        logger.info(f"Found {len(unique_texts)} unique text prompts to embed.")
        if not unique_texts:
            logger.warning("No text prompts found in annotations. Nothing to embed.")
            return

    except FileNotFoundError:
        logger.error(f"Annotations file not found: {annotations_csv_path}")
        return
    except Exception as e:
        logger.error(f"Error reading annotations file {annotations_csv_path}: {e}")
        return

    logger.info(f"Loading SBERT model: {sbert_model_name} onto device: {device}")
    try:
        sbert_model = SentenceTransformer(sbert_model_name, device=device)
    except Exception as e:
        logger.error(f"Failed to load SBERT model '{sbert_model_name}': {e}")
        return

    embeddings_dict = {}
    logger.info("Computing SBERT embeddings for unique texts...")
    for text in tqdm(unique_texts, desc="Embedding Texts"):
        try:
            embedding = sbert_model.encode(text, convert_to_tensor=True, show_progress_bar=False)
            embeddings_dict[text] = embedding.cpu() # Save on CPU to avoid CUDA memory issues
        except Exception as e:
            logger.warning(f"Could not compute embedding for text: '{text[:50]}...'. Error: {e}")
            embeddings_dict[text] = None

    # Filter out any texts that failed to compute embeddings
    final_embeddings_dict = {k: v for k, v in embeddings_dict.items() if v is not None}
    
    if not final_embeddings_dict:
        logger.error("No embeddings were successfully computed.")
        return

    logger.info(f"Successfully computed {len(final_embeddings_dict)} embeddings.")
    
    try:
        os.makedirs(os.path.dirname(output_embeddings_path), exist_ok=True)
        torch.save(final_embeddings_dict, output_embeddings_path)
        logger.info(f"SBERT embeddings saved to: {output_embeddings_path}")
    except Exception as e:
        logger.error(f"Failed to save SBERT embeddings to {output_embeddings_path}: {e}")