"""
Dataset and DataLoader for Text-to-Motion data with armature conditioning.
Handles loading motion data, text embeddings, armature information,
applies normalization, and prepares batches for training the ArmatureMDM.
"""
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
# from torch.nn.utils.rnn import pad_sequence # Not directly used in revised collate if padding manually

logger = logging.getLogger(__name__)

# --- Default Utility Functions for Diffusion Data Preparation ---
def default_uniform_timestep_sampler(num_diffusion_timesteps: int) -> int:
    """
    Samples a timestep uniformly at random from [0, num_diffusion_timesteps - 1].

    :param int num_diffusion_timesteps: The total number of diffusion timesteps (T).
    :return: int: A randomly sampled integer timestep t.
    """
    return torch.randint(0, num_diffusion_timesteps, (1,)).item()

def default_gaussian_noise_fn(target_x0_shape: tuple, device: torch.device) -> torch.Tensor:
    """
    Generates standard Gaussian noise tensor with the given shape and device.

    :param tuple target_x0_shape: The shape of the target_x0 tensor (e.g., (num_frames, num_features)).
    :param torch.device device: The device to create the noise tensor on.
    :return: torch.Tensor: A noise tensor epsilon.
    """
    return torch.randn(target_x0_shape, device=device)

def default_ddpm_noising_fn(target_x0: torch.Tensor,
                             epsilon: torch.Tensor,
                             t: int,
                             alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """
    Applies noise to target_x0 according to the DDPM forward process formula:
    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * epsilon.

    :param torch.Tensor target_x0: The clean data tensor.
    :param torch.Tensor epsilon: The noise tensor.
    :param int t: The current timestep (integer index).
    :param torch.Tensor alphas_cumprod: 1D Tensor of cumulative products of alphas (alpha_bar_t).
                                        Expected to be on the same device as target_x0 and epsilon.
    :return: torch.Tensor: The noised data tensor x_t.
    """
    def _extract_into_tensor(arr_tensor: torch.Tensor, timestep_int: int, broadcast_shape: tuple) -> torch.Tensor:
        """ Helper to extract a coefficient for a given timestep and broadcast it. """
        res = arr_tensor[timestep_int].float() # Indexing with int
        temp_res = res
        while len(temp_res.shape) < len(broadcast_shape):
            temp_res = temp_res[..., None]
        return temp_res.expand(broadcast_shape)

    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod)

    coeff1 = _extract_into_tensor(sqrt_alphas_cumprod_t, t, target_x0.shape)
    coeff2 = _extract_into_tensor(sqrt_one_minus_alphas_cumprod_t, t, target_x0.shape)
    
    return coeff1 * target_x0 + coeff2 * epsilon


def process_motion_file_with_normalization(
    file_path: str,
    num_expected_features: int,
    dataset_mean: np.ndarray,
    dataset_std: np.ndarray
) -> torch.Tensor:
    """
    Loads a single .npy motion file, flattens it per frame, and applies Z-score normalization.

    :param str file_path: Absolute path to the .npy motion file.
    :param int num_expected_features: The total number of features expected per frame after flattening.
    :param numpy.ndarray dataset_mean: The mean of the training dataset for each feature (shape [num_expected_features]).
    :param numpy.ndarray dataset_std: The standard deviation of the training dataset for each feature (shape [num_expected_features]).
    :return: torch.Tensor: The processed and normalized motion tensor (x0), shape [num_frames, num_expected_features].
    :raises FileNotFoundError: If the motion file does not exist.
    :raises ValueError: If motion data shape is unexpected or feature mismatch occurs.
    """
    if not os.path.exists(file_path):
        logger.error(f"Motion file not found during processing: {file_path}")
        raise FileNotFoundError(f"Motion file not found: {file_path}")
    try:
        motion_data_raw = np.load(file_path).astype(np.float32)
    except Exception as e:
        logger.error(f"Error loading motion file {file_path}: {e}")
        raise

    if motion_data_raw.ndim == 3: # e.g., [num_frames, num_joints, 3_xyz]
        # Optional: Root centering can be done here if desired before flattening
        # motion_data_raw = motion_data_raw - motion_data_raw[:, 0:1, :]
        motion_data_flat = motion_data_raw.reshape(motion_data_raw.shape[0], -1)
        if motion_data_flat.shape[1] != num_expected_features:
            raise ValueError(f"Feature mismatch in {file_path} after reshaping 3D data: "
                             f"expected {num_expected_features}, got {motion_data_flat.shape[1]}. "
                             f"Original shape {motion_data_raw.shape}.")
    elif motion_data_raw.ndim == 2 and motion_data_raw.shape[1] == num_expected_features:
        motion_data_flat = motion_data_raw # Already flat
    else:
        raise ValueError(f"Unexpected data shape in {file_path}: {motion_data_raw.shape}. "
                         f"Expected 3D (T, J, C) that flattens to (T, num_expected_features) "
                         f"or 2D (T, num_expected_features).")

    # Apply Z-score normalization
    std_safe = np.where(dataset_std == 0, 1e-8, dataset_std) # Avoid division by zero
    normalized_motion = (motion_data_flat - dataset_mean) / std_safe
    
    return torch.from_numpy(normalized_motion).float()


class MyTextToMotionDataset(Dataset):
    """
    Dataset for loading text prompts, corresponding motion data (as normalized x0),
    and armature IDs. It prepares data (x_noisy, target_x0, timesteps, conditions)
    for an x0-predicting diffusion model.
    """
    def __init__(self,
                 root_dir: str,
                 annotations_file_name: str,
                 motion_subdir: str,
                 precomputed_sbert_embeddings_path: str,
                 num_diffusion_timesteps: int,
                 alphas_cumprod: torch.Tensor,
                 num_motion_features: int,
                 dataset_mean_path: str,
                 dataset_std_path: str,
                 min_seq_len: int = 10,
                 max_seq_len: Optional[int] = 120,
                 timestep_sampler_fn: Callable = default_uniform_timestep_sampler,
                 noise_fn: Callable = default_gaussian_noise_fn,
                 noising_fn: Callable = default_ddpm_noising_fn,
                 data_device: str = 'cpu'):
        """
        Initializes the dataset.

        :param str root_dir: Root directory of the dataset.
        :param str annotations_file_name: Name of the CSV file containing annotations.
        :param str motion_subdir: Subdirectory within root_dir containing motion files (.npy).
        :param str precomputed_sbert_embeddings_path: Path to the .pt file of precomputed SBERT embeddings.
        :param int num_diffusion_timesteps: Total number of diffusion timesteps (T).
        :param torch.Tensor alphas_cumprod: Precomputed tensor of cumulative products of alphas (alpha_bar_t),
                                            shape [T], expected on data_device.
        :param int num_motion_features: Number of features per frame in the motion data.
        :param str dataset_mean_path: Path to the .npy file containing the dataset mean for normalization.
        :param str dataset_std_path: Path to the .npy file containing the dataset std for normalization.
        :param int min_seq_len: Minimum sequence length to accept. Shorter sequences are skipped.
        :param Optional[int] max_seq_len: Maximum sequence length. Longer sequences are truncated.
                                          If None, no truncation by max_seq_len is performed in __getitem__,
                                          relying on collate_fn for batch-level padding.
        :param Callable timestep_sampler_fn: Function to sample a timestep t.
        :param Callable noise_fn: Function to generate noise epsilon.
        :param Callable noising_fn: Function to apply noise (x0, eps, t, alphas_cumprod) -> xt.
        :param str data_device: Device to load/process data onto ('cpu' or 'cuda:X').
        """
        self.root_dir = Path(root_dir)
        self.annotations_path = self.root_dir / annotations_file_name
        self.motion_dir_abs = self.root_dir / motion_subdir
        
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.data_device = torch.device(data_device)
        self.alphas_cumprod = alphas_cumprod.to(self.data_device)
        
        self.num_motion_features = num_motion_features
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len if max_seq_len is not None else float('inf')
        
        self.timestep_sampler = timestep_sampler_fn
        self.noise_generator = noise_fn
        self.apply_noise = noising_fn
        
        try:
            logger.info(f"Loading precomputed SBERT embeddings from: {precomputed_sbert_embeddings_path}")
            self.sbert_embeddings_cache = torch.load(precomputed_sbert_embeddings_path, map_location=self.data_device)
            logger.info(f"Loaded {len(self.sbert_embeddings_cache)} SBERT embeddings.")
        except Exception as e:
            logger.error(f"Error loading SBERT embeddings from {precomputed_sbert_embeddings_path}: {e}", exc_info=True)
            raise

        try:
            self.dataset_mean = np.load(dataset_mean_path).astype(np.float32)
            self.dataset_std = np.load(dataset_std_path).astype(np.float32)
            if self.dataset_mean.shape != (self.num_motion_features,) or \
               self.dataset_std.shape != (self.num_motion_features,):
                raise ValueError(f"Shape of mean ({self.dataset_mean.shape}) or std ({self.dataset_std.shape}) "
                                 f"does not match num_motion_features ({self.num_motion_features}).")
            logger.info("Dataset mean and std for Z-score normalization loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading dataset mean/std from {dataset_mean_path}/{dataset_std_path}: {e}", exc_info=True)
            raise
            
        self._prepare_samples()

    def _prepare_samples(self):
        """Loads and filters samples from the annotations file."""
        self.samples: List[Dict[str, Any]] = []
        try:
            annotations_df = pd.read_csv(self.annotations_path)
        except Exception as e:
            logger.error(f"Error reading annotations file {self.annotations_path}: {e}", exc_info=True)
            raise

        for idx, row in annotations_df.iterrows():
            motion_filename = row.get('motion_filename')
            text_prompt = str(row.get('text', ''))
            armature_id = int(row.get('armature_id', -1)) # Use a default if not present

            if not motion_filename or not text_prompt or armature_id == -1:
                logger.warning(f"Skipping row {idx} due to missing motion_filename, text, or armature_id.")
                continue
                
            file_path = self.motion_dir_abs / motion_filename
            if not file_path.exists():
                logger.warning(f"Motion file {file_path} not found for row {idx}. Skipping sample.")
                continue
            if text_prompt not in self.sbert_embeddings_cache:
                logger.warning(f"Precomputed SBERT embedding not found for text: '{text_prompt[:50]}...' (row {idx}). Skipping sample.")
                continue
            
            # Optionally, pre-check motion length here if feasible, or rely on __getitem__
            # For example, by loading a small part of the .npy or storing lengths in annotations.
            # This can speed up _prepare_samples if many files are too short.

            self.samples.append({
                'motion_file_path': str(file_path),
                'text_prompt': text_prompt,
                'armature_id': armature_id
            })
        
        if not self.samples:
            raise ValueError(f"No valid samples loaded. Check annotations file '{self.annotations_path}', "
                             f"motion directory '{self.motion_dir_abs}', and SBERT cache.")
        logger.info(f"Prepared {len(self.samples)} valid samples from annotations.")

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Fetches, processes, and noises a single data sample for diffusion model training.

        :param int idx: Index of the sample to fetch.
        :return: Optional[Dict[str, Any]]: A dictionary containing the processed sample,
                 or None if processing fails after multiple attempts.
                 Keys: "target_x0", "x_noisy", "timesteps", "text_embeddings",
                       "armature_class_ids", "motion_actual_length".
        """
        max_attempts = 3 # Number of attempts to get a valid sample if current one fails
        for attempt in range(max_attempts):
            current_idx = (idx + attempt) % len(self.samples)
            sample_info = self.samples[current_idx]
            try:
                target_x0_normalized = process_motion_file_with_normalization(
                    sample_info['motion_file_path'],
                    self.num_motion_features,
                    self.dataset_mean,
                    self.dataset_std
                ).to(self.data_device)

                num_frames = target_x0_normalized.shape[0]

                if num_frames < self.min_seq_len:
                    if attempt == max_attempts - 1:
                        logger.debug(f"Last attempt for initial index {idx} failed min_seq_len check for {sample_info['motion_file_path']}.")
                    continue # Try next sample
                
                current_processed_len = num_frames
                if num_frames > self.max_seq_len:
                    # Example: truncate from the start, you might want random crop for augmentation
                    target_x0_final = target_x0_normalized[:self.max_seq_len]
                    current_processed_len = self.max_seq_len
                else:
                    target_x0_final = target_x0_normalized
                
                sbert_embedding = self.sbert_embeddings_cache[sample_info['text_prompt']].to(self.data_device)
                armature_id = sample_info['armature_id']
                
                t = self.timestep_sampler(self.num_diffusion_timesteps)
                epsilon = self.noise_generator(target_x0_final.shape, device=self.data_device)
                x_noisy = self.apply_noise(target_x0_final, epsilon, t, self.alphas_cumprod)
                
                return {
                    "target_x0": target_x0_final.float(),
                    "x_noisy": x_noisy.float(),
                    "timesteps": torch.tensor(t, dtype=torch.long, device=self.data_device),
                    "text_embeddings": sbert_embedding.float(),
                    "armature_class_ids": torch.tensor(armature_id, dtype=torch.long, device=self.data_device),
                    "motion_actual_length": torch.tensor(current_processed_len, dtype=torch.long, device=self.data_device)
                }
            except Exception as e:
                logger.warning(f"Error processing sample '{sample_info.get('motion_file_path', 'N/A')}' "
                               f"(index {current_idx}, attempt {attempt+1}): {e}", exc_info=False)
                if attempt == max_attempts - 1: # Last attempt
                    logger.error(f"All {max_attempts} attempts failed for initial index {idx}. Cannot fetch a valid sample.")
                    return None
        return None # Should only be reached if max_attempts is 0 or len(self.samples) is 0 (caught by __init__)


def collate_motion_data_revised(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """
    Collates a list of dataset items into a single batch dictionary.
    Filters out None items, pads sequences to the maximum length in the batch,
    and creates a validity mask (True for valid frames).

    :param List[Optional[Dict[str, Any]]] batch: A list of dictionaries from MyTextToMotionDatasetNormalized.__getitem__.
    :return: Optional[Dict[str, Any]]: A dictionary containing the collated batch data, or None if the batch is empty.
             The "motion_padding_mask_for_loss" is [bs, 1, 1, max_len_in_batch], True for VALID frames.
    """
    valid_batch_items = [item for item in batch if item is not None]
    if not valid_batch_items:
        logger.warning("collate_motion_data_revised received an empty or all-None batch.")
        return None

    collated_batch: Dict[str, Any] = {}
    # Define keys for different handling
    keys_to_pad_if_present = ["target_x0", "x_noisy"] # These are sequences
    keys_to_stack = ["timesteps", "text_embeddings", "armature_class_ids", "motion_actual_length"]
    
    # Determine max_len_in_batch from 'motion_actual_length' of valid items for sequence keys
    # This assumes 'motion_actual_length' is present and correct if sequence keys are present.
    actual_lengths = torch.stack([item["motion_actual_length"] for item in valid_batch_items])
    max_len_in_batch = actual_lengths.max().item()
    collated_batch["motion_actual_length"] = actual_lengths # Store the stacked actual lengths

    for key in valid_batch_items[0].keys():
        if key in keys_to_pad_if_present:
            sequences = [item[key] for item in valid_batch_items]
            padded_sequences = []
            for seq in sequences:
                # Pad sequences shorter than max_len_in_batch
                # Sequences should already be truncated to dataset's max_seq_len by __getitem__
                len_seq = seq.shape[0]
                if len_seq < max_len_in_batch:
                    padding_needed = max_len_in_batch - len_seq
                    padding_tensor = torch.zeros(padding_needed, seq.shape[1], dtype=seq.dtype, device=seq.device)
                    padded_seq = torch.cat([seq, padding_tensor], dim=0)
                elif len_seq > max_len_in_batch: # Should ideally not happen if __getitem__ truncates
                    padded_seq = seq[:max_len_in_batch]
                else:
                    padded_seq = seq
                padded_sequences.append(padded_seq)
            collated_batch[key] = torch.stack(padded_sequences)
        
        elif key in keys_to_stack and key != "motion_actual_length": # motion_actual_length already handled
            tensor_list = [item[key] for item in valid_batch_items]
            # Ensure all items are tensors before stacking
            if not all(isinstance(t, torch.Tensor) for t in tensor_list):
                try:
                    tensor_list = [torch.tensor(t) if not isinstance(t, torch.Tensor) else t for t in tensor_list]
                except Exception as e:
                    logger.error(f"Failed to convert items in '{key}' to tensors for stacking: {e}")
                    # Decide handling: skip key, return None, or raise error
                    # For now, let it raise if torch.stack fails
            collated_batch[key] = torch.stack(tensor_list)

        elif key == "text_prompt": # Keep text prompts if present (e.g., for debugging)
            collated_batch[key] = [item[key] for item in valid_batch_items]
        # other keys not explicitly handled will be omitted unless added here

    # Create validity mask (True for VALID frames, False for PADDED frames)
    # Shape: [bs, 1, 1, max_len_in_batch] as expected by the trainer for y_cond['mask']
    arange_tensor = torch.arange(max_len_in_batch, device=actual_lengths.device)
    # motion_validity_mask is [bs, max_len_in_batch]
    motion_validity_mask = arange_tensor.expand(len(valid_batch_items), max_len_in_batch) < actual_lengths.unsqueeze(1)
    collated_batch["motion_padding_mask_for_loss"] = motion_validity_mask.unsqueeze(1).unsqueeze(1) 
    
    return collated_batch