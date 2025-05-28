import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import os
import logging
from typing import Callable, Optional, List, Dict, Any
from dataset.dataset import TextToMotionDataset
from utils.diffusion_utils import default_uniform_timestep_sampler, default_gaussian_noise_fn, default_ddpm_noising_fn

logger = logging.getLogger(__name__)

# --- Default Utility Functions ---
def collate_motion_data(batch: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Collate function for motion data, handling padding for variable-length sequences.
    It filters out None items (samples that failed to load), pads tensor sequences
    to the maximum length in the batch, and creates a padding mask.

    :param batch: A list of dictionaries, where each dict is an output from 
                  MyTextToMotionDataset.__getitem__. Items can be None if 
                  __getitem__ failed for a particular sample.
    :return: A dictionary containing the collated batch data, with sequences
             padded and a 'motion_padding_mask' created. Returns an empty
             dictionary if the batch is empty after filtering None items.
    """
    # Filter out None items that might result from errors in __getitem__
    valid_batch_items = [item for item in batch if item is not None]

    if not valid_batch_items:
        logger.warning("collate_motion_data received an empty or all-None batch.")
        return {}

    collated_batch: Dict[str, Any] = {}
    first_item_keys = valid_batch_items[0].keys()

    for key in first_item_keys:
        if key == "text_conditions": 
            logger.warning("Collating 'text_conditions' (list of strings), expected 'text_embeddings' (tensor).")
            collated_batch[key] = [item[key] for item in valid_batch_items]
        elif torch.is_tensor(valid_batch_items[0][key]):
            if key in ["x_noisy", "target_x0"]:
                sequences = [item[key] for item in valid_batch_items]
                collated_batch[key] = pad_sequence(sequences, batch_first=True, padding_value=0.0)
            elif key == "text_embeddings":
                # Assuming text_embeddings are already tensors
                collated_batch[key] = torch.stack([item[key] for item in valid_batch_items])
            else:
                collated_batch[key] = torch.stack([item[key] for item in valid_batch_items])

    if "seq_len" in collated_batch and ("x_noisy" in collated_batch or "target_x0" in collated_batch):
        padded_sequence_key = "x_noisy" if "x_noisy" in collated_batch else "target_x0"
        max_len = collated_batch[padded_sequence_key].shape[1]
        
        seq_lengths_tensor = collated_batch["seq_len"]
        if not isinstance(seq_lengths_tensor, torch.Tensor):
            try:
                seq_lengths_tensor = torch.tensor(seq_lengths_tensor, dtype=torch.long)
            except TypeError:
                logger.error(f"seq_len in batch is not a tensor and could not be converted. Type: {type(seq_lengths_tensor)}")
    
        try:
            motion_padding_mask = torch.arange(max_len, device=seq_lengths_tensor.device).expand(len(valid_batch_items), max_len) >= seq_lengths_tensor.unsqueeze(1)
            collated_batch["motion_padding_mask"] = motion_padding_mask
        except Exception as e:
            logger.error(f"Error creating motion_padding_mask: {e}. Max_len: {max_len}, seq_lengths: {seq_lengths_tensor}", exc_info=True)
            raise

    if "seq_len" in collated_batch:
        del collated_batch["seq_len"]

    return collated_batch


# --- Motion Processing Function ---
def process_motion_file(file_path: str, num_expected_features: int) -> torch.Tensor:
    """
    Loads, preprocesses, and flattens a single .npz motion file.
    This should return the clean x0.
    :param file_path: Path to the .npz motion file.
    :param num_expected_features: Expected number of features after flattening (e.g., joints * coords).
    :return: Processed motion tensor x0 of shape (num_frames, num_motion_features).
    """
    try:
        joints_data = np.load(file_path)  # Expected (T, num_joints, 3)
    except Exception as e:
        logger.error(f"Error loading motion file {file_path}: {e}")
        raise
    
    # Example preprocessing: root centering
    if joints_data.ndim == 3 and joints_data.shape[2] == 3: # Basic check for (T, J, 3)
        joints_data = joints_data - joints_data[:, 0:1, :]  # Root centering
        # joints_data = joints_data / 1000.0  # Example: Scale mm to m if your data is in mm
                                           # Or apply other normalization (e.g., to [-1, 1])
    else:
        pass
        # Log a warning but proceed if possible, or raise an error if shape is critical
        #logger.warning(f"Unexpected data shape in {file_path}: {joints_data.shape}. Expected (T, J, 3). May affect preprocessing.")

    num_frames = joints_data.shape[0]
    try:
        # Flatten: (T, num_joints, 3) -> (T, num_joints * 3)
        processed_motion = joints_data.reshape(num_frames, -1)
    except Exception as e:
        logger.error(f"Error reshaping motion data from {file_path} (original shape: {joints_data.shape}): {e}")
        raise

    if processed_motion.shape[1] != num_expected_features:
        raise ValueError(f"Feature mismatch in {file_path}: expected {num_expected_features} features, got {processed_motion.shape[1]}.")

    return torch.from_numpy(processed_motion).float()


class MyTextToMotionDataset(TextToMotionDataset):
    """
    Concrete Dataset for loading text, motion (as x0), and armature IDs.
    Prepares data (x_noisy, target_x0, timesteps, conditions) for an x0-predicting diffusion model.
    Inherits from the abstract TextToMotionDataset.
    """
    def __init__(self,
                 root_dir: str, # Used as base for motion_dir and annotations_file if they are relative
                 annotations_file_name: str, # e.g., "annotations.csv"
                 motion_subdir: str,      # e.g., "motions_npy"
                 precomputed_sbert_embeddings_path: str, # Path to precomputed SBERT embeddings
                 num_diffusion_timesteps: int,
                 alphas_cumprod: torch.Tensor, # Precomputed alphas_cumprod (should be on data_device)
                 num_motion_features: int,
                 motion_processor_fn: Callable = process_motion_file,
                 min_seq_len: int = 10, # Minimum sequence length to accept
                 max_seq_len: Optional[int] = None, # Optional: max sequence length for truncation/filtering
                 timestep_sampler_fn: Callable = default_uniform_timestep_sampler,
                 noise_fn: Callable = default_gaussian_noise_fn,
                 noising_fn: Callable = default_ddpm_noising_fn,
                 data_device: str = 'cpu'
                ):
        """
        Initializes the dataset with paths, parameters, and customizable functions.
        :param root_dir: Root directory containing the dataset.
        :param annotations_file_name: Name of the CSV file with annotations (e.g., "annotations.csv").
        :param motion_subdir: Subdirectory containing motion files (e.g., "motions_npy").
        :param precomputed_sbert_embeddings_path: Path to precomputed SBERT embeddings for text conditions.
        :param num_diffusion_timesteps: Total number of diffusion timesteps (T).
        :param alphas_cumprod: Precomputed tensor of cumulative products of alphas (shape: (T,)).
        :param num_motion_features: Number of features per frame in the motion data (e.g., joints * coords).
        :param motion_processor_fn: Function to process motion files into tensors (default: process_motion_file).
        :param min_seq_len: Minimum sequence length to accept (default: 10).
        :param max_seq_len: Optional maximum sequence length for truncation (default: None, no truncation).
        :param timestep_sampler_fn: Function to sample a random timestep (default: default_uniform_timestep_sampler).
        :param noise_fn: Function to generate noise (default: default_gaussian_noise_fn).
        :param noising_fn: Function to apply noise to the clean motion data (default: default_ddpm_noising_fn).
        :param data_device: Device to use for data processing (default: 'cpu').
        """
        # Initialize paths
        self.annotations_path = os.path.join(root_dir, annotations_file_name)
        self.motion_dir_abs = os.path.join(root_dir, motion_subdir)
        
        # Store diffusion and processing parameters
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.data_device = torch.device(data_device)
        self.alphas_cumprod = alphas_cumprod.to(self.data_device)
        
        self.motion_processor = motion_processor_fn
        self.num_motion_features = num_motion_features
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        
        # Store customizable functions
        self.timestep_sampler = timestep_sampler_fn
        self.noise_generator = noise_fn
        self.apply_noise = noising_fn
        
        try:
            logger.info(f"Loading pre-computed SBERT embeddings from: {precomputed_sbert_embeddings_path}")
            self.sbert_embeddings_cache = torch.load(precomputed_sbert_embeddings_path, map_location=self.data_device)
            logger.info(f"Loaded {len(self.sbert_embeddings_cache)} pre-computed SBERT embeddings.")
        except FileNotFoundError:
            logger.error(f"Pre-computed SBERT embeddings file not found: {precomputed_sbert_embeddings_path}. "
                         "Please generate it first.")
            raise
        except Exception as e:
            logger.error(f"Error loading pre-computed SBERT embeddings from {precomputed_sbert_embeddings_path}: {e}")
            raise

        super().__init__(root_dir) 


    def _prepare_samples(self):
        """
        Prepares the list of samples by reading the annotations file.
        Each sample in self.samples will be a dictionary:
        {'motion_file_path': str, 'text_prompt': str, 'armature_id': int}
        This method is called by super().__init__.
        """
        self.samples = []
        try:
            annotations_df = pd.read_csv(self.annotations_path)
        except FileNotFoundError:
            logger.error(f"Annotations file not found: {self.annotations_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading annotations file {self.annotations_path}: {e}")
            raise

        for idx, row in annotations_df.iterrows():
            motion_filename = row['motion_filename']
            text_prompt = row['text']           
            armature_id = row['armature_id']    

            file_path = os.path.join(self.motion_dir_abs, motion_filename)
            if not os.path.exists(file_path):
                logger.warning(f"Motion file {file_path} for annotation '{motion_filename}' (row {idx}) not found. Skipping sample.")
                continue
            
            # Store a dictionary with all necessary info for __getitem__
            self.samples.append({
                'motion_file_path': file_path,
                'text_prompt': str(text_prompt),
                'armature_id': int(armature_id)
            })
        
        if not self.samples:
            raise ValueError(f"No valid samples loaded. Check annotations file '{self.annotations_path}' and motion directory '{self.motion_dir_abs}'.")
        logger.info(f"Prepared {len(self.samples)} samples from annotations.")

    def _load_animation(self, motion_file_path: str) -> torch.Tensor:
        """
        Loads and processes a motion file to get the clean x0 tensor.
        This overrides the abstract method.
        The 'animation' parameter from the abstract class's __getitem__ is interpreted
        as the motion_file_path here, passed from our concrete __getitem__.
        Return value is adapted: returns processed x0 tensor (motion features).
        :param motion_file_path: Path to the motion file (e.g., .npz).
        :return: Processed motion tensor x0 of shape (num_frames, num_motion_features).
        """
        return self.motion_processor(motion_file_path, num_expected_features=self.num_motion_features)

    def _load_text(self, text_prompt_string: str) -> str:
        """
        "Loads" text data by simply returning the provided string.
        This overrides the abstract method.
        The 'text' parameter from the abstract class's __getitem__ is interpreted
        as the text_prompt_string here.
        :param text_prompt_string: The text prompt string to be used as a condition.
        :return: The text prompt string as is.
        """
        return text_prompt_string

    def _load_action_descriptions(self) -> Dict[str, str]:
        """
        Loads action descriptions. Currently returns an empty dictionary.
        This can be expanded if action descriptions (mapping action names to text) are needed.
        ArmatureMDM currently uses armature_class_ids directly.
        This method is called by super().__init__.
        """
        logger.info("Action descriptions are not actively loaded by this dataset version (using armature_class_id).")
        return {} # Return empty dict as per abstract class if no specific loading is done

    def __len__(self):
        # This is implemented by the abstract TextToMotionDataset using self.samples
        return super().__len__()

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Fetches a single data sample, processes it for diffusion model training.
        This method overrides the one in the abstract class to return a complete dictionary.
        :param idx: Index of the sample to fetch.
        :return: A dictionary containing:
            - 'x_noisy': Noisy motion tensor (shape: (num_frames, num_motion_features))
        """
        # Simple retry logic: if a sample fails, try the next one.
        # For robustness, pre-filtering in _prepare_samples is better.
        for i in range(len(self.samples)): 
            current_idx = (idx + i) % len(self.samples) # Wrap around
            sample_info = self.samples[current_idx]
            
            try:
                # 1. Load clean x0 motion data using the overridden _load_animation
                # _load_animation uses self.motion_processor which should return a tensor
                target_x0 = self._load_animation(sample_info['motion_file_path'])
                target_x0 = target_x0.to(self.data_device) # Ensure it's on data_device

                num_frames = target_x0.shape[0]

                # 2. Filter by sequence length / truncate
                if num_frames < self.min_seq_len:
                    if i == len(self.samples) -1 : logger.debug(f"Last attempt for initial index {idx} failed min_seq_len check.")
                    continue # Try next sample
                
                if self.max_seq_len is not None and num_frames > self.max_seq_len:
                    target_x0 = target_x0[:self.max_seq_len] # Truncate (example: from start)
                    num_frames = self.max_seq_len
                
                # 3. "Load" text condition (already a string from sample_info)
                text_condition = self._load_text(sample_info['text_prompt'])
                
                # 4. Get armature ID
                armature_id = sample_info['armature_id']

                if text_condition not in self.sbert_embeddings_cache:
                    logger.warning(f"Pre-computed SBERT embedding not found for text: '{text_condition[:50]}...'. Skipping sample.")
                    if i == len(self.samples) -1: return None
                    continue
                
                sbert_embedding = self.sbert_embeddings_cache[text_condition].to(self.data_device)

                # 5. Sample a random timestep t using the customizable sampler
                t = self.timestep_sampler(self.num_diffusion_timesteps)

                # 6. Generate noise epsilon using the customizable noise generator
                epsilon = self.noise_generator(target_x0.shape, device=self.data_device)

                # 7. Calculate x_t (noisy motion) using the customizable noising function
                x_noisy = self.apply_noise(target_x0, epsilon, t, self.alphas_cumprod)
                
                return {
                    "x_noisy": x_noisy.float(),
                    "target_x0": target_x0.float(),
                    "timesteps": torch.tensor(t, dtype=torch.long, device=self.data_device),
                    "text_embeddings": sbert_embedding.float(),
                    "armature_class_ids": torch.tensor(armature_id, dtype=torch.long, device=self.data_device),
                    "seq_len": torch.tensor(num_frames, dtype=torch.long, device=self.data_device) # For padding mask by collate_fn
                }
            except Exception as e:
                if i == len(self.samples) -1: # If this was the last attempt
                    logger.error(f"All attempts failed for initial index {idx}. Could not fetch a valid sample.")
                    return None # Indicate failure to collate_fn
        
        return None # Should not be reached if self.samples is not empty

