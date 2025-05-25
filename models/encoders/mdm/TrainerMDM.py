import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Callable, Literal, Optional, Tuple 
import logging
from ArmatureMDM import ArmatureMDM

logger = logging.getLogger(__name__)

class MotionDataset(Dataset):
    def __init__(self, motion_files, texts, armature_ids, num_diffusion_timesteps, alphas_cumprod, motion_processor_fn, tokenizer_fn=None):
        """
        Conceptual Dataset for motion data.
        :param motion_files: List of paths to .npy motion files.
        :param texts: List of corresponding text prompts.
        :param armature_ids: List of corresponding armature class IDs.
        :param num_diffusion_timesteps: Total number of diffusion timesteps (T).
        :param alphas_cumprod: Cumulative products of (1-beta_t), used for noising.
        :param motion_processor_fn: Function to load and preprocess a single motion file into clean x0 tensor.
        :param tokenizer_fn: Optional function to tokenize text (if not handled by SBERT inside model).
        """
        self.motion_files = motion_files
        self.texts = texts
        self.armature_ids = armature_ids
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.alphas_cumprod = alphas_cumprod # Shape [T]
        self.motion_processor = motion_processor_fn
        # self.tokenizer = tokenizer_fn # If you tokenize text here

        assert len(self.motion_files) == len(self.texts) == len(self.armature_ids), \
            "Mismatch in lengths of inputs to Dataset."

    def __len__(self):
        return len(self.motion_files)

    def __getitem__(self, idx):
        motion_path = self.motion_files[idx]
        text_condition = self.texts[idx]
        armature_id = self.armature_ids[idx]

        # 1. Load and preprocess motion to get clean x0
        # target_x0 shape: (num_frames, num_motion_features)
        target_x0 = self.motion_processor(motion_path)
        num_frames = target_x0.shape[0]

        # 2. Sample a random timestep t
        # t is an integer from 0 to T-1 (or 1 to T depending on indexing)
        # For DDPM, t is usually sampled from [0, T-1] for training.
        t = torch.randint(0, self.num_diffusion_timesteps, (1,)).item() # A single integer timestep

        # 3. Generate noise epsilon
        epsilon = torch.randn_like(target_x0)

        # 4. Calculate x_t (noisy motion) using x0, epsilon, and t
        # x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * epsilon
        # Ensure alphas_cumprod is a tensor on the correct device
        sqrt_alpha_bar_t = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - self.alphas_cumprod[t])
        
        x_noisy = sqrt_alpha_bar_t * target_x0 + sqrt_one_minus_alpha_bar_t * epsilon
        
        # For padding (conceptual, you'll need a proper collate_fn in DataLoader for this)
        # motion_padding_mask = torch.ones(num_frames, dtype=torch.bool)

        return {
            "x_noisy": x_noisy.float(),  # Noisy input for the model
            "target_x0": target_x0.float(), # Clean target for the x0-prediction model
            "timesteps": torch.tensor(t, dtype=torch.long), # Timestep for the model
            "text_conditions": text_condition, # Raw text string
            "armature_class_ids": torch.tensor(armature_id, dtype=torch.long),
            # "motion_padding_mask": motion_padding_mask # If using padding
            # "target_noise": epsilon.float() # Not directly needed if model predicts x0
        }

# --- Noise Scheduler Example (Conceptual - you need your own) ---
def get_noise_schedule(num_diffusion_timesteps=1000, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    return betas, alphas, alphas_cumprod

# --- Motion Processor Example (Conceptual) ---
def process_motion_file(file_path, num_joints=22, num_coords=3):
    # This should load your .npy file, normalize, flatten etc.
    # The script you provided for visualization does:
    # joints = np.load(file_path) # (T, num_joints, 3)
    # joints = joints - joints[:, 0:1, :] # Root centering
    # joints_m = joints / 1000.0 # Scale (if needed, or ensure data is already scaled)
    # # Further normalization (e.g., to [-1, 1]) might be good.
    # # Flatten:
    # target_x0_np = joints_m.reshape(joints_m.shape[0], -1) # (T, num_joints * num_coords)
    # return torch.from_numpy(target_x0_np)
    # --- This is a placeholder - implement your full preprocessing ---
    print(f"Warning: process_motion_file is a placeholder. Implement actual data loading for {file_path}")
    # Example: return a dummy tensor of shape (sequence_length, num_features=66)
    sequence_length = 100 # Placeholder
    num_features = num_joints * num_coords
    return torch.randn(sequence_length, num_features)


class KinematicLossCalculator:
    """
    Class to compute kinematic losses (velocity, acceleration) on the predicted x0 from the ArmatureMDM model.
    This aligns with the MDM paper's approach of applying geometric losses to the predicted sample.
    The losses are computed on the predicted x0 directly, rather than deriving it from noise.
    This class is initialized with a function to get the bone mask, which is used to focus the loss on active features.
    """

    def __init__(self,
                    get_bone_mask_fn: Callable[[torch.Tensor, int, int, str], torch.Tensor],
                    device: str,
                    use_velocity_loss: bool = True,
                    velocity_loss_weight: float = 0.1,
                    use_acceleration_loss: bool = True,
                    acceleration_loss_weight: float = 0.1,
                    kinematic_loss_type: Literal["l1", "mse"] = "l1"
                ):
        """
        Initializes the KinematicLossCalculator with the necessary parameters.
        :param get_bone_mask_fn: Function to get the bone mask for active features.
        :param device: Device to run the calculations on (e.g., "cuda" or "cpu").
        :param use_velocity_loss: Whether to compute the velocity loss.
        :param velocity_loss_weight: Weight for the velocity loss.
        :param use_acceleration_loss: Whether to compute the acceleration loss.
        :param acceleration_loss_weight: Weight for the acceleration loss.
        :param kinematic_loss_type: Type of loss to use for kinematic losses ('l1' or 'mse').
        """
        self.get_bone_mask_fn = get_bone_mask_fn
        self.device = device
        self.use_velocity_loss = use_velocity_loss
        self.velocity_loss_weight = velocity_loss_weight
        self.use_acceleration_loss = use_acceleration_loss
        self.acceleration_loss_weight = acceleration_loss_weight
        self.kinematic_loss_type = kinematic_loss_type

        if not self.use_velocity_loss and not self.use_acceleration_loss:
            logger.warning("KinematicLossCalculator initialized but both velocity and acceleration losses are disabled.")


    def _calculate_derivative(self, motion_sequence: torch.Tensor) -> torch.Tensor:
        """
        Calculates the derivative of the motion sequence.
        :param motion_sequence: Tensor of shape [batch_size, seq_len, num_features].
        :return: Tensor of shape [batch_size, seq_len-1, num_features] representing the derivative.
        """
        if motion_sequence.shape[1] < 2: # Check if sequence is long enough for derivative
            # Return empty tensor with correct batch and feature dims, but 0 seq_len
            return torch.empty_like(motion_sequence[:, :0, :]) 
        return motion_sequence[:, 1:] - motion_sequence[:, :-1]


    def _calculate_masked_feature_loss(self, 
                                      prediction: torch.Tensor, 
                                      target: torch.Tensor, 
                                      mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates the masked feature loss between prediction and target.
        :param prediction: Tensor of predicted features [batch_size, seq_len, num_features].
        :param target: Tensor of target features [batch_size, seq_len, num_features].
        :param mask: Tensor of shape [batch_size, seq_len, num_features] indicating active features.
        :return: Masked loss tensor.
        """
        if prediction.shape[0] == 0 or prediction.shape[1] == 0: # If prediction is empty (e.g. from derivative on short seq)
            return torch.tensor(0.0, device=self.device)

        min_frames = prediction.shape[1] # prediction is already a derivative (e.g., velocity)
        # Target and mask should be for the same sequence length as the prediction
        target_adjusted = target[:, :min_frames, :]
        mask_adjusted = mask[:, :min_frames, :] # Mask also needs to match derivative length

        if self.kinematic_loss_type == "l1":
            loss = F.l1_loss(prediction, target_adjusted, reduction='none')
        elif self.kinematic_loss_type == "mse":
            loss = F.mse_loss(prediction, target_adjusted, reduction='none')
        else:
            raise ValueError(f"Unsupported kinematic_loss_type: {self.kinematic_loss_type}")

        masked_loss = loss * mask_adjusted
        num_active = mask_adjusted.sum()
        return masked_loss.sum() / (num_active + 1e-8) if num_active > 0 else torch.tensor(0.0, device=self.device)


    def compute_losses(self,
                       x_t: torch.Tensor,
                       timesteps: torch.Tensor,
                       predicted_x0_from_model: torch.Tensor, # Direct x0 output from the main model
                       target_x0: torch.Tensor, # Ground truth clean motion (for target derivatives)
                       scheduler_params: Dict[str, torch.Tensor],
                       armature_class_ids: torch.Tensor,
                       is_training_model: bool 
                    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes kinematic losses (velocity, acceleration) directly on the predicted_x0_from_model.
        This aligns with MDM paper's approach of applying geometric losses to the predicted sample.
        :param x_t: Original noisy input (may be less relevant now).
        :param timesteps: Original timesteps (may be less relevant now).
        :param predicted_x0_from_model: Direct output from the ArmatureMDM model (predicted clean motion).
        :param target_x0: Ground truth clean motion (used for target derivatives).
        :param scheduler_params: Scheduler parameters (may be less relevant now).
        :param armature_class_ids: Tensor of armature class IDs for the batch.
        :param is_training_model: Boolean indicating if the model is in training mode.
        :return: Tuple (total_kinematic_loss, losses_dict).
        """
        if not (self.use_velocity_loss or self.use_acceleration_loss):
            # Return 0 loss, ensuring requires_grad is set correctly if in training model
            return torch.tensor(0.0, device=self.device, requires_grad=is_training_model), {}

        # The predicted_x0_from_model IS the clean motion prediction. No need to derive it.
        # current_predicted_clean_motion = self._predict_x0(x_t, predicted_noise, ...) # REMOVED
        current_predicted_clean_motion = predicted_x0_from_model 

        bs, num_frames_x0, num_total_motion_features = current_predicted_clean_motion.shape

        # The bone_mask should correspond to the full x0 sequence length
        bone_mask_for_x0_derivatives = self.get_bone_mask_fn(
            armature_class_ids, num_total_motion_features, num_frames_x0, self.device
        )

        total_kin_loss = torch.tensor(0.0, device=self.device) # requires_grad will propagate
        losses_dict = {}

        # --- Velocity Loss ---
        # Calculated on current_predicted_clean_motion and target_x0
        if self.use_velocity_loss and num_frames_x0 > 1:
            pred_vel = self._calculate_derivative(current_predicted_clean_motion)
            # target_x0 has the same num_frames as current_predicted_clean_motion
            target_vel = self._calculate_derivative(target_x0) 

            if pred_vel.shape[1] > 0: # If derivative resulted in a non-empty sequence
                # Mask for velocity is for a sequence of length num_frames_x0 - 1
                bone_mask_vel = bone_mask_for_x0_derivatives[:, :pred_vel.shape[1], :]
                vel_loss = self._calculate_masked_feature_loss(pred_vel, target_vel, bone_mask_vel)
                total_kin_loss += self.velocity_loss_weight * vel_loss
                losses_dict['velocity_loss'] = vel_loss.item()

        # --- Acceleration Loss ---
        # Calculated on current_predicted_clean_motion and target_x0
        if self.use_acceleration_loss and num_frames_x0 > 2:
            # Recompute pred_vel and target_vel if not already available or if pred_vel was empty
            if not (self.use_velocity_loss and num_frames_x0 > 1 and 'pred_vel' in locals() and pred_vel.shape[1] > 0):
                # Need to ensure 'pred_vel' and 'target_vel' from above block are accessible or recompute
                current_pred_vel_for_accel = self._calculate_derivative(current_predicted_clean_motion)
                current_target_vel_for_accel = self._calculate_derivative(target_x0)
            else: # Velocities already computed and valid
                current_pred_vel_for_accel = pred_vel
                current_target_vel_for_accel = target_vel

            if current_pred_vel_for_accel.shape[1] > 0: # Check if velocity calculation was valid
                pred_accel = self._calculate_derivative(current_pred_vel_for_accel)
                target_accel = self._calculate_derivative(current_target_vel_for_accel)

                if pred_accel.shape[1] > 0: # If acceleration derivative resulted in non-empty sequence
                    # Mask for acceleration is for a sequence of length num_frames_x0 - 2
                    bone_mask_accel = bone_mask_for_x0_derivatives[:, :pred_accel.shape[1], :]
                    accel_loss = self._calculate_masked_feature_loss(pred_accel, target_accel, bone_mask_accel)
                    total_kin_loss += self.acceleration_loss_weight * accel_loss
                    losses_dict['acceleration_loss'] = accel_loss.item()

        # Ensure requires_grad is properly set for the total_kin_loss if it's zero
        # and the model is in training mode (so backpropagation doesn't break if this is the only loss)
        if is_training_model and not total_kin_loss.requires_grad and total_kin_loss.is_leaf :
             # If total_kin_loss is a zero scalar and it's a leaf, clone and set requires_grad
             # This can happen if all weightings are zero or all sub-losses are zero.
             if total_kin_loss == 0.0:
                 total_kin_loss = total_kin_loss.clone().requires_grad_(True)

        return total_kin_loss, losses_dict


class ArmatureMDMTrainer:
    """
    Trainer class for the ArmatureMDM model, handling training and evaluation.
    This class manages the training loop, loss calculations, and optional kinematic losses.
    It uses a provided function to get the bone mask for active features, allowing for flexible training setups.
    """

    def __init__(self,
                    model: ArmatureMDM,
                    optimizer: optim.Optimizer,
                    get_bone_mask_fn: Callable[[torch.Tensor, int, int, str], torch.Tensor],
                    device: str = "cuda" if torch.cuda.is_available() else "cpu",
                    lr_scheduler: Optional[Any] = None,
                    main_loss_type: Literal["l1", "mse"] = "mse",  # for x0 prediction.
                    cfg_drop_prob: float = 0.1, # Probability of dropping conditions for CFG training
                    kinematic_loss_calculator: Optional[KinematicLossCalculator] = None,
                    early_stopping_patience: int = 10,
                    early_stopping_min_delta: float = 0.01,
                    model_save_path: Optional[str] = 'armature_mdm_best_model.pth'  # Path to save the best model
                ):
        """
        Initializes the ArmatureMDMTrainer with the model, optimizer, and other parameters.
        :param model: Instance of ArmatureMDM model.
        :param optimizer: Optimizer for training the model.
        :param get_bone_mask_fn: Function to get bone mask for active features.
        :param device: Device to run the model on (default: "cuda" if available).
        :param lr_scheduler: Optional learning rate scheduler.
        :param main_loss_type: Type of loss for x0 prediction ('l1' or 'mse').
        :param cfg_drop_prob: Probability of dropping conditions for CFG training.
        :param kinematic_loss_calculator: Optional KinematicLossCalculator instance for kinematic losses.
        :param early_stopping_patience: Number of epochs with no improvement after which training will be stopped.
        :param early_stopping_min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        :param model_save_path: Path to save the best model during training.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.get_bone_mask_fn = get_bone_mask_fn
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.main_loss_type = main_loss_type
        self.cfg_drop_prob = cfg_drop_prob
        self.kinematic_loss_calculator = kinematic_loss_calculator

        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.model_save_path = model_save_path
        
        self._early_stopping_counter = 0
        self._best_val_loss = float('inf')

        # Logger warnings (unchanged)
        if not isinstance(self.model, ArmatureMDM):
            logger.warning("The provided model might not be an instance of the expected ArmatureMDM class.")
        if not hasattr(self.model, 'sbert_model') or self.model.sbert_model is None:
            logger.warning("SBERT model attribute not found or not loaded in ArmatureMDM.")


    def _calculate_masked_x0_loss(self,
                                predicted_x0: torch.Tensor,  # Model's direct output (predicted clean motion)
                                target_x0: torch.Tensor,     # Ground truth clean motion
                                bone_mask: torch.Tensor      # User's bone mask
                                ) -> torch.Tensor:
        """
        Calculates the masked L1 or MSE loss between predicted x0 and target x0.
        This corresponds to L_simple in the MDM paper if using MSE.
        The bone_mask allows focusing the loss on specific joints/features.
        :param predicted_x0: Predicted clean motion [bs, num_frames, num_features].
        :param target_x0: Ground truth clean motion [bs, num_frames, num_features].
        :param bone_mask: Bone mask [bs, num_frames, num_features] indicating active features.
        :return: Masked loss tensor.
        """
        if self.main_loss_type == "l1":
            element_wise_loss = F.l1_loss(predicted_x0, target_x0, reduction='none')
        elif self.main_loss_type == "mse":
            element_wise_loss = F.mse_loss(predicted_x0, target_x0, reduction='none')
        else:
            raise ValueError(f"Unsupported main_loss_type: {self.main_loss_type}. Choose 'l1' or 'mse'.")

        masked_loss = element_wise_loss * bone_mask
        num_active_elements = bone_mask.sum()

        if num_active_elements == 0:
            logger.warning("Bone mask sum is zero in _calculate_masked_x0_loss. Loss will be zero.")
            # Return a zero loss tensor that requires grad if the model is in training
            return torch.tensor(0.0, device=predicted_x0.device, requires_grad=predicted_x0.requires_grad)
        return masked_loss.sum() / num_active_elements
    

    def _run_batch(self, 
                batch_data: Dict[str, Any], 
                is_training_model: bool # True if model.train(), False if model.eval()
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Runs a single batch through the model and computes the loss.
        :param batch_data: Dictionary containing batch data.
        :param is_training_model: Boolean indicating if the model is in training mode.
        :return: Tuple (total_loss, loss_components).
        """
        x_noisy = batch_data["x_noisy"].to(self.device) # This is x_t, the noisy input to the model
        timesteps = batch_data["timesteps"].to(self.device) # Diffusion timesteps
        target_x0_for_main_loss = batch_data["target_x0"].to(self.device) # This is the ground truth clean motion, now the primary target

        text_conditions = batch_data["text_conditions"] # List of strings
        armature_class_ids = batch_data["armature_class_ids"].to(self.device) # Tensor of armature class IDs

        # --- Classifier-Free Guidance (CFG) Training Logic ---
        # Randomly drop conditions during training to enable CFG sampling.
        # MDM paper randomly sets condition c to null (∅) for 10% of samples.
        # We can achieve this by setting both uncond_text and uncond_armature to True.

        # Start with uncond flags from batch_data, if provided (e.g., for specific eval cases)
        cfg_uncond_text = batch_data.get('uncond_text', False)
        cfg_uncond_armature = batch_data.get('uncond_armature', False)

        if is_training_model and self.cfg_drop_prob > 0:
            # Overall probability to drop the entire condition 'c' (text AND armature)
            if torch.rand(1).item() < self.cfg_drop_prob:
                cfg_uncond_text = True
                cfg_uncond_armature = True
            # Else, you could implement more granular dropping (e.g., only text, only armature)
            # For simplicity, this implements dropping the full condition 'c'.
            # The ArmatureMDM's _apply_dropout_mask(force_mask=True) will handle zeroing out.

        motion_padding_mask = batch_data.get('motion_padding_mask')
        if motion_padding_mask is not None:
            motion_padding_mask = motion_padding_mask.to(self.device)

        # --- Model Forward Pass ---
        # The model now predicts the clean sample x0 directly
        predicted_x0 = self.model(
            x=x_noisy,  # Input is x_t (noisy motion)
            timesteps=timesteps,
            text_conditions=text_conditions,
            armature_class_ids=armature_class_ids,
            uncond_text=cfg_uncond_text,      # Pass CFG flag for text
            uncond_armature=cfg_uncond_armature, # Pass CFG flag for armature
            motion_padding_mask=motion_padding_mask
        ) # Output is predicted_x0

        bs, num_frames, num_total_motion_features = predicted_x0.shape # Or use x_noisy.shape

        # Get the bone mask for the full sequence length (of x0)
        bone_mask_full_seq = self.get_bone_mask_fn(
            armature_class_ids, num_total_motion_features, num_frames, self.device
        )

        # --- Main Loss Calculation (on predicted x0) ---
        # Corresponds to L_simple in MDM paper plus your bone masking
        main_x0_loss = self._calculate_masked_x0_loss(
            predicted_x0,          # Model's prediction of clean motion
            target_x0_for_main_loss, # Ground truth clean motion
            bone_mask_full_seq     # Your custom bone mask
        )
        current_total_loss = main_x0_loss
        loss_components = {"main_x0_loss": main_x0_loss.item()} # Updated loss name

        # --- Optional Kinematic Losses ---
        # These are applied to the predicted x0, consistent with MDM's geometric losses
        if self.kinematic_loss_calculator:
            # target_x0_for_main_loss is the same ground truth x0 needed by kinematic calculator

            # scheduler_params from batch_data might be less relevant now for kinematic_loss_calculator
            # if it no longer needs to derive x0 from noise.
            # x_t (x_noisy) and timesteps might also be less relevant for it.
            # Pass them if your kinematic loss implementation still uses them for some reason.
            scheduler_params_for_kin_calc = batch_data.get("scheduler_params", {}) 

            kin_loss, kin_losses_dict = self.kinematic_loss_calculator.compute_losses(
                x_t=x_noisy, # Original noisy input (may or may not be needed by kin_calc now)
                timesteps=timesteps, # Original timesteps (may or may not be needed)
                predicted_x0_from_model=predicted_x0, # Pass the model's direct x0 prediction
                target_x0=target_x0_for_main_loss,    # Ground truth clean motion
                scheduler_params=scheduler_params_for_kin_calc, # Pass if still used by kin_calc
                armature_class_ids=armature_class_ids,
                is_training_model=is_training_model 
            )
            current_total_loss += kin_loss
            loss_components.update(kin_losses_dict)

        return current_total_loss, loss_components


    def train_epoch(self, data_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Trains the model for one epoch.
        :param data_loader: DataLoader for the training data.
        :return: Tuple (average loss, dictionary of average loss components).
        """
        self.model.train()
        epoch_total_loss = 0.0
        epoch_loss_components_sum: Dict[str, float] = {} # To average individual losses
        num_batches = len(data_loader)

        for batch_idx, batch_data in enumerate(data_loader):
            self.optimizer.zero_grad()
            batch_total_loss, batch_loss_components = self._run_batch(batch_data, is_training_model=True)
            
            if torch.isnan(batch_total_loss): # Check for NaN loss
                logger.warning(f"NaN loss detected at batch {batch_idx}. Skipping update. Components: {batch_loss_components}")
                # Potentially skip optimizer step or handle error
                # For now, we'll just log and it will propagate if not handled
                # To prevent propagation of NaN to accumulated loss, we can skip this batch's loss
                if not torch.isnan(batch_total_loss): # Re-check, as 0/0 can become NaN if num_active is 0
                    epoch_total_loss += batch_total_loss.item()
                    for key, val in batch_loss_components.items():
                        epoch_loss_components_sum[key] = epoch_loss_components_sum.get(key, 0.0) + val
                continue # Skip backprop and step if loss is NaN


            batch_total_loss.backward()
            self.optimizer.step()
            
            epoch_total_loss += batch_total_loss.item()
            for key, val in batch_loss_components.items():
                epoch_loss_components_sum[key] = epoch_loss_components_sum.get(key, 0.0) + val


            if (batch_idx + 1) % max(1, num_batches // 10) == 0:
                log_components = ", ".join([f"{k}: {v:.4f}" for k,v in batch_loss_components.items()])
                logger.info(f"  Batch {batch_idx+1}/{num_batches}, Total Loss: {batch_total_loss.item():.4f} ({log_components})")
        
        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0
        avg_loss_components = {k: v / num_batches if num_batches > 0 else 0 for k, v in epoch_loss_components_sum.items()}
        
        if self.lr_scheduler:
            self.lr_scheduler.step()
        return avg_epoch_loss, avg_loss_components

    def evaluate_epoch(self, data_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Evaluates the model for one epoch.
        :param data_loader: DataLoader for the evaluation data.
        :return: Tuple (average loss, dictionary of average loss components).
        """
        self.model.eval()
        epoch_total_loss = 0.0
        epoch_loss_components_sum: Dict[str, float] = {}
        num_batches = len(data_loader)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                batch_total_loss, batch_loss_components = self._run_batch(batch_data, is_training_model=False)
                
                if not torch.isnan(batch_total_loss): # Only add if not NaN
                    epoch_total_loss += batch_total_loss.item()
                    for key, val in batch_loss_components.items():
                        epoch_loss_components_sum[key] = epoch_loss_components_sum.get(key, 0.0) + val

                if (batch_idx + 1) % max(1, num_batches // 10) == 0:
                    log_components = ", ".join([f"{k}: {v:.4f}" for k,v in batch_loss_components.items()])
                    logger.debug(f"  Eval Batch {batch_idx+1}/{num_batches}, Total Loss: {batch_total_loss.item():.4f} ({log_components})")
        
        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0
        avg_loss_components = {k: v / num_batches if num_batches > 0 else 0 for k, v in epoch_loss_components_sum.items()}
        return avg_epoch_loss, avg_loss_components

    def train(self, 
              train_loader: DataLoader, 
              num_epochs: int, 
              val_loader: Optional[DataLoader] = None
            ):
        """
        Trains the model for a specified number of epochs.
        :param train_loader: DataLoader for the training data.
        :param num_epochs: Number of epochs to train.
        :param val_loader: Optional DataLoader for validation data.
        """
        logger.info(f"Starting ArmatureMDM training for {num_epochs} epochs on device: {self.device}.")
        logger.info(f"Early stopping patience: {self.early_stopping_patience} epochs, min delta: {self.early_stopping_min_delta}")
        logger.info(f"Best model will be saved to: {self.model_save_path}")

        # Reset early stopping counters for a new training run
        self._early_stopping_counter = 0
        self._best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            logger.info(f"--- Training Epoch {epoch}/{num_epochs} ---")
            avg_train_loss, avg_train_components = self.train_epoch(train_loader)
            log_train_components_str = ", ".join([f"Avg {k}: {v:.4f}" for k,v in avg_train_components.items()])
            logger.info(f"Epoch {epoch} Training Summary: Avg Total Loss: {avg_train_loss:.4f} ({log_train_components_str})")

            if val_loader:
                logger.info(f"--- Validating Epoch {epoch}/{num_epochs} ---")
                avg_val_loss, avg_val_components = self.evaluate_epoch(val_loader)
                log_val_components_str = ", ".join([f"Avg {k}: {v:.4f}" for k,v in avg_val_components.items()])
                logger.info(f"Epoch {epoch} Validation Summary: Avg Total Loss: {avg_val_loss:.4f} ({log_val_components_str})")

                # Early Stopping Check
                if avg_val_loss < self._best_val_loss - self.early_stopping_min_delta:
                    self._best_val_loss = avg_val_loss
                    self._early_stopping_counter = 0 # Reset counter
                    logger.info(f"New best validation loss: {self._best_val_loss:.4f}. Saving model...")
                    try:
                        # Save model checkpoint (can include epoch, optimizer state etc. if needed)
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'val_loss': self._best_val_loss,
                            # You can add more info like model_params here
                        }, self.model_save_path)
                        logger.info(f"Model saved to {self.model_save_path}")
                    except Exception as e:
                        logger.error(f"Error saving model: {e}")
                else:
                    self._early_stopping_counter += 1
                    logger.info(f"Validation loss did not improve significantly. Early stopping counter: {self._early_stopping_counter}/{self.early_stopping_patience}")

                if self._early_stopping_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break # Exit training loop
            
            else: # No validation loader, save model at the end of each epoch or based on train loss (less ideal)
                if epoch % 10 == 0: # Example: save every 10 epochs if no validation
                     logger.info(f"No validation loader. Saving model checkpoint at epoch {epoch}...")
                     temp_save_path = self.model_save_path.replace(".pth", f"_epoch{epoch}.pth")
                     try:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            # 'train_loss': avg_train_loss, # Optional
                        }, temp_save_path)
                        logger.info(f"Model checkpoint saved to {temp_save_path}")
                     except Exception as e:
                        logger.error(f"Error saving model checkpoint: {e}")


            if self.lr_scheduler:
                # Common practice: step LR scheduler based on validation loss or after each epoch
                # If using ReduceLROnPlateau, step with validation loss
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and val_loader:
                    self.lr_scheduler.step(avg_val_loss)
                elif not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step() # For schedulers like StepLR, CosineAnnealingLR etc.
                
                if hasattr(self.optimizer, 'param_groups'):
                     current_lr = self.optimizer.param_groups[0]['lr']
                     logger.info(f"Current learning rate: {current_lr:.6e}")
        
        logger.info("ArmatureMDM training finished.")
        if not val_loader: # If no validation, save the final model
            logger.info(f"Saving final model (trained for {num_epochs} epochs without validation-based early stopping)...")
            final_save_path = self.model_save_path.replace(".pth", "_final.pth")
            try:
                torch.save({
                    'epoch': num_epochs,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, final_save_path)
                logger.info(f"Final model saved to {final_save_path}")
            except Exception as e:
                logger.error(f"Error saving final model: {e}")

