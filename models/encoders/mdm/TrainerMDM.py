import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Callable, Literal, Optional, Tuple 
import logging
from ArmatureMDM import ArmatureMDM

logger = logging.getLogger(__name__)

class KinematicLossCalculator:
    def __init__(self,
                    get_bone_mask_fn: Callable[[torch.Tensor, int, int, str], torch.Tensor],
                    device: str,
                    use_velocity_loss: bool = True, # Default to True if this class is instanced
                    velocity_loss_weight: float = 0.1,
                    use_acceleration_loss: bool = True, # Default to True
                    acceleration_loss_weight: float = 0.1,
                    kinematic_loss_type: Literal["l1", "mse"] = "l1"
                ):
        """
        Calculates kinematic losses (velocity, acceleration) for motion sequences.

        :param get_bone_mask_fn: Function to get bone_mask for active features.
        :param device: The device to perform calculations on.
        :param use_velocity_loss: Whether to compute velocity loss.
        :param velocity_loss_weight: Weight for the velocity loss.
        :param use_acceleration_loss: Whether to compute acceleration loss.
        :param acceleration_loss_weight: Weight for the acceleration loss.
        :param kinematic_loss_type: Base loss type for kinematic losses ('l1' or 'mse').
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

    def _predict_x0(self, 
                    x_t: torch.Tensor, 
                    noise_pred: torch.Tensor,
                    sqrt_alphas_cumprod_t: torch.Tensor, 
                    sqrt_one_minus_alphas_cumprod_t: torch.Tensor) -> torch.Tensor:
        """ 
        Predicts x0 (clean data) from x_t and predicted noise. 
        :param x_t: Noisy motion input [bs, num_frames, num_features].
        :param noise_pred: Noise predicted by the model [bs, num_frames, num_features].
        :param sqrt_alphas_cumprod_t: [bs] tensor of sqrt(alpha_bar_t) for each t in batch.
        :param sqrt_one_minus_alphas_cumprod_t: [bs] tensor of sqrt(1-alpha_bar_t) for each t in batch.
        :return: Predicted x0 [bs, num_frames, num_features].
        """
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1).to(self.device)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1).to(self.device)
        pred_x0 = (x_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / (sqrt_alphas_cumprod_t + 1e-8)
        return pred_x0

    def _calculate_derivative(self, motion_sequence: torch.Tensor) -> torch.Tensor:
        """
        Calculates the first derivative (e.g., velocity).
        :param motion_sequence: [bs, num_frames, num_features].
        :return: Velocity [bs, num_frames-1, num_features].
        """
        if motion_sequence.shape[1] < 2:
            return torch.empty_like(motion_sequence[:, :0, :]) # Return empty if not enough frames
        return motion_sequence[:, 1:] - motion_sequence[:, :-1]

    def _calculate_masked_feature_loss(self, 
                                      prediction: torch.Tensor, 
                                      target: torch.Tensor, 
                                      mask: torch.Tensor) -> torch.Tensor:
        """ 
        Calculates masked L1 or MSE loss on generic features.
        :param prediction: Predicted motion [bs, num_frames, num_features].
        :param target: Target motion [bs, num_frames, num_features].
        :param mask: Feature mask [bs, num_frames, num_features].
        :return: Masked loss. 
        """
        if prediction.shape[0] == 0: # If prediction is empty (e.g. from derivative on short seq)
            return torch.tensor(0.0, device=self.device)

        # Ensure mask and target are compatible with prediction's sequence length
        min_frames = prediction.shape[1]
        target_adjusted = target[:, :min_frames, :]
        mask_adjusted = mask[:, :min_frames, :]
        
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
                       timesteps: torch.Tensor, # Unused here, but often part of the batch
                       predicted_noise: torch.Tensor,
                       target_x0: torch.Tensor, # Ground truth clean motion
                       scheduler_params: Dict[str, torch.Tensor], # Contains sqrt_alphas & sqrt_one_minus_alphas
                       armature_class_ids: torch.Tensor,
                       is_training_model: bool # To set requires_grad for 0-loss tensor if needed
                    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the total weighted kinematic loss and a dictionary of individual losses.

        :param x_t: Noisy motion input to the main model [bs, num_frames, num_features].
        :param timesteps: Diffusion timesteps [bs].
        :param predicted_noise: Noise predicted by the main model [bs, num_frames, num_features].
        :param target_x0: Ground truth clean motion [bs, num_frames, num_features].
        :param scheduler_params: Dict containing 'sqrt_alphas_cumprod_t' and 
                                 'sqrt_one_minus_alphas_cumprod_t' tensors of shape [bs].
        :param armature_class_ids: Armature IDs for generating bone_mask [bs].
        :param is_training_model: Boolean, true if the main model is in training mode.
        :return: Tuple (total_weighted_kinematic_loss, dictionary_of_individual_losses).
        """
        if not (self.use_velocity_loss or self.use_acceleration_loss):
            return torch.tensor(0.0, device=self.device, requires_grad=is_training_model), {}

        bs, num_frames, num_total_motion_features = predicted_noise.shape
        
        sqrt_alphas_cumprod_t = scheduler_params['sqrt_alphas_cumprod_t'].to(self.device)
        sqrt_one_minus_alphas_cumprod_t = scheduler_params['sqrt_one_minus_alphas_cumprod_t'].to(self.device)

        predicted_x0 = self._predict_x0(x_t, predicted_noise, 
                                        sqrt_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t)
        
        # Full bone mask, will be sliced for derivatives
        bone_mask_full = self.get_bone_mask_fn(
            armature_class_ids, num_total_motion_features, num_frames, self.device
        )

        total_kin_loss = torch.tensor(0.0, device=self.device, requires_grad=is_training_model)
        losses_dict = {}

        # Velocity Loss
        if self.use_velocity_loss and num_frames > 1:
            pred_vel = self._calculate_derivative(predicted_x0)
            target_vel = self._calculate_derivative(target_x0)
            if pred_vel.shape[1] > 0: # Check if velocity could be computed
                bone_mask_vel = bone_mask_full[:, 1:, :] # Adjust mask for velocity sequence length
                vel_loss = self._calculate_masked_feature_loss(pred_vel, target_vel, bone_mask_vel)
                total_kin_loss += self.velocity_loss_weight * vel_loss
                losses_dict['velocity_loss'] = vel_loss.item()
        
        # Acceleration Loss
        if self.use_acceleration_loss and num_frames > 2:
            # Recompute velocities if not already done (or if pred_vel was empty)
            if not (self.use_velocity_loss and num_frames > 1 and pred_vel.shape[1] > 0) :
                current_pred_vel = self._calculate_derivative(predicted_x0)
                current_target_vel = self._calculate_derivative(target_x0)
            else: # Velocities were already computed and valid
                 current_pred_vel = pred_vel
                 current_target_vel = target_vel
            
            if current_pred_vel.shape[1] > 0: # Check if velocity calculation was valid before taking another derivative
                pred_accel = self._calculate_derivative(current_pred_vel)
                target_accel = self._calculate_derivative(current_target_vel)
                if pred_accel.shape[1] > 0: # Check if acceleration could be computed
                    bone_mask_accel = bone_mask_full[:, 2:, :] # Adjust mask for acceleration sequence length
                    accel_loss = self._calculate_masked_feature_loss(pred_accel, target_accel, bone_mask_accel)
                    total_kin_loss += self.acceleration_loss_weight * accel_loss
                    losses_dict['acceleration_loss'] = accel_loss.item()
        
        return total_kin_loss, losses_dict


class ArmatureMDMTrainer:
    def __init__(self,
                    model: ArmatureMDM,
                    optimizer: optim.Optimizer,
                    get_bone_mask_fn: Callable[[torch.Tensor, int, int, str], torch.Tensor],
                    device: str = "cuda" if torch.cuda.is_available() else "cpu",
                    lr_scheduler: Optional[Any] = None,
                    noise_loss_type: Literal["l1", "mse"] = "l1",
                    kinematic_loss_calculator: Optional[KinematicLossCalculator] = None
                ):
        """
        Initializes the ArmatureMDMTrainer.
        :param model: The ArmatureMDM model to be trained.
        :param optimizer: The optimizer for training the model.
        :param get_bone_mask_fn: Function to get bone mask for active features.
        :param device: Device to run the model on (default: "cuda" if available).
        :param lr_scheduler: Optional learning rate scheduler.
        :param noise_loss_type: Type of loss for noise prediction ('l1' or 'mse').
        :kinematic_loss_calculator: Optional KinematicLossCalculator instance for kinematic losses.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.get_bone_mask_fn = get_bone_mask_fn
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.noise_loss_type = noise_loss_type
        self.kinematic_loss_calculator = kinematic_loss_calculator

        if not isinstance(self.model, ArmatureMDM): # Check against the actual class name
            logger.warning("The provided model might not be an instance of the expected ArmatureMDM class.")
        if not hasattr(self.model, 'sbert_model') or self.model.sbert_model is None:
            logger.warning("SBERT model attribute not found or not loaded in the ArmatureMDM instance. "
                           "Ensure it's handled correctly within ArmatureMDM if text conditioning is used.")

    def _calculate_masked_noise_loss(self, 
                                    predicted_noise: torch.Tensor, 
                                    target_noise: torch.Tensor, 
                                    bone_mask: torch.Tensor
                                    ) -> torch.Tensor:
        """
        Calculates the masked noise loss (L1 or MSE) based on the bone mask.
        :param predicted_noise: Predicted noise from the model [bs, num_frames, num_features].
        :param target_noise: Target noise [bs, num_frames, num_features].
        :param bone_mask: Bone mask [bs, num_frames, num_features].
        :return: Masked noise loss.
        """
        if self.noise_loss_type == "l1":
            element_wise_loss = F.l1_loss(predicted_noise, target_noise, reduction='none')
        elif self.noise_loss_type == "mse":
            element_wise_loss = F.mse_loss(predicted_noise, target_noise, reduction='none')
        else:
            raise ValueError(f"Unsupported noise_loss_type: {self.noise_loss_type}. Choose 'l1' or 'mse'.")
        masked_loss = element_wise_loss * bone_mask
        num_active_elements = bone_mask.sum()
        if num_active_elements == 0:
            logger.warning("Bone mask sum is zero in _calculate_masked_noise_loss. Loss will be zero.")
            return torch.tensor(0.0, device=predicted_noise.device, requires_grad=self.model.training)
        return masked_loss.sum() / num_active_elements

    def _run_batch(self, 
                   batch_data: Dict[str, Any], 
                   is_training_model: bool
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Helper function to run a single batch for either training or evaluation.
        Returns the total loss for the batch and a dictionary of individual loss components.
        
        Batch data must now include: "target_x0", and a "scheduler_params" dict
        with "sqrt_alphas_cumprod_t", "sqrt_one_minus_alphas_cumprod_t" if kinematic losses are used.
        :param batch_data: Dictionary containing batch data.
        :param is_training_model: Boolean indicating if the model is in training mode.
        :return: Tuple (total_loss, loss_components).
        """
        x_noisy = batch_data["x_noisy"].to(self.device)
        timesteps = batch_data["timesteps"].to(self.device)
        target_noise = batch_data["target_noise"].to(self.device)
        text_conditions = batch_data["text_conditions"]
        armature_class_ids = batch_data["armature_class_ids"].to(self.device)
        
        uncond_text = batch_data.get('uncond_text', False)
        uncond_armature = batch_data.get('uncond_armature', False)
        motion_padding_mask = batch_data.get('motion_padding_mask')
        if motion_padding_mask is not None:
            motion_padding_mask = motion_padding_mask.to(self.device)

        predicted_noise = self.model(
            x=x_noisy, timesteps=timesteps, text_conditions=text_conditions,
            armature_class_ids=armature_class_ids, uncond_text=uncond_text,
            uncond_armature=uncond_armature, motion_padding_mask=motion_padding_mask
        )
        
        bs, num_frames, num_total_motion_features = predicted_noise.shape
        bone_mask_full_seq = self.get_bone_mask_fn(
            armature_class_ids, num_total_motion_features, num_frames, self.device
        )

        # Standard Noise Loss (Primary)
        noise_loss = self._calculate_masked_noise_loss(predicted_noise, target_noise, bone_mask_full_seq)
        current_total_loss = noise_loss
        loss_components = {"noise_loss": noise_loss.item()}

        # Optional Kinematic Losses
        if self.kinematic_loss_calculator:
            if "target_x0" not in batch_data or "scheduler_params" not in batch_data:
                logger.warning("KinematicLossCalculator is active, but 'target_x0' or 'scheduler_params' "
                               "missing in batch_data. Skipping kinematic losses for this batch.")
            else:
                target_x0 = batch_data["target_x0"].to(self.device)
                scheduler_params = {k: v.to(self.device) for k, v in batch_data["scheduler_params"].items()}

                kin_loss, kin_losses_dict = self.kinematic_loss_calculator.compute_losses(
                    x_t=x_noisy,
                    timesteps=timesteps, # Passed for completeness, not directly used by _predict_x0 in this version
                    predicted_noise=predicted_noise,
                    target_x0=target_x0,
                    scheduler_params=scheduler_params,
                    armature_class_ids=armature_class_ids,
                    is_training_model=is_training_model # Pass model's training state
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
        for epoch in range(1, num_epochs + 1):
            logger.info(f"--- Training Epoch {epoch}/{num_epochs} ---")
            avg_train_loss, avg_train_components = self.train_epoch(train_loader)
            log_train_components = ", ".join([f"Avg {k}: {v:.4f}" for k,v in avg_train_components.items()])
            logger.info(f"Epoch {epoch} Avg Training Total Loss: {avg_train_loss:.4f} ({log_train_components})")

            if val_loader:
                logger.info(f"--- Validating Epoch {epoch}/{num_epochs} ---")
                avg_val_loss, avg_val_components = self.evaluate_epoch(val_loader) 
                log_val_components = ", ".join([f"Avg {k}: {v:.4f}" for k,v in avg_val_components.items()])
                logger.info(f"Epoch {epoch} Avg Validation Total Loss: {avg_val_loss:.4f} ({log_val_components})")
            
            if self.lr_scheduler and hasattr(self.optimizer, 'param_groups'):
                 current_lr = self.optimizer.param_groups[0]['lr']
                 logger.info(f"Current learning rate: {current_lr:.6f}")
        logger.info("ArmatureMDM training finished.")