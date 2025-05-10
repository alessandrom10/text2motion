import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset # For type hinting and example
from typing import List, Dict, Any, Callable, Literal, Optional # Added Callable
import logging
from ArmatureMDM import ArmatureMDM, PositionalEncoding, TimestepEmbedder

logger = logging.getLogger(__name__)

class ArmatureMDMTrainer:
    """
    Trainer class for the ArmatureMDM model, incorporating a masked loss function
    based on armature class IDs.
    """
    def __init__(self,
                    model: nn.Module, # Should be an instance of ArmatureMDM
                    optimizer: optim.Optimizer,
                    get_bone_mask_fn: Callable[[torch.Tensor, int, int, str], torch.Tensor],
                    device: str = "cuda" if torch.cuda.is_available() else "cpu",
                    lr_scheduler: Optional[Any] = None,
                    loss_type: Literal["l1", "mse"] = "l1" # Default to L1 loss
                ):
        """
        Initializes the ArmatureMDMTrainer.

        :param model: The ArmatureMDM model instance to train.
        :param optimizer: The optimizer for the model's parameters.
        :param get_bone_mask_fn: A function that takes (armature_class_ids_batch, num_total_motion_features,
                                 num_frames, device) and returns a bone_mask tensor of shape
                                 [bs, num_frames, num_total_motion_features] with 1s for active features
                                 and 0s for inactive ones.
        :param device: The device to train on ('cuda' or 'cpu').
        :param lr_scheduler: Optional learning rate scheduler.
        :param loss_type: Type of base loss to use before masking ('l1' or 'mse').
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.get_bone_mask_fn = get_bone_mask_fn
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.loss_type = loss_type

        if not isinstance(self.model, ArmatureMDM): # Runtime check for correct model type
             logger.warning("The provided model might not be an instance of ArmatureMDM, "
                           "ensure it's compatible with the trainer's expectations.")


    def _calculate_masked_loss(self, 
                               predicted_noise: torch.Tensor, 
                               target_noise: torch.Tensor, 
                               bone_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss (L1 or MSE) only on the active features defined by the bone_mask.

        :param predicted_noise: Tensor from model [bs, num_frames, num_total_motion_features].
        :param target_noise: Ground truth noise tensor [bs, num_frames, num_total_motion_features].
        :param bone_mask: Binary mask [bs, num_frames, num_total_motion_features], 1 for active.
        :return: Scalar loss value.
        """
        if self.loss_type == "l1":
            element_wise_loss = F.l1_loss(predicted_noise, target_noise, reduction='none')
        elif self.loss_type == "mse":
            element_wise_loss = F.mse_loss(predicted_noise, target_noise, reduction='none')
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}. Choose 'l1' or 'mse'.")

        masked_loss = element_wise_loss * bone_mask
        
        # Normalize by the number of active elements to get a meaningful mean
        num_active_elements = bone_mask.sum()
        if num_active_elements == 0: # Avoid division by zero if mask is all zeros
            return torch.tensor(0.0, device=predicted_noise.device, requires_grad=True) 
            
        final_loss = masked_loss.sum() / num_active_elements
        return final_loss

    def train_epoch(self, data_loader: DataLoader) -> float:
        """
        Trains the model for one epoch.
        Assumes data_loader yields batches of (x_noisy, timesteps, target_noise, conditions_dict)
        where conditions_dict contains 'text_conditions', 'armature_class_ids',
        and optionally 'motion_padding_mask', 'uncond_text', 'uncond_armature'.
        It also needs info for get_bone_mask_fn, implicitly through armature_class_ids.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(data_loader)

        for batch_idx, batch_data in enumerate(data_loader):
            # Adapt this unpacking based on actual DataLoader output
            # Example: x_noisy, timesteps, target_noise, text_conditions, armature_ids, motion_mask_opt = batch_data
            # For now, let's assume a more structured batch_data:
            x_noisy = batch_data["x_noisy"].to(self.device)
            timesteps = batch_data["timesteps"].to(self.device)
            target_noise = batch_data["target_noise"].to(self.device)
            text_conditions = batch_data["text_conditions"] # List of strings, SBERT handles device
            armature_class_ids = batch_data["armature_class_ids"].to(self.device)
            
            # Optional conditioning flags and masks from batch_data
            uncond_text = batch_data.get('uncond_text', False)
            uncond_armature = batch_data.get('uncond_armature', False)
            motion_padding_mask = batch_data.get('motion_padding_mask')
            if motion_padding_mask is not None:
                motion_padding_mask = motion_padding_mask.to(self.device)

            self.optimizer.zero_grad()

            predicted_noise = self.model(
                x=x_noisy,
                timesteps=timesteps,
                text_conditions=text_conditions,
                armature_class_ids=armature_class_ids,
                uncond_text=uncond_text,
                uncond_armature=uncond_armature,
                motion_padding_mask=motion_padding_mask
            )
            
            # Get the bone mask for the current batch
            # The shape of predicted_noise is [bs, num_frames, num_total_motion_features]
            bs, num_frames, num_total_motion_features = predicted_noise.shape
            bone_mask = self.get_bone_mask_fn(
                armature_class_ids, 
                num_total_motion_features, 
                num_frames, # Pass num_frames to allow per-frame masking if needed by get_bone_mask_fn
                self.device
            )

            loss = self._calculate_masked_loss(predicted_noise, target_noise, bone_mask)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % max(1, num_batches // 10) == 0: # Log ~10 times per epoch
                logger.info(f"  Batch {batch_idx+1}/{num_batches}, Training Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        if self.lr_scheduler:
            self.lr_scheduler.step()
        return avg_loss

    def evaluate_epoch(self, data_loader: DataLoader) -> float:
        """
        Evaluates the model for one epoch on a validation set.
        :param data_loader: DataLoader for validation data.
        :return: Average loss over the validation set.
        """
        self.model.eval() # Set ArmatureMDM to evaluation mode
        total_loss = 0.0
        num_batches = len(data_loader)

        with torch.no_grad(): # Disable gradient calculations
            for batch_idx, batch_data in enumerate(data_loader):
                x_noisy = batch_data["x_noisy"].to(self.device)
                timesteps = batch_data["timesteps"].to(self.device)
                target_noise = batch_data["target_noise"].to(self.device)
                text_conditions = batch_data["text_conditions"]
                armature_class_ids = batch_data["armature_class_ids"].to(self.device)
                
                uncond_text = batch_data.get('uncond_text', False) # CFG flags might also apply in eval
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
                bone_mask = self.get_bone_mask_fn(
                    armature_class_ids, num_total_motion_features, num_frames, self.device
                )

                loss = self._calculate_masked_loss(predicted_noise, target_noise, bone_mask)
                total_loss += loss.item()

                if (batch_idx + 1) % max(1, num_batches // 10) == 0: # Optional: log eval batch progress
                    logger.debug(f"  Eval Batch {batch_idx+1}/{num_batches}, Eval Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def train(self, 
              train_loader: DataLoader, 
              num_epochs: int, 
              val_loader: Optional[DataLoader] = None):
        """
        Full training loop, with optional validation.
        :param train_loader: DataLoader for training data.
        :param num_epochs: Number of epochs to train for.
        :param val_loader: Optional DataLoader for validation data.
        """
        logger.info(f"Starting ArmatureMDM training for {num_epochs} epochs on device: {self.device}.")
        for epoch in range(1, num_epochs + 1):
            logger.info(f"--- Training Epoch {epoch}/{num_epochs} ---")
            train_loss = self.train_epoch(train_loader)
            logger.info(f"Epoch {epoch} Avg Training Loss: {train_loss:.4f}")

            if val_loader:
                logger.info(f"--- Validating Epoch {epoch}/{num_epochs} ---")
                val_loss = self.evaluate_epoch(val_loader) 
                logger.info(f"Epoch {epoch} Avg Validation Loss: {val_loss:.4f}")
            
            if self.lr_scheduler and hasattr(self.optimizer, 'param_groups'):
                 current_lr = self.optimizer.param_groups[0]['lr']
                 logger.info(f"Current learning rate: {current_lr:.6f}")
        logger.info("ArmatureMDM training finished.")
