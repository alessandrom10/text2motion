import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Callable, List, Literal, Optional, Tuple 
import logging
from tqdm.auto import tqdm
from ArmatureMDM import ArmatureMDM

logger = logging.getLogger(__name__)


class MDMGeometricLosses(nn.Module):
    """
    Calculates geometric losses as described in the MDM paper:
    L_pos (if applicable, assuming x0 is already positions), L_vel, and L_foot.
    The losses are MSE based, as per the paper.
    """
    def __init__(self,
                 lambda_pos: float = 0.0, # Weight for position loss
                 lambda_vel: float = 1.0, # Weight for velocity loss
                 lambda_foot: float = 1.0, # Weight for foot contact loss
                 num_joints: int = 22,     # Total number of joints in the representation
                 features_per_joint: int = 3, # e.g., 3 for XYZ
                 foot_joint_indices: Optional[List[int]] = None, # Indices of left and right foot joints
                 device: str = 'cpu',
                 # Optional: if you want to apply your get_bone_mask_fn to these losses too
                 get_bone_mask_fn: Optional[Callable[[torch.Tensor, int, int, str], torch.Tensor]] = None
                ):
        """
        Initializes the MDMGeometricLosses module.
        :param lambda_pos: Weight for position loss (L_pos).
        :param lambda_vel: Weight for velocity loss (L_vel).
        :param lambda_foot: Weight for foot contact loss (L_foot).
        :param num_joints: Total number of joints in the motion representation.
        :param features_per_joint: Number of features per joint (e.g., 3 for XYZ).
        :param foot_joint_indices: Optional list of indices for foot joints (e.g., [10, 11] for L_Foot, R_Foot).
                                   If None, defaults to HumanML3D-like indices.
        :param device: Device to run the calculations on (e.g., "cuda" or "cpu").
        :param get_bone_mask_fn: Optional function to get a bone mask for active features.
                                 Should take (armature_class_ids, num_total_features, num_frames, device) and return a mask.
        """
        super().__init__()
        self.lambda_pos = lambda_pos
        self.lambda_vel = lambda_vel
        self.lambda_foot = lambda_foot
        
        self.num_joints = num_joints
        self.features_per_joint = features_per_joint # Should be 3 for XYZ
        if self.features_per_joint != 3:
            logger.warning(f"MDMGeometricLosses expects features_per_joint=3 for XYZ. "
                           f"Received {self.features_per_joint}. FK might be needed if not XYZ.")

        if foot_joint_indices is None:
            # Default HumanML3D-like foot joint indices (L_Foot, R_Foot)
            # Adjust these if your joint order is different!
            # Based on t2m_kinematic_chain: L_Foot=10, R_Foot=11
            self.foot_joint_indices = [10, 11] 
            logger.info(f"Using default foot joint indices: {self.foot_joint_indices}")
        else:
            self.foot_joint_indices = foot_joint_indices
            logger.info(f"Using provided foot joint indices: {self.foot_joint_indices}")
            
        self.device = device
        self.get_bone_mask_fn = get_bone_mask_fn # For overall armature-based masking

        if self.lambda_foot > 0 and not self.foot_joint_indices:
            logger.warning("lambda_foot > 0 but no foot_joint_indices provided. Foot contact loss will be zero.")
        
        logger.info(f"MDMGeometricLosses initialized with weights: "
                    f"Pos={self.lambda_pos}, Vel={self.lambda_vel}, FootContact={self.lambda_foot}")

    def _calculate_derivative(self, motion_sequence: torch.Tensor) -> torch.Tensor:
        """ Calculates the first derivative (velocity). motion_sequence: [bs, num_frames, features] """
        if motion_sequence.shape[1] < 2:
            return torch.empty_like(motion_sequence[:, :0, :])
        return motion_sequence[:, 1:] - motion_sequence[:, :-1]

    def compute_losses(self,
                       predicted_x0: torch.Tensor, # Shape: [bs, num_frames, num_total_features]
                       target_x0: torch.Tensor,    # Shape: [bs, num_frames, num_total_features]
                       # foot_contact_gt: Shape [bs, num_frames, num_joints], binary 1 for contact
                       foot_contact_gt: Optional[torch.Tensor] = None, 
                       armature_class_ids: Optional[torch.Tensor] = None, # For get_bone_mask_fn
                       armature_config_data: Optional[Dict] = None
                      ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the weighted sum of geometric losses.

        :param predicted_x0: Model's prediction of clean motion (joint positions).
        :param target_x0: Ground truth clean motion (joint positions).
        :param foot_contact_gt: Ground truth binary foot contact mask (1 for contact, 0 for air).
                                 Shape: [bs, num_frames, num_joints]. Required if lambda_foot > 0.
        :param armature_class_ids: Tensor of armature class IDs for applying get_bone_mask_fn.
        :param armature_config_data: Optional configuration data for the armature (e.g., bone names, joint indices).
        :return: Tuple (total_weighted_geometric_loss, dictionary_of_individual_losses).
        """
        bs, num_frames, num_total_features = predicted_x0.shape
        
        if num_total_features != self.num_joints * self.features_per_joint:
            logger.error(f"Feature dimension mismatch! num_total_features ({num_total_features}) "
                         f"!= num_joints ({self.num_joints}) * features_per_joint ({self.features_per_joint}). "
                         f"Cannot reliably apply geometric losses.")
            return torch.tensor(0.0, device=self.device, requires_grad=predicted_x0.requires_grad), {}

        losses = {}
        total_geometric_loss = torch.tensor(0.0, device=self.device, requires_grad=predicted_x0.requires_grad)

        # Optional overall bone mask based on armature_class_id
        overall_bone_mask = None
        if self.get_bone_mask_fn is not None and armature_class_ids is not None:
            overall_bone_mask = self.get_bone_mask_fn(
                armature_class_ids, 
                num_total_features, 
                num_frames, 
                str(self.device),
                armature_config_data=armature_config_data
            ) # Shape: [bs, num_frames, num_total_features]


        # 1. L_pos (Position Loss) - MDM paper uses this if predicting rotations, FK is identity if predicting positions.
        # Your main_x0_loss already covers this if it's MSE on positions.
        # We include it here for completeness or if you want to weight it differently or apply a different mask.
        if self.lambda_pos > 0:
            # Assuming predicted_x0 and target_x0 are already joint positions (FK is identity)
            pos_loss_elementwise = F.mse_loss(predicted_x0, target_x0, reduction='none')
            if overall_bone_mask is not None:
                pos_loss_elementwise = pos_loss_elementwise * overall_bone_mask
                num_active = overall_bone_mask.sum()
            else:
                num_active = torch.tensor(pos_loss_elementwise.numel(), device=self.device)

            l_pos = pos_loss_elementwise.sum() / (num_active + 1e-8) if num_active > 0 else torch.tensor(0.0, device=self.device)
            total_geometric_loss = total_geometric_loss + self.lambda_pos * l_pos
            losses['geom_pos_loss'] = l_pos.item()

        # 2. L_vel (Velocity Loss)
        if self.lambda_vel > 0 and num_frames > 1:
            pred_vel = self._calculate_derivative(predicted_x0) # [bs, num_frames-1, num_features]
            target_vel = self._calculate_derivative(target_x0) # [bs, num_frames-1, num_features]

            if pred_vel.shape[1] > 0: # If velocity could be computed
                vel_loss_elementwise = F.mse_loss(pred_vel, target_vel, reduction='none')
                
                current_mask = None
                if overall_bone_mask is not None:
                    # Apply mask, considering sequence length is num_frames-1 for velocity
                    current_mask = overall_bone_mask[:, 1:, :] # Or overall_bone_mask[:, :pred_vel.shape[1], :]
                    vel_loss_elementwise = vel_loss_elementwise * current_mask
                    num_active = current_mask.sum()
                else:
                    num_active = torch.tensor(vel_loss_elementwise.numel(), device=self.device)

                l_vel = vel_loss_elementwise.sum() / (num_active + 1e-8) if num_active > 0 else torch.tensor(0.0, device=self.device)
                total_geometric_loss = total_geometric_loss + self.lambda_vel * l_vel
                losses['geom_vel_loss'] = l_vel.item()

        # 3. L_foot (Foot Contact Loss)
        if self.lambda_foot > 0 and num_frames > 1 and self.foot_joint_indices:
            if foot_contact_gt is None:
                logger.warning("Foot contact loss (lambda_foot > 0) but foot_contact_gt is None. Skipping L_foot.")
            elif foot_contact_gt.shape[2] != self.num_joints:
                 logger.warning(f"Foot contact GT shape mismatch. Expected {self.num_joints} joints, got {foot_contact_gt.shape[2]}. Skipping L_foot.")
            else:
                # Reshape predicted_x0 to [bs, num_frames, num_joints, features_per_joint]
                pred_x0_reshaped = predicted_x0.reshape(bs, num_frames, self.num_joints, self.features_per_joint)
                
                # Predicted velocities of all joints
                pred_all_joint_vel = self._calculate_derivative(pred_x0_reshaped) # [bs, num_frames-1, num_joints, feat_per_joint]

                # Select velocities of foot joints
                pred_foot_vel = pred_all_joint_vel[:, :, self.foot_joint_indices, :] # [bs, num_frames-1, num_foot_joints, feat_per_joint]
                
                # Ground truth foot contact mask $f_i$ for the N-1 velocity frames, for foot joints
                # foot_contact_gt is [bs, num_frames, num_joints]
                # We need it for num_frames-1 and for the specific foot_joint_indices
                # The paper's loss is on (vel * f_i). It penalizes velocity if f_i=1.
                # So, we select foot velocities where contact is True (f_i=1).
                
                # Align foot_contact_gt with velocity frames (num_frames-1)
                # Use $f_i$ to penalize $v_i = x_{i+1} - x_i$. So, $f_i$ should correspond to frame $i$.
                # $v_0$ uses $x_1, x_0$. $f_0$ applies to $v_0$.
                # $v_{N-2}$ uses $x_{N-1}, x_{N-2}$. $f_{N-2}$ applies to $v_{N-2}$.
                # So, foot_contact_gt should be sliced for frames [0 to N-2] to match velocity frames.
                foot_contact_mask_for_vel = foot_contact_gt[:, :num_frames-1, :] # [bs, num_frames-1, num_joints]
                
                # Select contact mask for foot joints
                active_foot_contact = foot_contact_mask_for_vel[:, :, self.foot_joint_indices] # [bs, num_frames-1, num_foot_joints]
                active_foot_contact = active_foot_contact.unsqueeze(-1).expand_as(pred_foot_vel) # Expand to XYZ dims: [bs, N-1, n_feet, feat_per_joint]

                # Penalize foot velocities WHEN foot_contact is 1 (active_foot_contact is 1)
                # The loss is || (predicted_foot_velocity) * (foot_contact_is_1) ||^2
                # which means we want predicted_foot_velocity to be 0 when foot_contact_is_1.
                # This is equivalent to an MSE loss between (predicted_foot_velocity * foot_contact_is_1) and 0.
                foot_vel_to_penalize = pred_foot_vel * active_foot_contact # Zeroes out velocities when contact is 0
                
                l_foot_elementwise = F.mse_loss(foot_vel_to_penalize, 
                                                torch.zeros_like(foot_vel_to_penalize), 
                                                reduction='none')
                
                # Optional: Apply overall_bone_mask to foot contact loss
                # This is more complex as overall_bone_mask is [bs, N, total_features]
                # We need to select features corresponding to foot_joint_indices and slice for N-1 frames.
                if overall_bone_mask is not None:
                    # Create a specific mask for the foot joint features from overall_bone_mask
                    # This part needs careful implementation if you want fine-grained armature masking on foot loss.
                    # For simplicity now, we apply loss over all components of active foot joints.
                    # If you need to mask specific XYZ of specific feet based on overall_bone_mask:
                    mask_for_foot_loss = torch.zeros_like(foot_vel_to_penalize, device=self.device)
                    temp_overall_bone_mask_vel = overall_bone_mask[:, 1:, :] # for N-1 frames
                    for j_idx, actual_joint_idx in enumerate(self.foot_joint_indices):
                        start_feat = actual_joint_idx * self.features_per_joint
                        end_feat = start_feat + self.features_per_joint
                        mask_for_foot_loss[:, :, j_idx, :] = temp_overall_bone_mask_vel[:, :, start_feat:end_feat]
                    l_foot_elementwise = l_foot_elementwise * mask_for_foot_loss
                    num_active = mask_for_foot_loss.sum() * active_foot_contact.sum() / (active_foot_contact.numel() + 1e-8) # Approximate active for overall mask
                else:
                    # Normalize by number of elements where foot contact is 1
                    num_active = active_foot_contact.sum()

                l_foot = l_foot_elementwise.sum() / (num_active + 1e-8) if num_active > 0 else torch.tensor(0.0, device=self.device)
                total_geometric_loss = total_geometric_loss + self.lambda_foot * l_foot
                losses['geom_foot_loss'] = l_foot.item()
        
        if hasattr(total_geometric_loss, 'requires_grad') and predicted_x0.requires_grad and not total_geometric_loss.requires_grad and total_geometric_loss == 0.0:
            # Ensure loss requires grad if model is in training and loss is zero,
            # to prevent issues if this is the only loss component with non-zero weight.
            total_geometric_loss = total_geometric_loss.clone().requires_grad_(True)

        return total_geometric_loss, losses


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
                       is_training_model: bool,
                       armature_config_data: Optional[Dict] = None
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
        :param armature_config_data: Optional configuration data for the armature (e.g., bone names, joint indices).
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
            armature_class_ids, 
            num_total_motion_features, 
            num_frames_x0, 
            self.device,
            armature_config_data=armature_config_data
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
                total_kin_loss = total_kin_loss + self.velocity_loss_weight * vel_loss
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
                    total_kin_loss = total_kin_loss + self.acceleration_loss_weight * accel_loss
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
                    get_bone_mask_fn: Callable[..., torch.Tensor],
                    config: Dict[str, Any],
                    armature_config_data: Optional[Dict] = None,
                    device: str = "cuda" if torch.cuda.is_available() else "cpu",
                    lr_scheduler: Optional[Any] = None,
                    kinematic_loss_calculator: Optional[Any] = None,
                    mdm_geometric_loss_calculator: Optional[Any] = None,
                    model_save_path: Optional[str] = 'armature_mdm_best_model.pth'
                ):
        """
        Initializes the ArmatureMDMTrainer with the model, optimizer, and configuration.
        :param model: The ArmatureMDM model to be trained.
        :param optimizer: Optimizer for training the model.
        :param get_bone_mask_fn: Function to get the bone mask for active features.
        :param config: Full configuration dictionary containing training hyperparameters, loss settings, etc.
        :param armature_config_data: Optional configuration data for the armature (e.g., bone names, joint indices).
        :param device: Device to run the training on (e.g., "cuda" or "cpu").
        :param lr_scheduler: Optional learning rate scheduler for the optimizer.
        :param kinematic_loss_calculator: Optional instance of KinematicLossCalculator for computing kinematic losses.
        :param mdm_geometric_loss_calculator: Optional instance of MDMGeometricLosses for computing geometric losses.
        :param model_save_path: Path to save the best model during training.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.get_bone_mask_fn = get_bone_mask_fn
        self.armature_config_data = armature_config_data
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.config = config # Store the full config

        training_hyperparams_cfg = config.get('training_hyperparameters', {})
        self.main_loss_type = training_hyperparams_cfg.get('main_loss_type_trainer', "mse")
        self.cfg_drop_prob = training_hyperparams_cfg.get('cfg_drop_prob_trainer', 0.1)

        # Timestep loss weighting configuration
        main_x0_loss_cfg = config.get('main_x0_loss_config', {})
        self.loss_weighting_config = main_x0_loss_cfg.get('timestep_weighting', {'scheme': 'none'})
        self.loss_weighting_scheme = self.loss_weighting_config.get('scheme', 'none')
        self.min_snr_gamma_value = self.loss_weighting_config.get('min_snr_gamma_value', 5.0)

        # Noise schedule parameters for loss weighting
        diffusion_hyperparams_cfg = config.get('diffusion_hyperparameters', {})
        self.num_diffusion_timesteps = diffusion_hyperparams_cfg.get('num_diffusion_timesteps', 1000)

        if self.loss_weighting_scheme != "none":
            beta_start = diffusion_hyperparams_cfg.get('beta_start', 0.0001)
            beta_end = diffusion_hyperparams_cfg.get('beta_end', 0.02)
            # Generate alphas_cumprod on the trainer's device
            betas = torch.linspace(beta_start, beta_end, self.num_diffusion_timesteps,
                                   dtype=torch.float32, device=self.device)
            alphas = 1.0 - betas
            self.alphas_cumprod_trainer = torch.cumprod(alphas, axis=0) # Shape [T]

            if self.loss_weighting_scheme in ["snr_plus_one", "min_snr_gamma"]:
                self.snr_values = self.alphas_cumprod_trainer / (1.0 - self.alphas_cumprod_trainer + 1e-8) # Shape [T]
        
        logger.info(f"ArmatureMDMTrainer initialized. Main loss type: {self.main_loss_type}")
        logger.info(f"Timestep loss weighting scheme: '{self.loss_weighting_scheme}'")
        if self.loss_weighting_scheme == "min_snr_gamma":
            logger.info(f" -> min_snr_gamma_value: {self.min_snr_gamma_value}")

        self.kinematic_loss_calculator = kinematic_loss_calculator
        self.mdm_geometric_loss_calculator = mdm_geometric_loss_calculator
        
        early_stopping_cfg = config.get('early_stopping', {})
        self.early_stopping_patience = early_stopping_cfg.get('early_stopping_patience', 10)
        self.early_stopping_min_delta = early_stopping_cfg.get('early_stopping_min_delta', 0.0001)
        self.model_save_path = model_save_path
        
        self._early_stopping_counter = 0
        self._best_val_loss = float('inf')
        self.completed_epochs_count = 0


    def _get_timestep_loss_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Calculates loss weights based on the provided timesteps and the configured weighting scheme.
        Assumes self.alphas_cumprod_trainer and self.snr_values are initialized if needed.
        :param timesteps: Tensor of timesteps for which to calculate weights.
        :return: Tensor of weights for each timestep, shape [bs] where bs is the batch size.
        """
        if self.loss_weighting_scheme == "none":
            return torch.ones_like(timesteps, dtype=torch.float32, device=self.device)
        
        # Ensure timesteps are long for indexing
        long_timesteps = timesteps.long()

        if self.loss_weighting_scheme == "snr_plus_one":
            selected_snr = self.snr_values[long_timesteps]
            weights = selected_snr + 1.0
        elif self.loss_weighting_scheme == "inv_sigma_squared":
            # sigma_squared_t = 1 - alpha_bar_t
            selected_alphas_cumprod = self.alphas_cumprod_trainer[long_timesteps]
            weights = 1.0 / (1.0 - selected_alphas_cumprod + 1e-8) # Epsilon for stability
        elif self.loss_weighting_scheme == "min_snr_gamma":
            selected_snr = self.snr_values[long_timesteps]
            weights = torch.minimum(selected_snr, torch.tensor(self.min_snr_gamma_value, device=self.device))
        else:
            logger.warning(f"Unknown weighting scheme '{self.loss_weighting_scheme}' in _get_timestep_loss_weights. Defaulting to no weights.")
            return torch.ones_like(timesteps, dtype=torch.float32, device=self.device)
        
        # Optional: Normalize weights to have a mean of 1 for the batch.
        # This can help stabilize training if weights vary a lot.
        weights = weights / (weights.mean() + 1e-8)
        return weights

    def _calculate_masked_x0_loss(self,
                                 predicted_x0: torch.Tensor,
                                 target_x0: torch.Tensor,
                                 bone_mask: torch.Tensor,
                                 # Optional: per-sample weights for timestep-based weighting
                                 sample_timestep_weights: Optional[torch.Tensor] = None
                                 ) -> torch.Tensor:
        """
        Calculates the masked L1 or MSE loss between predicted x0 and target x0.
        This corresponds to L_simple in the MDM paper if using MSE.
        The bone_mask allows focusing the loss on specific joints/features.
        Optionally applies per-sample timestep weights if provided.

        :param predicted_x0: Tensor of predicted x0 values, shape [bs, num_frames, num_features].
        :param target_x0: Tensor of target x0 values, shape [bs, num_frames, num_features].
        :param bone_mask: Tensor of shape [bs, num_frames, num_features] indicating active features.
        :param sample_timestep_weights: Optional tensor of shape [bs] for per-sample timestep weights.
        :return: Scalar tensor representing the final loss for the batch.
        """
        if self.main_loss_type == "l1":
            element_wise_loss = F.l1_loss(predicted_x0, target_x0, reduction='none')
        elif self.main_loss_type == "mse":
            element_wise_loss = F.mse_loss(predicted_x0, target_x0, reduction='none')
        else:
            raise ValueError(f"Unsupported main_loss_type: {self.main_loss_type}. Choose 'l1' or 'mse'.")

        # Apply bone mask
        masked_element_wise_loss = element_wise_loss * bone_mask  # Shape: [bs, num_frames, num_features]

        # Calculate mean loss per sample, normalized by active elements for that sample
        sum_loss_per_sample = masked_element_wise_loss.sum(dim=[1, 2])  # Sum over frames and features -> [bs]
        num_active_per_sample = bone_mask.sum(dim=[1, 2])  # -> [bs]
        
        # Handle cases where a sample might have no active elements (though unlikely with valid masks)
        # Clamp num_active_per_sample to avoid division by zero, default loss to 0 for such samples.
        is_inactive_sample = (num_active_per_sample == 0)
        num_active_per_sample_safe = torch.clamp(num_active_per_sample, min=1e-8)
        
        mean_loss_per_sample = sum_loss_per_sample / num_active_per_sample_safe
        mean_loss_per_sample[is_inactive_sample] = 0.0 # Ensure loss is 0 if no elements were active

        if is_inactive_sample.any():
             logger.warning(f"{(is_inactive_sample.sum().item())} samples had zero active elements in bone_mask "
                            f"within _calculate_masked_x0_loss.")

        # Apply timestep-based sample weights if provided
        if sample_timestep_weights is not None:
            if sample_timestep_weights.shape[0] != mean_loss_per_sample.shape[0]:
                raise ValueError(f"Shape mismatch: sample_timestep_weights ({sample_timestep_weights.shape}) "
                                 f"and mean_loss_per_sample ({mean_loss_per_sample.shape})")
            weighted_loss_per_sample = mean_loss_per_sample * sample_timestep_weights
        else:
            weighted_loss_per_sample = mean_loss_per_sample # No timestep weighting

        # Final loss for the batch is the mean of these (potentially weighted) per-sample losses
        final_batch_loss = weighted_loss_per_sample.mean()
        
        return final_batch_loss

    def _get_tqdm_postfix_names(self, loss_components: Dict[str, float]) -> Dict[str, float]:
        """
        Helper function to shorten loss names for tqdm display.
        This function maps long loss component names to shorter versions for better readability in tqdm progress bars.
        :param loss_components: Dictionary of loss components with their values.
        :return: Dictionary with shortened names for tqdm display.
        """
        # Define your mapping here
        # The keys are the original loss component names, values are the short names
        name_map = {
            f"main_x0_loss ({self.main_loss_type}{'_ts_w' if self.loss_weighting_scheme != 'none' else ''})": "main_x0",
            "main_x0_loss_unweighted_avg": "main_x0_uw",
            "avg_timestep_weight": "ts_w_avg",
            "velocity_loss": "vl",
            "acceleration_loss": "ac_l",
            "geom_pos_loss": "gpos_l",
            "geom_vel_loss": "gvel_l",
            "geom_foot_loss": "gfoot_l"
        }
        
        short_postfix = {}
        for k, v in loss_components.items():
            short_name = name_map.get(k, k)
            if len(short_name) > 10 and short_name == k:
                parts = short_name.split('_')
                if len(parts) > 1 :
                    short_name = "".join(p[0] for p in parts if p) 
                else:
                    short_name = short_name[:8] + ".."


            short_postfix[short_name] = f"{v:.4f}"
        return short_postfix

    def _run_batch(self,
                    batch_data: Dict[str, Any],
                    is_training_model: bool # True if model.train(), False if model.eval()
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Runs a single batch through the model and computes the total loss and its components.

        :param batch_data: Dictionary containing the batch data, including:
            - "x_noisy": Noisy input tensor of shape [bs, num_frames, num_features].
            - "timesteps": Tensor of timesteps for the batch, shape [bs].
            - "target_x0": Target clean motion tensor of shape [bs, num_frames, num_features].
            - "text_embeddings": Text embeddings for the batch, shape [bs, embedding_dim].
            - "armature_class_ids": Tensor of armature class IDs for the batch, shape [bs].
        :param is_training_model: Boolean indicating if the model is in training mode.
        :return: Tuple containing:
            - current_total_loss: Scalar tensor representing the total loss for the batch.
            - loss_components: Dictionary with individual loss components for logging.
        """
        x_noisy = batch_data["x_noisy"].to(self.device)
        timesteps = batch_data["timesteps"].to(self.device)
        target_x0_for_main_loss = batch_data["target_x0"].to(self.device)
        text_embeddings_batch = batch_data["text_embeddings"].to(self.device)
        armature_class_ids = batch_data["armature_class_ids"].to(self.device)

        # Classifier-Free Guidance (CFG) logic
        cfg_uncond_text = batch_data.get('uncond_text', False)
        cfg_uncond_armature = batch_data.get('uncond_armature', False)
        if is_training_model and self.cfg_drop_prob > 0:
            if torch.rand(1).item() < self.cfg_drop_prob:
                cfg_uncond_text = True
                cfg_uncond_armature = True

        motion_padding_mask = batch_data.get('motion_padding_mask')
        if motion_padding_mask is not None:
            motion_padding_mask = motion_padding_mask.to(self.device)

        # Model Forward Pass
        predicted_x0 = self.model(
            x=x_noisy, timesteps=timesteps, text_embeddings_batch=text_embeddings_batch,
            armature_class_ids=armature_class_ids, uncond_text=cfg_uncond_text,
            uncond_armature=cfg_uncond_armature, motion_padding_mask=motion_padding_mask
        )

        _bs, num_frames, num_total_motion_features = predicted_x0.shape

        bone_mask_full_seq = self.get_bone_mask_fn(
            armature_class_ids, num_total_motion_features, num_frames,
            str(self.device), armature_config_data=self.armature_config_data
        )

        # --- Main x0 Loss Calculation (with optional timestep weighting) ---
        # Get timestep weights based on the configured scheme
        timestep_loss_w = self._get_timestep_loss_weights(timesteps) if self.loss_weighting_scheme != "none" else None
        
        main_x0_loss = self._calculate_masked_x0_loss(
            predicted_x0,
            target_x0_for_main_loss,
            bone_mask_full_seq,
            sample_timestep_weights=timestep_loss_w # Pass weights here
        )

        current_total_loss = main_x0_loss
        loss_components = {
            f"main_x0_loss ({self.main_loss_type}{'_ts_w' if self.loss_weighting_scheme != 'none' else ''})": main_x0_loss.item()
        }
        if self.loss_weighting_scheme != "none" and timestep_loss_w is not None:
            loss_components["avg_timestep_weight"] = timestep_loss_w.mean().item()
            # To log the unweighted version for comparison if weighting is active:
            unweighted_main_x0_loss_for_logging = self._calculate_masked_x0_loss(
                predicted_x0, target_x0_for_main_loss, bone_mask_full_seq, sample_timestep_weights=None
            )
            loss_components["main_x0_loss_unweighted_avg"] = unweighted_main_x0_loss_for_logging.item()

        # --- Optional Kinematic Losses ---
        if self.kinematic_loss_calculator:
            # Kinematic losses are weighted by their own internal lambda weights.
            # armature_config_data is passed for its internal bone mask generation.
            scheduler_params_for_kin_calc = batch_data.get("scheduler_params", {})
            kin_loss, kin_losses_dict = self.kinematic_loss_calculator.compute_losses(
                x_t=x_noisy, timesteps=timesteps, predicted_x0_from_model=predicted_x0,
                target_x0=target_x0_for_main_loss, scheduler_params=scheduler_params_for_kin_calc,
                armature_class_ids=armature_class_ids, is_training_model=is_training_model,
                armature_config_data=self.armature_config_data
            )
            current_total_loss = current_total_loss + kin_loss
            loss_components.update(kin_losses_dict)

        # --- Optional MDM Geometric Losses ---
        if self.mdm_geometric_loss_calculator:
            # MDM Geometric losses are weighted by their own internal lambda weights.
            # armature_config_data is passed for its internal bone mask generation.
            foot_contact_gt = batch_data.get("foot_contact_ground_truth")
            if foot_contact_gt is not None:
                foot_contact_gt = foot_contact_gt.to(self.device)
            
            mdm_geom_loss, mdm_geom_losses_dict = self.mdm_geometric_loss_calculator.compute_losses(
                predicted_x0=predicted_x0, target_x0=target_x0_for_main_loss,
                foot_contact_gt=foot_contact_gt, armature_class_ids=armature_class_ids,
                armature_config_data=self.armature_config_data
            )
            current_total_loss = current_total_loss + mdm_geom_loss
            loss_components.update(mdm_geom_losses_dict)
            
        return current_total_loss, loss_components

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

        self._early_stopping_counter = 0
        self._best_val_loss = float('inf')
        self.completed_epochs_count = 0
        
        training_history = {'train_loss': [], 'val_loss': [], 'val_metric': []} # For plotting

        for epoch in tqdm(range(1, num_epochs + 1), desc="Training Epoch", unit="epoch", leave=False): # leave=False for cleaner logs
            self.completed_epochs_count = epoch
            logger.info(f"--- Training Epoch {epoch}/{num_epochs} ---")
            avg_train_loss, avg_train_components = self.train_epoch(train_loader)
            log_train_components_str = ", ".join([f"Avg {k}: {v:.4f}" for k,v in avg_train_components.items()])
            logger.info(f"Epoch {epoch} Training Summary: Avg Total Loss: {avg_train_loss:.4f} ({log_train_components_str})")
            training_history['train_loss'].append(avg_train_loss)

            if val_loader:
                logger.info(f"--- Validating Epoch {epoch}/{num_epochs} ---")
                avg_val_loss, avg_val_components = self.evaluate_epoch(val_loader)
                log_val_components_str = ", ".join([f"Avg {k}: {v:.4f}" for k,v in avg_val_components.items()])
                logger.info(f"Epoch {epoch} Validation Summary: Avg Total Loss: {avg_val_loss:.4f} ({log_val_components_str})")
                training_history['val_loss'].append(avg_val_loss)
                # Assuming avg_val_loss is the main metric for early stopping and lr scheduling for now
                training_history['val_metric'].append(avg_val_loss)

                if avg_val_loss < self._best_val_loss - self.early_stopping_min_delta:
                    self._best_val_loss = avg_val_loss
                    self._early_stopping_counter = 0
                    logger.info(f"New best validation loss: {self._best_val_loss:.4f}. Saving model...")
                    try:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'val_loss': self._best_val_loss,
                            'config': self.config # Save config with the model
                        }, self.model_save_path)

                        logger.info(f"Model saved to {self.model_save_path}")

                    except Exception as e:
                        logger.error(f"Error saving model: {e}")
                else:
                    self._early_stopping_counter += 1
                    logger.info(f"Validation loss did not improve significantly. Early stopping counter: {self._early_stopping_counter}/{self.early_stopping_patience}")
                if self._early_stopping_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break 
            else: 
                if epoch % 10 == 0: 
                     logger.info(f"No validation loader. Saving model checkpoint at epoch {epoch}...")
                     temp_save_path = self.model_save_path.replace(".pth", f"_epoch{epoch}.pth")
                     try:

                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'config': self.config
                        }, temp_save_path)

                        logger.info(f"Model checkpoint saved to {temp_save_path}")
                     except Exception as e:
                        logger.error(f"Error saving model checkpoint: {e}")

            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and val_loader:
                    self.lr_scheduler.step(avg_val_loss)
                elif not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step()
                
                if hasattr(self.optimizer, 'param_groups'):
                     current_lr = self.optimizer.param_groups[0]['lr']
                     logger.info(f"Current learning rate: {current_lr:.6e}")
        
        logger.info(f"ArmatureMDM training finished after {self.completed_epochs_count} epochs.")
        if not val_loader and self.completed_epochs_count == num_epochs : # Only save final if full epochs run without validation
            logger.info(f"Saving final model (trained for {num_epochs} epochs without validation-based early stopping)...")
            final_save_path = self.model_save_path # Overwrite or create new name
            if os.path.exists(self.model_save_path) and self._best_val_loss == float('inf'): # If best model wasn't saved due to no val
                final_save_path = self.model_save_path.replace(".pth", "_final_no_val.pth")
            try:
                torch.save({
                    'epoch': self.completed_epochs_count,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config
                }, final_save_path)

                logger.info(f"Final model saved to {final_save_path}")
            except Exception as e:
                logger.error(f"Error saving final model: {e}")
        return training_history 


    def train_epoch(self, data_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Trains the model for one epoch using the provided DataLoader.
        :param data_loader: DataLoader for the training data.
        :return: Tuple containing the average loss for the epoch and a dictionary of loss components.
        """
        self.model.train()
        epoch_total_loss = 0.0
        epoch_loss_components_sum: Dict[str, float] = {}
        num_batches = len(data_loader)
        
        # Corrected tqdm description to use self.completed_epochs_count if available, or a generic message
        current_epoch_display = self.completed_epochs_count if hasattr(self, 'completed_epochs_count') and self.completed_epochs_count > 0 else "Current"
        batch_iterator = tqdm(data_loader, desc=f"Tr Ep {current_epoch_display}", total=num_batches, leave=True, ncols=None)

        for batch_idx, batch_data in enumerate(batch_iterator):
            self.optimizer.zero_grad()
            batch_total_loss, batch_loss_components = self._run_batch(batch_data, is_training_model=True)
            
            if torch.isnan(batch_total_loss):
                logger.warning(f"NaN loss detected at batch {batch_idx}. Skipping update. Components: {batch_loss_components}")
                continue

            batch_total_loss.backward()
            # Optional: Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_total_loss += batch_total_loss.item()
            for key, val in batch_loss_components.items():
                epoch_loss_components_sum[key] = epoch_loss_components_sum.get(key, 0.0) + val

            if (batch_idx + 1) % max(1, num_batches // 10) == 0: # Log every 10%
                log_components_str = ", ".join([f"{k}: {v:.4f}" for k,v in batch_loss_components.items()])
                logger.info(f"  Batch {batch_idx+1}/{num_batches}, Total Loss: {batch_total_loss.item():.4f} ({log_components_str})")
        
            tqdm_postfix_short = self._get_tqdm_postfix_names(batch_loss_components)
            batch_iterator.set_postfix(loss=f"{batch_total_loss.item():.4f}", **tqdm_postfix_short)

        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        avg_loss_components = {k: v / num_batches if num_batches > 0 else 0.0 for k, v in epoch_loss_components_sum.items()}
        
        return avg_epoch_loss, avg_loss_components

    def evaluate_epoch(self, data_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Evaluates the model for one epoch using the provided DataLoader.
        :param data_loader: DataLoader for the validation data.
        :return: Tuple containing the average loss for the epoch and a dictionary of loss components.
        """
        self.model.eval()
        epoch_total_loss = 0.0
        epoch_loss_components_sum: Dict[str, float] = {}
        num_batches = len(data_loader)

        current_epoch_display = self.completed_epochs_count if hasattr(self, 'completed_epochs_count') and self.completed_epochs_count > 0 else "Current"
        batch_iterator = tqdm(data_loader, desc=f"Val Ep {current_epoch_display}", total=num_batches, leave=True, ncols=None)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(batch_iterator):
                batch_total_loss, batch_loss_components = self._run_batch(batch_data, is_training_model=False)
                
                if not torch.isnan(batch_total_loss):
                    epoch_total_loss += batch_total_loss.item()
                    for key, val in batch_loss_components.items():
                        epoch_loss_components_sum[key] = epoch_loss_components_sum.get(key, 0.0) + val
                else:
                    tqdm_postfix_short = {"status": "NaN_loss"}
                    batch_iterator.set_postfix(loss="NaN", **tqdm_postfix_short)
                    continue

                if (batch_idx + 1) % max(1, num_batches // 10) == 0:
                    log_components_str = ", ".join([f"{k}: {v:.4f}" for k,v in batch_loss_components.items()])
                    logger.debug(f"  Eval Batch {batch_idx+1}/{num_batches}, Total Loss: {batch_total_loss.item():.4f} ({log_components_str})")
        
                if not torch.isnan(batch_total_loss):
                    tqdm_postfix_short = self._get_tqdm_postfix_names(batch_loss_components)
                    batch_iterator.set_postfix(loss=f"{batch_total_loss.item():.4f}", **tqdm_postfix_short)

        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        avg_loss_components = {k: v / num_batches if num_batches > 0 else 0.0 for k, v in epoch_loss_components_sum.items()}

        return avg_epoch_loss, avg_loss_components