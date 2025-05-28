import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Callable, List, Optional, Tuple
import logging
import numpy as np
from tqdm import tqdm
import os
import math

from AMDM import ArmatureMDM # Aggiunto per math.ceil

# Assumiamo che ArmatureMDM (l'ultima versione allineata) e le sue dipendenze
# (PositionalEncoding, TimestepEmbedder, InputProcessMDM, OutputProcessMDM)
# siano definite qui o importate correttamente.
# from your_model_file import ArmatureMDM, PositionalEncoding, TimestepEmbedder, InputProcessMDM, OutputProcessMDM

logger = logging.getLogger(__name__)

def mean_flat(tensor): # Da diffusion.nn
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class UniformSampler: # Semplice UniformSampler da diffusion.resample
    def __init__(self, diffusion_process_obj):
        self.diffusion = diffusion_process_obj
        # Assumiamo che diffusion_process_obj.num_timesteps esista
        self._weights = np.ones([getattr(self.diffusion, 'num_timesteps', 1000)])

    def weights(self):
        return self._weights

    def sample(self, batch_size, device):
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights_pt = torch.from_numpy(weights_np).float().to(device)
        return indices, weights_pt


class GaussianDiffusionTrainerUtil: # Classe helper per parametri di diffusione
    def __init__(self, betas: np.ndarray):
        self.betas = betas
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        # ... (altri parametri necessari da GaussianDiffusion se usati)
        logger.info(f"GaussianDiffusionTrainerUtil initialized with {self.num_timesteps} timesteps.")

    def q_sample(self, x_start, t, noise=None): # Da GaussianDiffusion.q_sample
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        
        # _extract_into_tensor è una funzione helper in MDM. La replichiamo qui per semplicità.
        def _extract_into_tensor(arr, timesteps, broadcast_shape):
            res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
            while len(res.shape) < len(broadcast_shape):
                res = res[..., None]
            return res.expand(broadcast_shape)

        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

class ArmatureMDMTrainerRevised:
    def __init__(self,
                 config: Dict[str, Any],
                 model: ArmatureMDM, # La tua ArmatureMDM (ultima versione)
                 diffusion_util: GaussianDiffusionTrainerUtil, # Oggetto per q_sample e parametri beta/alpha
                 train_loader: DataLoader,
                 get_bone_mask_fn: Callable[..., torch.Tensor],
                 armature_config_data: Optional[Dict] = None,
                 val_loader: Optional[DataLoader] = None,
                 # Parametri aggiuntivi che erano nel tuo TrainerMDM originale
                 lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None 
                ):

        self.args = config # Salva l'intera configurazione
        self.model = model
        self.diffusion_util = diffusion_util # Oggetto che gestisce i parametri di diffusione
        self.data_loader = train_loader
        self.val_data_loader = val_loader
        self.get_bone_mask_fn = get_bone_mask_fn
        self.armature_config_data = armature_config_data
        self.lr_scheduler = lr_scheduler

        # Training hyperparameters (allineati a MDM e alla tua config)
        train_cfg = self.args.get('training_hyperparameters', {})
        self.batch_size = train_cfg.get('batch_size', 32)
        self.lr = train_cfg.get('learning_rate', 1e-4)
        self.weight_decay = train_cfg.get('weight_decay', 0.0)
        self.adam_beta1 = train_cfg.get('adam_beta1', 0.9) # Valore comune
        self.adam_beta2 = train_cfg.get('adam_beta2', 0.999)
        self.num_epochs = train_cfg.get('num_epochs', 200) # Aumentato come suggerito
        
        self.log_interval_epochs = train_cfg.get('log_interval_epochs', 1) # Log a fine epoca
        self.save_interval_epochs = train_cfg.get('save_interval_epochs', 10) # Salva ogni N epoche
        
        self.cfg_drop_prob_trainer = train_cfg.get('cfg_drop_prob_trainer', 0.15)

        paths_cfg = self.args.get('paths', {})
        self.model_save_dir = paths_cfg.get('model_save_dir', './save')
        self.model_filename_base = paths_cfg.get('model_filename', 'armature_mdm_trained.pth').replace('.pth', '')
        os.makedirs(self.model_save_dir, exist_ok=True)

        self.current_epoch = 0
        self.completed_steps = 0 # Contatore di step totali se vuoi loggare/salvare per step

        self.device = torch.device(train_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)

        self.opt = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(self.adam_beta1, self.adam_beta2)
        )

        # Schedule Sampler (Uniforme per semplicità, come in MDM TrainingLoop)
        self.schedule_sampler = UniformSampler(self.diffusion_util)

        # Configurazione Loss
        self.main_loss_type = train_cfg.get('main_loss_type_trainer', "mse")
        main_x0_loss_cfg = self.args.get('main_x0_loss_config', {})
        self.loss_weighting_scheme = main_x0_loss_cfg.get('timestep_weighting', {}).get('scheme', 'none')
        self.min_snr_gamma_value = main_x0_loss_cfg.get('min_snr_gamma_value', 5.0)
        
        if self.loss_weighting_scheme != "none":
            # Precalcola valori SNR se necessario per la pesatura (come nel tuo trainer originale)
            diff_hyperparams = self.args.get('diffusion_hyperparameters', {})
            beta_start = diff_hyperparams.get('beta_start', 0.0001)
            beta_end = diff_hyperparams.get('beta_end', 0.02)
            num_timesteps = diff_hyperparams.get('num_diffusion_timesteps', 1000)
            
            betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32, device=self.device)
            alphas = 1.0 - betas
            self.alphas_cumprod_for_loss_weighting = torch.cumprod(alphas, axis=0)
            if self.loss_weighting_scheme in ["snr_plus_one", "min_snr_gamma"]:
                self.snr_for_loss_weighting = self.alphas_cumprod_for_loss_weighting / \
                                             (1.0 - self.alphas_cumprod_for_loss_weighting + 1e-8)

        # Pesi Loss Ausiliarie (dalla tua config)
        kin_cfg = self.args.get('kinematic_losses', {})
        self.use_kinematic_losses = kin_cfg.get('use_kinematic_losses', False)
        self.lambda_kin_vel = kin_cfg.get('velocity_loss_weight', 0.0)
        self.lambda_kin_accel = kin_cfg.get('acceleration_loss_weight', 0.0)
        self.kinematic_loss_type = kin_cfg.get('kinematic_loss_type', 'l1')

        geom_cfg = self.args.get('mdm_geometric_losses', {})
        self.use_mdm_geometric_losses = geom_cfg.get('use_mdm_geometric_losses', False)
        self.lambda_geom_pos = geom_cfg.get('lambda_pos', 0.0)
        self.lambda_geom_vel = geom_cfg.get('lambda_vel', 0.1) # Default aumentato
        self.lambda_geom_foot = geom_cfg.get('lambda_foot', 0.0)
        self.foot_joint_indices = geom_cfg.get('foot_joint_indices', [10, 11]) # Esempio
        
        model_hyperparams = self.args.get('model_hyperparameters', {})
        self.num_joints_for_geom = model_hyperparams.get('num_joints_for_geom', 22)
        self.features_per_joint_for_geom = model_hyperparams.get('features_per_joint_for_geom', 3)

        # Early stopping
        early_stop_cfg = self.args.get('early_stopping', {})
        self.early_stopping_patience = early_stop_cfg.get('early_stopping_patience', 10)
        self.early_stopping_min_delta = early_stop_cfg.get('early_stopping_min_delta', 0.0001)
        self._early_stopping_counter = 0
        self._best_val_loss = float('inf')
        self.model_save_path = os.path.join(self.model_save_dir, f"{self.model_filename_base}_best.pth")

        logger.info("ArmatureMDMTrainerRevised initialized.")
        logger.info(f"Main loss: {self.main_loss_type}, Timestep weighting: {self.loss_weighting_scheme}")
        logger.info(f"Kinematic losses: Use={self.use_kinematic_losses}, Vel_w={self.lambda_kin_vel}, Accel_w={self.lambda_kin_accel}")
        logger.info(f"Geometric losses: Use={self.use_mdm_geometric_losses}, Pos_w={self.lambda_geom_pos}, Vel_w={self.lambda_geom_vel}, Foot_w={self.lambda_geom_foot}")


    def _get_timestep_loss_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        if self.loss_weighting_scheme == "none" or not hasattr(self, 'alphas_cumprod_for_loss_weighting'):
            return torch.ones_like(timesteps, dtype=torch.float32, device=self.device)
        long_timesteps = timesteps.long()
        if self.loss_weighting_scheme == "snr_plus_one":
            selected_snr = self.snr_for_loss_weighting[long_timesteps]
            weights = selected_snr + 1.0
        elif self.loss_weighting_scheme == "min_snr_gamma":
            selected_snr = self.snr_for_loss_weighting[long_timesteps]
            weights = torch.minimum(selected_snr, torch.tensor(self.min_snr_gamma_value, device=self.device))
        else:
            logger.warning(f"Unknown weighting scheme '{self.loss_weighting_scheme}'. Defaulting to ones.")
            return torch.ones_like(timesteps, dtype=torch.float32, device=self.device)
        return weights / (weights.mean() + 1e-8) # Normalize

    def _get_combined_mask(self, temporal_mask, bone_mask, target_shape):
        bs, nframes, nfeatures = target_shape[0], target_shape[1], target_shape[2]
        if temporal_mask is not None: # temporal_mask: [bs,1,1,nfr_orig_tempo], True=valido
            current_nframes_in_mask = temporal_mask.shape[-1]
            # Adatta la maschera temporale alla lunghezza corrente (es. per velocità/accel)
            safe_nframes = min(nframes, current_nframes_in_mask)
            processed_temporal_mask = temporal_mask[..., :safe_nframes].squeeze(1).permute(0, 2, 1)[:,:nframes,:] # -> [bs, nframes, 1]
        else:
            processed_temporal_mask = torch.ones(bs, nframes, 1, device=bone_mask.device, dtype=torch.bool)
        
        # bone_mask: [bs, nfr_orig_bone, nfeatures], 1.0=attivo. Adatta nfr_orig_bone a nframes.
        safe_nframes_bone = min(nframes, bone_mask.shape[1])
        processed_bone_mask = bone_mask[:, :safe_nframes_bone, :].bool() # -> [bs, nframes, nfeatures]

        # Estendi la maschera più corta se necessario
        if processed_temporal_mask.shape[1] > processed_bone_mask.shape[1]: # Se bone_mask è più corta (improbabile)
            padding = torch.zeros(bs, processed_temporal_mask.shape[1] - processed_bone_mask.shape[1], nfeatures,
                                  dtype=torch.bool, device=bone_mask.device)
            processed_bone_mask = torch.cat((processed_bone_mask, padding), dim=1)
        elif processed_bone_mask.shape[1] > processed_temporal_mask.shape[1]: # Se temporal_mask è più corta
             padding = torch.zeros(bs, processed_bone_mask.shape[1] - processed_temporal_mask.shape[1], 1,
                                  dtype=torch.bool, device=bone_mask.device)
             processed_temporal_mask = torch.cat((processed_temporal_mask, padding), dim=1)


        combined_mask_float = processed_temporal_mask.float() * processed_bone_mask.float()
        return combined_mask_float

    def _calculate_masked_loss(self, prediction, target, combined_mask, loss_type, sample_timestep_weights=None):
        if loss_type == "mse":
            element_wise_loss = F.mse_loss(prediction, target, reduction='none')
        elif loss_type == "l1":
            element_wise_loss = F.l1_loss(prediction, target, reduction='none')
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        
        # Assicurati che combined_mask abbia le stesse dimensioni spaziali/feature di element_wise_loss
        # element_wise_loss è [bs, nframes, nfeatures]
        # combined_mask è [bs, nframes, nfeatures]
        masked_element_wise_loss = element_wise_loss * combined_mask
        
        sum_loss_per_sample = masked_element_wise_loss.sum(dim=list(range(1, masked_element_wise_loss.ndim)))
        num_active_elements_per_sample = combined_mask.sum(dim=list(range(1, combined_mask.ndim)))
        
        num_active_per_sample_safe = torch.clamp(num_active_elements_per_sample, min=1e-8)
        mean_loss_per_sample = sum_loss_per_sample / num_active_per_sample_safe
        
        is_inactive_sample = (num_active_elements_per_sample == 0)
        mean_loss_per_sample[is_inactive_sample] = 0.0

        if sample_timestep_weights is not None:
            weighted_loss_per_sample = mean_loss_per_sample * sample_timestep_weights
        else:
            weighted_loss_per_sample = mean_loss_per_sample
            
        final_batch_loss = weighted_loss_per_sample.mean()
        return final_batch_loss

    def _get_derivatives(self, motion_sequence: torch.Tensor):
        # motion_sequence: [bs, nframes, nfeatures]
        vel = torch.empty(motion_sequence.shape[0], 0, motion_sequence.shape[2], device=motion_sequence.device, dtype=motion_sequence.dtype)
        accel = torch.empty(motion_sequence.shape[0], 0, motion_sequence.shape[2], device=motion_sequence.device, dtype=motion_sequence.dtype)

        if motion_sequence.shape[1] >= 2:
            vel = motion_sequence[:, 1:] - motion_sequence[:, :-1]
            if vel.shape[1] >= 2:
                accel = vel[:, 1:] - vel[:, :-1]
        return vel, accel

    def _compute_losses_dict(self, x_start_gt, predicted_x0, t, y_cond, bone_mask):
        """Calcola tutte le componenti della loss."""
        terms = {}
        current_total_loss = torch.tensor(0.0, device=self.device)
        
        temporal_mask = y_cond.get('mask') # [bs, 1, 1, nframes_gt], True per validi

        # Loss principale
        combined_mask_x0 = self._get_combined_mask(temporal_mask, bone_mask, x_start_gt.shape)
        timestep_weights = self._get_timestep_loss_weights(t)
        
        main_loss_val = self._calculate_masked_loss(
            predicted_x0, x_start_gt, combined_mask_x0, self.main_loss_type,
            sample_timestep_weights=timestep_weights
        )
        terms[f"main_{self.main_loss_type}"] = main_loss_val.item()
        current_total_loss += main_loss_val
        if timestep_weights is not None:
            terms["avg_ts_w"] = timestep_weights.mean().item()

        # Loss ausiliarie
        pred_vel, pred_accel = self._get_derivatives(predicted_x0)
        target_vel, target_accel = self._get_derivatives(x_start_gt)

        if self.use_kinematic_losses:
            if self.lambda_kin_vel > 0 and pred_vel.shape[1] > 0:
                mask_vel_temp = temporal_mask[..., :pred_vel.shape[1]] if temporal_mask is not None else None
                bone_mask_vel = bone_mask[:, :pred_vel.shape[1], :]
                combined_mask_kin_vel = self._get_combined_mask(mask_vel_temp, bone_mask_vel, target_vel.shape)
                loss_val = self._calculate_masked_loss(pred_vel, target_vel, combined_mask_kin_vel, self.kinematic_loss_type)
                terms["kin_vel"] = loss_val.item()
                current_total_loss += self.lambda_kin_vel * loss_val

            if self.lambda_kin_accel > 0 and pred_accel.shape[1] > 0:
                mask_accel_temp = temporal_mask[..., :pred_accel.shape[1]] if temporal_mask is not None else None
                bone_mask_accel = bone_mask[:, :pred_accel.shape[1], :]
                combined_mask_kin_accel = self._get_combined_mask(mask_accel_temp, bone_mask_accel, target_accel.shape)
                loss_val = self._calculate_masked_loss(pred_accel, target_accel, combined_mask_kin_accel, self.kinematic_loss_type)
                terms["kin_accel"] = loss_val.item()
                current_total_loss += self.lambda_kin_accel * loss_val
        
        if self.use_mdm_geometric_losses:
            if self.lambda_geom_pos > 0: # L_rcxyz in MDM
                loss_val = self._calculate_masked_loss(predicted_x0, x_start_gt, combined_mask_x0, "mse")
                terms["geom_pos"] = loss_val.item()
                current_total_loss += self.lambda_geom_pos * loss_val
            
            if self.lambda_geom_vel > 0 and pred_vel.shape[1] > 0: # L_vel in MDM
                mask_vel_temp_geom = temporal_mask[..., :pred_vel.shape[1]] if temporal_mask is not None else None
                bone_mask_vel_geom = bone_mask[:, :pred_vel.shape[1], :]
                combined_mask_geom_vel = self._get_combined_mask(mask_vel_temp_geom, bone_mask_vel_geom, target_vel.shape)
                loss_val = self._calculate_masked_loss(pred_vel, target_vel, combined_mask_geom_vel, "mse")
                terms["geom_vel"] = loss_val.item()
                current_total_loss += self.lambda_geom_vel * loss_val

            if self.lambda_geom_foot > 0 and y_cond.get("foot_contact_ground_truth") is not None and pred_vel.shape[1] > 0 : # L_fc in MDM
                # Implementazione L_foot (complessa, richiede mapping preciso features/piedi e bone_mask)
                # Per ora, la omettiamo per brevità, ma la logica andrebbe qui,
                # adattando dalla tua classe MDMGeometricLosses precedente.
                # Si userebbero self.foot_joint_indices, self.num_joints_for_geom, self.features_per_joint_for_geom
                # e si dovrebbe estrarre la porzione rilevante della bone_mask.
                # terms["geom_foot"] = loss_foot_val.item()
                # current_total_loss += self.lambda_geom_foot * loss_foot_val
                pass

        terms["loss"] = current_total_loss.item() # Loss totale finale
        return current_total_loss, terms


    def run_step(self, batch_data):
        """ Corrisponde a TrainLoop.run_step """
        self.model.train()
        self.opt.zero_grad()

        x_start_gt = batch_data["target_x0"].to(self.device) # Ground truth motion
        
        y_cond = {
            'text_embeddings_batch': batch_data["text_embeddings"].to(self.device),
            'armature_class_ids': batch_data["armature_class_ids"].to(self.device),
            'mask': batch_data.get('motion_padding_mask_for_loss').to(self.device) if batch_data.get('motion_padding_mask_for_loss') is not None else None,
            'foot_contact_ground_truth': batch_data.get("foot_contact_ground_truth").to(self.device) if batch_data.get("foot_contact_ground_truth") is not None else None,
            'uncond': False, 'uncond_text': False, 'uncond_armature': False
        }

        t, _ = self.schedule_sampler.sample(x_start_gt.shape[0], self.device) # Campiona timesteps
        
        # Applica CFG dropout flags in training
        if self.cfg_drop_prob_trainer > 0:
            if torch.rand(1).item() < self.cfg_drop_prob_trainer:
                y_cond['uncond'] = True # Usato da _mask_cond nel modello
                # Potresti anche impostare y_cond['uncond_text'] e y_cond['uncond_armature'] = True

        noise = torch.randn_like(x_start_gt)
        x_t = self.diffusion_util.q_sample(x_start_gt, t, noise=noise) # Crea input rumoroso

        # Esegui il forward pass e calcola le loss
        # Il modello ArmatureMDM ora si aspetta x_t, t, e y_cond
        predicted_x0 = self.model(x_t, t, y_cond)

        bs, nframes, nfeatures = x_start_gt.shape
        bone_mask = self.get_bone_mask_fn(
            y_cond['armature_class_ids'], nfeatures, nframes, str(self.device),
            armature_config_data=self.armature_config_data
        )
        
        # Calcola tutte le loss
        total_loss_tensor, losses_dict_batch_float = self._compute_losses_dict(
            x_start_gt, predicted_x0, t, y_cond, bone_mask
        )
        
        total_loss_tensor.backward()
        self.opt.step()
        # self._anneal_lr() # Se implementato
        
        return losses_dict_batch_float


    def run_loop(self): # Corrisponde a TrainLoop.run_loop
        logger.info(f"Starting training for {self.num_epochs} epochs.")
        
        for epoch_idx in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch_idx
            logger.info(f"--- Epoch {self.current_epoch + 1}/{self.num_epochs} ---")
            
            epoch_loss_components_sum = {}
            num_batches_in_epoch = len(self.data_loader)

            # La barra tqdm per le epoche è gestita qui
            # La barra tqdm per i batch è stata rimossa come da tua richiesta precedente
            for batch_idx, batch_data in enumerate(self.data_loader):
                losses_batch = self.run_step(batch_data)
                self.completed_steps += 1

                for key, val in losses_batch.items():
                    epoch_loss_components_sum[key] = epoch_loss_components_sum.get(key, 0.0) + val
                
                # Log per step (MDM style, meno frequente del log per batch)
                # if self.completed_steps % self.args.get('training_hyperparameters',{}).get('log_interval_mdm_style', 1000) == 0:
                #     log_str_step = f"Step {self.completed_steps}: " + " | ".join([f"{k}: {v:.4f}" for k,v in losses_batch.items()])
                #     logger.info(log_str_step)
            
            # Log di fine epoca
            avg_epoch_losses = {k: v / num_batches_in_epoch for k, v in epoch_loss_components_sum.items()}
            log_train_summary = f"Epoch {self.current_epoch + 1} Training Summary: " + \
                                ", ".join([f"Avg {k}: {v:.4f}" for k,v in avg_epoch_losses.items()])
            logger.info(log_train_summary)

            # Validazione a fine epoca
            if self.val_data_loader is not None:
                avg_val_loss = self.evaluate() # evaluate() gestirà il logging della validazione
                if self.lr_scheduler and isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(avg_val_loss)
                
                if avg_val_loss < self._best_val_loss - self.early_stopping_min_delta:
                    self._best_val_loss = avg_val_loss
                    self._early_stopping_counter = 0
                    logger.info(f"New best validation loss: {self._best_val_loss:.4f}. Saving best model...")
                    self.save_checkpoint("best")
                else:
                    self._early_stopping_counter += 1
                    logger.info(f"Validation loss did not improve. Counter: {self._early_stopping_counter}/{self.early_stopping_patience}")
                    if self._early_stopping_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping triggered at epoch {self.current_epoch + 1}.")
                        break
            
            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Salva checkpoint periodicamente (basato su epoche invece che step per semplicità qui)
            if (self.current_epoch + 1) % self.args.get('training_hyperparameters',{}).get('save_interval_epochs', 10) == 0:
                self.save_checkpoint(f"epoch_{self.current_epoch+1}")
            
        self.save_checkpoint("final") # Salvataggio finale
        logger.info(f"Training finished after {self.current_epoch + 1} epochs / {self.completed_steps} steps.")


    def evaluate(self): # Simile al tuo evaluate_epoch
        self.model.eval()
        total_val_loss = 0.0
        val_loss_components_sum = {}
        
        with torch.no_grad():
            for batch_data in self.val_data_loader:
                x_start_gt = batch_data["target_x0"].to(self.device)
                y_cond = {
                    'text_embeddings_batch': batch_data["text_embeddings"].to(self.device),
                    'armature_class_ids': batch_data["armature_class_ids"].to(self.device),
                    'mask': batch_data.get('motion_padding_mask_for_loss').to(self.device) if batch_data.get('motion_padding_mask_for_loss') is not None else None,
                    'foot_contact_ground_truth': batch_data.get("foot_contact_ground_truth").to(self.device) if batch_data.get("foot_contact_ground_truth") is not None else None,
                    'uncond': False, 'uncond_text': False, 'uncond_armature': False # No CFG dropout for validation loss
                }
                t, _ = self.schedule_sampler.sample(x_start_gt.shape[0], self.device)
                
                # Forward pass per ottenere predicted_x0
                predicted_x0 = self.model(self.diffusion_util.q_sample(x_start_gt, t, noise=torch.randn_like(x_start_gt)), t, y_cond)

                bs, nframes, nfeatures = x_start_gt.shape
                bone_mask = self.get_bone_mask_fn(
                    y_cond['armature_class_ids'], nfeatures, nframes, str(self.device),
                    armature_config_data=self.armature_config_data
                )
                
                # Calcola loss per questo batch di validazione
                _, losses_dict_val_batch = self._compute_losses_dict(
                    x_start_gt, predicted_x0, t, y_cond, bone_mask
                )

                total_val_loss += losses_dict_val_batch["loss"] # "loss" è la loss totale combinata
                for key, val in losses_dict_val_batch.items():
                    val_loss_components_sum[key] = val_loss_components_sum.get(key, 0.0) + val
        
        avg_val_loss = total_val_loss / len(self.val_data_loader) if len(self.val_data_loader) > 0 else 0
        log_val_summary = f"Epoch {self.current_epoch + 1} Validation Summary: " + \
                          ", ".join([f"Avg {k}: {v / len(self.val_data_loader) if len(self.val_data_loader) > 0 else 0:.4f}" for k,v in val_loss_components_sum.items()])
        logger.info(log_val_summary)
        
        self.model.train()
        return avg_val_loss # Ritorna la loss totale media per early stopping / LR scheduler

    def save_checkpoint(self, suffix: str):
        filename = f"{self.model_filename_base}_{suffix}.pth"
        save_path = os.path.join(self.model_save_dir, filename)
        logger.info(f"Saving model checkpoint to {save_path} (Epoch: {self.current_epoch+1}, Step: {self.completed_steps})")
        
        state_to_save = {
            'epoch': self.current_epoch,
            'completed_steps': self.completed_steps,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'config': self.args, # Salva la configurazione usata
            '_best_val_loss': self._best_val_loss, # Per resume
            '_early_stopping_counter': self._early_stopping_counter # Per resume
        }
        if self.lr_scheduler is not None:
            state_to_save['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        torch.save(state_to_save, save_path)
        logger.info(f"Checkpoint saved: {save_path}")

    # _anneal_lr e altri metodi helper da MDM TrainLoop possono essere aggiunti qui se necessario
    # Esempio:
    # def _anneal_lr(self):
    #     if not self.lr_anneal_steps:
    #         return
    #     frac_done = self.completed_steps / self.lr_anneal_steps
    #     lr = self.lr * (1 - frac_done)
    #     for param_group in self.opt.param_groups:
    #         param_group["lr"] = lr