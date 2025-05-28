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

from AMDM import ArmatureMDM

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)


class GaussianDiffusionSamplerUtil: # Classe helper per parametri di campionamento
    def __init__(self, betas: np.ndarray, model_mean_type: str, model_var_type: str, loss_type: str): # Aggiunti model_mean_type etc.
        self.betas = betas
        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
        # Questi dovrebbero corrispondere a ModelMeanType, ModelVarType, LossType di MDM
        self.model_mean_type_enum = self._parse_model_mean_type(model_mean_type)
        self.model_var_type_enum = self._parse_model_var_type(model_var_type)
        # self.loss_type_enum = self._parse_loss_type(loss_type) # Non serve per il campionamento
        logger.info(f"GaussianDiffusionSamplerUtil initialized for {self.num_timesteps} timesteps.")

    def _parse_model_mean_type(self, mean_type_str: str):
        if mean_type_str.upper() == "START_X": return "START_X" # Simula enum
        elif mean_type_str.upper() == "EPSILON": return "EPSILON"
        else: raise ValueError(f"Unknown model_mean_type: {mean_type_str}")

    def _parse_model_var_type(self, var_type_str: str):
        if var_type_str.upper() == "FIXED_SMALL": return "FIXED_SMALL"
        elif var_type_str.upper() == "FIXED_LARGE": return "FIXED_LARGE"
        else: raise ValueError(f"Unknown model_var_type: {var_type_str}")


    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        # ... (come definito prima) ...
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def _predict_xstart_from_eps(self, x_t, t, eps): # Da GaussianDiffusion
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def p_mean_variance(self, model_func, x_t, t, model_kwargs=None, clip_denoised=True): # Adattato da GaussianDiffusion
        """ Calcola la media e la varianza della p(x_{t-1} | x_t) e la predizione di x_0. """
        if model_kwargs is None: model_kwargs = {}
        
        model_output_x0 = model_func(x_t, t, model_kwargs) # Il nostro modello predice x0

        # Assumendo che model_output_x0 sia la predizione di x_start
        # (dato che il tuo ArmatureMDM è progettato per predire x0)
        if self.model_mean_type_enum != "START_X":
             raise NotImplementedError("Sampling logic here assumes model predicts START_X (x0)")
        
        pred_xstart = model_output_x0
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1) # MDM clippa x_start

        # Calcola la media della posterior q(x_{t-1} | x_t, x_0) usando la x_0 predetta
        model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x_t, t)

        # Varianza (MDM usa FIXED_SMALL o FIXED_LARGE di solito per il campionamento)
        if self.model_var_type_enum == "FIXED_SMALL":
            model_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
            model_log_variance = self._extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        elif self.model_var_type_enum == "FIXED_LARGE":
            model_variance = self._extract_into_tensor(np.append(self.posterior_variance[1], self.betas[1:]), t, x_t.shape)
            model_log_variance = self._extract_into_tensor(np.log(np.append(self.posterior_variance[1], self.betas[1:])), t, x_t.shape)
        else:
            raise NotImplementedError(f"Variance type {self.model_var_type_enum} not implemented for sampling")

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def q_posterior_mean_variance(self, x_start, x_t, t): # Da GaussianDiffusion
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # posterior_variance (calcolato in __init__)
        # posterior_log_variance_clipped (calcolato in __init__)
        return posterior_mean, None, None # Varianza e log_var non usate direttamente qui, prese da p_mean_variance

    @torch.no_grad()
    def p_sample_loop(self, model_func, shape, model_kwargs=None, device=None, clip_denoised=True, progress=False, const_noise=False):
        """ Simile a GaussianDiffusion.p_sample_loop """
        if device is None: device = next(model_func.__self__.parameters()).device # model_func è un metodo legato
        
        img = torch.randn(*shape, device=device) # Immagine/moto iniziale rumoroso
        
        indices = list(range(self.num_timesteps))[::-1]
        if progress: indices = tqdm(indices, desc="Sampling Loop")

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            
            # Prepara y_cond per il forward pass del modello
            # Aggiungi CFG scale qui se necessario (MDM lo fa esternamente)
            current_model_kwargs = model_kwargs # y_cond

            out = self.p_mean_variance(model_func, img, t, model_kwargs=current_model_kwargs, clip_denoised=clip_denoised)
            
            noise = torch.randn_like(img)
            if const_noise and i != 0: # Applica solo se non è l'ultimo step e const_noise è True
                 # Usa lo stesso rumore per tutti gli elementi del batch e mantienilo costante (o quasi)
                 # Questa è una semplificazione. MDM ha logica più complessa se usa const_noise.
                 # Solitamente, se const_noise, noise viene generato una volta all'inizio.
                 # Per ora, lo ricalcoliamo ma potremmo fissarlo esternamente.
                 # La logica MDM per const_noise in p_sample non è banale da replicare qui senza il modello completo.
                 # Semplifichiamo: se const_noise è True, il rumore sarà lo stesso per tutti gli elementi del batch in questo step.
                 if 'const_noise_val' not in locals() or i == indices[0]: # Genera all'inizio o se non esiste
                     const_noise_val = torch.randn_like(img[[0]]).repeat(img.shape[0], 1, 1, 1)
                 if i != 0 : noise = const_noise_val


            nonzero_mask = (t != 0).float().view(-1, *([1] * (len(img.shape) - 1))) # No noise at t=0
            
            sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            img = sample
            
        return img # Questo è il predicted_x0 finale


@torch.no_grad()
def generate_motion_mdm_style(
    armature_mdm_model: ArmatureMDM, # Il tuo modello ArmatureMDM (ultima versione)
    diffusion_sampler_util: GaussianDiffusionSamplerUtil, # Oggetto per il campionamento
    y_conditions: Dict[str, Any], # Dizionario con 'text_embeddings_batch', 'armature_class_ids', (opzionale 'mask')
                                   # e flags per CFG: 'cfg_scale', 'uncond_text', 'uncond_armature'
    num_frames: int,
    device: str = 'cuda',
    clip_denoised: bool = True,
    progress: bool = True,
    const_noise: bool = False # Se usare rumore costante (MDM ha questa opzione)
):
    """
    Genera un campione di movimento usando la logica di campionamento DDPM (predizione x0)
    ispirata a MDM.
    """
    armature_mdm_model.eval()
    armature_mdm_model.to(device)

    batch_size = y_conditions['text_embeddings_batch'].shape[0]
    num_motion_features = armature_mdm_model.input_feats # Dal modello

    shape = (batch_size, num_frames, num_motion_features) # Attenzione: ArmatureMDM internamente permuta a [nfr, bs, feat]
                                                        # ma il suo input/output esterno è [bs, nfr, feat]

    # --- Classifier-Free Guidance (CFG) ---
    # MDM la implementa avvolgendo il modello in ClassifierFreeSampleModel.
    # Qui, la simuliamo passando y_cond due volte al modello se cfg_scale > 1.
    cfg_scale = y_conditions.get('cfg_scale', 1.0) # Default a 1.0 (no CFG o condizionale puro)

    if cfg_scale != 1.0:
        # Prepara y_cond per la predizione condizionata
        y_cond_input = {k: v for k, v in y_conditions.items() if k != 'cfg_scale'}
        y_cond_input['uncond'] = False # Flag per il modello
        y_cond_input['uncond_text'] = False
        y_cond_input['uncond_armature'] = False

        # Prepara y_uncond per la predizione incondizionata
        y_uncond_input = {k: v for k, v in y_conditions.items() if k != 'cfg_scale'}
        y_uncond_input['uncond'] = True # Flag per il modello
        y_uncond_input['uncond_text'] = True # Maschera testo
        y_uncond_input['uncond_armature'] = True # Maschera armatura
        
        # Funzione modello che internamente gestisce CFG
        def model_fn_cfg(x_t, t, y_private_cond): # y_private_cond è il dizionario di condizioni specifico
            # y_private_cond non conterrà cfg_scale, è già gestito qui
            return armature_mdm_model(x_t, t, y_private_cond)

        def combined_model_fn(x_t, t, _): # L'ultimo arg non usato, le y sono catturate dallo scope
            out_cond = model_fn_cfg(x_t, t, y_cond_input)
            out_uncond = model_fn_cfg(x_t, t, y_uncond_input)
            return out_uncond + cfg_scale * (out_cond - out_uncond)
        
        model_to_sample_from = combined_model_fn
        # model_kwargs per p_sample_loop non serviranno per 'y' perché catturate in combined_model_fn
        final_model_kwargs = {} 

    else: # No CFG o cfg_scale = 1.0 (solo condizionale)
        y_cond_input = {k: v for k, v in y_conditions.items() if k != 'cfg_scale'}
        y_cond_input['uncond'] = False
        y_cond_input['uncond_text'] = y_conditions.get('uncond_text', False)
        y_cond_input['uncond_armature'] = y_conditions.get('uncond_armature', False)

        # model_to_sample_from = armature_mdm_model.forward # Passa direttamente il metodo forward
        # Questo non funziona perché .forward non è legato all'istanza in questo modo.
        # Creiamo una lambda.
        model_to_sample_from = lambda x_t, t, y_lambda_cond: armature_mdm_model(x_t, t, y_lambda_cond)
        final_model_kwargs = y_cond_input


    # Esegui il loop di campionamento
    # `p_sample_loop` di diffusion_sampler_util si aspetta che model_func sia una funzione
    # che prende (x_t, t, model_kwargs) e restituisce la predizione di x0.
    # La nostra `combined_model_fn` o la lambda per `armature_mdm_model` fanno questo.
    generated_motion = diffusion_sampler_util.p_sample_loop(
        model_func=model_to_sample_from,
        shape=shape,
        model_kwargs=final_model_kwargs, # Passa il dizionario y corretto
        device=device,
        clip_denoised=clip_denoised,
        progress=progress,
        const_noise=const_noise
    )
    # L'output di p_sample_loop è già il predicted_x0 finale, forma [bs, num_frames, features]
    
    logger.info(f"Generazione completata. Shape output: {generated_motion.shape}")
    return generated_motion