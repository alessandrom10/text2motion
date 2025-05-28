import torch
from torch.utils.data import Dataset # Rimosso DataLoader se non usato qui direttamente
# from torch.nn.utils.rnn import pad_sequence # pad_sequence non è usato in collate_motion_data_revised
import pandas as pd
import numpy as np
import os
import logging
from typing import Callable, Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Funzioni di Utilità Default per la Diffusione ---
def default_uniform_timestep_sampler(num_diffusion_timesteps: int) -> int:
    return torch.randint(0, num_diffusion_timesteps, (1,)).item()

def default_gaussian_noise_fn(target_x0_shape: tuple, device: torch.device) -> torch.Tensor:
    return torch.randn(target_x0_shape, device=device)

def default_ddpm_noising_fn(target_x0: torch.Tensor,
                             epsilon: torch.Tensor,
                             t: int, # t è un intero
                             alphas_cumprod: torch.Tensor) -> torch.Tensor: # alphas_cumprod è un tensore
    """
    Applica rumore a target_x0 secondo la formula del processo forward DDPM.
    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * epsilon
    """
    # _extract_into_tensor è una funzione helper interna
    def _extract_into_tensor(arr_tensor: torch.Tensor, timestep_int: int, broadcast_shape: tuple) -> torch.Tensor:
        # arr_tensor è atteso essere un tensore PyTorch 1D (es. sqrt_alphas_cumprod)
        # ed è già sul dispositivo corretto (ereditato da alphas_cumprod).
        # timestep_int è un intero scalare.
        
        res = arr_tensor[timestep_int].float() # Indicizzazione con un intero
        
        temp_res = res
        while len(temp_res.shape) < len(broadcast_shape):
            temp_res = temp_res[..., None]
        return temp_res.expand(broadcast_shape)

    # alphas_cumprod è già un tensore sul data_device corretto
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod)

    coeff1 = _extract_into_tensor(sqrt_alphas_cumprod_t, t, target_x0.shape)
    coeff2 = _extract_into_tensor(sqrt_one_minus_alphas_cumprod_t, t, target_x0.shape)
    
    return coeff1 * target_x0 + coeff2 * epsilon

# --- Funzione di Processamento del Movimento ---
def process_motion_file_with_normalization(
    file_path: str,
    num_expected_features: int,
    dataset_mean: np.ndarray, 
    dataset_std: np.ndarray   
) -> torch.Tensor:
    """
    Carica, normalizza (Z-score), e appiattisce un singolo file di movimento .npy.
    Restituisce x0 pulito e normalizzato.
    """
    try:
        motion_data_raw = np.load(file_path).astype(np.float32) 
    except Exception as e:
        logger.error(f"Errore nel caricamento del file di movimento {file_path}: {e}")
        raise
    
    if motion_data_raw.ndim == 3: 
        if motion_data_raw.shape[1] * motion_data_raw.shape[2] == num_expected_features:
            # Opzionale: Root centering (valuta se farlo prima della normalizzazione Z-score)
            # motion_data_raw = motion_data_raw - motion_data_raw[:, 0:1, :] 
            motion_data_flat = motion_data_raw.reshape(motion_data_raw.shape[0], -1)
        else:
            raise ValueError(f"Shape del file {file_path} ({motion_data_raw.shape}) non compatibile con "
                             f"num_expected_features ({num_expected_features}) se appiattita da (T, J, C).")
    elif motion_data_raw.ndim == 2 and motion_data_raw.shape[1] == num_expected_features:
        motion_data_flat = motion_data_raw
    else:
        raise ValueError(f"Shape del file {file_path} ({motion_data_raw.shape}) non gestita. "
                         f"Attesa (T, J, C) o (T, num_expected_features).")

    if motion_data_flat.shape[1] != num_expected_features:
        raise ValueError(f"Mismatch di feature in {file_path}: attese {num_expected_features}, ottenute {motion_data_flat.shape[1]}.")

    std_safe = np.where(dataset_std == 0, 1e-8, dataset_std) # Evita divisione per zero
    normalized_motion = (motion_data_flat - dataset_mean) / std_safe
    
    return torch.from_numpy(normalized_motion).float()


class MyTextToMotionDatasetNormalized(Dataset):
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
            logger.info(f"Caricamento SBERT embeddings precalcolati da: {precomputed_sbert_embeddings_path}")
            self.sbert_embeddings_cache = torch.load(precomputed_sbert_embeddings_path, map_location=self.data_device)
            logger.info(f"Caricati {len(self.sbert_embeddings_cache)} SBERT embeddings.")
        except Exception as e:
            logger.error(f"Errore nel caricamento SBERT embeddings da {precomputed_sbert_embeddings_path}: {e}")
            raise

        try:
            self.dataset_mean = np.load(dataset_mean_path).astype(np.float32)
            self.dataset_std = np.load(dataset_std_path).astype(np.float32)
            if self.dataset_mean.shape[0] != self.num_motion_features or \
               self.dataset_std.shape[0] != self.num_motion_features:
                raise ValueError(f"Shape di mean/std ({self.dataset_mean.shape}, {self.dataset_std.shape}) "
                                 f"non corrisponde a num_motion_features ({self.num_motion_features})")
            logger.info(f"Caricati mean e std del dataset per normalizzazione.")
        except Exception as e:
            logger.error(f"Errore nel caricamento di mean/std del dataset da {dataset_mean_path}/{dataset_std_path}: {e}")
            raise
            
        self._prepare_samples()

    def _prepare_samples(self):
        self.samples = []
        try:
            annotations_df = pd.read_csv(self.annotations_path)
        except Exception as e:
            logger.error(f"Errore nel leggere il file di annotazioni {self.annotations_path}: {e}")
            raise

        for idx, row in annotations_df.iterrows():
            motion_filename = row['motion_filename']
            text_prompt = str(row['text'])
            armature_id = int(row['armature_id'])
            file_path = self.motion_dir_abs / motion_filename
            
            if not file_path.exists():
                logger.warning(f"File di movimento {file_path} non trovato per riga {idx}. Salto campione.")
                continue
            if text_prompt not in self.sbert_embeddings_cache:
                logger.warning(f"Embedding SBERT non trovato per testo: '{text_prompt[:50]}...' (riga {idx}). Salto campione.")
                continue
            
            self.samples.append({
                'motion_file_path': str(file_path),
                'text_prompt': text_prompt,
                'armature_id': armature_id
            })
        
        if not self.samples:
            raise ValueError("Nessun campione valido caricato. Controlla annotazioni e path.")
        logger.info(f"Preparati {len(self.samples)} campioni validi.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        max_attempts = 3
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
                    if attempt == max_attempts - 1 and current_idx == (idx + max_attempts -1) % len(self.samples) : 
                        logger.debug(f"Ultimo tentativo per idx iniziale {idx} fallito check min_seq_len per {sample_info['motion_file_path']}.")
                    continue 
                
                # Gestione troncamento/padding alla max_seq_len del dataset (se specificata)
                # Il padding effettivo a max_len_in_batch avviene nel collate_fn
                if num_frames > self.max_seq_len:
                    target_x0_final = target_x0_normalized[:self.max_seq_len]
                    current_processed_len = self.max_seq_len
                else:
                    target_x0_final = target_x0_normalized
                    current_processed_len = num_frames
                
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
                logger.warning(f"Errore nel processare campione {sample_info.get('motion_file_path', 'N/A')} (idx {current_idx}, tentativo {attempt+1}): {e}", exc_info=False)
                if attempt == max_attempts - 1:
                    logger.error(f"Tutti i tentativi falliti per l'indice iniziale {idx}. Restituisco None.")
                    return None
        return None


def collate_motion_data_revised(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    valid_batch_items = [item for item in batch if item is not None]
    if not valid_batch_items:
        logger.warning("collate_motion_data_revised ha ricevuto un batch vuoto o tutto None.")
        return None 

    collated_batch: Dict[str, Any] = {}
    keys_to_pad = ["target_x0", "x_noisy"]
    keys_to_stack = ["timesteps", "text_embeddings", "armature_class_ids", "motion_actual_length"]

    # Estrai e stacka prima le lunghezze effettive
    actual_lengths_list = []
    for item in valid_batch_items:
        val = item["motion_actual_length"]
        if not isinstance(val, torch.Tensor): val = torch.tensor(val, dtype=torch.long)
        actual_lengths_list.append(val)
    actual_lengths = torch.stack(actual_lengths_list)
    collated_batch["motion_actual_length"] = actual_lengths # Salva subito quelle stackate
    
    max_len_in_batch = actual_lengths.max().item()

    for key in valid_batch_items[0].keys():
        if key in keys_to_pad:
            sequences = [item[key] for item in valid_batch_items]
            padded_sequences = []
            for seq in sequences:
                len_seq = seq.shape[0]
                # Applica padding se la sequenza è più corta di max_len_in_batch
                # Applica troncamento se è più lunga (non dovrebbe succedere se __getitem__ tronca a max_seq_len)
                if len_seq < max_len_in_batch:
                    padding_needed = max_len_in_batch - len_seq
                    padding_tensor = torch.zeros(padding_needed, seq.shape[1], dtype=seq.dtype, device=seq.device)
                    padded_seq = torch.cat([seq, padding_tensor], dim=0)
                elif len_seq > max_len_in_batch: # Troncamento di sicurezza
                    padded_seq = seq[:max_len_in_batch]
                else:
                    padded_seq = seq
                padded_sequences.append(padded_seq)
            collated_batch[key] = torch.stack(padded_sequences)
        
        elif key in keys_to_stack and key != "motion_actual_length": # Già gestito
            tensor_list = []
            for item in valid_batch_items:
                item_val = item[key]
                if not isinstance(item_val, torch.Tensor):
                    try:
                        item_val = torch.tensor(item_val) 
                    except Exception as e:
                        logger.error(f"Impossibile convertire item['{key}'] in tensore: {item_val}, tipo: {type(item_val)}. Errore: {e}")
                tensor_list.append(item_val)
            collated_batch[key] = torch.stack(tensor_list)

        elif key == "text_prompt": 
            collated_batch[key] = [item[key] for item in valid_batch_items]

    # Crea la maschera di validità (True per VALIDO, False per PADDED)
    # Forma: [bs, 1, 1, max_len_in_batch] come atteso dal trainer per y_cond['mask']
    arange_tensor = torch.arange(max_len_in_batch, device=actual_lengths.device)
    motion_validity_mask = arange_tensor.expand(len(valid_batch_items), max_len_in_batch) < actual_lengths.unsqueeze(1)
    collated_batch["motion_padding_mask_for_loss"] = motion_validity_mask.unsqueeze(1).unsqueeze(1)
    
    # Rimuovi motion_actual_length se non vuoi che sia passato oltre (MDM non lo usa direttamente nel model_kwargs['y'])
    # Ma può essere utile per debug o per la maschera nel trainer, quindi per ora la lascio.
    # Se ArmatureMDMTrainerRevised la prende da y_cond['mask'], allora non serve qui.
    # if "motion_actual_length" in collated_batch and "motion_padding_mask_for_loss" in collated_batch:
    #     del collated_batch["motion_actual_length"] # Opzionale

    return collated_batch