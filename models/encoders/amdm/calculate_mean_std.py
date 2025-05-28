import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

# Setup del logger per questo script
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

def process_single_motion_file_for_stats(
    file_path: str,
    num_expected_features: int
) -> Optional[np.ndarray]:
    """
    Carica e appiattisce un singolo file di movimento .npy per il calcolo delle statistiche.
    Non applica la normalizzazione Z-score qui, solo il reshape.
    Il root centering è opzionale e dipende se vuoi che mean/std siano calcolati su dati centrati.
    """
    try:
        motion_data_raw = np.load(file_path).astype(np.float32)
    except Exception as e:
        logger.error(f"Errore nel caricamento del file di movimento {file_path}: {e}")
        return None
    
    # Appiattimento (se necessario)
    if motion_data_raw.ndim == 3: # Assumiamo [T, num_joints, 3] o simile
        if motion_data_raw.shape[1] * motion_data_raw.shape[2] == num_expected_features:
            # Opzionale: Root centering (se vuoi che le stats siano su dati centrati)
            # motion_data_raw = motion_data_raw - motion_data_raw[:, 0:1, :] 
            motion_data_flat = motion_data_raw.reshape(motion_data_raw.shape[0], -1)
        else:
            logger.warning(f"Shape del file {file_path} ({motion_data_raw.shape}) non compatibile con "
                           f"num_expected_features ({num_expected_features}) se appiattita da (T, J, C). Salto.")
            return None
    elif motion_data_raw.ndim == 2 and motion_data_raw.shape[1] == num_expected_features:
        motion_data_flat = motion_data_raw
    else:
        logger.warning(f"Shape del file {file_path} ({motion_data_raw.shape}) non gestita. "
                       f"Attesa (T, J, 3) o (T, num_expected_features). Salto.")
        return None

    if motion_data_flat.shape[1] != num_expected_features:
        logger.warning(f"Mismatch di feature in {file_path}: attese {num_expected_features}, ottenute {motion_data_flat.shape[1]}. Salto.")
        return None
            
    return motion_data_flat


def calculate_and_save_mean_std(config_path: str):
    """
    Calcola e salva mean.npy e std.npy per il training set.
    """
    logger.info(f"Avvio calcolo mean/std usando la configurazione: {config_path}")

    # 1. Carica Configurazione
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Errore nel caricamento del file di configurazione {config_path}: {e}")
        return

    paths_cfg = config.get('paths', {})
    model_cfg = config.get('model_hyperparameters', {})
    dataset_cfg = config.get('dataset_parameters', {})

    data_root_dir = Path(paths_cfg.get('data_root_dir'))
    annotations_train_file = data_root_dir / paths_cfg.get('annotations_file_name') # Usa il file di training
    motion_subdir = paths_cfg.get('motion_subdir')
    motion_dir_abs = data_root_dir / motion_subdir

    # Determina num_motion_features dalla configurazione
    if "flat" in model_cfg.get('data_rep', ""):
        num_motion_features = model_cfg.get('num_motion_features_actual')
    else: # structured or default
        num_motion_features = model_cfg.get('njoints') * model_cfg.get('nfeats_per_joint')
    
    if num_motion_features is None:
        logger.error("num_motion_features (o njoints/nfeats_per_joint) non specificato correttamente "
                       "nella sezione model_hyperparameters della configurazione.")
        return
    
    logger.info(f"Numero di feature attese per frame: {num_motion_features}")

    # 2. Leggi le Annotazioni del Training Set
    try:
        annotations_df = pd.read_csv(annotations_train_file)
        logger.info(f"Caricate {len(annotations_df)} annotazioni da {annotations_train_file}")
    except Exception as e:
        logger.error(f"Errore nel leggere il file di annotazioni {annotations_train_file}: {e}")
        return

    all_motion_data_list = []
    processed_files = 0
    failed_files = 0

    logger.info("Processamento file di movimento dal training set...")
    for idx, row in tqdm(annotations_df.iterrows(), total=len(annotations_df), desc="Processing motions"):
        motion_filename = row['motion_filename']
        file_path = motion_dir_abs / motion_filename
        
        if not file_path.exists():
            logger.warning(f"File di movimento {file_path} non trovato. Salto.")
            failed_files += 1
            continue
        
        motion_features = process_single_motion_file_for_stats(str(file_path), num_motion_features)
        
        if motion_features is not None:
            all_motion_data_list.append(motion_features)
            processed_files += 1
        else:
            failed_files += 1

    if not all_motion_data_list:
        logger.error("Nessun dato di movimento valido processato. Controlla i file e i path.")
        return
        
    logger.info(f"Processati {processed_files} file con successo, {failed_files} falliti o saltati.")

    # 3. Concatena tutti i dati e calcola Mean/Std
    try:
        full_motion_dataset_np = np.concatenate(all_motion_data_list, axis=0)
        logger.info(f"Dataset completo concatenato con shape: {full_motion_dataset_np.shape}") # Dovrebbe essere [totale_frame_validi, num_motion_features]
    except Exception as e:
        logger.error(f"Errore durante la concatenazione dei dati di movimento: {e}")
        logger.error("Questo potrebbe accadere se i file .npy hanno un numero di feature inconsistente "
                     "nonostante i controlli, o se sono corrotti.")
        return

    dataset_mean = np.mean(full_motion_dataset_np, axis=0)
    dataset_std = np.std(full_motion_dataset_np, axis=0)
    
    # Sostituisci std=0 con un valore piccolo per evitare divisioni per zero durante la normalizzazione
    dataset_std[dataset_std == 0] = 1e-8 
    logger.info(f"Media calcolata con shape: {dataset_mean.shape}")
    logger.info(f"Deviazione standard calcolata con shape: {dataset_std.shape}")

    # 4. Salva Mean e Std
    mean_save_path = data_root_dir / dataset_cfg.get('dataset_mean_filename', 'mean.npy')
    std_save_path = data_root_dir / dataset_cfg.get('dataset_std_filename', 'std.npy')

    try:
        np.save(mean_save_path, dataset_mean)
        logger.info(f"Media del dataset salvata in: {mean_save_path}")
        np.save(std_save_path, dataset_std)
        logger.info(f"Deviazione standard del dataset salvata in: {std_save_path}")
    except Exception as e:
        logger.error(f"Errore durante il salvataggio dei file mean/std: {e}")
        return

    logger.info("Calcolo di mean.npy e std.npy completato.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and save mean and std for the motion dataset.")
    parser.add_argument(
        '--config', type=str,
        default=r"C:\PythonProjects\text2motion\config\diffusion_config.yaml", # Assicurati che questo path sia corretto
        help='Path to the YAML configuration file used for training (to get dataset paths and params).'
    )
    args = parser.parse_args()

    config_file_arg_path = Path(args.config)
    if not config_file_arg_path.is_absolute():
        script_dir_path = Path(__file__).resolve().parent
        resolved_config_path = script_dir_path / config_file_arg_path
        if not resolved_config_path.exists():
            resolved_config_path = Path(os.getcwd()) / config_file_arg_path
    else:
        resolved_config_path = config_file_arg_path

    if not resolved_config_path.exists():
        sys.stderr.write(f"ERRORE: Il file di configurazione '{args.config}' non è stato trovato.\n"
                         f"Path tentati: {config_file_arg_path}, {resolved_config_path}\n")
        sys.exit(1)
        
    calculate_and_save_mean_std(str(resolved_config_path))