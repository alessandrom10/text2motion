import argparse
import logging
import math
import os
import sys
from pathlib import Path
import yaml
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sentence_transformers import SentenceTransformer # Per la generazione

# --- Modifica dei Path di Sistema ---
current_script_dir = Path(__file__).resolve().parent 
project_root = current_script_dir.parents[2] 

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# Assumendo che AMDM.py e AMDMTrainer.py siano nella stessa cartella di train.py (mdm/)
if str(current_script_dir) not in sys.path:
     sys.path.insert(0, str(current_script_dir))
if str(project_root / "dataset") not in sys.path:
    sys.path.insert(0, str(project_root / "dataset"))
if str(project_root / "utils") not in sys.path:
    sys.path.insert(0, str(project_root / "utils"))


# --- Importa le tue definizioni ---
from AMDM import (
    ArmatureMDM,
    PositionalEncoding, 
    TimestepEmbedder,   
    InputProcess,       
    OutputProcess       
)
from AMDMTrainer import (
    ArmatureMDMTrainerRevised,
    GaussianDiffusionTrainerUtil,
    UniformSampler
)
from dataset_loader import ( 
    MyTextToMotionDatasetNormalized,
    collate_motion_data_revised
)
from diffusion import ( 
    GaussianDiffusionSamplerUtil,
    generate_motion_mdm_style
)
from diffusion_utils import ( 
     load_armature_config, setup_logging, get_bone_mask_for_armature,
     create_motion_animation,
     T2M_KINEMATIC_CHAIN
)

# --- Beta Schedule (ispirato a MDM) ---
def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1.): #
    if schedule_name == "linear":
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        def betas_for_alpha_bar(num_diffusion_timesteps_inner, alpha_bar_fn, max_beta=0.999): #
            betas_out = []
            for i in range(num_diffusion_timesteps_inner):
                t1 = i / num_diffusion_timesteps_inner
                t2 = (i + 1) / num_diffusion_timesteps_inner
                betas_out.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
            return np.array(betas_out)

        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2, #
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

logger = logging.getLogger(__name__)


# --- FUNZIONE PRINCIPALE ---
def main_training_and_generation_loop(config_path: Path, generation_text: str, generation_armature_id: int):
    # 1. Carica Configurazione
    logger.info(f"Caricamento configurazione da: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Errore nel caricamento del file di configurazione {config_path}: {e}")
        return
    logger.info("Configurazione caricata.")

    paths_cfg = config.get('paths', {})
    model_cfg = config.get('model_hyperparameters', {})
    train_cfg = config.get('training_hyperparameters', {})
    diffusion_cfg = config.get('diffusion_hyperparameters', {})
    dataset_cfg = config.get('dataset_parameters', {})
    
    run_name = config.get('run_name', 'armature_mdm_run')
    model_save_dir_relative = Path(paths_cfg.get('model_save_dir', 'experiments_output')) 
    model_save_dir = project_root / model_save_dir_relative / run_name
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(str(model_save_dir), run_name)
    device = torch.device(train_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Utilizzo del dispositivo: {device}")

    armature_config_file_path_str = paths_cfg.get('armature_config_file', 'config/armature_config.json')
    armature_config_file_abs = project_root / armature_config_file_path_str
    
    loaded_armature_config = load_armature_config(str(armature_config_file_abs))
    if loaded_armature_config is None:
        logger.error(f"Impossibile caricare la configurazione dell'armatura da {armature_config_file_abs}. Interruzione.")
        return

    betas_for_dataset_np = get_named_beta_schedule(
        schedule_name=diffusion_cfg.get('noise_schedule_mdm', 'linear'),
        num_diffusion_timesteps=diffusion_cfg.get('num_diffusion_timesteps', 1000)
    )
    alphas_for_dataset_np = 1.0 - betas_for_dataset_np
    alphas_cumprod_for_dataset = torch.tensor(np.cumprod(alphas_for_dataset_np, axis=0), dtype=torch.float32)

    logger.info("Preparazione dataset e DataLoader...")
    data_root_path_str = paths_cfg.get('data_root_dir')
    data_root_abs = Path(data_root_path_str)
    if not data_root_abs.is_absolute():
        data_root_abs = project_root / data_root_path_str
    
    dataset_mean_path_str = str(data_root_abs / dataset_cfg.get('dataset_mean_filename', 'mean.npy'))
    dataset_std_path_str = str(data_root_abs / dataset_cfg.get('dataset_std_filename', 'std.npy'))

    if not Path(dataset_mean_path_str).exists() or not Path(dataset_std_path_str).exists():
        logger.error(f"File mean ({dataset_mean_path_str}) o std ({dataset_std_path_str}) non trovati. "
                       "Esegui lo script calculate_mean_std.py.")
        return

    if "flat" in model_cfg.get('data_rep', ""):
        dataset_num_motion_features = model_cfg.get('num_motion_features_actual')
    else: 
        dataset_num_motion_features = model_cfg.get('njoints') * model_cfg.get('nfeats_per_joint')
    if dataset_num_motion_features is None:
        logger.error("num_motion_features (o njoints/nfeats_per_joint) non specificato in model_hyperparameters.")
        return

    train_dataset = MyTextToMotionDatasetNormalized(
        root_dir=str(data_root_abs),
        annotations_file_name=paths_cfg.get('annotations_file_name'),
        motion_subdir=paths_cfg.get('motion_subdir'),
        precomputed_sbert_embeddings_path=str(project_root / paths_cfg.get('precomputed_sbert_train_path')),
        num_diffusion_timesteps=diffusion_cfg.get('num_diffusion_timesteps', 1000),
        alphas_cumprod=alphas_cumprod_for_dataset,
        num_motion_features=dataset_num_motion_features,
        dataset_mean_path=dataset_mean_path_str,
        dataset_std_path=dataset_std_path_str,
        min_seq_len=dataset_cfg.get('min_seq_len_dataset'),
        max_seq_len=dataset_cfg.get('max_seq_len_dataset'),
        data_device=dataset_cfg.get('dataset_device', 'cpu')
    )
    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg.get('batch_size'), shuffle=True,
        num_workers=train_cfg.get('num_dataloader_workers', 0), collate_fn=collate_motion_data_revised
    )
    logger.info(f"Dataset di training caricato: {len(train_dataset)} campioni.")

    val_loader = None
    if paths_cfg.get('val_annotations_file_name'):
        val_dataset = MyTextToMotionDatasetNormalized(
            root_dir=str(data_root_abs),
            annotations_file_name=paths_cfg.get('val_annotations_file_name'),
            motion_subdir=paths_cfg.get('motion_subdir'),
            precomputed_sbert_embeddings_path=str(project_root / paths_cfg.get('precomputed_sbert_val_path')),
            num_diffusion_timesteps=diffusion_cfg.get('num_diffusion_timesteps', 1000),
            alphas_cumprod=alphas_cumprod_for_dataset,
            num_motion_features=dataset_num_motion_features,
            dataset_mean_path=dataset_mean_path_str,
            dataset_std_path=dataset_std_path_str,
            min_seq_len=dataset_cfg.get('min_seq_len_dataset'),
            max_seq_len=dataset_cfg.get('max_seq_len_dataset'),
            data_device=dataset_cfg.get('dataset_device', 'cpu')
        )
        val_loader = DataLoader(
            val_dataset, batch_size=train_cfg.get('batch_size'), shuffle=False,
            num_workers=train_cfg.get('num_dataloader_workers', 0), collate_fn=collate_motion_data_revised
        )
        logger.info(f"Dataset di validazione caricato: {len(val_dataset)} campioni.")

    logger.info("Inizializzazione modello ArmatureMDM...")
    armature_mdm_model = ArmatureMDM(
        data_rep=model_cfg.get('data_rep'),
        njoints=model_cfg.get('njoints'),
        nfeats_per_joint=model_cfg.get('nfeats_per_joint'),
        num_motion_features=model_cfg.get('num_motion_features_actual'),
        latent_dim=model_cfg.get('latent_dim'),
        ff_size=model_cfg.get('ff_size'),
        num_layers=model_cfg.get('num_layers'),
        num_heads=model_cfg.get('num_heads'),
        dropout=model_cfg.get('dropout'),
        activation=model_cfg.get('activation', 'gelu'),
        sbert_embedding_dim=model_cfg.get('sbert_embedding_dim'),
        max_armature_classes=model_cfg.get('max_armature_classes'),
        armature_embedding_dim=model_cfg.get('armature_embedding_dim'),
        armature_mlp_hidden_dims=model_cfg.get('armature_mlp_hidden_dims'),
        max_seq_len_pos_enc=model_cfg.get('max_seq_len_pos_enc'),
        # >>> MODIFICA QUI: Fornisci un valore di default se la chiave non è nella config <<<
        cond_mask_prob=model_cfg.get('text_cond_dropout_prob', 0.1), 
        armature_cond_mask_prob=model_cfg.get('armature_cond_dropout_prob', 0.1),
        # <<< FINE MODIFICA <<<
        arch=model_cfg.get('arch', 'trans_enc'),
        dataset_name=dataset_cfg.get('dataset_name_for_mdm', 'custom'),
        clip_dim=model_cfg.get('sbert_embedding_dim'), # Allineato con SBERT
        batch_first_transformer=model_cfg.get('batch_first_transformer', False)
    ).to(device)
    logger.info(f"Modello ArmatureMDM creato. Parametri: {sum(p.numel() for p in armature_mdm_model.parameters() if p.requires_grad):,}")

    diffusion_util_trainer = GaussianDiffusionTrainerUtil(betas=betas_for_dataset_np)

    temp_optimizer_for_scheduler = optim.AdamW(armature_mdm_model.parameters(), lr=train_cfg.get('learning_rate'))
    lr_scheduler_instance = None
    if train_cfg.get('use_lr_scheduler', False) and val_loader is not None :
        lr_scheduler_instance = optim.lr_scheduler.ReduceLROnPlateau(
            temp_optimizer_for_scheduler, 'min',
            factor=train_cfg.get('lr_scheduler_factor', 0.5),
            patience=config.get('early_stopping', {}).get('patience', 10) // 2,
            verbose=True )

    trainer = ArmatureMDMTrainerRevised(
        config=config, model=armature_mdm_model, diffusion_util=diffusion_util_trainer,
        train_loader=train_loader, get_bone_mask_fn=get_bone_mask_for_armature,
        armature_config_data=loaded_armature_config, val_loader=val_loader,
        lr_scheduler=lr_scheduler_instance )
    
    if lr_scheduler_instance:
         trainer.opt = temp_optimizer_for_scheduler
    else:
        trainer.opt = optim.AdamW(armature_mdm_model.parameters(), lr=train_cfg.get('learning_rate'),
                                   weight_decay=train_cfg.get('weight_decay',0.0),
                                   betas=(train_cfg.get('adam_beta1', 0.9), train_cfg.get('adam_beta2', 0.999)))
        # if lr_scheduler_instance:  # Questa riga era ridondante se lo scheduler non è ReduceLROnPlateau
        #     lr_scheduler_instance.optimizer = trainer.opt


    logger.info("Inizio addestramento...")
    #trainer.run_loop()
    logger.info("Addestramento completato.")
    
    logger.info("Inizio fase di generazione...")
    
    final_model_filename = f"{trainer.model_filename_base}_final.pth"
    best_model_filename = f"{trainer.model_filename_base}_best.pth"
    trainer_save_directory_path = Path(trainer.model_save_dir) 
    
    path_to_best_ckpt = trainer_save_directory_path / best_model_filename
    path_to_final_ckpt = trainer_save_directory_path / final_model_filename
    model_load_path = path_to_best_ckpt if path_to_best_ckpt.exists() else path_to_final_ckpt

    if not model_load_path.exists():
        logger.error(f"Nessun modello addestrato ({best_model_filename} o {final_model_filename}) trovato in {trainer.model_save_dir}. Interruzione generazione.")
        return

    logger.info(f"Caricamento modello per generazione da: {model_load_path}")
    checkpoint = torch.load(str(model_load_path), map_location=device)
    loaded_config_from_ckpt = checkpoint.get('config', config)
    model_cfg_loaded = loaded_config_from_ckpt.get('model_hyperparameters', model_cfg)
    
    if "flat" in model_cfg_loaded.get('data_rep', ""):
        gen_actual_input_feats_loaded = model_cfg_loaded.get('num_motion_features_actual')
    else:
        gen_actual_input_feats_loaded = model_cfg_loaded.get('njoints') * model_cfg_loaded.get('nfeats_per_joint')

    generation_model = ArmatureMDM(
        data_rep=model_cfg_loaded.get('data_rep'),
        njoints=model_cfg_loaded.get('njoints'),
        nfeats_per_joint=model_cfg_loaded.get('nfeats_per_joint'),
        num_motion_features=gen_actual_input_feats_loaded,
        latent_dim=model_cfg_loaded.get('latent_dim'),
        ff_size=model_cfg_loaded.get('ff_size'),
        num_layers=model_cfg_loaded.get('num_layers'),
        num_heads=model_cfg_loaded.get('num_heads'),
        dropout=model_cfg_loaded.get('dropout', 0.1),
        activation=model_cfg_loaded.get('activation', 'gelu'),
        sbert_embedding_dim=model_cfg_loaded.get('sbert_embedding_dim'),
        max_armature_classes=model_cfg_loaded.get('max_armature_classes'),
        armature_embedding_dim=model_cfg_loaded.get('armature_embedding_dim'),
        armature_mlp_hidden_dims=model_cfg_loaded.get('armature_mlp_hidden_dims'),
        max_seq_len_pos_enc=model_cfg_loaded.get('max_seq_len_pos_enc', 5000),
        # >>> MODIFICA QUI (anche per la generazione): Fornisci un valore di default <<<
        cond_mask_prob=model_cfg_loaded.get('text_cond_dropout_prob', 0.1),
        armature_cond_mask_prob=model_cfg_loaded.get('armature_cond_dropout_prob', 0.1),
        # <<< FINE MODIFICA <<<
        arch=model_cfg_loaded.get('arch', 'trans_enc'),
        dataset_name=loaded_config_from_ckpt.get('dataset_parameters', dataset_cfg).get('dataset_name_for_mdm', 'custom'),
        clip_dim=model_cfg_loaded.get('sbert_embedding_dim'),
        batch_first_transformer=model_cfg_loaded.get('batch_first_transformer', False)
    ).to(device)
    generation_model.load_state_dict(checkpoint['model_state_dict'])
    generation_model.eval()
    logger.info("Modello per generazione caricato.")

    diffusion_sampler = GaussianDiffusionSamplerUtil(
        betas=betas_for_dataset_np, # Riusa betas_np_for_trainer
        model_mean_type=diffusion_cfg.get('model_mean_type_mdm', 'START_X'),
        model_var_type=diffusion_cfg.get('model_var_type_mdm', 'FIXED_SMALL'),
        loss_type="MSE" 
    )

    try:
        sbert_processor_gen = SentenceTransformer(model_cfg.get('sbert_model_name'), device=device)
        gen_text_embedding = sbert_processor_gen.encode(generation_text, convert_to_tensor=True)
    except Exception as e:
        logger.error(f"Errore SBERT per testo di generazione: {e}. Uso dummy tensor.")
        gen_text_embedding = torch.randn(model_cfg.get('sbert_embedding_dim')).to(device)

    y_conditions_for_generation = {
        'text_embeddings_batch': gen_text_embedding.unsqueeze(0).to(device),
        'armature_class_ids': torch.tensor([generation_armature_id], dtype=torch.long).to(device),
        'mask': None, 
        'cfg_scale': config.get('generation_params', {}).get('cfg_scale', 2.5),
    }
    num_frames_to_generate = config.get('generation_params', {}).get('num_frames_to_generate', 120)

    logger.info(f"Generazione movimento per: '{generation_text}', armatura ID: {generation_armature_id}")
    generated_motion = generate_motion_mdm_style(
        armature_mdm_model=generation_model,
        diffusion_sampler_util=diffusion_sampler,
        y_conditions=y_conditions_for_generation,
        num_frames=num_frames_to_generate,
        device=device
    ) # Rimosso clip_denoised, progress, const_noise perché non erano definiti nel main. Aggiungili se necessario.
    logger.info(f"Movimento generato con shape: {generated_motion.shape}")

    gen_output_dir = model_save_dir / "generated_samples"
    gen_output_dir.mkdir(exist_ok=True)
    animation_filename = f"gen_arm{generation_armature_id}_cfg{y_conditions_for_generation['cfg_scale']}_{run_name}.gif"
    animation_output_path = gen_output_dir / animation_filename

    motion_np = generated_motion.squeeze(0).cpu().numpy()
    
    if hasattr(train_dataset, 'dataset_mean') and hasattr(train_dataset, 'dataset_std'):
        mean_for_denorm = train_dataset.dataset_mean
        std_for_denorm = np.where(train_dataset.dataset_std == 0, 1e-8, train_dataset.dataset_std)
        motion_np_denormalized = motion_np * std_for_denorm + mean_for_denorm
    else:
        logger.warning("Mean/std del dataset non trovati per de-normalizzazione. Visualizzo output normalizzato.")
        motion_np_denormalized = motion_np

    num_j_viz = model_cfg_loaded.get('num_joints_for_geom', generation_model.njoints)
    feat_p_j_viz = model_cfg_loaded.get('features_per_joint_for_geom', generation_model.nfeats)

    if motion_np_denormalized.shape[1] == num_j_viz * feat_p_j_viz:
        motion_reshaped = motion_np_denormalized.reshape(num_frames_to_generate, num_j_viz, feat_p_j_viz)
        motion_to_animate = motion_reshaped

        logger.info(f"Creazione animazione: {animation_output_path}")
        create_motion_animation(
            motion_data_frames=motion_to_animate,
            kinematic_chain=T2M_KINEMATIC_CHAIN, 
            output_filename=str(animation_output_path),
            fps=config.get('generation_params', {}).get('render_fps', 20) )
        logger.info(f"Animazione salvata: {animation_output_path}")
    else:
        logger.warning(f"Shape output denormalizzato ({motion_np_denormalized.shape[1]}) non adatta per reshape a ({num_j_viz} joints, {feat_p_j_viz} feat/joint). Attese {num_j_viz * feat_p_j_viz} features. Salto animazione.")
    
    logger.info("Script main completato.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ArmatureMDM and Generate Motion.")
    parser.add_argument(
        '--config', type=str,
        default="config/diffusion_config.yaml",
        help='Path to the YAML configuration file (relative to project root or absolute).'
    )
    parser.add_argument(
        '--gen_text', type=str, default="the man is closing and opening umbrella",
        help="Text prompt for generation after training."
    )
    parser.add_argument(
        '--gen_arm_id', type=int, default=1,
        help="Armature ID for generation after training."
    )
    cli_args = parser.parse_args()

    config_file_on_disk = project_root / cli_args.config
    if not config_file_on_disk.exists():
        config_file_on_disk = Path(cli_args.config) # Prova come path assoluto/relativo alla CWD
        if not config_file_on_disk.exists():
            sys.stderr.write(f"ERRORE: Il file di configurazione '{cli_args.config}' non è stato trovato.\n")
            sys.exit(1)

    main_training_and_generation_loop(config_file_on_disk, cli_args.gen_text, cli_args.gen_arm_id)