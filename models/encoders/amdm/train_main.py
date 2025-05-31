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
from sentence_transformers import SentenceTransformer

# --- System Path Modifications ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parents[2]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(current_script_dir) not in sys.path:
     sys.path.insert(0, str(current_script_dir))
if str(project_root / "dataset") not in sys.path:
    sys.path.insert(0, str(project_root / "dataset"))
if str(project_root / "utils") not in sys.path:
    sys.path.insert(0, str(project_root / "utils"))

# --- Importing Modules ---
from TAMDM import ArmatureMDM
from AMDMTrainer import ADMTrainer, GaussianDiffusionTrainerUtil
from dataset_loader import MyTextToMotionDataset, collate_motion_data_revised
from motion_generator import MotionGenerator, get_named_beta_schedule
from diffusion_utils import load_armature_config, setup_logging, get_bone_mask_for_armature


logger = logging.getLogger(__name__)


# --- MAIN FUNCTION ---
def main_training_and_generation_loop(config_path: Path, generation_text: str, generation_armature_id: int) -> None:
    """
    Main loop for training the ArmatureMDM model and then generating a sample motion.

    :param config_path: Path to the YAML configuration file.
    :param generation_text: Text prompt for the generation phase.
    :param generation_armature_id: Armature ID for the generation phase.
    """
    # 1. Load Configuration
    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading configuration file {config_path}: {e}")
        return
    logger.info("Configuration loaded.")

    paths_cfg = config.get('paths', {})
    model_cfg = config.get('model_hyperparameters', {})
    train_cfg = config.get('training_hyperparameters', {})
    diffusion_cfg = config.get('diffusion_hyperparameters', {})
    dataset_cfg = config.get('dataset_parameters', {})
    gen_params_cfg = config.get('generation_parameters', {})

    run_name = config.get('run_name', 'armature_mdm_run')
    model_save_dir_relative = Path(paths_cfg.get('model_save_dir', 'experiments_output'))
    model_save_dir = project_root / model_save_dir_relative / run_name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(model_save_dir), run_name)
    device = torch.device(train_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")

    armature_config_file_path_str = paths_cfg.get('armature_config_file', 'config/armature_config.json')
    armature_config_file_abs = project_root / armature_config_file_path_str

    loaded_armature_config = load_armature_config(str(armature_config_file_abs))
    if loaded_armature_config is None:
        logger.error(f"Could not load armature configuration from {armature_config_file_abs}. Aborting.")
        return

    betas_for_dataset_np = get_named_beta_schedule(
        schedule_name=diffusion_cfg.get('noise_schedule_mdm', 'cosine'),
        num_diffusion_timesteps=diffusion_cfg.get('num_diffusion_timesteps', 1000)
    )

    alphas_for_dataset_np = 1.0 - betas_for_dataset_np
    alphas_cumprod_for_dataset = torch.tensor(np.cumprod(alphas_for_dataset_np, axis=0), dtype=torch.float32)

    logger.info("Preparing dataset and DataLoader...")
    data_root_path_str = paths_cfg.get('data_root_dir')
    data_root_abs = Path(data_root_path_str)
    if not data_root_abs.is_absolute():
        data_root_abs = project_root / data_root_path_str

    dataset_mean_path_str = str(data_root_abs / dataset_cfg.get('dataset_mean_filename', 'mean.npy'))
    dataset_std_path_str = str(data_root_abs / dataset_cfg.get('dataset_std_filename', 'std.npy'))

    if not Path(dataset_mean_path_str).exists() or not Path(dataset_std_path_str).exists():
        logger.error(f"Mean file ({dataset_mean_path_str}) or std file ({dataset_std_path_str}) not found. "
                       "Run the calculate_mean_std.py script.")
        return

    if "flat" in model_cfg.get('data_rep', ""):
        dataset_num_motion_features = model_cfg.get('num_motion_features_actual')
    else:
        dataset_num_motion_features = model_cfg.get('njoints') * model_cfg.get('nfeats_per_joint')
    if dataset_num_motion_features is None:
        logger.error("num_motion_features (or njoints/nfeats_per_joint) not specified in model_hyperparameters.")
        return

    train_subset_size = dataset_cfg.get('subset_size', None)
    if train_subset_size is not None and train_subset_size <= 0:
        train_subset_size = None
    
    train_shuffle_subset = dataset_cfg.get('shuffle_subset_after_sampling', True)

    train_dataset = MyTextToMotionDataset(
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
        subset_size=train_subset_size,
        shuffle_subset=train_shuffle_subset,
        data_device=dataset_cfg.get('dataset_device', 'cpu')
    )
    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg.get('batch_size'), shuffle=True,
        num_workers=train_cfg.get('num_dataloader_workers', 0), collate_fn=collate_motion_data_revised
    )
    logger.info(f"Training data loaded: {len(train_dataset)} samples (subset_size: {train_subset_size}, shuffle_subset: {train_shuffle_subset}).")

    val_loader = None
    if paths_cfg.get('val_annotations_file_name'):
        val_subset_size = dataset_cfg.get('val_subset_size', None)
        if val_subset_size is not None and val_subset_size <= 0:
            val_subset_size = None
        
        val_shuffle_subset = dataset_cfg.get('val_shuffle_subset_after_sampling', True)

        val_dataset = MyTextToMotionDataset(
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
            subset_size=val_subset_size,
            shuffle_subset=val_shuffle_subset,
            data_device=dataset_cfg.get('dataset_device', 'cpu')
        )
        val_loader = DataLoader(
            val_dataset, batch_size=train_cfg.get('batch_size'), shuffle=False,
            num_workers=train_cfg.get('num_dataloader_workers', 0), collate_fn=collate_motion_data_revised
        )
        logger.info(f"Validation data loaded: {len(val_dataset)} samples (subset_size: {val_subset_size}, shuffle_subset: {val_shuffle_subset}).")

    logger.info("Initializing ArmatureMDM model...")
    armature_mdm_model_base = ArmatureMDM(
        data_rep=model_cfg.get('data_rep'),
        njoints=model_cfg.get('njoints'),
        nfeats_per_joint=model_cfg.get('nfeats_per_joint'),
        num_motion_features=dataset_num_motion_features,
        latent_dim=model_cfg.get('latent_dim'),
        arch=model_cfg.get('arch', 'trans_enc'),
        num_layers=model_cfg.get('num_layers'),
        num_heads=model_cfg.get('num_heads'),
        ff_size=model_cfg.get('ff_size'),
        dropout=model_cfg.get('dropout'),
        activation=model_cfg.get('activation', 'gelu'),
        batch_first_transformer=model_cfg.get('batch_first_transformer', False),

        conditioning_integration_mode=model_cfg.get('conditioning_integration_mode', "mlp"),
        armature_integration_policy=model_cfg.get('armature_integration_policy', "add_refined"),
        conditioning_transformer_config=model_cfg.get('conditioning_transformer_config', {}),

        sbert_embedding_dim=model_cfg.get('sbert_embedding_dim'),
        max_armature_classes=model_cfg.get('max_armature_classes'),
        armature_embedding_dim=model_cfg.get('armature_embedding_dim'),
        armature_mlp_hidden_dims=model_cfg.get('armature_mlp_hidden_dims'),
        max_seq_len_pos_enc=model_cfg.get('max_seq_len_pos_enc'),
        text_cond_mask_prob=model_cfg.get('text_cond_mask_prob', model_cfg.get('cond_mask_prob', 0.1)),
        armature_cond_mask_prob=model_cfg.get('armature_cond_mask_prob', 0.1),
    )

    armature_mdm_model = armature_mdm_model_base.to(device)
    is_data_parallel = False

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        armature_mdm_model = torch.nn.DataParallel(armature_mdm_model_base) # DataParallel will distribute the model across available GPUs
        armature_mdm_model.to(device) # move the model to the device after wrapping in DataParallel
        is_data_parallel = True

    logger.info(f"ArmatureMDM model created. Parameters: {sum(p.numel() for p in armature_mdm_model.parameters() if p.requires_grad):,}")

    diffusion_util_trainer = GaussianDiffusionTrainerUtil(betas=betas_for_dataset_np)

    # LR Scheduler Setup
    temp_optimizer_for_scheduler = optim.AdamW(armature_mdm_model.parameters(), lr=train_cfg.get('learning_rate'))
    lr_scheduler_instance = None
    if train_cfg.get('use_lr_scheduler', False) and val_loader is not None :
        lr_scheduler_instance = optim.lr_scheduler.ReduceLROnPlateau(
            temp_optimizer_for_scheduler, 'min',
            factor=train_cfg.get('lr_scheduler_factor', 0.5),
            patience=config.get('early_stopping', {}).get('early_stopping_patience', 50) // 2,
            verbose=True )

    trainer = ADMTrainer(
        config=config,
        original_config_path=config_path,
        model=armature_mdm_model, 
        diffusion_util=diffusion_util_trainer,
        train_loader=train_loader, 
        get_bone_mask_fn=get_bone_mask_for_armature,
        model_save_dir=model_save_dir,
        armature_config_data=loaded_armature_config, 
        val_loader=val_loader,
        lr_scheduler=lr_scheduler_instance)

    # Optimizer Setup
    if is_data_parallel:
        params_to_adam = armature_mdm_model.module.parameters()
    else:
        params_to_adam = armature_mdm_model.parameters()

    if lr_scheduler_instance and isinstance(lr_scheduler_instance, torch.optim.lr_scheduler.ReduceLROnPlateau):
        trainer.opt = temp_optimizer_for_scheduler
    else:
        trainer.opt = optim.AdamW(params_to_adam, lr=train_cfg.get('learning_rate'),
                                    weight_decay=train_cfg.get('weight_decay',0.0),
                                    betas=(train_cfg.get('adam_beta1', 0.9), train_cfg.get('adam_beta2', 0.999))
                                )
        if lr_scheduler_instance:
            lr_scheduler_instance.optimizer = trainer.opt


    logger.info("Starting training...")
    trainer.run_loop()
    logger.info("Training completed.")

    # 2. Generation Phase
    logger.info("Starting post-training generation phase...")

    final_model_filename = f"{trainer.model_filename_base}_final.pth"
    best_model_filename = f"{trainer.model_filename_base}_best.pth"
    trainer_save_directory_path = Path(trainer.model_save_dir)

    path_to_best_ckpt = trainer_save_directory_path / best_model_filename
    path_to_final_ckpt = trainer_save_directory_path / final_model_filename
    model_load_path = path_to_best_ckpt if path_to_best_ckpt.exists() else path_to_final_ckpt

    if not model_load_path.exists():
        logger.error(f"No trained model ({best_model_filename} or {final_model_filename}) found in {trainer.model_save_dir}. Aborting generation.")
        return

    try:
        logger.info(f"Initializing MotionGenerator with model: {model_load_path}")
        # config_path is the main YAML config path
        motion_generator = MotionGenerator(
            config_path=config_path, 
            model_checkpoint_path=model_load_path,
            device_str=str(device)
        )

        num_frames_to_generate = gen_params_cfg.get('num_frames_to_generate', 120)
        cfg_scale_final_gen = gen_params_cfg.get('cfg_scale', 2.5)
        const_noise_final_gen = gen_params_cfg.get('const_noise_for_sampling', False) # Get from gen_params if specific
        render_fps_final_gen = gen_params_cfg.get('render_fps', 20)


        logger.info(f"Generating motion for: '{generation_text}', armature ID: {generation_armature_id}")
        generated_motion_np = motion_generator.generate_single_motion(
            text_prompt=generation_text,
            armature_id=generation_armature_id,
            num_frames=num_frames_to_generate,
            cfg_scale=cfg_scale_final_gen,
            const_noise_for_sampling=const_noise_final_gen
            # clip_denoised, progress_bar use defaults or can be exposed
        )

        if generated_motion_np is not None:
            gen_output_dir = model_save_dir / "generated_samples_after_training" # Specific subdir
            gen_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Sanitize prompt for filename
            safe_prompt_snippet = "".join(c if c.isalnum() else "_" for c in generation_text[:50])
            animation_filename = f"gen_arm{generation_armature_id}_cfg{cfg_scale_final_gen}_{run_name}_{safe_prompt_snippet}.gif"
            animation_output_path = gen_output_dir / animation_filename

            motion_generator.save_motion_as_gif(
                motion_data_frames=generated_motion_np,
                output_path_abs=animation_output_path,
                armature_id=generation_armature_id, # For kinematic chain
                title=generation_text,
                fps=render_fps_final_gen
            )
        else:
            logger.warning("Post-training generation failed to produce motion.")

    except Exception as e:
        logger.error(f"Error during post-training generation phase: {e}", exc_info=True)

    logger.info("Main script (train_main.py) completed.")


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
        config_file_on_disk = Path(cli_args.config) # Try as absolute path or relative to CWD
        if not config_file_on_disk.exists():
            sys.stderr.write(f"ERROR: Configuration file '{cli_args.config}' not found.\n")
            sys.exit(1)

    main_training_and_generation_loop(config_file_on_disk, cli_args.gen_text, cli_args.gen_arm_id)