import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
from typing import Any, Dict, List
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..')) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ArmatureMDM import ArmatureMDM
from TrainerMDM import ArmatureMDMTrainer, KinematicLossCalculator

from dataset.motion_dataset_loader import (
    MyTextToMotionDataset, 
    process_motion_file,
    collate_motion_data
)

from utils.diffusion_utils import (
    setup_logging,
    get_noise_schedule,
    plot_training_history,
    get_bone_mask_for_armature
)

logger = logging.getLogger(__name__) # Gets a logger named after the current module

def generate_annotation_file(
    data_root: str,
    motion_subdir_name: str,
    text_subdir_name: str,
    output_csv_filename: str,
    default_armature_id: int = 1
) -> None:
    """
    Scans motion and text directories, pairs them, and creates a CSV annotation file.

    :param data_root: Root directory where 'motion_subdir' and 'text_subdir' are located.
    :param motion_subdir_name: Subdirectory containing .npy motion files.
    :param text_subdir_name: Subdirectory containing .txt text files.
    :param output_csv_filename: Name of the CSV file to create (e.g., "annotations.csv").
                                This path will be relative to 'data_root'.
    :param default_armature_id: The armature ID to assign to all samples.
    """
    logger.info(f"Starting annotation file generation: {output_csv_filename}")
    motion_dir_abs = os.path.join(data_root, motion_subdir_name)
    text_dir_abs = os.path.join(data_root, text_subdir_name)
    output_csv_path = os.path.join(data_root, output_csv_filename)

    if not os.path.isdir(motion_dir_abs):
        logger.error(f"Motion directory not found: {motion_dir_abs}")
        return
    if not os.path.isdir(text_dir_abs):
        logger.error(f"Text directory not found: {text_dir_abs}")
        return

    paired_samples_data = []
    motion_files = sorted([f for f in os.listdir(motion_dir_abs) if f.endswith(".npy")]) 
    logger.info(f"Found {len(motion_files)} .npy files in '{motion_dir_abs}'.")

    for motion_filename in motion_files:
        base_name = os.path.splitext(motion_filename)[0]
        text_file_name = base_name + ".txt"
        text_file_path_abs = os.path.join(text_dir_abs, text_file_name)
        
        if os.path.exists(text_file_path_abs):
            try:
                with open(text_file_path_abs, 'r', encoding='utf-8') as f:
                    text_prompt = f.read().strip()
                if text_prompt:
                    paired_samples_data.append({
                        'motion_filename': motion_filename,
                        'text': text_prompt,
                        'armature_id': default_armature_id
                    })
                else:
                    logger.warning(f"Empty text file: {text_file_path_abs} for motion {motion_filename}. Skipping.")
            except Exception as e:
                logger.warning(f"Could not read text file {text_file_path_abs}: {e}. Skipping.")
        else:
            logger.warning(f"Matching text file not found for {motion_filename} at {text_file_path_abs}. Skipping.")

    if not paired_samples_data:
        logger.warning(f"No paired samples found to write to {output_csv_path}.")
        return

    try:
        df = pd.DataFrame(paired_samples_data)
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True) # Ensure directory exists
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Successfully created annotation file with {len(df)} entries: {output_csv_path}")
    except Exception as e:
        logger.error(f"Failed to write annotation file {output_csv_path}: {e}", exc_info=True)


# --- Main Training Orchestration Function ---
def run_training_pipeline(config: Dict[str, Any]) -> None:
    """
    Orchestrates the entire training pipeline using parameters from the config dictionary.
    """
    run_name = config.get('run_name', 'default_run')
    paths_config = config.get('paths', {})
    model_save_dir = paths_config.get('model_save_dir', './trained_models_output')
    
    setup_logging(model_save_dir, run_name)
    
    logger.info(f"======= Starting Training Run: {run_name} =======")
    training_hparams = config.get('training_hyperparameters', {})
    device_name = training_hparams.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)
    logger.info(f"Using device: {device}")
    logger.info(f"Full configuration loaded: {config}")

    dataset_params = config.get('dataset_parameters', {})
    diffusion_params = config.get('diffusion_hyperparameters', {})
    model_hparams = config.get('model_hyperparameters', {})

    # 1. Get Noise Schedule (alphas_cumprod for dataset)
    alphas_cumprod_device = torch.device(dataset_params.get('dataset_device', 'cpu'))
    alphas_cumprod = get_noise_schedule( 
        diffusion_params.get('num_diffusion_timesteps', 1000),
        beta_start=diffusion_params.get('beta_start', 0.0001),
        beta_end=diffusion_params.get('beta_end', 0.02),
        device=alphas_cumprod_device
    )

    # 2. Create Datasets and DataLoaders
    logger.info("Loading training data...")
    try:
        train_dataset = MyTextToMotionDataset(
            root_dir=paths_config.get('data_root_dir', './data/'),
            annotations_file_name=paths_config.get('annotations_file_name', 'annotations_train.csv'),
            motion_subdir=paths_config.get('motion_subdir', 'new_joints'),
            num_diffusion_timesteps=diffusion_params.get('num_diffusion_timesteps', 1000),
            alphas_cumprod=alphas_cumprod,
            num_motion_features=model_hparams.get('num_motion_features', 66),
            motion_processor_fn=process_motion_file,
            min_seq_len=dataset_params.get('min_seq_len_dataset', 10),
            max_seq_len=dataset_params.get('max_seq_len_dataset'), 
            data_device=str(alphas_cumprod_device) 
        )
        if len(train_dataset) == 0:
            logger.error("Training dataset is empty. Please check annotations and data paths. Aborting.")
            return
        train_loader = DataLoader(
            train_dataset, batch_size=training_hparams.get('batch_size', 32), shuffle=True, 
            num_workers=training_hparams.get('num_dataloader_workers', 0), collate_fn=collate_motion_data
        )
        logger.info(f"Training data loaded: {len(train_dataset)} samples.")
    except Exception as e:
        logger.error(f"Failed to create training dataset: {e}", exc_info=True)
        raise

    val_loader = None
    if paths_config.get('val_annotations_file_name'):
        logger.info("Loading validation data...")
        try:
            val_dataset = MyTextToMotionDataset(
                root_dir=paths_config.get('data_root_dir', './data/'),
                annotations_file_name=paths_config['val_annotations_file_name'],
                motion_subdir=paths_config.get('motion_subdir', 'new_joints'),
                num_diffusion_timesteps=diffusion_params.get('num_diffusion_timesteps', 1000),
                alphas_cumprod=alphas_cumprod,
                num_motion_features=model_hparams.get('num_motion_features', 66),
                motion_processor_fn=process_motion_file,
                min_seq_len=dataset_params.get('min_seq_len_dataset', 10),
                max_seq_len=dataset_params.get('max_seq_len_dataset'),
                data_device=str(alphas_cumprod_device)
            )
            if len(val_dataset) > 0:
                val_loader = DataLoader(
                    val_dataset, batch_size=training_hparams.get('batch_size', 32), shuffle=False, 
                    num_workers=training_hparams.get('num_dataloader_workers', 0), collate_fn=collate_motion_data
                )
                logger.info(f"Validation data loaded: {len(val_dataset)} samples.")
            else: logger.warning("Validation dataset is empty.")
        except Exception as e:
            logger.warning(f"Failed to create validation dataset: {e}. Proceeding without validation.", exc_info=True)
    else: logger.info("No 'val_annotations_file_name' in config. Proceeding without validation.")

    # 3. Setup Model, Optimizer, Scheduler (as in your script)
    model = ArmatureMDM(
        num_motion_features=model_hparams.get('num_motion_features', 66),
        latent_dim=model_hparams.get('latent_dim', 768),
        ff_size=model_hparams.get('ff_size', 1024),
        num_layers=model_hparams.get('num_layers', 8),
        num_heads=model_hparams.get('num_heads', 4),
        dropout=model_hparams.get('dropout', 0.1),
        activation=model_hparams.get('activation', 'gelu'),
        sbert_model_name=model_hparams.get('sbert_model_name', 'all-mpnet-base-v2'),
        max_armature_classes=model_hparams.get('max_armature_classes', 10),
        armature_embedding_dim=model_hparams.get('armature_embedding_dim', 64),
        text_time_conditioning_policy=model_hparams.get('text_time_cond_policy', "add"),
        armature_integration_policy=model_hparams.get('armature_integration_policy', "add"),
        timestep_embedder_type=model_hparams.get('timestep_embedder_type', "mlp"),
        # Add other params your ArmatureMDM expects from your ArmatureMDM.py, e.g.:
        text_cond_dropout_prob=model_hparams.get('text_cond_dropout_prob', 0.1),
        armature_cond_dropout_prob=model_hparams.get('armature_cond_dropout_prob', 0.1),
        max_seq_len=model_hparams.get('max_seq_len', 5000),
        use_final_algebraic_refinement_encoder=model_hparams.get('use_final_algebraic_refinement_encoder', True),
        algebraic_refinement_hidden_dim=model_hparams.get('algebraic_refinement_hidden_dim'), # Can be None
        timestep_gru_num_layers=model_hparams.get('timestep_gru_num_layers', 1),
        timestep_gru_dropout=model_hparams.get('timestep_gru_dropout', 0.0)
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=training_hparams.get('learning_rate', 1e-4))
    lr_scheduler = None
    early_stopping_params = config.get('early_stopping', {})
    if training_hparams.get('use_lr_scheduler', True):
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', 
            factor=training_hparams.get('lr_scheduler_factor', 0.5),
            patience=early_stopping_params.get('patience', 10) // 3,
            verbose=False
        )
    logger.info(f"Model initialized. Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 4. Setup Kinematic Loss Calculator (as in your script)
    kinematic_calculator = None
    kinematic_params = config.get('kinematic_losses', {}) 
    if kinematic_params.get('use_kinematic_losses', False):
        if kinematic_params.get('velocity_loss_weight', 0.0) > 0 or \
           kinematic_params.get('acceleration_loss_weight', 0.0) > 0:
            kinematic_calculator = KinematicLossCalculator(
                get_bone_mask_fn=get_bone_mask_for_armature,
                device=str(device), 
                use_velocity_loss=(kinematic_params.get('velocity_loss_weight', 0.0) > 0),
                velocity_loss_weight=kinematic_params.get('velocity_loss_weight', 0.0),
                use_acceleration_loss=(kinematic_params.get('acceleration_loss_weight', 0.0) > 0),
                acceleration_loss_weight=kinematic_params.get('acceleration_loss_weight', 0.0),
                kinematic_loss_type=kinematic_params.get('kinematic_loss_type', 'l1')
            )
    logger.info(f"Kinematic losses {'ENABLED' if kinematic_calculator else 'DISABLED'}")

    # 5. Initialize Trainer (as in your script)
    trainer = ArmatureMDMTrainer(
        model=model, optimizer=optimizer, get_bone_mask_fn=get_bone_mask_for_armature,
        device=str(device), lr_scheduler=lr_scheduler,
        main_loss_type=training_hparams.get('main_loss_type', "mse"),
        cfg_drop_prob=training_hparams.get('cfg_drop_prob_trainer', 0.1),
        kinematic_loss_calculator=kinematic_calculator,
        early_stopping_patience=early_stopping_params.get('patience', 10),
        early_stopping_min_delta=early_stopping_params.get('min_delta', 0.0001),
        model_save_path=os.path.join(model_save_dir, paths_config.get('model_filename', 'model.pth'))
    )

    # 6. Start Training
    num_epochs_to_run = training_hparams.get('num_epochs', 50)
    logger.info(f"Attempting to start training for {num_epochs_to_run} epochs...")
    
    history = trainer.train( 
        train_loader=train_loader,
        num_epochs=num_epochs_to_run,
        val_loader=val_loader
    )

    completed_epochs = trainer.completed_epochs_count if hasattr(trainer, 'completed_epochs_count') else num_epochs_to_run
    logger.info(f"======= Training Run Finished ({completed_epochs} epochs completed) =======")

    # 7. Plotting results
    if history:
        plot_training_history( # From your utils
            history,
            main_val_metric_name=training_hparams.get('main_validation_metric_name', "Val Main Loss"),
            model_save_dir=model_save_dir,
            run_name=run_name
        )
    else: logger.warning("No history returned from trainer to plot.")


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if config is None:
            logger.warning(f"Config file {config_path} is empty or invalid. Returning empty config.")
            return {}
        logger.info(f"Config loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config {config_path}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading config {config_path}: {e}", exc_info=True)
        raise

# --- Updated Main Entry Point ---
def main_entry_point():
    parser = argparse.ArgumentParser(description="Train ArmatureMDM or Generate Annotations.")
    parser.add_argument(
        '--config', 
        type=str, 
        default=str(Path(__file__).resolve().parents[3] / 'config' / 'diffusion_config.yaml'), # Simpler default path
        help='Path to the YAML configuration file for training.'
    )
    parser.add_argument(
        '--generate_annotations',
        action='store_true',
        help='If set, generates the annotation CSV files and exits. Requires config for paths.'
    )
    parser.add_argument(
        '--default_armature_id',
        type=int,
        default=1,
        help='Default armature ID to use when generating annotations.'
    )
    args = parser.parse_args()

    try:
        config = load_config_from_yaml(args.config)
        if not config: 
            logger.error("Failed to load a valid configuration. Exiting.")
            return

        paths_config = config.get('paths', {})
        data_root = paths_config.get('data_root_dir', './data/')

        if args.generate_annotations:
            logger.info("Annotation generation mode selected.")
            # Generate for training set
            train_ann_filename = paths_config.get('annotations_file_name')
            if train_ann_filename:
                generate_annotation_file(
                    data_root=data_root,
                    motion_subdir_name=paths_config.get('motion_subdir', 'new_joints'),
                    text_subdir_name=config.get('text_subdir_name_for_generation', 'texts'), # New config key or hardcode
                    output_csv_filename=train_ann_filename,
                    default_armature_id=args.default_armature_id
                )
            else:
                logger.warning("Config 'paths.annotations_file_name' not specified. Skipping training annotation generation.")

            val_ann_filename = paths_config.get('val_annotations_file_name')
            if val_ann_filename:
                generate_annotation_file(
                    data_root=data_root,
                    motion_subdir_name=paths_config.get('motion_subdir_val', paths_config.get('motion_subdir', 'new_joints')), # Allow specific val motion_subdir
                    text_subdir_name=config.get('text_subdir_name_for_generation_val', config.get('text_subdir_name_for_generation', 'texts')), # Allow specific val text_subdir
                    output_csv_filename=val_ann_filename,
                    default_armature_id=args.default_armature_id
                )
            else:
                logger.info("Config 'paths.val_annotations_file_name' not specified. Skipping validation annotation generation.")
            
            logger.info("Annotation generation process finished.")

        else:
            logger.info("Training mode selected.")
            run_training_pipeline(config)

    except FileNotFoundError:
        logger.error(f"Ensure the config file specified ('{args.config}') exists.")
    except Exception as e:
        logger.critical(f"A critical error occurred in the main execution flow: {e}", exc_info=True)
    finally:
        logger.info("Main script execution attempt completed.")


if __name__ == "__main__":
    main_entry_point()