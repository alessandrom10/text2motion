import argparse
import logging
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
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
from TrainerMDM import ArmatureMDMTrainer, KinematicLossCalculator, MDMGeometricLosses

from dataset.motion_dataset_loader import (
    MyTextToMotionDataset, 
    process_motion_file,
    collate_motion_data
)

from utils.diffusion_utils import (
    load_armature_config,
    setup_logging,
    get_noise_schedule,
    plot_training_history,
    get_bone_mask_for_armature,
    generate_and_save_sbert_embeddings
)

logger = logging.getLogger(__name__) # Gets a logger named after the current module

def scan_and_pair_all_available_samples(
    data_root: str,
    motion_subdir_name: str,
    text_subdir_name: str,
    default_armature_id: int
) -> List[Dict[str, Any]]:
    """
    Scans motion and text directories to find all valid, paired samples.
    Returns a list of dictionaries, each containing 'motion_filename' (base name), 
    'text', and 'armature_id'.
    :param data_root: Base directory where motion and text subdirs are located.
    :param motion_subdir_name: Name of the subdirectory containing motion files.
    :param text_subdir_name: Name of the subdirectory containing text files.
    :param default_armature_id: Default armature ID to use for all samples.
    :return: List of dictionaries with paired sample information.
    """
    logger.info(f"Scanning for all available samples: Motion subdir='{motion_subdir_name}', Text subdir='{text_subdir_name}' in '{data_root}'")
    all_paired_samples_info = []
    motion_dir_abs_path = os.path.join(data_root, motion_subdir_name)
    text_dir_abs_path = os.path.join(data_root, text_subdir_name)

    if not os.path.isdir(motion_dir_abs_path):
        logger.error(f"Motion directory not found: {motion_dir_abs_path}"); return []
    if not os.path.isdir(text_dir_abs_path):
        logger.error(f"Text directory not found: {text_dir_abs_path}"); return []

    motion_filenames_with_ext = sorted([f for f in os.listdir(motion_dir_abs_path) if f.endswith(".npy")])
    logger.info(f"Found {len(motion_filenames_with_ext)} total .npy files in '{motion_dir_abs_path}'.")

    processed_motion_files_count = 0
    total_extracted_text_pairs_count = 0

    for motion_filename_ext in motion_filenames_with_ext:
        base_name = os.path.splitext(motion_filename_ext)[0]
        text_file_name_ext = base_name + ".txt"
        text_file_path_abs = os.path.join(text_dir_abs_path, text_file_name_ext)
        
        if os.path.exists(text_file_path_abs):
            try:
                num_descriptions_in_file = 0
                with open(text_file_path_abs, 'r', encoding='utf-8') as f:
                    for line_number, line_content in enumerate(f):
                        line_content_stripped = line_content.strip()
                        if not line_content_stripped: # Skip empty lines
                            continue

                        parts = line_content_stripped.split('#')
                        if not parts: # Should not happen if line_content_stripped is not empty
                            logger.warning(f"Unexpected empty line content after strip in {text_file_path_abs}, line {line_number + 1}. Skipping.")
                            continue
                        
                        natural_language_text = parts[0].strip()
                        
                        if natural_language_text:
                            all_paired_samples_info.append({
                                'motion_filename': motion_filename_ext, 
                                'text': natural_language_text, # Only the natural language part
                                'armature_id': default_armature_id
                            })
                            total_extracted_text_pairs_count += 1
                            num_descriptions_in_file +=1
                        else:
                            logger.warning(f"Empty natural language text extracted from line {line_number + 1} in {text_file_path_abs}. Line: '{line_content_stripped[:100]}...'")
                if num_descriptions_in_file > 0:
                    processed_motion_files_count +=1
                else:
                    logger.warning(f"No valid descriptions found in {text_file_path_abs} for motion {motion_filename_ext}.")

            except Exception as e:
                logger.warning(f"Could not read or parse text file {text_file_path_abs}: {e}. Skipping this motion file.")
        else:
            logger.warning(f"Matching text file not found for {motion_filename_ext} at {text_file_path_abs}. Skipping.")
    
    logger.info(f"Successfully processed {processed_motion_files_count} motion files, yielding {total_extracted_text_pairs_count} (motion_file, single_natural_language_description) pairs.")
    return all_paired_samples_info

# --- Generate Annotation File from a LIST of samples ---
def generate_annotation_file_from_sample_list(
    samples_list: List[Dict[str, Any]],
    data_root: str,
    output_csv_filename: str 
) -> None:
    """
    Creates a CSV annotation file from a provided list of sample dictionaries.
    The 'motion_filename' in the sample_list should be the filename (e.g., "000000.npy").
    The output CSV will be saved in data_root/output_csv_filename.
    :param samples_list: List of dictionaries, each with 'motion_filename', 'text', and 'armature_id'.
    :param data_root: Base directory where the output CSV will be saved.
    :param output_csv_filename: Name of the output CSV file (e.g., "annotations_train.csv").
    :return: None
    """
    if not samples_list:
        logger.warning(f"No samples provided to write to '{output_csv_filename}'. Skipping file generation.")
        return

    output_csv_full_path = os.path.join(data_root, output_csv_filename)
    logger.info(f"Generating annotation file: {output_csv_full_path} with {len(samples_list)} samples.")

    try:
        df = pd.DataFrame(samples_list)
        # Ensure the directory for the output CSV exists
        os.makedirs(os.path.dirname(output_csv_full_path), exist_ok=True)
        df.to_csv(output_csv_full_path, index=False)
        logger.info(f"Successfully created annotation file: {output_csv_full_path}")
    except Exception as e:
        logger.error(f"Failed to write annotation file {output_csv_full_path}: {e}", exc_info=True)


# --- Main Training Orchestration Function ---
def run_training_pipeline(config: Dict[str, Any]) -> None:
    """
    Orchestrates the entire training pipeline using parameters from the config dictionary.
    """
    run_name = config.get('run_name', 'default_run')
    paths_config = config.get('paths', {})
    model_save_dir = paths_config.get('model_save_dir', './trained_models_output')

    # Load armature configuration
    default_armature_cfg_file = Path(project_root) / 'config' / 'armature_config.json'
    armature_config_file = paths_config.get('armature_config_file', str(default_armature_cfg_file))
    
    if not os.path.isabs(armature_config_file):
        armature_config_file = os.path.join(project_root, armature_config_file)

    loaded_armature_config_data = load_armature_config(armature_config_file) # load_armature_config from utils

    if loaded_armature_config_data is None:
        logger.warning("Armature configuration could not be loaded. Bone masking might not work as expected.")
        # Potentially set a default or raise an error if it's critical
        loaded_armature_config_data = {} # Empty dict as fallback

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

    # 2. Precomputed SBERT Embeddings
    sbert_train_emb_path = os.path.join(paths_config.get('data_root_dir', './data/'), 
                                        paths_config.get('precomputed_sbert_train_filename', 'sbert_embeddings_train.pt'))

    if not os.path.exists(sbert_train_emb_path):
        logger.info(f"Precomputed SBERT embeddings for train not found at {sbert_train_emb_path}.")
        # Call the function to generate SBERT embeddings
        logger.info("Generating SBERT embeddings for training data...")
        generate_and_save_sbert_embeddings(
            annotations_csv_path=os.path.join(paths_config['data_root_dir'], paths_config['annotations_file_name']),
            sbert_model_name=model_hparams['sbert_model_name'],
            output_embeddings_path=sbert_train_emb_path,
            device=str(device)
        )
        if not os.path.exists(sbert_train_emb_path):
            logger.error("Failed to generate or find precomputed SBERT train embeddings. Aborting.")
            return
    
    sbert_val_emb_path = None
    if paths_config.get('val_annotations_file_name'):
        sbert_val_emb_path = os.path.join(paths_config.get('data_root_dir', './data/'),
                                          paths_config.get('precomputed_sbert_val_filename', 'sbert_embeddings_val.pt'))
        if not os.path.exists(sbert_val_emb_path):
            logger.info(f"Precomputed SBERT embeddings for validation not found at {sbert_val_emb_path}.")
            # Call the function to generate SBERT embeddings for validation
            logger.info("Generating SBERT embeddings for validation data...")
            generate_and_save_sbert_embeddings(
                annotations_csv_path=os.path.join(paths_config['data_root_dir'], paths_config['val_annotations_file_name']),
                sbert_model_name=model_hparams['sbert_model_name'],
                output_embeddings_path=sbert_val_emb_path,
                device=str(device)
            )
            if not os.path.exists(sbert_val_emb_path):
                logger.error("Failed to generate or find precomputed SBERT validation embeddings. Aborting.")
                return
            
    # 3. Create Datasets and DataLoaders
    logger.info("Loading training data...")
    try:
        train_dataset = MyTextToMotionDataset(
            root_dir=paths_config.get('data_root_dir', './data/'),
            annotations_file_name=paths_config.get('annotations_file_name', 'annotations_train.csv'),
            motion_subdir=paths_config.get('motion_subdir', 'new_joints'),
            precomputed_sbert_embeddings_path=sbert_train_emb_path,
            num_diffusion_timesteps=diffusion_params.get('num_diffusion_timesteps', 50),
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
                precomputed_sbert_embeddings_path=sbert_val_emb_path,
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

    # 4. Setup Model, Optimizer, Scheduler (as in your script)
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

    # 5. Setup Kinematic Loss Calculator (as in your script)
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

    mdm_geom_loss_calc = None
    mdm_geom_cfg = config.get('mdm_geometric_losses', {})
    if mdm_geom_cfg.get('use_mdm_geometric_losses', False):
        logger.info("Initializing MDMGeometricLosses calculator...")
        mdm_geom_loss_calc = MDMGeometricLosses(
            lambda_pos=mdm_geom_cfg.get('lambda_pos', 0.0),
            lambda_vel=mdm_geom_cfg.get('lambda_vel', 0.0), # Set to 0 if you only want L_foot
            lambda_foot=mdm_geom_cfg.get('lambda_foot', 0.0), # Set >0 to activate
            num_joints=model_hparams.get('num_joints_for_geom', 22),
            features_per_joint=model_hparams.get('features_per_joint_for_geom', 3),
            foot_joint_indices=mdm_geom_cfg.get('foot_joint_indices', [10, 11]), # Example
            device=str(device),
            get_bone_mask_fn=get_bone_mask_for_armature # Optional: for further masking geometric losses
        )
    logger.info(f"MDMGeometricLosses calculator {'ENABLED' if mdm_geom_loss_calc else 'DISABLED'}")

    # 6. Initialize Trainer (as in your script)
    trainer = ArmatureMDMTrainer(
        model=model, 
        optimizer=optimizer, 
        get_bone_mask_fn=get_bone_mask_for_armature,
        armature_config_data=loaded_armature_config_data,
        device=str(device), lr_scheduler=lr_scheduler,
        main_loss_type=training_hparams.get('main_loss_type', "mse"),
        cfg_drop_prob=training_hparams.get('cfg_drop_prob_trainer', 0.1),
        kinematic_loss_calculator=kinematic_calculator,
        mdm_geometric_loss_calculator=mdm_geom_loss_calc,
        early_stopping_patience=early_stopping_params.get('patience', 10),
        early_stopping_min_delta=early_stopping_params.get('min_delta', 0.0001),
        model_save_path=os.path.join(model_save_dir, paths_config.get('model_filename', 'model.pth'))
    )

    # 7. Start Training
    num_epochs_to_run = training_hparams.get('num_epochs', 50)
    logger.info(f"Attempting to start training for {num_epochs_to_run} epochs...")
    
    history = trainer.train( 
        train_loader=train_loader,
        num_epochs=num_epochs_to_run,
        val_loader=val_loader
    )

    completed_epochs = trainer.completed_epochs_count if hasattr(trainer, 'completed_epochs_count') else num_epochs_to_run
    logger.info(f"======= Training Run Finished ({completed_epochs} epochs completed) =======")

    # 8. Plotting results
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
    parser.add_argument(
        '--val_split_ratio', type=float, default=0.2, # Added for 80/20 split
        help='Ratio of data to use for the validation set (e.g., 0.2 for 20%).'
    )
    args = parser.parse_args()

    try:
        config = load_config_from_yaml(args.config)
        if not config: logger.error("Failed to load configuration. Exiting."); return

        paths_cfg = config.get('paths', {})
        data_root_cfg = paths_cfg.get('data_root_dir', './data/') # Fallback if not in config

        if args.generate_annotations:
            logger.info("Annotation generation mode selected.")
            
            # 1. Scan all available data first
            all_samples = scan_and_pair_all_available_samples(
                data_root=data_root_cfg,
                motion_subdir_name=paths_cfg.get('motion_subdir', 'new_joints'), # Use config, fallback
                text_subdir_name=config.get('text_subdir_name_for_prep', 'texts'), # Use config, fallback
                default_armature_id=args.default_armature_id
            )

            if not all_samples:
                logger.error("No samples found to generate annotations. Exiting.")
                return

            # 2. Split the list of all samples into training and validation sets
            if args.val_split_ratio > 0 and args.val_split_ratio < 1:
                train_samples_list, val_samples_list = train_test_split(
                    all_samples, 
                    test_size=args.val_split_ratio, 
                    random_state=42,
                    shuffle=True
                )
                logger.info(f"Split all {len(all_samples)} samples into {len(train_samples_list)} train and {len(val_samples_list)} validation samples.")
            else: # No validation split or invalid ratio
                train_samples_list = all_samples
                val_samples_list = []
                logger.info(f"Using all {len(all_samples)} samples for training. No validation set will be generated from split.")

            # 3. Generate annotation file for the training set
            train_ann_filename = paths_cfg.get('annotations_file_name')
            if train_ann_filename and train_samples_list:
                generate_annotation_file_from_sample_list(
                    samples_list=train_samples_list,
                    data_root=data_root_cfg,
                    output_csv_filename=train_ann_filename
                )
            elif not train_ann_filename:
                logger.warning("Config 'paths.annotations_file_name' not specified. Skipping training annotation generation.")
            elif not train_samples_list:
                logger.warning("Training sample list is empty after split. Skipping training annotation generation.")


            # 4. Generate annotation file for the validation set
            val_ann_filename = paths_cfg.get('val_annotations_file_name')
            if val_ann_filename and val_samples_list:
                generate_annotation_file_from_sample_list(
                    samples_list=val_samples_list,
                    data_root=data_root_cfg, # Output CSV will be in data_root_cfg
                    output_csv_filename=val_ann_filename
                )
            elif not val_ann_filename:
                logger.info("Config 'paths.val_annotations_file_name' not specified. Skipping validation annotation generation.")
            elif not val_samples_list:
                logger.info("Validation sample list is empty. Skipping validation annotation generation.")
            
            logger.info("Annotation generation process finished.")

        else: # If not generating annotations, proceed to training
            logger.info("Training mode selected.")
            # Ensure annotation files exist before trying to train
            train_ann_full_path = os.path.join(data_root_cfg, paths_cfg.get('annotations_file_name', ''))
            if not os.path.exists(train_ann_full_path):
                logger.error(f"Training annotation file not found: {train_ann_full_path}. "
                             "Please generate annotations first using the --generate_annotations flag.")
                return
            
            val_ann_filename_cfg = paths_cfg.get('val_annotations_file_name')
            if val_ann_filename_cfg: # If a validation file is configured
                val_ann_full_path = os.path.join(data_root_cfg, val_ann_filename_cfg)
                if not os.path.exists(val_ann_full_path):
                    logger.warning(f"Validation annotation file '{val_ann_full_path}' configured but not found. "
                                   "Proceeding without validation or generate it first.")
                    # Optionally, you could force run_training_pipeline to not use val_loader
                    # by modifying the config dict passed to it, or handle in run_training_pipeline
            
            run_training_pipeline(config)

    except FileNotFoundError:
        logger.error(f"Ensure the config file specified ('{args.config}') exists.")
    except Exception as e:
        logger.critical(f"A critical error occurred in the main execution flow: {e}", exc_info=True)
    finally:
        logger.info("Main script execution attempt completed.")


if __name__ == "__main__":
    main_entry_point()