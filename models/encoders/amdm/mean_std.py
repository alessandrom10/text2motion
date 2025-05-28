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

# Set up logger for this script
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

def process_single_motion_file_for_stats(
    file_path: str,
    num_expected_features: int
) -> Optional[np.ndarray]:
    """
    Load and flatten a single .npy motion file for statistics computation.
    Does not apply Z-score normalization; only reshapes the data.
    Root centering is optional and can be applied if needed.

    :param file_path: Path to the .npy motion file.
    :param num_expected_features: Expected number of features per frame.
    :return: Flattened motion data as a 2D numpy array, or None if the file is invalid.
    """
    try:
        motion_data_raw = np.load(file_path).astype(np.float32)
    except Exception as e:
        logger.error(f"Error loading motion file {file_path}: {e}")
        return None

    # Flatten if necessary
    if motion_data_raw.ndim == 3:
        if motion_data_raw.shape[1] * motion_data_raw.shape[2] == num_expected_features:
            # Optional root centering
            # motion_data_raw = motion_data_raw - motion_data_raw[:, 0:1, :]
            motion_data_flat = motion_data_raw.reshape(motion_data_raw.shape[0], -1)
        else:
            logger.warning(f"Shape mismatch in file {file_path} ({motion_data_raw.shape}), skipping.")
            return None
    elif motion_data_raw.ndim == 2 and motion_data_raw.shape[1] == num_expected_features:
        motion_data_flat = motion_data_raw
    else:
        logger.warning(f"Unhandled shape in file {file_path} ({motion_data_raw.shape}), skipping.")
        return None

    if motion_data_flat.shape[1] != num_expected_features:
        logger.warning(f"Feature mismatch in file {file_path}: expected {num_expected_features}, got {motion_data_flat.shape[1]}, skipping.")
        return None

    return motion_data_flat

def calculate_and_save_mean_std(config_path: str):
    """
    Calculate and save mean.npy and std.npy for the training dataset.

    :param config_path: Path to the YAML configuration file.
    """
    logger.info(f"Starting mean/std calculation with config: {config_path}")

    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading configuration file {config_path}: {e}")
        return

    paths_cfg = config.get('paths', {})
    model_cfg = config.get('model_hyperparameters', {})
    dataset_cfg = config.get('dataset_parameters', {})

    data_root_dir = Path(paths_cfg.get('data_root_dir'))
    annotations_train_file = data_root_dir / paths_cfg.get('annotations_file_name')
    motion_subdir = paths_cfg.get('motion_subdir')
    motion_dir_abs = data_root_dir / motion_subdir

    if "flat" in model_cfg.get('data_rep', ""):
        num_motion_features = model_cfg.get('num_motion_features_actual')
    else:
        num_motion_features = model_cfg.get('njoints') * model_cfg.get('nfeats_per_joint')

    if num_motion_features is None:
        logger.error("num_motion_features or njoints/nfeats_per_joint is not specified correctly in the config.")
        return

    logger.info(f"Expected features per frame: {num_motion_features}")

    # Load training annotations
    try:
        annotations_df = pd.read_csv(annotations_train_file)
        logger.info(f"Loaded {len(annotations_df)} annotations from {annotations_train_file}")
    except Exception as e:
        logger.error(f"Error reading annotation file {annotations_train_file}: {e}")
        return

    all_motion_data_list = []
    processed_files = 0
    failed_files = 0

    logger.info("Processing motion files from training set...")
    for idx, row in tqdm(annotations_df.iterrows(), total=len(annotations_df), desc="Processing motions"):
        motion_filename = row['motion_filename']
        file_path = motion_dir_abs / motion_filename

        if not file_path.exists():
            logger.warning(f"Motion file {file_path} not found, skipping.")
            failed_files += 1
            continue

        motion_features = process_single_motion_file_for_stats(str(file_path), num_motion_features)

        if motion_features is not None:
            all_motion_data_list.append(motion_features)
            processed_files += 1
        else:
            failed_files += 1

    if not all_motion_data_list:
        logger.error("No valid motion data processed. Check the files and paths.")
        return

    logger.info(f"Successfully processed {processed_files} files, {failed_files} failed or skipped.")

    # Concatenate and compute mean/std
    try:
        full_motion_dataset_np = np.concatenate(all_motion_data_list, axis=0)
        logger.info(f"Full dataset shape after concatenation: {full_motion_dataset_np.shape}")
    except Exception as e:
        logger.error(f"Error concatenating motion data: {e}")
        return

    dataset_mean = np.mean(full_motion_dataset_np, axis=0)
    dataset_std = np.std(full_motion_dataset_np, axis=0)
    dataset_std[dataset_std == 0] = 1e-8
    logger.info(f"Mean shape: {dataset_mean.shape}, Std shape: {dataset_std.shape}")

    # Save mean and std
    mean_save_path = data_root_dir / dataset_cfg.get('dataset_mean_filename', 'mean.npy')
    std_save_path = data_root_dir / dataset_cfg.get('dataset_std_filename', 'std.npy')

    try:
        np.save(mean_save_path, dataset_mean)
        logger.info(f"Mean saved to: {mean_save_path}")
        np.save(std_save_path, dataset_std)
        logger.info(f"Standard deviation saved to: {std_save_path}")
    except Exception as e:
        logger.error(f"Error saving mean/std files: {e}")
        return

    logger.info("Mean and std calculation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and save mean and std for the motion dataset.")
    parser.add_argument(
        '--config', type=str,
        default=str(Path(__file__).resolve().parent.parent.parent.parent / "config/diffusion_config.yaml"),
        help='Path to the YAML configuration file used for training.'
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
        sys.stderr.write(f"ERROR: Configuration file '{args.config}' not found.\n"
                         f"Tried paths: {config_file_arg_path}, {resolved_config_path}\n")
        sys.exit(1)

    calculate_and_save_mean_std(str(resolved_config_path))