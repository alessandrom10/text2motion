import argparse
import logging
import sys
from pathlib import Path
import yaml
from typing import Any, Dict, List, Optional
import torch
import numpy as np

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

if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))
from dataset_loader import MyTextToMotionDataset
from diffusion_utils import setup_logging, load_armature_config

from motion_generator import MotionGenerator

logger = logging.getLogger(__name__)

def run_batch_generation(
    config_path: Path,
    model_checkpoint_path: Path,
    output_base_dir: Path,
    dataset_split: str,
    num_samples_to_generate: int,
    cfg_scale: float,
    num_frames_override: Optional[int] = None,
    generation_run_name: Optional[str] = None,
    const_noise_for_sampling: bool = False,
    progress_bar_batch: bool = True
) -> None:
    """
    Generates motions for a subset of a dataset using the MotionGenerator class.
    :param config_path: Path to the main YAML configuration file.
    :param model_checkpoint_path: Path to the pre-trained model checkpoint (.pth).
    :param output_base_dir: Base directory to save generated motions.
    :param dataset_split: Which dataset split to use for prompts ('train', 'val', 'test').
    :param num_samples_to_generate: Number of samples to generate from the dataset.
    :param cfg_scale: Classifier-Free Guidance scale for generation.
    :param num_frames_override: Optional override for the number of frames in generated motions.
    :param generation_run_name: Optional name for this generation run (output subdirectory).
    :param const_noise_for_sampling: Use constant noise during sampling for reproducibility (if supported by sampler).
    :param progress_bar_batch: Whether to show a progress bar during batch generation.
    :return: None
    """
    # 1. Load Main Configuration (primarily for paths and dataset params)
    logger.info(f"Loading main configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f: config = yaml.safe_load(f)
    except Exception as e: logger.error(f"Error loading config {config_path}: {e}"); return
    logger.info("Main configuration loaded.")

    paths_cfg = config.get('paths', {})
    dataset_cfg = config.get('dataset_parameters', {})
    gen_params_from_config = config.get('generation_params', {})

    # Output directory setup
    gen_run_name_final = generation_run_name if generation_run_name else f"generated_{dataset_split}_cfg{cfg_scale}"
    output_dir_for_run = output_base_dir / gen_run_name_final
    output_dir_for_run.mkdir(parents=True, exist_ok=True) # MotionGenerator will also ensure subdirs
    
    setup_logging(str(output_dir_for_run), f"batch_generation_log_{gen_run_name_final}")
    # Device will be handled by MotionGenerator based on config or auto-detect
    device_main_cfg = config.get('training_hyperparameters', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')


    # 2. Initialize MotionGenerator
    logger.info(f"Initializing MotionGenerator with model: {model_checkpoint_path}")
    try:
        motion_generator = MotionGenerator(
            config_path=config_path, # Pass the main YAML config path
            model_checkpoint_path=model_checkpoint_path,
            device_str=device_main_cfg # Can explicitly pass device
        )
    except Exception as e:
        logger.error(f"Failed to initialize MotionGenerator: {e}", exc_info=True)
        return
    
    # 3. Load Dataset (to get prompts, armature IDs, etc.)
    logger.info(f"Preparing dataset for '{dataset_split}' split to extract generation requests...")
    data_root_path_str = paths_cfg.get('data_root_dir')
    data_root_abs = Path(data_root_path_str) if Path(data_root_path_str).is_absolute() else project_root / data_root_path_str


    annotations_file_config_key : str
    if dataset_split == "train":
        annotations_file_config_key = paths_cfg.get('annotations_file_name', 'train_annotations.jsonl')
    elif dataset_split == "val":
        annotations_file_config_key = paths_cfg.get('val_annotations_file_name', 'val_annotations.jsonl')
    elif dataset_split == "test":
        annotations_file_config_key = paths_cfg.get('test_annotations_file_name', 'test_annotations.jsonl')
    else:
        logger.error(f"Unsupported dataset_split: {dataset_split}. Use 'train', 'val', or 'test'.")
        return

    try:
        dataset_num_motion_features = motion_generator.model_cfg_loaded.get('num_motion_features_actual',
            motion_generator.model_cfg_loaded.get('njoints') * motion_generator.model_cfg_loaded.get('nfeats_per_joint'))

        dataset_for_prompts = MyTextToMotionDataset(
            root_dir=str(data_root_abs),
            annotations_file_name=annotations_file_config_key,
            motion_subdir=paths_cfg.get('motion_subdir'),
            precomputed_sbert_embeddings_path=str(project_root / paths_cfg.get(f'precomputed_sbert_{dataset_split}_path', "")),
            num_diffusion_timesteps=motion_generator.diffusion_cfg.get('num_diffusion_timesteps', 1000),
            alphas_cumprod=torch.tensor(np.cumprod(1.0 - motion_generator.betas_np, axis=0), dtype=torch.float32),
            num_motion_features=dataset_num_motion_features,
            dataset_mean_path=str(data_root_abs / dataset_cfg.get('dataset_mean_filename', 'mean.npy')),
            dataset_std_path=str(data_root_abs / dataset_cfg.get('dataset_std_filename', 'std.npy')),
            min_seq_len=dataset_cfg.get('min_seq_len_dataset'),
            max_seq_len=dataset_cfg.get('max_seq_len_dataset'),
            subset_size=None,
            shuffle_subset=False,
            data_device='cpu'
        )
        logger.info(f"Dataset for '{dataset_split}' split loaded: {len(dataset_for_prompts)} total samples available.")
    except Exception as e:
        logger.error(f"Failed to load dataset for prompts: {e}", exc_info=True)
        return

    # 4. Prepare Generation Requests
    generation_requests: List[Dict[str, Any]] = []
    # Use the 'samples' attribute directly from MyTextToMotionDataset
    if not hasattr(dataset_for_prompts, 'samples') or not dataset_for_prompts.samples:
        logger.error("Dataset does not have a 'samples' attribute or it's empty. "
                     "Ensure MyTextToMotionDataset._prepare_samples populates self.samples.")
        return

    available_samples = dataset_for_prompts.samples
    samples_to_process = min(num_samples_to_generate, len(available_samples))
    
    logger.info(f"Preparing {samples_to_process} generation requests from dataset '{dataset_split}'...")

    for i in range(samples_to_process):
        try:
            # Get sample details from the 'samples' list populated by MyTextToMotionDataset
            sample_info = available_samples[i] # Each item is a dict
            
            text_prompt = sample_info.get('text_prompt')
            armature_id = sample_info.get('armature_id')
            motion_file_path_str = sample_info.get('motion_file_path')

            if not text_prompt or armature_id is None or not motion_file_path_str:
                logger.warning(f"Skipping dataset sample {i} (from dataset's internal list) due to "
                               f"missing text_prompt, armature_id, or motion_file_path: {sample_info}")
                continue
            
            motion_file_path = Path(motion_file_path_str)
            base_filename_part = motion_file_path.stem
            original_length = None

            # Optionally, get original_length by loading the motion file
            # This adds I/O, consider if it's critical or if num_frames_override/default is sufficient
            if num_frames_override is None: # Only load if we might use original_length
                if motion_file_path.exists():
                    try:
                        # Temporarily load just to get shape for length
                        motion_data_for_length_check = np.load(motion_file_path)
                        original_length = motion_data_for_length_check.shape[0]
                    except Exception as e_load:
                        logger.warning(f"Could not load motion file {motion_file_path} to get original length: {e_load}. Will use default frames.")
                else:
                    logger.warning(f"Motion file {motion_file_path} not found for length check. Will use default frames.")
            
            generation_requests.append({
                'text': text_prompt,
                'armature_id': int(armature_id), # Ensure it's int
                'original_length': original_length,
                'base_filename_part': base_filename_part,
                'dataset_idx': i # For traceability if needed
            })
        except Exception as e:
            logger.error(f"Error processing dataset sample info at index {i} for generation: {e}. Skipping.", exc_info=True)
            continue
    
    if not generation_requests:
        logger.info("No valid generation requests prepared from the dataset. Exiting.")
        return

    # 5. Call MotionGenerator to process the batch
    default_frames_from_config = gen_params_from_config.get('num_frames_to_generate', 120)
    default_fps_from_config = gen_params_from_config.get('render_fps', 20)

    motion_generator.generate_and_save_from_metadata_list(
        generation_requests=generation_requests,
        base_output_dir=output_dir_for_run, # Specific output dir for this run
        cfg_scale=cfg_scale, # From CLI args
        default_num_frames=default_frames_from_config,
        num_frames_override=num_frames_override, # From CLI args
        render_fps=default_fps_from_config, # Or allow CLI override for fps too
        progress_bar=progress_bar_batch,
        const_noise_for_sampling=const_noise_for_sampling
    )

    logger.info(f"Batch generation completed. Outputs in: {output_dir_for_run}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate motions from dataset samples using MotionGenerator.")
    parser.add_argument('--config', type=str, default='config/diffusion_config.yaml', help='Path to the main YAML configuration file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model checkpoint (.pth).')
    parser.add_argument('--output_dir', type=str, default="batch_generated_motions", help='Base directory to save generated GIFs.')
    parser.add_argument('--dataset_split', type=str, default="train", choices=["train", "val", "test"], help="Dataset split for prompts.")
    parser.add_argument('--num_samples', type=int, default=20, help="Number of samples to generate.")
    parser.add_argument('--cfg_scale', type=float, default=2.5, help="Classifier-Free Guidance scale.")
    parser.add_argument('--num_frames', type=int, default=None, help="Override motion length. Default from dataset/config.")
    parser.add_argument('--run_name', type=str, default=None, help="Optional name for this generation run (output subdirectory).")
    parser.add_argument('--const_noise', action='store_true', help="Use constant noise during sampling for reproducibility (if supported by sampler).")
    parser.add_argument('--no_progress_bar', action='store_true', help="Disable the progress bar for batch generation.")


    cli_args = parser.parse_args()

    # Resolve config_path and model_path relative to project_root if not absolute
    def resolve_path(input_path_str: str, is_dir: bool = False) -> Path:
        p = Path(input_path_str)
        if p.is_absolute(): return p
        resolved_p = project_root / p
        if is_dir: return resolved_p # For directories, existence check is done by mkdir
        if resolved_p.exists(): return resolved_p
        # If not found relative to project_root, try relative to CWD (original behavior)
        return p 

    final_config_path = resolve_path(cli_args.config)
    if not final_config_path.exists():
        sys.stderr.write(f"ERROR: Configuration file '{final_config_path}' (abs) or '{cli_args.config}' (rel) not found.\n")
        sys.exit(1)

    final_model_path = resolve_path(cli_args.model_path)
    if not final_model_path.exists():
        sys.stderr.write(f"ERROR: Model checkpoint '{final_model_path}' (abs) or '{cli_args.model_path}' (rel) not found.\n")
        sys.exit(1)

    final_output_dir = resolve_path(cli_args.output_dir, is_dir=True)

    run_batch_generation(
        config_path=final_config_path,
        model_checkpoint_path=final_model_path,
        output_base_dir=final_output_dir,
        dataset_split=cli_args.dataset_split,
        num_samples_to_generate=cli_args.num_samples,
        cfg_scale=cli_args.cfg_scale,
        num_frames_override=cli_args.num_frames,
        generation_run_name=cli_args.run_name,
        const_noise_for_sampling=cli_args.const_noise,
        progress_bar_batch=not cli_args.no_progress_bar
    )