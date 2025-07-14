"""
Train a diffusion model on motion data.
"""

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from models.diffusion.training_loop import TrainLoop
from dataset.dataset_diffusion.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion_general_skeleton


def prepare_save_dir(args):
    """
    Determine and prepare the directory where model checkpoints and configs will be saved.
    """
    save_dir = args.save_dir

    if save_dir is None:
        # Auto-generate a directory name based on model config
        model_name = f"{args.model_prefix}_dataset_truebones_bs_{args.batch_size}_latentdim_{args.latent_dim}"
        base_path = os.path.join(os.getcwd(), "save")
        # Check if the base path exists, create it if not
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        existing = [m for m in os.listdir(base_path) if m.startswith(model_name)]
        if existing and not args.overwrite:
            model_name = f"{model_name}_{len(existing)}"
        save_dir = os.path.join(base_path, model_name)
        args.save_dir = save_dir

    # Safety checks
    if os.path.exists(save_dir) and not args.overwrite:
        raise FileExistsError(f"Save directory '{save_dir}' already exists. Use --overwrite to allow reuse.")
    os.makedirs(save_dir, exist_ok=True)

    # Save training arguments to file
    args_path = os.path.join(save_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    return save_dir


def main():
    # Parse training arguments and set seeds
    args = train_args()
    fixseed(args.seed)

    # Prepare save directory
    prepare_save_dir(args)

    # Setup distributed training (if applicable)
    dist_util.setup_dist(args.device)

    # Create data loader
    print("Creating data loader...")
    sbert_loader = get_dataset_loader(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        split='train',
        temporal_window=args.temporal_window,
        t5_name="t5-base",
        balanced=False,  # SBERT loader does not need to be balanced
        objects_subset=args.objects_subset,
        dataset_type='PairedMotionDataset'  # Use PairedMotionDataset for SBERT pairing
    )

    # Create model and diffusion components
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_general_skeleton(args)
    model.to(dist_util.dev())

    # Start training
    print("Training...")
    TrainLoop(args, model, diffusion, sbert_loader).run_loop()


if __name__ == "__main__":
    main()
