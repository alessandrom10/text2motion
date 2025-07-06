import re
import os
from os.path import join as pjoin
from typing import Optional
from utils import dist_util
from utils.model_util import load_model
import blobfile as bf
import torch

def parse_resume_step_from_filename(filename: str) -> int:
    match = re.search(r'model(\d+)\.pt$', filename)
    return int(match.group(1)) if match else 0


def log_loss_dict(diffusion, ts, losses, logger):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for t, val in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", val)

def _log_step(trainer, logger):
    logger.logkv("step", trainer.total_step())
    logger.logkv("samples", (trainer.total_step() + 1) * trainer.global_batch)

def _log_metrics(trainer, cond, logger):
    print(f"Object type: {cond['y']['object_type']}")
    for k, v in logger.get_current().dumpkvs().items():
        if k == 'loss':
            print(f"Step [{trainer.total_step()}]: loss [{v:.5f}]")

def save(trainer, logger):

    state_dict = trainer.model.state_dict() if trainer.use_fp16 else \
        trainer.mp_trainer.master_params_to_state_dict(trainer.mp_trainer.master_params)

    if trainer.args.use_ema:
        state_dict_avg = trainer.model_avg.state_dict()
        state_dict = {'model': state_dict, 'model_avg': state_dict_avg}

    logger.log("Saving model checkpoint...")
    with bf.BlobFile(pjoin(trainer.save_dir, trainer.ckpt_file_name()), "wb") as f:
        torch.save(state_dict, f)

    opt_state = {'opt': trainer.opt.state_dict()}
    if trainer.use_fp16:
        opt_state['scaler'] = trainer.scaler.state_dict()
    with bf.BlobFile(pjoin(trainer.save_dir, f"opt{trainer.total_step():09d}.pt"), "wb") as f:
        torch.save(opt_state, f)

def _load_and_sync_parameters(trainer, logger):
    trainer.resume_checkpoint = find_resume_checkpoint(trainer) or trainer.resume_checkpoint
    if not trainer.resume_checkpoint:
        return

    trainer.resume_step = parse_resume_step_from_filename(trainer.resume_checkpoint)
    logger.log(f"Loading model from checkpoint: {trainer.resume_checkpoint}")
    state = dist_util.load_state_dict(trainer.resume_checkpoint, map_location=trainer.device)

    if 'model_avg' in state:
        load_model(trainer.model, state['model'])
        load_model(trainer.model_avg, state['model_avg'])
    else:
        load_model(trainer.model, state)
        if trainer.args.use_ema:
            trainer.model_avg.load_state_dict(trainer.model.state_dict())

def _load_optimizer_state(trainer, logger):
    opt_path = find_resume_opt_checkpoint(trainer)
    if not bf.exists(opt_path):
        return
    logger.log(f"Loading optimizer state from {opt_path}")
    state = dist_util.load_state_dict(opt_path, map_location=trainer.device)
   
    state = state['opt']
    #if trainer.use_fp16 and 'scaler' in state:
    #    trainer.scaler.load_state_dict(state['scaler'])
    #    state = state['opt']

    trainer.opt.load_state_dict(state)
    for group in trainer.opt.param_groups:
        group['weight_decay'] = trainer.weight_decay

def find_resume_checkpoint(trainer) -> Optional[str]:
    files = os.listdir(trainer.save_dir)
    ckpts = {int(m.group(1)): f for f in files if (m := re.match(r'model(\d+).pt$', f))}
    return pjoin(trainer.save_dir, ckpts[max(ckpts)]) if ckpts else None

def find_resume_opt_checkpoint(trainer) -> Optional[str]:
    files = os.listdir(trainer.save_dir)
    ckpts = {int(m.group(1)): f for f in files if (m := re.match(r'opt(\d+).pt$', f))}
    return pjoin(trainer.save_dir, ckpts[max(ckpts)]) if ckpts else None