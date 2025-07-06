import os
import functools
from os.path import join as pjoin

import torch
from torch.optim import AdamW
from tqdm import tqdm

from utils.diffusion import logger
from utils.diffusion.fp16_util import MixedPrecisionTrainer
from utils.diffusion.resample import LossAwareSampler, create_named_schedule_sampler
from utils import dist_util
from utils.checkpoint import (log_loss_dict,save,_load_and_sync_parameters,_load_optimizer_state,_log_step,_log_metrics)

torch.autograd.set_detect_anomaly(True)


class TrainLoop:
    def __init__(self, args, model, diffusion, data):
        self.args = args
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data

        self.device = torch.device(dist_util.dev() if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.resume_checkpoint = args.resume_checkpoint
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.use_fp16 = False
        self.fp16_scale_growth = 1e-3

        _load_and_sync_parameters(self, logger)

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=10000, gamma=0.99
        )

        if self.resume_step:
            _load_optimizer_state(self, logger)

        self.schedule_sampler = create_named_schedule_sampler('uniform', self.diffusion)

        self.model.train()
        self.use_ddp = False
        self.ddp_model = self.model

    def run_loop(self):
        print(f"Training for {self.num_steps} steps...")

        while self.total_step() < self.num_steps:
            print(f"Epoch start — step {self.total_step()}")
            for motion, cond in tqdm(self.data):
                if self.lr_anneal_steps and self.total_step() >= self.lr_anneal_steps:
                    break

                motion = motion.to(self.device)
                cond['y'] = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in cond['y'].items()}

                self.run_step(motion, cond)

                if self.total_step() % self.log_interval == 0:
                    _log_metrics(self, cond, logger)

                if self.total_step() % self.save_interval == 0 or self.total_step() == self.num_steps - 1:
                    save(self, logger)
                    self.model.train()

                self.step += 1
                if self.total_step() >= self.num_steps:
                    break

    def run_step(self, batch, cond, epoch=-1):
        self.forward_backward(batch, cond, epoch)
        self.mp_trainer.optimize(self.opt, self.lr_scheduler)
        self._anneal_lr()
        _log_step(self, logger)

    def forward_backward(self, batch, cond, epoch):
        self.mp_trainer.zero_grad()
        micro = batch
        micro_cond = cond
        t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)

        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model, micro, t, model_kwargs=micro_cond
        )

        losses = compute_losses()
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

        loss = (losses["loss"] * weights).mean()
        log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()}, logger)
        self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = self.total_step() / self.lr_anneal_steps
        new_lr = self.lr * (1 - frac_done)
        for group in self.opt.param_groups:
            group["lr"] = new_lr

    def total_step(self):
        return self.step + self.resume_step + 1 if self.resume_step else self.step

    def ckpt_file_name(self):
        return f"model{(self.step + self.resume_step):09d}.pt"
