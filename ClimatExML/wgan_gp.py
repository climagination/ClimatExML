import torch
import torch.nn as nn
from torchmetrics.functional import mean_absolute_error, mean_squared_error
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure
from lightning.pytorch.loggers import CometLogger

from ClimatExML.losses import crps_empirical #added crps_empirical from losses.py
from ClimatExML.models import HRStreamGenerator, Critic
from ClimatExML.trainer_utils import go_downhill, configure_figure, compute_gradient_penalty
from ClimatExML.base_trainer import BaseSuperResolutionTrainer


import os
import comet_ml
import lightning as pl
import torch
from ClimatExML.models import HRStreamGenerator, Critic
import torch.nn as nn
from ClimatExML.logging_tools import gen_grid_images

from torchmetrics.functional import (
    mean_absolute_error,
    mean_squared_error,
)
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure
from omegaconf.dictconfig import DictConfig
import matplotlib.pyplot as plt


class SuperResolutionWGANGP(BaseSuperResolutionTrainer):
    def __init__(
        self,
        tracking,
        hardware,
        hyperparameters,
        invariant,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.log_every_n_steps = tracking.log_every_n_steps
        self.validation_log_every_n_steps = tracking.validation_log_every_n_steps
        self.save_dir = tracking.save_dir
        self.experiment_name = tracking.experiment_name

        self.learning_rate = hyperparameters.learning_rate
        self.b1 = hyperparameters.b1
        self.b2 = hyperparameters.b2
        self.gp_lambda = hyperparameters.gp_lambda
        self.n_critic = hyperparameters.n_critic
        self.alpha = hyperparameters.alpha
        self.is_noise = hyperparameters.noise_injection

        self.lr_shape = invariant.lr_shape
        self.hr_shape = invariant.hr_shape
        self.hr_invariant_shape = invariant.hr_invariant_shape

        n_covariates, lr_dim, _ = self.lr_shape
        n_predictands, hr_dim, _ = self.hr_shape
        n_hr_covariates = self.hr_invariant_shape[0]

        self.G = HRStreamGenerator(
            noise=self.is_noise,
            filters=lr_dim,
            fine_dims=hr_dim,
            channels=n_covariates,
            channels_hr_cov=n_hr_covariates,
            n_predictands=n_predictands,
        )
        self.C = Critic(lr_dim, hr_dim, n_predictands)
        self.automatic_optimization = False

    def unpack_batch(self, batch):
        lr, hr, hr_cov = batch[0]
        return lr.float(), hr.float(), hr_cov.float()

    def losses(self, set_type, hr, sr, mean_sr, mean_hr):
        return {
            f"{set_type} MAE": mean_absolute_error(sr, hr),
            f"{set_type} MSE": mean_squared_error(sr, hr),
            #f"{set_type} MSSIM": multiscale_structural_similarity_index_measure(sr, hr),
            f"{set_type} Wasserstein Distance": mean_hr - mean_sr,
        }

    def training_step(self, batch, batch_idx):
        lr, hr, hr_cov = self.unpack_batch(batch)
        g_opt, c_opt = self.optimizers()
        self.toggle_optimizer(c_opt)

        sr = self.G(lr, hr_cov).detach()
        mean_sr = self.C(sr).mean()
        mean_hr = self.C(hr).mean()
        gp = compute_gradient_penalty(self.C, hr, sr)
        loss_c = mean_sr - mean_hr + self.gp_lambda * gp
        go_downhill(self, loss_c, c_opt)

        #copied over from wgan_gp_stoch.py 
        # the n realisation part and crps part are not in wgan_gp.py
        if (batch_idx + 1) % self.n_critic == 0:
            self.toggle_optimizer(g_opt)
            sr = self.G(lr, hr_cov)
            n_realisation = 4 # related to batch size needs to be egual in config
            ls1 = [i for i in range(lr.shape[0])]
            dat_lr = [lr[i,...].unsqueeze(0).repeat(n_realisation,1,1,1) for i in ls1]
            dat_hr = [hr[i,...] for i in ls1]
            dat_sr = [self.G(lr,hr_cov[0:n_realisation,...]) for lr in dat_lr]
            crps_ls = [crps_empirical(sr,hr) for sr,hr in zip(dat_sr,dat_hr)]
            crps = torch.cat(crps_ls)

            loss_g = (
                -torch.mean(self.C(sr).detach())
                + self.alpha * torch.mean(crps)
            )

            go_downhill(self,loss_g, g_opt) #added self as its own argument here with Seamus

        self.log_dict(
            self.losses("Train", hr, sr.detach(), mean_sr.detach(), mean_hr.detach()),
            sync_dist=True,
        )

        if (batch_idx + 1) % self.log_every_n_steps == 0:
            configure_figure(self.G, self.logger, "Train", lr, hr, hr_cov)

    def validation_step(self, batch, batch_idx):
        lr, hr, hr_cov = batch
        sr = self.G(lr, hr_cov).detach()
        mean_sr = self.C(sr).mean()
        mean_hr = self.C(hr).mean()
        self.log_dict(
            self.losses("Validation", hr, sr, mean_sr, mean_hr),
            sync_dist=True,
        )

        if (batch_idx + 1) % self.validation_log_every_n_steps == 0:
            configure_figure(self.G, self.logger, "Validation", lr, hr, hr_cov)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        opt_c = torch.optim.Adam(self.C.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        return opt_g, opt_c

    def on_train_epoch_end(self):
        g_path = f"{self.save_dir}/generator_{self.experiment_name}.pt"
        c_path = f"{self.save_dir}/critic_{self.experiment_name}.pt"

        g_scripted = torch.jit.script(self.G)
        c_scripted = torch.jit.script(self.C)
        g_scripted.save(g_path)
        c_scripted.save(c_path)

        self.logger.experiment.log_model("Generator", g_path, overwrite=True)
        self.logger.experiment.log_model("Critic", c_path, overwrite=True)
