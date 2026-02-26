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
        self.drift_epsilon = hyperparameters.drift_epsilon
        self.n_critic = hyperparameters.n_critic
        self.alpha = hyperparameters.alpha
        self.is_noise = hyperparameters.noise_injection
        self.n_realisations = hyperparameters.n_realisations

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

    def losses(self, set_type, gen_loss, adv_loss, cont_loss, mean_sr, mean_hr, crit_loss=None, grad_pen=None):
        losses_dict = {
            f"{set_type} Generator Loss": gen_loss,
            f"{set_type} Adversarial Loss" : adv_loss,
            f"{set_type} Content_loss" : cont_loss,
            f"{set_type} Wasserstein Distance": mean_hr - mean_sr,
        }
        if set_type == 'Train':
            losses_dict[f"{set_type} Critic Loss"] = crit_loss
            losses_dict[f"{set_type} Gradient Penalty"] = grad_pen
        return losses_dict

    def training_step(self, batch, batch_idx):
        lr, hr, hr_cov = self.unpack_batch(batch)
        g_opt, c_opt = self.optimizers()
        self.toggle_optimizer(c_opt)

        sr = self.G(lr, hr_cov).detach()
        mean_sr = self.C(sr).mean()
        mean_hr = self.C(hr).mean()
        gp = compute_gradient_penalty(self.C, hr, sr)
        drift = (mean_hr ** 2) * self.drift_epsilon

        loss_c = mean_sr - mean_hr + self.gp_lambda * gp + drift
        go_downhill(self, loss_c, c_opt)

        # Train generator every n_critic iterations
        if (batch_idx + 1) % self.n_critic == 0:
            self.toggle_optimizer(g_opt)
            
            # Generate single realization for adversarial loss
            sr = self.G(lr, hr_cov)
            
            # Prepare data for CRPS computation (multiple realizations per sample)
            batch_size = lr.shape[0]
            
            # Repeat each sample n_realisation times to generate ensemble
            dat_lr = [lr[i].unsqueeze(0).repeat(self.n_realisations, 1, 1, 1) 
                    for i in range(batch_size)]
            dat_hr_cov = [hr_cov[i].unsqueeze(0).repeat(self.n_realisations, 1, 1, 1) 
                        for i in range(batch_size)]
            dat_hr = [hr[i] for i in range(batch_size)]
            
            # Generate ensemble realizations and compute CRPS for each sample
            dat_sr = [self.G(lr_rep, cov_rep) 
                    for lr_rep, cov_rep in zip(dat_lr, dat_hr_cov)]
            crps_ls = [crps_empirical(sr_ens, hr_true) 
                    for sr_ens, hr_true in zip(dat_sr, dat_hr)]
            crps = torch.cat(crps_ls)

            crps_mean = torch.mean(crps)

            cont_loss = self.alpha * crps_mean

            # Get adversarial loss component
            adv_loss = torch.mean(self.C(sr))
            
            # Combined loss: adversarial + content (CRPS)
            loss_g = -adv_loss + cont_loss
            
            go_downhill(self, loss_g, g_opt)

            self.log_dict(
                self.losses(set_type="Train",
                            gen_loss=loss_g.detach(),
                            adv_loss=adv_loss.detach(),
                            cont_loss=cont_loss.detach(),
                            mean_sr=mean_sr.detach(),
                            mean_hr=mean_hr.detach(),
                            crit_loss=loss_c.detach(),
                            grad_pen=self.gp_lambda * gp.detach(),
                            ),
                sync_dist=True,
            )

        if (batch_idx + 1) % self.log_every_n_steps == 0:
            configure_figure(self.G, self.logger, "Train", lr, hr, hr_cov)

    def validation_step(self, batch, batch_idx):
        lr, hr, hr_cov = batch
        sr = self.G(lr, hr_cov).detach()
        mean_sr = self.C(sr).mean()
        mean_hr = self.C(hr).mean()

        # Compute CRPS on validation set
        batch_size = lr.shape[0]
        dat_lr = [lr[i].unsqueeze(0).repeat(self.n_realisations, 1, 1, 1) 
                for i in range(batch_size)]
        dat_hr_cov = [hr_cov[i].unsqueeze(0).repeat(self.n_realisations, 1, 1, 1) 
                    for i in range(batch_size)]
        dat_hr = [hr[i] for i in range(batch_size)]
        
        dat_sr = [self.G(lr_rep, cov_rep) 
                for lr_rep, cov_rep in zip(dat_lr, dat_hr_cov)]
        crps_ls = [crps_empirical(sr_ens, hr_true) 
                for sr_ens, hr_true in zip(dat_sr, dat_hr)]
        crps = torch.cat(crps_ls)
        crps_mean = torch.mean(crps)
        cont_loss = self.alpha * crps_mean

        adv_loss = torch.mean(self.C(sr))
        loss_g = -adv_loss + cont_loss

        self.log_dict(
            self.losses(set_type="Validation",
                        gen_loss=loss_g.detach(),
                        adv_loss=adv_loss.detach(),
                        cont_loss=cont_loss.detach(),
                        mean_sr=mean_sr.detach(),
                        mean_hr=mean_hr.detach(),
                        ),
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
