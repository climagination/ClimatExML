import os
import comet_ml
import lightning as pl
import torch
from ClimatExML.models import Generator, HRStreamGenerator, Critic
import torch.nn as nn
from ClimatExML.losses import crps_empirical
from ClimatExML.logging_tools import gen_grid_images

from torchmetrics.functional import (
    mean_absolute_error,
    mean_squared_error,
)
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure
from omegaconf.dictconfig import DictConfig
import matplotlib.pyplot as plt


class SuperResolutionWGANGP(pl.LightningModule):
    tracking: DictConfig
    hardware: DictConfig
    hyperparameters: DictConfig
    invariant: DictConfig
    is_noise: bool

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
        self.num_workers = hardware.num_workers
        self.log_every_n_steps = tracking.log_every_n_steps
        self.validation_log_every_n_steps = tracking.validation_log_every_n_steps

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
        #torch.manual_seed(1) - maybe unessecary 
        # networks
        n_covariates, lr_dim, _ = self.lr_shape
        n_predictands, hr_dim, _ = self.hr_shape

        n_hr_covariates = self.hr_invariant_shape[0]
        self.G = HRStreamGenerator(
            self.is_noise, lr_dim, hr_dim, n_covariates, n_hr_covariates, n_predictands
        )

        self.C = Critic(lr_dim, hr_dim, n_predictands)
        self.automatic_optimization = False

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        current_batch_size = real_samples.size(0)
        # Calculate interpolation

        # gradient penalty
        gp_alpha = (
            torch.rand(current_batch_size, 1, 1, 1, requires_grad=True)
            .expand_as(real_samples)
            .to(real_samples)
        )

        interpolated = gp_alpha * real_samples.data + (1 - gp_alpha) * fake_samples.data

        # Calculate probability of interpolated examples
        critic_interpolated = self.C(interpolated)

        # self.register_buffer("gp_ones", torch.ones(critic_interpolated.size(), requires_grad=True))

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(critic_interpolated.size(), requires_grad=True).to(
                real_samples
            ),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(current_batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)

        # Return gradient penalty
        return ((gradients_norm - 1) ** 2).mean()

    def losses(self, set_type, hr, sr, mean_sr, mean_hr):
        return {
            f"{set_type} MAE": mean_absolute_error(sr, hr),
            f"{set_type} MSE": mean_squared_error(sr, hr),
            f"{set_type} MSSIM": multiscale_structural_similarity_index_measure(sr, hr),
            f"{set_type} Wasserstein Distance": mean_hr - mean_sr,
        }

    def configure_figure(self, set_type, lr, hr, hr_cov, n_examples=3, cmap="viridis"):
        use_hr_cov = self.hr_invariant_shape is not None

        for var in range(hr.shape[1]):
            fig = plt.figure(figsize=(30, 10))
            fig = gen_grid_images(
                var,
                fig,
                self.G,
                lr,
                hr,
                hr_cov,
                use_hr_cov,
                n_examples,
                cmap=cmap,
            )
            self.logger.experiment.log_figure(
                figure_name=f"{set_type}_images_{var}", figure=fig, overwrite=True
            )
            plt.close(fig)

    def training_step(self, batch, batch_idx):

        # train generator
        lr, hr, hr_cov = batch[0]
        lr = lr.float()
        hr = hr.float()
        hr_cov = hr_cov.float()

        g_opt, c_opt = self.optimizers()
        self.toggle_optimizer(c_opt)

        sr = self.G(lr, hr_cov).detach()
        gradient_penalty = self.compute_gradient_penalty(hr, sr)
        mean_sr = torch.mean(self.C(sr))
        mean_hr = torch.mean(self.C(hr))
        loss_c = mean_sr - mean_hr + self.gp_lambda * gradient_penalty

        self.go_downhill(loss_c, c_opt)

        if (batch_idx + 1) % self.n_critic == 0:
            self.toggle_optimizer(g_opt)
            sr = self.G(lr, hr_cov)
            n_realisation = 4
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

            self.go_downhill(loss_g, g_opt)

        self.log_dict(
            self.losses("Train", hr, sr.detach(), mean_sr.detach(), mean_hr.detach()),
            sync_dist=True,
        )

        if (batch_idx + 1) % self.log_every_n_steps == 0:
            self.configure_figure(
                "Train",
                lr,
                hr,
                hr_cov,
                n_examples=3,
                cmap="viridis",
            )
        #self.epoch_num = batch_idx

    def validation_step(self, batch, batch_idx):
        lr, hr, hr_cov = batch

        sr = self.G(lr, hr_cov).detach()
        mean_sr = torch.mean(self.C(sr).detach())
        mean_hr = torch.mean(self.C(hr).detach())
        self.log_dict(
            self.losses("Validation", hr, sr, mean_sr, mean_hr),
            sync_dist=True,
        )

        if (batch_idx + 1) % self.validation_log_every_n_steps == 0:
            self.configure_figure(
                "Validation",
                lr,
                hr,
                hr_cov,
                n_examples=3,
                cmap="viridis",
            )

    def on_train_epoch_end(
        self,
    ):
        # save files in working directory for inference
        epoch = self.current_epoch
        if(epoch % 5 == 0):
            g_path = f"{os.environ['OUTPUT_DIR']}generator_{epoch}.pt"
            c_path = f"{os.environ['OUTPUT_DIR']}critic.pt"

            g_scripted = torch.jit.script(self.G)
            c_scripted = torch.jit.script(self.C)
            g_scripted.save(g_path)
            c_scripted.save(c_path)
            self.logger.experiment.log_model("Generator", g_path, overwrite=True)
            self.logger.experiment.log_model("Critic", c_path, overwrite=True)

    def go_downhill(self, loss, opt):
        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()
        self.untoggle_optimizer(opt)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2)
        )
        opt_d = torch.optim.Adam(
            self.C.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2)
        )
        return opt_g, opt_d
