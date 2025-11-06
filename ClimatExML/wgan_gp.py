import torch
import torch.nn as nn
import os
import comet_ml
import lightning as pl
import torch
from torchmetrics.functional import mean_absolute_error, mean_squared_error
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure
from lightning.pytorch.loggers import CometLogger

from ClimatExML.losses import crps_empirical, lsd
from ClimatExML.models import HRStreamGenerator, Critic
from ClimatExML.trainer_utils import go_downhill, configure_figure, compute_gradient_penalty
from ClimatExML.base_trainer import BaseSuperResolutionTrainer

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
        self.batch_size = hyperparameters.batch_size

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
        """
        Compute a set of image restoration/regression losses for a given dataset split.

        Args:
            set_type (str): A label for the dataset split (e.g., "train", "val", "test").
            hr (array-like or torch.Tensor): Ground-truth high-resolution images.
                Must be broadcast-compatible with sr for elementwise error calculations.
            sr (array-like or torch.Tensor): Model-produced super-resolved (or predicted) images,
                with the same shape and dtype compatibility as hr.
            mean_sr (float): Precomputed mean pixel/value of sr across the batch (or dataset),
                used here to compute an approximate Wasserstein distance.
            mean_hr (float): Precomputed mean pixel/value of hr across the batch (or dataset).

        Returns:
            dict: A mapping from human-readable loss names to scalar loss values:
                - "<set_type> MAE": Mean absolute error between sr and hr.
                - "<set_type> MSE": Mean squared error between sr and hr.
                - "<set_type> Wasserstein Distance": Approximated as mean_hr - mean_sr.
              (Note: multiscale SSIM is present in the source but currently commented out.)

        Raises:
            ValueError: If hr and sr are not compatible for elementwise error computation
                (e.g., mismatched shapes that cannot be broadcast).
        """
        return {
            f"{set_type} MAE": mean_absolute_error(sr, hr),
            f"{set_type} MSE": mean_squared_error(sr, hr),
            #f"{set_type} MSSIM": multiscale_structural_similarity_index_measure(sr, hr),
            f"{set_type} Wasserstein Distance": mean_hr - mean_sr,
            f"{set_type} Log-Spectral Distance": torch.mean(torch.stack([lsd(sr[i][0], hr[i][0], d=4.0) for i in range(sr.shape[0])])),
        }

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step for WGAN-GP.

        Workflow:
        - Unpack the batch.
        - Update the critic n_critic times per generator update:
            * Generate super-resolved (SR) samples with the generator, detached
              so gradients do not flow into the generator during critic updates.
            * Compute critic scores for fake and real HR, compute gradient penalty,
              form critic loss and step the critic optimizer.
        - Every n_critic steps, update the generator:
            * Generate SR samples (not detached) for gradient flow into the generator.
            * Create multiple realizations per low-resolution input to compute an
              empirical CRPS (continuous ranked probability score) term across samples.
            * Generator loss is composed of the negative critic score (to fool critic)
              plus a CRPS regularizer scaled by self.alpha.
        - Log training metrics and optionally save visualization figures.

        Args:
            batch: A batch from the dataloader (expected to be unpackable by unpack_batch).
            batch_idx: Index of the batch within the current epoch.

        Returns:
            None (logging and optimizer steps are handled internally).
        """
        # Unpack inputs and move to float (device handling assumed elsewhere)
        lr, hr, hr_cov = self.unpack_batch(batch)

        # Get optimizers from Lightning; order expected as (generator_opt, critic_opt)
        g_opt, c_opt = self.optimizers()

        # --- Critic update ---
        # Tell Lightning we are using the critic optimizer (for automatic state handling)
        self.toggle_optimizer(c_opt)

        # Generate SR samples but detach so the generator is not updated while training critic
        sr = self.G(lr, hr_cov).detach()

        # Compute mean critic outputs for fake and real samples (scalars)
        mean_sr = self.C(sr).mean()
        mean_hr = self.C(hr).mean()

        # Compute gradient penalty for WGAN-GP (penalizes norm deviation of critic gradients)
        gp = compute_gradient_penalty(self.C, hr, sr)

        # Critic loss: E[C(fake)] - E[C(real)] + lambda * GP
        loss_c = mean_sr - mean_hr + self.gp_lambda * gp

        # Apply gradient descent step to minimize critic loss
        go_downhill(self, loss_c, c_opt)

        # --- Generator update (every n_critic steps) ---
        if (batch_idx + 1) % self.n_critic == 0:
            # Switch to generator optimizer context
            self.toggle_optimizer(g_opt)

            # Generate SR without detaching so gradients flow back to generator
            sr = self.G(lr, hr_cov)

            # Prepare multiple realizations per LR sample for CRPS computation.
            # n_realisation controls how many stochastic realizations we create.
            n_realisation = 6

            # Build repeated LR inputs: for each sample in the batch, create n_realisation copies
            ls1 = [i for i in range(lr.shape[0])]
            dat_lr = [lr[i, ...].unsqueeze(0).repeat(n_realisation, 1, 1, 1) for i in ls1]

            # Keep corresponding HR targets (single truth per LR sample)
            dat_hr = [hr[i, ...] for i in ls1]

            # For each repeated LR block, generate a set of SR realizations.
            # Note: the inner comprehension reuses the name `lr` from the outer scope;
            # this mirrors the original structure and yields a list of tensors shaped (n_realisation, ...)
            dat_sr = [self.G(lr, hr_cov[0: n_realisation, ...]) for lr in dat_lr]

            # Compute CRPS per sample by comparing the ensemble dat_sr to the single HR truth
            crps_ls = [crps_empirical(sr, hr) for sr, hr in zip(dat_sr, dat_hr)]
            crps = torch.cat(crps_ls)

            # Generator loss: encourage high critic score for generated samples (i.e., minimize -E[C(G)])
            # plus an alpha-weighted CRPS regularization term that promotes ensemble calibration
            loss_g = (
                -torch.mean(self.C(sr).detach())
                + self.alpha * torch.mean(crps)
            )

            # Step the generator optimizer to minimize generator loss
            go_downhill(self, loss_g, g_opt)

        # Log common training metrics (MAE, MSE, Wasserstein estimate, etc.)
        self.log_dict(
            self.losses("Train", hr, sr.detach(), mean_sr.detach(), mean_hr.detach()),
            sync_dist=True,
        )

        # Optionally create and upload diagnostic figures at configured intervals
        if (batch_idx + 1) % self.log_every_n_steps == 0:
            configure_figure(self.G, self.logger, "Train", lr, hr, hr_cov)

    def validation_step(self, batch, batch_idx):
        lr, hr, hr_cov = batch
        lr, hr, hr_cov = lr.float(), hr.float(), hr_cov.float()
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
