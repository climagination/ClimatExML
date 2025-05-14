import torch
import torch.nn.functional as F
import lightning as pl

from torchmetrics.functional import mean_squared_error
from ClimatExML.models import HRStreamGenerator
from ClimatExML.trainer_utils import configure_figure
from ClimatExML.base_trainer import BaseSuperResolutionTrainer


class CNNTrainer(BaseSuperResolutionTrainer):
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
        self.lr_shape = invariant.lr_shape
        self.hr_shape = invariant.hr_shape
        self.hr_invariant_shape = invariant.hr_invariant_shape

        n_covariates, lr_dim, _ = self.lr_shape
        n_predictands, hr_dim, _ = self.hr_shape
        n_hr_covariates = self.hr_invariant_shape[0]

        self.G = HRStreamGenerator(
            noise=False,
            filters=64,
            fine_dims=hr_dim,
            channels=n_covariates,
            channels_hr_cov=n_hr_covariates,
            n_predictands=n_predictands,
        )

    def unpack_batch(self, batch):
        lr, hr, hr_cov = batch
        return lr.float(), hr.float(), hr_cov.float()

    def training_step(self, batch, batch_idx):
        lr, hr, hr_cov = self.unpack_batch(batch)
        sr = self.G(lr, hr_cov)
        loss = mean_squared_error(sr, hr)
        self.log("Train MSE", loss, sync_dist=True)

        if (batch_idx + 1) % self.log_every_n_steps == 0:
            configure_figure(self.G, self.logger, "Train", lr, hr, hr_cov)

        return loss

    def validation_step(self, batch, batch_idx):
        lr, hr, hr_cov = batch
        sr = self.G(lr, hr_cov)
        loss = mean_squared_error(sr, hr)
        self.log("Validation MSE", loss, sync_dist=True)

        if (batch_idx + 1) % self.validation_log_every_n_steps == 0:
            configure_figure(self.G, self.logger, "Validation", lr, hr, hr_cov)

    def configure_optimizers(self):
        return torch.optim.Adam(self.G.parameters(), lr=self.learning_rate)

    def on_train_epoch_end(self):
        g_path = f"{self.save_dir}/generator_{self.experiment_name}.pt"
        g_scripted = torch.jit.script(self.G)
        g_scripted.save(g_path)
        self.logger.experiment.log_model("Generator", g_path, overwrite=True)