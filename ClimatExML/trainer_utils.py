from typing import Optional
import torch
from torch import Tensor, optim
import matplotlib.pyplot as plt
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import CometLogger

from ClimatExML.logging_tools import gen_grid_images


def go_downhill(lightning_module: LightningModule, loss: Tensor, opt: optim.Optimizer) -> None:
    """
    Performs a manual optimization step for manual_optimization=True trainers.

    Args:
        lightning_module (LightningModule): The model using manual optimization.
        loss (Tensor): The computed loss.
        opt (Optimizer): The optimizer to step.
    """
    lightning_module.manual_backward(loss)
    opt.step()
    opt.zero_grad()
    lightning_module.untoggle_optimizer(opt)


def configure_figure(
    generator: torch.nn.Module,
    logger: CometLogger,
    set_type: str,
    lr: Tensor,
    hr: Tensor,
    hr_cov: Optional[Tensor],
    use_hr_cov: bool = True,
    n_examples: int = 3,
    cmap: str = "viridis",
) -> None:
    """
    Logs side-by-side plots of SR, HR, and inputs for visualization.

    Args:
        generator (nn.Module): The generator model (e.g., self.G).
        logger (CometLogger): The experiment logger.
        set_type (str): One of "Train", "Validation", etc.
        lr (Tensor): Low-resolution input tensor.
        hr (Tensor): High-resolution target tensor.
        hr_cov (Optional[Tensor]): High-res invariant input (or dummy).
        use_hr_cov (bool): Whether the generator uses HR covariates.
        n_examples (int): Number of examples to plot.
        cmap (str): Colormap for the images.
    """
    for var in range(hr.shape[1]):
        fig = plt.figure(figsize=(30, 10))
        fig = gen_grid_images(
            var,
            fig,
            generator,
            lr,
            hr,
            hr_cov,
            use_hr_cov,
            n_examples,
            cmap=cmap,
        )
        logger.experiment.log_figure(
            figure_name=f"{set_type}_images_{var}", figure=fig, overwrite=True
        )
        plt.close(fig)


def compute_gradient_penalty(critic: torch.nn.Module, real_samples: Tensor, fake_samples: Tensor) -> Tensor:
    """
    Computes the gradient penalty for WGAN-GP.

    Args:
        critic (nn.Module): The critic model.
        real_samples (Tensor): Real high-resolution samples.
        fake_samples (Tensor): Generated samples from the generator.

    Returns:
        Tensor: Scalar gradient penalty loss.
    """
    batch_size = real_samples.size(0)
    device = real_samples.device

    alpha = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True).expand_as(real_samples)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)

    critic_output = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=critic_output,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_output, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
    return ((gradients_norm - 1) ** 2).mean()
