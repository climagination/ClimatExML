import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


def wass_loss(real, fake, device):
    return real - fake


def content_loss(hr: torch.Tensor, fake: torch.Tensor) -> float:
    """Calculates the L1 loss (pixel wise error) between both
    samples. Note that this is done on the high resolution
    (or super resolved fields)
    Args:
        hr (Tensor): Tensor containing batch of ground truth data
        fake (Tensor): Tensory containing batch of fake data
        device: device to be run on
    Returns:
        content_loss (float): Single value corresponding to L1.
    """
    criterion_pixelwise = nn.L1Loss()
    # content_loss = criterion_pixelwise(hr/hr.std(), fake/fake.std())
    content_loss = criterion_pixelwise(hr, fake)

    return content_loss


def content_MSELoss(
    hr: torch.Tensor, fake: torch.Tensor, device: torch.device
) -> float:
    """Calculates the L1 loss (pixel wise error) between both
    samples. Note that this is done on the high resolution (or super resolved fields)
    Args:
        hr (Tensor): Tensor containing batch of ground truth data
        fake (Tensor): Tensory containing batch of fake data
        device: device to be run on
    Returns:
        content_loss (float): Single value corresponding to L1.
    """
    criterion_pixelwise = nn.MSELoss().to(device)
    content_loss = criterion_pixelwise(hr, fake)
    return content_loss


# adding crps to losses.py from wgan_gp_stoch.py old version 
def crps_empirical(pred, truth): ##adapted from https://docs.pyro.ai/en/stable/_modules/pyro/ops/stats.html#crps_empirical
    
    if pred.shape[1:] != (1,) * (pred.dim() - truth.dim() - 1) + truth.shape:
        raise ValueError(
            "Expected pred to have one extra sample dim on left. "
            "Actual shapes: {} versus {}".format(pred.shape, truth.shape)
        )
    opts = dict(device=pred.device, dtype=pred.dtype)
    num_samples = pred.size(0)
    if num_samples == 1:
        return (pred[0] - truth).abs()

    pred = pred.sort(dim=0).values
    diff = pred[1:] - pred[:-1]
    weight = torch.arange(1, num_samples, **opts) * torch.arange(
        num_samples - 1, 0, -1, **opts
    )
    weight = weight.reshape(weight.shape + (1,) * (diff.dim() - 1))

    return (pred - truth).abs().mean(0) - (diff * weight).sum(0) / num_samples**2

