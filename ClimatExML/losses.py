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


def _power_spectrum(x):
    x_fft = torch.fft.fftn(x, norm='ortho')
    energy = torch.abs(x_fft).pow(2)
    return energy


def _wave_number_radial(dim, d=1.0):
    freq = torch.fft.fftfreq(dim, d=d)
    grid_h, grid_w = torch.meshgrid(freq, freq, indexing="ij")  # must be 'ij'
    wave_radial = torch.sqrt(grid_h.pow(2) + grid_w.pow(2))
    return wave_radial


def rapsd(x, d=1.0):
    """
    Compute the radially averaged power spectral density (RAPSD) of a 2D square field.

    The RAPSD describes how the variance (energy) of a spatial field is distributed
    across spatial frequencies (wavenumbers), averaged over all directions. It is
    particularly useful in analyzing the spatial structure or scaling behavior of
    geophysical, meteorological, or image data.

    Parameters
    ----------
    x : torch.Tensor
        A 2D square tensor representing the spatial field (e.g., an image or gridded data).
        Must have shape (N, N).
    d : float, optional
        The grid spacing or physical distance between adjacent grid points.
        Defaults to 1.0.

    Returns
    -------
    bin_avgs : torch.Tensor
        The average power spectral density within each radial wavenumber bin.
    bins_mids : torch.Tensor
        The midpoints of the radial wavenumber bins.
    bin_counts : torch.Tensor
        The number of Fourier components contributing to each radial bin.

    Notes
    -----
    - The function computes the 2D Fourier transform of the input, converts it
      to an energy spectrum (squared magnitude), and bins it radially based on
      isotropic wavenumber magnitude.
    - The result provides a one-dimensional representation of the power spectrum,
      where `bins_mids` correspond to the isotropic wavenumber and `bin_avgs`
      gives the corresponding average spectral power.
    """

    if not (x.dim() == 2 and x.shape[0] == x.shape[1]):
        raise ValueError("Input x must be a square 2D tensor")

    if x.dtype == torch.bfloat16 or x.dtype == torch.float16:
        x = x.to(torch.float32)

    dim = x.shape[0]
    wavenumber = _wave_number_radial(dim, d=d)

    delta = wavenumber[0][1]
    freq_max = 1 / (2 * d)

    bins_edges = torch.arange(delta / 2, freq_max + delta / 2, delta)
    bins_edges = torch.cat((torch.tensor([0.0]), bins_edges))
    bins_mids = 0.5 * (bins_edges[1:] + bins_edges[:-1])
    bin_counts, _ = torch.histogram(wavenumber, bins=bins_edges)

    energy = _power_spectrum(x)
    bin_sums, _ = torch.histogram(wavenumber.cpu(), bins=bins_edges.cpu(), weight=energy.cpu())
    bin_avgs = bin_sums.to(energy.device) / bin_counts.to(energy.device)

    return bin_avgs, bins_mids, bin_counts


def lsd(sr, hr, d=1.0):
    """Compute the log-spectral distance between two power spectra."""

    if sr.dim() != 2 or hr.dim() != 2:
        raise ValueError(f"Input tensors must be square 2D tensors. Got shapes SR: {sr.shape} and HR: {hr.shape}.")
    ps, _, _ = rapsd(sr, d=d)
    ps_ref, _, _ = rapsd(hr, d=d)
    lsd = 10*torch.sqrt(torch.mean((torch.log10(ps) - torch.log10(ps_ref)).pow(2)))
    return lsd
