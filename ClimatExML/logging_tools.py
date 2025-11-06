# Plots matplotlib grids and saves to file
import torch
import time

# import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
from numpy.random import RandomState


def gen_grid_images(
    var,
    fig,
    G,
    lr: torch.Tensor,
    hr: torch.Tensor,
    hr_cov: torch.Tensor,
    use_hr_cov: bool,
    n_examples: int = 3,
    cmap="viridis",
) -> None:
    """
    Plots a grid of images and saves them to file
    Args:
        coarse (torch.Tensor): The coarse input.
        fake (torch.Tensor): The fake input.
        real (torch.Tensor): The real input.
    Returns:
        None
    """

    batch_size = lr.size(0)
    prng = RandomState(int(time.time()))

    random = prng.randint(0, batch_size, size=(n_examples,))

    if use_hr_cov:
        sr = G(lr[random, ...], hr_cov[random, ...]).detach()
    else:
        sr = G(lr[random, ...]).detach()

    lr_grid = torchvision.utils.make_grid(lr[random, var:var+1, ...], nrow=n_examples, padding=5)[0, ...]
    sr_grid = torchvision.utils.make_grid(sr[:, var:var+1, ...], nrow=n_examples)[0, ...]
    hr_grid = torchvision.utils.make_grid(hr[random, var:var+1, ...], nrow=n_examples)[0, ...]

    fig.suptitle("Training Samples")

    # Plot the coarse and fake samples
    subfigs = fig.subfigures(nrows=n_examples, ncols=1)

    ax = make_subfig(
        subfigs,
        0,
        f"Low Resolution Fields Min: {lr_grid.min()} Max: {lr_grid.max()}",
        lr_grid,
        cmap,
    )
    ax = make_subfig(
        subfigs,
        1,
        f"Super Resolved Fields Min: {sr_grid.min()} Max: {sr_grid.max()}",
        sr_grid,
        cmap,
    )
    ax = make_subfig(
        subfigs,
        2,
        f"Ground Truth Fields Min: {hr_grid.min()} Max: {hr_grid.max()}",
        hr_grid,
        cmap,
    )

    return fig


def make_subfig(subfigs, idx, title, grid, cmap):
    """Function to plot a subfigure of the tensor grid
    Args:
        subfigs (matplotlib.figure.SubFigure): The subfigure to plot on.
        idx (int): The index of the subfigure.
        title (str): The title of the subfigure.
        grid (torch.Tensor): The grid to plot.
        cmap (str): The colormap to use.

    Returns:
        matplotlib.axes.Axes: The axes of the subfigure.
    """
    subfigs[idx].suptitle(title)
    result = subfigs[idx].subplots(1, 1)
    result.imshow(grid.float().cpu().detach().numpy(), origin="lower", cmap=cmap)

    return result
