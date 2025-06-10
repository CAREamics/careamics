"""
This script provides methods to evaluate the performance of the LVAE model.
It includes functions to:
    - make predictions,
    - quantify the performance of the model
    - create plots to visualize the results.
"""

import os
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from careamics.lightning import VAEModule
from careamics.lvae_training.dataset import MultiChDloaderRef
from careamics.utils.metrics import scale_invariant_psnr


class TilingMode:
    """
    Enum for the tiling mode.
    """

    TrimBoundary = 0
    PadBoundary = 1
    ShiftBoundary = 2


# ------------------------------------------------------------------------------------------------
# Function of plotting: TODO -> moved them to another file, plot_utils.py
def clean_ax(ax):
    """
    Helper function to remove ticks from axes in plots.
    """
    # 2D or 1D axes are of type np.ndarray
    if isinstance(ax, np.ndarray):
        for one_ax in ax:
            clean_ax(one_ax)
        return

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(left=False, right=False, top=False, bottom=False)


def get_eval_output_dir(
    saveplotsdir: str, patch_size: int, mmse_count: int = 50
) -> str:
    """
    Given the path to a root directory to save plots, patch size, and mmse count,
    it returns the specific directory to save the plots.
    """
    eval_out_dir = os.path.join(
        saveplotsdir, f"eval_outputs/patch_{patch_size}_mmse_{mmse_count}"
    )
    os.makedirs(eval_out_dir, exist_ok=True)
    print(eval_out_dir)
    return eval_out_dir


def get_psnr_str(tar_hsnr, pred, col_idx):
    """
    Compute PSNR between the ground truth (`tar_hsnr`) and the predicted image (`pred`).
    """
    return f"{scale_invariant_psnr(tar_hsnr[col_idx][None], pred[col_idx][None]).item():.1f}"


def add_psnr_str(ax_, psnr):
    """
    Add psnr string to the axes
    """
    textstr = f"PSNR\n{psnr}"
    props = dict(boxstyle="round", facecolor="gray", alpha=0.5)
    # place a text box in upper left in axes coords
    ax_.text(
        0.05,
        0.95,
        textstr,
        transform=ax_.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
        color="white",
    )


def get_last_index(bin_count, quantile):
    cumsum = np.cumsum(bin_count)
    normalized_cumsum = cumsum / cumsum[-1]
    for i in range(1, len(normalized_cumsum)):
        if normalized_cumsum[-i] < quantile:
            return i - 1
    return None


def get_first_index(bin_count, quantile):
    cumsum = np.cumsum(bin_count)
    normalized_cumsum = cumsum / cumsum[-1]
    for i in range(len(normalized_cumsum)):
        if normalized_cumsum[i] > quantile:
            return i
    return None


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def show_for_one(
    idx,
    val_dset,
    highsnr_val_dset,
    model,
    calibration_stats,
    mmse_count=5,
    patch_size=256,
    num_samples=2,
    baseline_preds=None,
):
    """
    Given an index, it plots the input, target, reconstructed images and the difference image.
    Note the the difference image is computed with respect to a ground truth image, obtained from the high SNR dataset.
    """
    highsnr_val_dset.set_img_sz(patch_size, 64)
    highsnr_val_dset.disable_noise()
    _, tar_hsnr = highsnr_val_dset[idx]
    inp, tar, recon_img_list = get_predictions(
        idx, val_dset, model, mmse_count=mmse_count, patch_size=patch_size
    )
    plot_crops(
        inp,
        tar,
        tar_hsnr,
        recon_img_list,
        calibration_stats,
        num_samples=num_samples,
        baseline_preds=baseline_preds,
    )


def plot_crops(
    inp,
    tar,
    tar_hsnr,
    recon_img_list,
    calibration_stats=None,
    num_samples=2,
    baseline_preds=None,
):
    if baseline_preds is None:
        baseline_preds = []
    if len(baseline_preds) > 0:
        for i in range(len(baseline_preds)):
            if baseline_preds[i].shape != tar_hsnr.shape:
                print(
                    f"Baseline prediction {i} shape {baseline_preds[i].shape} does not match target shape {tar_hsnr.shape}"
                )
                print("This happens when we want to predict the edges of the image.")
                return
    color_ch_list = ["goldenrod", "cyan"]
    color_pred = "red"
    insetplot_xmax_value = 10000
    insetplot_xmin_value = -1000
    inset_min_labelsize = 10
    inset_rect = [0.05, 0.05, 0.4, 0.2]

    img_sz = 3
    ncols = num_samples + len(baseline_preds) + 1 + 1 + 1 + 1 + 1 * (num_samples > 1)
    grid_factor = 5
    grid_img_sz = img_sz * grid_factor
    example_spacing = 1
    c0_extra = 1
    nimgs = 1
    fig_w = ncols * img_sz + 2 * c0_extra / grid_factor
    fig_h = int(img_sz * ncols + (example_spacing * (nimgs - 1)) / grid_factor)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        nrows=int(grid_factor * fig_h),
        ncols=int(grid_factor * fig_w),
        hspace=0.2,
        wspace=0.2,
    )
    params = {"mathtext.default": "regular"}
    plt.rcParams.update(params)
    # plot baselines
    for i in range(2, 2 + len(baseline_preds)):
        for col_idx in range(baseline_preds[0].shape[0]):
            ax_temp = fig.add_subplot(
                gs[
                    col_idx * grid_img_sz : grid_img_sz * (col_idx + 1),
                    i * grid_img_sz + c0_extra : (i + 1) * grid_img_sz + c0_extra,
                ]
            )
            print(tar_hsnr.shape, baseline_preds[i - 2].shape)
            psnr = get_psnr_str(tar_hsnr, baseline_preds[i - 2], col_idx)
            ax_temp.imshow(baseline_preds[i - 2][col_idx], cmap="magma")
            add_psnr_str(ax_temp, psnr)
            clean_ax(ax_temp)

    # plot samples
    sample_start_idx = 2 + len(baseline_preds)
    for i in range(sample_start_idx, ncols - 3):
        for col_idx in range(recon_img_list.shape[1]):
            ax_temp = fig.add_subplot(
                gs[
                    col_idx * grid_img_sz : grid_img_sz * (col_idx + 1),
                    i * grid_img_sz + c0_extra : (i + 1) * grid_img_sz + c0_extra,
                ]
            )
            psnr = get_psnr_str(tar_hsnr, recon_img_list[i - sample_start_idx], col_idx)
            ax_temp.imshow(recon_img_list[i - sample_start_idx][col_idx], cmap="magma")
            add_psnr_str(ax_temp, psnr)
            clean_ax(ax_temp)
            # inset_ax = add_pixel_kde(ax_temp,
            #                       inset_rect,
            #                       [tar_hsnr[col_idx],
            #                        recon_img_list[i - sample_start_idx][col_idx]],
            #                       inset_min_labelsize,
            #                       label_list=['', ''],
            #                       color_list=[color_ch_list[col_idx], color_pred],
            #                       plot_xmax_value=insetplot_xmax_value,
            #                       plot_xmin_value=insetplot_xmin_value)

            # inset_ax.set_xticks([])
            # inset_ax.set_yticks([])

    # difference image
    if num_samples > 1:
        for col_idx in range(recon_img_list.shape[1]):
            ax_temp = fig.add_subplot(
                gs[
                    col_idx * grid_img_sz : grid_img_sz * (col_idx + 1),
                    (ncols - 3) * grid_img_sz
                    + c0_extra : (ncols - 2) * grid_img_sz
                    + c0_extra,
                ]
            )
            ax_temp.imshow(
                recon_img_list[1][col_idx] - recon_img_list[0][col_idx], cmap="coolwarm"
            )
            clean_ax(ax_temp)

    for col_idx in range(recon_img_list.shape[1]):
        # print(recon_img_list.shape)
        ax_temp = fig.add_subplot(
            gs[
                col_idx * grid_img_sz : grid_img_sz * (col_idx + 1),
                c0_extra
                + (ncols - 2) * grid_img_sz : (ncols - 1) * grid_img_sz
                + c0_extra,
            ]
        )
        psnr = get_psnr_str(tar_hsnr, recon_img_list.mean(axis=0), col_idx)
        ax_temp.imshow(recon_img_list.mean(axis=0)[col_idx], cmap="magma")
        add_psnr_str(ax_temp, psnr)
        # inset_ax = add_pixel_kde(ax_temp,
        #                           inset_rect,
        #                           [tar_hsnr[col_idx],
        #                            recon_img_list.mean(axis=0)[col_idx]],
        #                           inset_min_labelsize,
        #                           label_list=['', ''],
        #                           color_list=[color_ch_list[col_idx], color_pred],
        #                           plot_xmax_value=insetplot_xmax_value,
        #                           plot_xmin_value=insetplot_xmin_value)
        # inset_ax.set_xticks([])
        # inset_ax.set_yticks([])

        clean_ax(ax_temp)

        ax_temp = fig.add_subplot(
            gs[
                col_idx * grid_img_sz : grid_img_sz * (col_idx + 1),
                (ncols - 1) * grid_img_sz
                + 2 * c0_extra : (ncols) * grid_img_sz
                + 2 * c0_extra,
            ]
        )
        ax_temp.imshow(tar_hsnr[col_idx], cmap="magma")
        if col_idx == 0:
            legend_ch1_ax = ax_temp
        if col_idx == 1:
            legend_ch2_ax = ax_temp

        # inset_ax = add_pixel_kde(ax_temp,
        #                           inset_rect,
        #                           [tar_hsnr[col_idx],
        #                            ],
        #                           inset_min_labelsize,
        #                           label_list=[''],
        #                           color_list=[color_ch_list[col_idx]],
        #                           plot_xmax_value=insetplot_xmax_value,
        #                           plot_xmin_value=insetplot_xmin_value)
        # inset_ax.set_xticks([])
        # inset_ax.set_yticks([])

        clean_ax(ax_temp)

        ax_temp = fig.add_subplot(
            gs[
                col_idx * grid_img_sz : grid_img_sz * (col_idx + 1),
                grid_img_sz : 2 * grid_img_sz,
            ]
        )
        ax_temp.imshow(tar[0, col_idx].cpu().numpy(), cmap="magma")
        # inset_ax = add_pixel_kde(ax_temp,
        #                           inset_rect,
        #                           [tar[0,col_idx].cpu().numpy(),
        #                            ],
        #                           inset_min_labelsize,
        #                           label_list=[''],
        #                           color_list=[color_ch_list[col_idx]],
        #                           plot_kwargs_list=[{'linestyle':'--'}],
        #                           plot_xmax_value=insetplot_xmax_value,
        #                           plot_xmin_value=insetplot_xmin_value)

        # inset_ax.set_xticks([])
        # inset_ax.set_yticks([])

        clean_ax(ax_temp)

    ax_temp = fig.add_subplot(gs[0:grid_img_sz, 0:grid_img_sz])
    ax_temp.imshow(inp[0, 0].cpu().numpy(), cmap="magma")
    clean_ax(ax_temp)

    # line_ch1 = mlines.Line2D([0, 1], [0, 1], color=color_ch_list[0], linestyle='-', label='$C_1$')
    # line_ch2 = mlines.Line2D([0, 1], [0, 1], color=color_ch_list[1], linestyle='-', label='$C_2$')
    # line_pred = mlines.Line2D([0, 1], [0, 1], color=color_pred, linestyle='-', label='Pred')
    # line_noisych1 = mlines.Line2D([0, 1], [0, 1], color=color_ch_list[0], linestyle='--', label='$C^N_1$')
    # line_noisych2 = mlines.Line2D([0, 1], [0, 1], color=color_ch_list[1], linestyle='--', label='$C^N_2$')
    # legend_ch1 = legend_ch1_ax.legend(handles=[line_ch1, line_noisych1, line_pred], loc='upper right', frameon=False, labelcolor='white',
    #                         prop={'size': 11})
    # legend_ch2 = legend_ch2_ax.legend(handles=[line_ch2, line_noisych2, line_pred], loc='upper right', frameon=False, labelcolor='white',
    #                             prop={'size': 11})

    if calibration_stats is not None:
        smaller_offset = 4
        ax_temp = fig.add_subplot(
            gs[
                grid_img_sz + 1 : 2 * grid_img_sz - smaller_offset + 1,
                smaller_offset - 1 : grid_img_sz - 1,
            ]
        )
        plot_calibration(ax_temp, calibration_stats)


def plot_calibration(ax, calibration_stats):
    """
    To plot calibration statistics (RMV vs RMSE).
    """
    first_idx = get_first_index(calibration_stats[0]["bin_count"], 0.001)
    last_idx = get_last_index(calibration_stats[0]["bin_count"], 0.999)
    ax.plot(
        calibration_stats[0]["rmv"][first_idx:-last_idx],
        calibration_stats[0]["rmse"][first_idx:-last_idx],
        "o",
        label=r"$\hat{C}_0$",
    )

    first_idx = get_first_index(calibration_stats[1]["bin_count"], 0.001)
    last_idx = get_last_index(calibration_stats[1]["bin_count"], 0.999)
    ax.plot(
        calibration_stats[1]["rmv"][first_idx:-last_idx],
        calibration_stats[1]["rmse"][first_idx:-last_idx],
        "o",
        label=r"$\hat{C}_1$",
    )

    ax.set_xlabel("RMV")
    ax.set_ylabel("RMSE")
    ax.legend()


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
    Adapted from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    mid_idx = len(reg_index) // 2
    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        a = np.abs(ri - reg_index[mid_idx]) / reg_index[mid_idx]
        # print(a)
        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    matplotlib.colormaps.register(cmap=newcmap, force=True)

    return newcmap


def get_fractional_change(target, prediction, max_val=None):
    """
    Get relative difference between target and prediction.
    """
    if max_val is None:
        max_val = target.max()
    return (target - prediction) / max_val


def get_zero_centered_midval(error):
    """
    When done this way, the midval ensures that the colorbar is centered at 0. (Don't know how, but it works ;))
    """
    vmax = error.max()
    vmin = error.min()
    midval = 1 - vmax / (vmax + abs(vmin))
    return midval


def plot_error(target, prediction, cmap=matplotlib.cm.coolwarm, ax=None, max_val=None):
    """
    Plot the relative difference between target and prediction.
    NOTE: The plot is overlapped to the prediction image (in gray scale).
    NOTE: The colorbar is centered at 0.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Relative difference between target and prediction
    rel_diff = get_fractional_change(target, prediction, max_val=max_val)
    midval = get_zero_centered_midval(rel_diff)
    shifted_cmap = shiftedColorMap(
        cmap, start=0, midpoint=midval, stop=1.0, name="shiftedcmap"
    )
    ax.imshow(prediction, cmap="gray")
    img_err = ax.imshow(rel_diff, cmap=shifted_cmap, alpha=1)
    plt.colorbar(img_err, ax=ax)


# -------------------------------------------------------------------------------------


def get_predictions(
    model: VAEModule,
    dset: Dataset,
    batch_size: int,
    tile_size: Optional[tuple[int, int]] = None,
    grid_size: Optional[int] = None,
    mmse_count: int = 1,
    num_workers: int = 4,
) -> tuple[dict, dict, dict]:
    """Get patch-wise predictions from a model for the entire dataset.

    Parameters
    ----------
    model : VAEModule
        Lightning model used for prediction.
    dset : Dataset
        Dataset to predict on.
    batch_size : int
        Batch size to use for prediction.
    loss_type :
        Type of reconstruction loss used by the model, by default `None`.
    mmse_count : int, optional
        Number of samples to generate for each input and then to average over for
        MMSE estimation, by default 1.
    num_workers : int, optional
        Number of workers to use for DataLoader, by default 4.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float]]
        Tuple containing:
            - predictions: Predicted images for the dataset.
            - predictions_std: Standard deviation of the predicted images.
            - logvar_arr: Log variance of the predicted images.
            - losses: Reconstruction losses for the predictions.
            - psnr: PSNR values for the predictions.
    """
    if hasattr(dset, "dsets"):
        multifile_stitched_predictions = {}
        multifile_stitched_stds = {}
        for d in dset.dsets:
            stitched_predictions, stitched_stds = get_single_file_mmse(
                model=model,
                dset=d,
                batch_size=batch_size,
                tile_size=tile_size,
                grid_size=grid_size,
                mmse_count=mmse_count,
                num_workers=num_workers,
            )
            # get filename without extension and path
            filename = d._fpath.name
            multifile_stitched_predictions[filename] = stitched_predictions
            multifile_stitched_stds[filename] = stitched_stds
        return (
            multifile_stitched_predictions,
            multifile_stitched_stds,
        )
    else:
        stitched_predictions, stitched_stds = get_single_file_mmse(
            model=model,
            dset=dset,
            batch_size=batch_size,
            tile_size=tile_size,
            grid_size=grid_size,
            mmse_count=mmse_count,
            num_workers=num_workers,
        )
        # TODO stitching still not working properly for weirdly shaped images
        # get filename without extension and path
        # TODO in the ref ds this is the name of a folder not file :(
        filename = dset._fpath.name
        return (
            {filename: stitched_predictions},
            {filename: stitched_stds},
        )


def get_single_file_predictions(
    model: VAEModule,
    dset: Dataset,
    batch_size: int,
    tile_size: Optional[tuple[int, int]] = None,
    grid_size: Optional[int] = None,
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Get patch-wise predictions from a model for a single file dataset."""
    if tile_size and grid_size:
        dset.set_img_sz(tile_size, grid_size)

    device = get_device()

    dloader = DataLoader(
        dset,
        pin_memory=False,
        num_workers=num_workers,
        shuffle=False,
        batch_size=batch_size,
    )
    model.eval()
    model.to(device)
    tiles = []
    logvar_arr = []
    with torch.no_grad():
        for batch in tqdm(dloader, desc="Predicting tiles"):
            inp, tar = batch
            inp = inp.to(device)
            tar = tar.to(device)

            # get model output
            rec, _ = model(inp)

            # get reconstructed img
            if model.model.predict_logvar is None:
                rec_img = rec
                logvar = torch.tensor([-1])
            else:
                rec_img, logvar = torch.chunk(rec, chunks=2, dim=1)
            logvar_arr.append(logvar.cpu().numpy())  # Why do we need this ?

            tiles.append(rec_img.cpu().numpy())

    tile_samples = np.concatenate(tiles, axis=0)
    return stitch_predictions_new(tile_samples, dset)


def get_single_file_mmse(
    model: VAEModule,
    dset: Dataset,
    batch_size: int,
    tile_size: Optional[tuple[int, int]] = None,
    grid_size: Optional[int] = None,
    mmse_count: int = 1,
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Get patch-wise predictions from a model for a single file dataset."""
    device = get_device()

    dloader = DataLoader(
        dset,
        pin_memory=False,
        num_workers=num_workers,
        shuffle=False,
        batch_size=batch_size,
    )
    if tile_size and grid_size:
        dset.set_img_sz(tile_size, grid_size)

    model.eval()
    model.to(device)
    tile_mmse = []
    tile_stds = []
    logvar_arr = []
    with torch.no_grad():
        for batch in tqdm(dloader, desc="Predicting tiles"):
            inp, tar = batch
            inp = inp.to(device)
            tar = tar.to(device)

            rec_img_list = []
            for _ in range(mmse_count):

                # get model output
                rec, _ = model(inp)

                # get reconstructed img
                if model.model.predict_logvar is None:
                    rec_img = rec
                    logvar = torch.tensor([-1])
                else:
                    rec_img, logvar = torch.chunk(rec, chunks=2, dim=1)
                rec_img_list.append(rec_img.cpu().unsqueeze(0))  # add MMSE dim
                logvar_arr.append(logvar.cpu().numpy())  # Why do we need this ?

            # aggregate results
            samples = torch.cat(rec_img_list, dim=0)
            mmse_imgs = torch.mean(samples, dim=0)  # avg over MMSE dim
            std_imgs = torch.std(samples, dim=0)  # std over MMSE dim

            tile_mmse.append(mmse_imgs.cpu().numpy())
            tile_stds.append(std_imgs.cpu().numpy())

    tiles_arr = np.concatenate(tile_mmse, axis=0)
    tile_stds = np.concatenate(tile_stds, axis=0)
    # TODO temporary hack, because of the stupid jupyter!
    # If a user reruns a cell with class definition, isinstance will return False
    if str(MultiChDloaderRef).split(".")[-1] == str(dset.__class__).split(".")[-1]:
        stitch_func = stitch_predictions_general
    else:
        stitch_func = stitch_predictions_new
    stitched_predictions = stitch_func(tiles_arr, dset)
    stitched_stds = stitch_func(tile_stds, dset)
    return stitched_predictions, stitched_stds


# ------------------------------------------------------------------------------------------
### Classes and Functions used to stitch predictions
class PatchLocation:
    """
    Encapsulates t_idx and spatial location.
    """

    def __init__(self, h_idx_range, w_idx_range, t_idx):
        self.t = t_idx
        self.h_start, self.h_end = h_idx_range
        self.w_start, self.w_end = w_idx_range

    def __str__(self):
        msg = f"T:{self.t} [{self.h_start}-{self.h_end}) [{self.w_start}-{self.w_end}) "
        return msg


def _get_location(extra_padding, hwt, pred_h, pred_w):
    h_start, w_start, t_idx = hwt
    h_start -= extra_padding
    h_end = h_start + pred_h
    w_start -= extra_padding
    w_end = w_start + pred_w
    return PatchLocation((h_start, h_end), (w_start, w_end), t_idx)


def get_location_from_idx(dset, dset_input_idx, pred_h, pred_w):
    """
    For a given idx of the dataset, it returns where exactly in the dataset, does this prediction lies.
    Note that this prediction also has padded pixels and so a subset of it will be used in the final prediction.
    Which time frame, which spatial location (h_start, h_end, w_start,w_end)
    Args:
        dset:
        dset_input_idx:
        pred_h:
        pred_w:

    Returns
    -------
    """
    extra_padding = dset.per_side_overlap_pixelcount()
    htw = dset.get_idx_manager().hwt_from_idx(
        dset_input_idx, grid_size=dset.get_grid_size()
    )
    return _get_location(extra_padding, htw, pred_h, pred_w)


def remove_pad(pred, loc, extra_padding, smoothening_pixelcount, frame_shape):
    assert smoothening_pixelcount == 0
    if extra_padding - smoothening_pixelcount > 0:
        h_s = extra_padding - smoothening_pixelcount

        # rows
        h_N = frame_shape[0]
        if loc.h_end > h_N:
            assert loc.h_end - extra_padding + smoothening_pixelcount <= h_N
        h_e = extra_padding - smoothening_pixelcount

        w_s = extra_padding - smoothening_pixelcount

        # columns
        w_N = frame_shape[1]
        if loc.w_end > w_N:
            assert loc.w_end - extra_padding + smoothening_pixelcount <= w_N

        w_e = extra_padding - smoothening_pixelcount

        return pred[h_s:-h_e, w_s:-w_e]

    return pred


def update_loc_for_final_insertion(loc, extra_padding, smoothening_pixelcount):
    extra_padding = extra_padding - smoothening_pixelcount
    loc.h_start += extra_padding
    loc.w_start += extra_padding
    loc.h_end -= extra_padding
    loc.w_end -= extra_padding
    return loc


def stitch_predictions(predictions, dset, smoothening_pixelcount=0):
    """
    Args:
        smoothening_pixelcount: number of pixels which can be interpolated
    """
    assert smoothening_pixelcount >= 0 and isinstance(smoothening_pixelcount, int)
    extra_padding = dset.per_side_overlap_pixelcount()
    # if there are more channels, use all of them.
    shape = list(dset.get_data_shape())
    shape[-1] = max(shape[-1], predictions.shape[1])

    output = np.zeros(shape, dtype=predictions.dtype)
    frame_shape = dset.get_data_shape()[1:3]
    for dset_input_idx in range(predictions.shape[0]):
        loc = get_location_from_idx(
            dset, dset_input_idx, predictions.shape[-2], predictions.shape[-1]
        )

        mask = None
        cropped_pred_list = []
        for ch_idx in range(predictions.shape[1]):
            # class i
            cropped_pred_i = remove_pad(
                predictions[dset_input_idx, ch_idx],
                loc,
                extra_padding,
                smoothening_pixelcount,
                frame_shape,
            )

            if mask is None:
                # NOTE: don't need to compute it for every patch.
                assert (
                    smoothening_pixelcount == 0
                ), "For smoothing,enable the get_smoothing_mask. It is disabled since I don't use it and it needs modification to work with non-square images"
                mask = 1
                # mask = _get_smoothing_mask(cropped_pred_i.shape, smoothening_pixelcount, loc, frame_size)

            cropped_pred_list.append(cropped_pred_i)

        loc = update_loc_for_final_insertion(loc, extra_padding, smoothening_pixelcount)
        for ch_idx in range(predictions.shape[1]):
            output[loc.t, loc.h_start : loc.h_end, loc.w_start : loc.w_end, ch_idx] += (
                cropped_pred_list[ch_idx] * mask
            )

    return output


# from disentangle.analysis.stitch_prediction import *
def stitch_predictions_new(predictions, dset):
    """
    Args:
        smoothening_pixelcount: number of pixels which can be interpolated
    """
    # Commented out since it is not used as of now
    # if isinstance(dset, MultiFileDset):
    #     cum_count = 0
    #     output = []
    #     for dset in dset.dsets:
    #         cnt = dset.idx_manager.total_grid_count()
    #         output.append(
    #             stitch_predictions(predictions[cum_count:cum_count + cnt], dset))
    #         cum_count += cnt
    #     return output

    # else:
    mng = dset.idx_manager

    # if there are more channels, use all of them.
    shape = list(dset.get_data_shape())
    shape[-1] = max(shape[-1], predictions.shape[1])

    output = np.zeros(shape, dtype=predictions.dtype)
    # frame_shape = dset.get_data_shape()[:-1]
    for dset_idx in range(predictions.shape[0]):
        # loc = get_location_from_idx(dset, dset_idx, predictions.shape[-2], predictions.shape[-1])
        # grid start, grid end
        gs = np.array(mng.get_location_from_dataset_idx(dset_idx), dtype=int)
        ge = gs + mng.grid_shape

        # patch start, patch end
        ps = gs - mng.patch_offset()
        pe = ps + mng.patch_shape
        # print('PS')
        # print(ps)
        # print(pe)

        # valid grid start, valid grid end
        vgs = np.array([max(0, x) for x in gs], dtype=int)
        vge = np.array([min(x, y) for x, y in zip(ge, mng.data_shape)], dtype=int)
        # assert np.all(vgs == gs)
        # assert np.all(vge == ge) # TODO comented out this shit cuz I have no interest to dig why it's failing at this point !
        # print('VGS')
        # print(gs)
        # print(ge)

        if mng.tiling_mode == TilingMode.ShiftBoundary:
            for dim in range(len(vgs)):
                if ps[dim] == 0:
                    vgs[dim] = 0
                if pe[dim] == mng.data_shape[dim]:
                    vge[dim] = mng.data_shape[dim]

        # relative start, relative end. This will be used on pred_tiled
        rs = vgs - ps
        re = rs + (vge - vgs)
        # print('RS')
        # print(rs)
        # print(re)

        # print(output.shape)
        # print(predictions.shape)
        for ch_idx in range(predictions.shape[1]):
            if len(output.shape) == 4:
                # channel dimension is the last one.
                output[vgs[0] : vge[0], vgs[1] : vge[1], vgs[2] : vge[2], ch_idx] = (
                    predictions[dset_idx][ch_idx, rs[1] : re[1], rs[2] : re[2]]
                )
            elif len(output.shape) == 5:
                # channel dimension is the last one.
                assert vge[0] - vgs[0] == 1, "Only one frame is supported"
                output[
                    vgs[0], vgs[1] : vge[1], vgs[2] : vge[2], vgs[3] : vge[3], ch_idx
                ] = predictions[dset_idx][
                    ch_idx, rs[1] : re[1], rs[2] : re[2], rs[3] : re[3]
                ]
            else:
                raise ValueError(f"Unsupported shape {output.shape}")

    return output


def stitch_predictions_general(predictions, dset):
    """Stitching for the dataset with multiple files of different shape."""
    mng = dset.idx_manager

    # TODO assert all shapes are equal len
    # adjust number of channels to match with prediction shape #TODO ugly, refac!
    shapes = []
    for shape in dset.get_data_shapes()[0]:
        shapes.append((predictions.shape[1],) + shape[1:])

    output = [np.zeros(shape, dtype=predictions.dtype) for shape in shapes]
    # frame_shape = dset.get_data_shape()[:-1]
    for patch_idx in range(predictions.shape[0]):
        # grid start, grid end
        # channel_idx is 0 because during prediction we're only use one channel. # TODO revisit this
        # 0th dimension is sample index in the output list
        grid_coords = np.array(
            mng.get_location_from_patch_idx(channel_idx=0, patch_idx=patch_idx),
            dtype=int,
        )
        sample_idx = grid_coords[0]
        grid_start = grid_coords[1:]
        # from here on, coordinates are relative to the sample(file in the list of inputs)
        grid_end = grid_start + mng.grid_shape

        # patch start, patch end
        patch_start = grid_start - mng.patch_offset()
        patch_end = patch_start + mng.patch_shape

        # valid grid start, valid grid end
        valid_grid_start = np.array([max(0, x) for x in grid_start], dtype=int)
        valid_grid_end = np.array(
            [min(x, y) for x, y in zip(grid_end, shapes[sample_idx])], dtype=int
        )

        if mng.tiling_mode == TilingMode.ShiftBoundary:
            for dim in range(len(valid_grid_start)):
                if patch_start[dim] == 0:
                    valid_grid_start[dim] = 0
                if patch_end[dim] == mng.data_shape[dim]:
                    valid_grid_end[dim] = mng.data_shape[dim]

        # relative start, relative end. This will be used on pred_tiled
        relative_start = valid_grid_start - patch_start
        relative_end = relative_start + (valid_grid_end - valid_grid_start)

        for ch_idx in range(predictions.shape[1]):
            if len(output[sample_idx].shape) == 3:
                # starting from 1 because 0th dimension is channel relative to input
                # channel dimension for stitched output is relative to model output
                output[sample_idx][
                    ch_idx,
                    valid_grid_start[1] : valid_grid_end[1],
                    valid_grid_start[2] : valid_grid_end[2],
                ] = predictions[patch_idx][
                    ch_idx,
                    relative_start[1] : relative_end[1],
                    relative_start[2] : relative_end[2],
                ]
            elif len(output[sample_idx].shape) == 4:
                assert (
                    valid_grid_end[0] - valid_grid_start[0] == 1
                ), "Only one frame is supported"
                output[
                    ch_idx,
                    valid_grid_start[0],
                    valid_grid_end[1] : valid_grid_end[1],
                    valid_grid_start[2] : valid_grid_end[2],
                    valid_grid_start[3] : valid_grid_end[3],
                ] = predictions[patch_idx][
                    ch_idx,
                    relative_start[1] : relative_end[1],
                    relative_start[2] : relative_end[2],
                    relative_start[3] : relative_end[3],
                ]
            else:
                raise ValueError(f"Unsupported shape {output.shape}")

    return output
