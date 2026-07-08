import os
from typing import Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from scripts.plotting_utils import (
    _add_channel_image,
    _add_metrics_info,
    _derive_vmaxs,
    _derive_vmins,
    _get_multichannel_cmap,
    _crop_image,
    ColorRepo,
    ColormapRepo,
    match_range
)

def plot_multichannel_image(
    img: NDArray,
    suptitle: Optional[str] = None,
    z_idx: Optional[int] = None,
    x_ROI: Optional[tuple[int, int]] = None,
    y_ROI: Optional[tuple[int, int]] = None,
    clip: Optional[float] = None,
    contrast_lims: Optional[Sequence[Optional[tuple[float, float]]]] = None,
    n_rows: int = 1,
    cmaps: Optional[Union[ColormapRepo, Sequence[ColormapRepo]]] = None,
    multicolor_cmaps: bool = False,
    diverging_cmaps: bool = False,
    diverging_colors: Optional[Sequence[ColorRepo]] = None,
    diverging_midvalues: Optional[Sequence[float]] = None,
    diverging_symmetric_norm: bool = True,
    save_fpath: Optional[str] = None,
    dpi: int = 300,
    transparent: bool = False
) -> None:
    """Plot a multichannel image.

    Parameters
    ----------
    img : NDArray
        Multichannel image. Shape is (C, [Z], Y, X), C is the number of channels.
    suptitle : Optional[str]
        Title for the multichannel image. Default is None.
    z_idx : Optional[int]
        Z-slice index to plot in 3D images. Default is None.
    x_ROI : Optional[tuple[int, int]]
        X-axis ROI. Default is None.
    y_ROI : Optional[tuple[int, int]]
        Y-axis ROI. Default is None.
    clip : Optional[float]
        Clip lower bound for the image, for example used to remove background.
        Default is None.
    contrast_lims : Optional[Sequence[Optional[tuple[float, float]]]]
        Contrast limits for each channel. If you want to leave the default limits for
        some channel, set it to None (e.g., [(0, 100), None]). Default is None.
    n_rows: int
        Number of rows in the plot. Default is 1.
    cmaps : Optional[Union[ColormapRepo, Sequence[ColormapRepo]]]
        Explicit colormap(s). A single entry is broadcast to all channels. If provided
        it overrides ``diverging_cmaps`` and ``multicolor_cmaps``. Default is None.
    multicolor_cmaps : bool
        Use a distinct colormap per channel (cycles through ``ColormapRepo``).
        Set to ``False`` for grayscale when ``diverging_cmaps`` is also ``False``.
        Default is ``False``.
    diverging_cmaps : bool
        Use diverging colormaps (black at center, red for negatives). Default is ``False``.
    diverging_colors : Optional[Sequence[ColorRepo]]
        Explicit positive colors for diverging colormaps. Overrides ``multicolor_cmaps``
        when set. Default is None.
    diverging_midvalues : Optional[Sequence[float]]
        Midvalues for diverging colormaps. Overrides default of 0. Default is None.
    diverging_symmetric_norm : bool
        When ``True`` (default), the negative half of each diverging colormap mirrors
        the positive half (symmetric normalization). When ``False``, ``vmin`` per
        channel is derived from ``contrast_lims`` or the channel image minimum,
        enabling independent control over each half.
    save_fpath : Optional[str]
        Path to save the plot. Default is None.
    dpi : int
        Dots per inch for saved plot. Default is 300.
    transparent : bool
        Whether to save the plot with a transparent background. Default is False.
    """
    C = img.shape[0]
    if img.ndim - 1 == 3:
        assert z_idx is not None, "`z_idx` must be provided for 3D images."
    if contrast_lims is not None:
        assert len(contrast_lims) == C, (
            "Length of `contrast_lims` must match number of channels."
        )

    # crop image
    img = _crop_image(img, z_idx, x_ROI, y_ROI)

    # clip image
    if clip is not None:
        img = img.clip(min=clip)

    n_cols = int(np.ceil(C / n_rows))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), constrained_layout=True
    )
    axes = axes.flatten() if C > 1 else [axes]
    fig.patch.set_facecolor('black')
    if suptitle:
        fig.suptitle(suptitle, fontsize=20, color="white")

    # get colormaps
    vmins = _derive_vmins(img, contrast_lims) if not diverging_symmetric_norm else None
    cmap_list = _get_multichannel_cmap(
        n_channels=C,
        cmaps=cmaps,
        multicolor_cmaps=multicolor_cmaps,
        diverging_cmaps=diverging_cmaps,
        diverging_colors=diverging_colors,
        vmaxs=_derive_vmaxs(img, contrast_lims),
        vmins=vmins,
        midvalues=diverging_midvalues,
    )
    cmap_list, norms = zip(*cmap_list)

    # plot each channel
    for i in range(C):
        _add_channel_image(
            img[i], axes[i], fig,
            cmap=cmap_list[i],
            norm=norms[i] if norms[i] is not None else None,
            title=None,
            vlims=contrast_lims[i] if contrast_lims else None,
            add_colorbar=False,
            add_histogram=True
        )

    # save plot
    if save_fpath:
        os.makedirs(os.path.dirname(save_fpath), exist_ok=True)
        print(f"Saving plot to {save_fpath}")
        ext = save_fpath.split('.')[-1]
        plt.savefig(
            save_fpath,
            format=ext,
            dpi=dpi,
            transparent=transparent,
            bbox_inches="tight",
            pad_inches=0.02
        )


def plot_multichannel_image_comparison(
    img1: NDArray,
    img2: NDArray,
    suptitle: Optional[str] = None,
    title1: Optional[str] = None,
    title2: Optional[str] = None,
    metrics_avg: Optional[dict[str, dict[str, float]]] = None,
    metrics_per_img: Optional[dict[str, dict[str, dict[str, float]]]] = None,
    z_idx: Optional[int] = None,
    x_ROI: Optional[tuple[int, int]] = None,
    y_ROI: Optional[tuple[int, int]] = None,
    clip1: Optional[Union[float, Sequence[float]]] = None,
    clip2: Optional[Union[float, Sequence[float]]] = None,
    contrast_lims1: Optional[Sequence[Optional[tuple[float, float]]]] = None,
    contrast_lims2: Optional[Sequence[Optional[tuple[float, float]]]] = None,
    cmaps: Optional[Union[ColormapRepo, Sequence[ColormapRepo]]] = None,
    multicolor_cmaps: bool = False,
    diverging_cmaps: bool = False,
    diverging_colors: Optional[Sequence[ColorRepo]] = None,
    diverging_midvalues: Optional[Sequence[float]] = None,
    diverging_symmetric_norm: bool = True,
    img_fname: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot a comparison between multichannel images.

    Parameters
    ----------
    img1 : NDArray
        First multichannel image. Shape is (C, [Z], Y, X), C is the number of channels.
    img2 : NDArray
        Second multichannel image. Shape is (C, [Z], Y, X), C is the number of channels.
    suptitle : Optional[str]
        Overall title for the comparison plot. Default is None.
    title1 : Optional[str]
        Title for the first multichannel image. Default is None.
    title2 : Optional[str]
        Title for the second multichannel image. Default is None.
    metrics_avg : Optional[dict[str, dict[str, float]]]
        Metrics dictionary of `img1` vs. `img2`. Default is None.
    metrics_per_img : Optional[dict[str, dict[str, dict[str, float]]]]
        Metrics dictionary of `img1` vs. `img2` for each image. Default is None.
    z_idx : Optional[int]
        Z-slice index to plot in 3D images. Default is None.
    x_ROI : Optional[tuple[int, int]]
        X-axis ROI. Default is None.
    y_ROI : Optional[tuple[int, int]]
        Y-axis ROI. Default is None.
    clip1 : Optional[Union[float, Sequence[float]]]
        Clip lower bound for the first image. Default is None.
    clip2 : Optional[Union[float, Sequence[float]]]
        Clip lower bound for the second image. Default is None.
    contrast_lims1 : Optional[Sequence[Optional[tuple[float, float]]]]
        Contrast limits for each channel in the first image. Default is None.
    contrast_lims2 : Optional[Sequence[Optional[tuple[float, float]]]]
        Contrast limits for each channel in the second image. Default is None.
    cmaps : Optional[Union[ColormapRepo, Sequence[ColormapRepo]]]
        Explicit colormap(s). A single entry is broadcast to all channels. If provided
        it overrides ``diverging_cmaps`` and ``multicolor_cmaps``. Default is None.
    multicolor_cmaps : bool
        Use a distinct colormap per channel (cycles through ``ColormapRepo``).
        Set to ``False`` for grayscale when ``diverging_cmaps`` is also ``False``.
        Default is ``False``.
    diverging_cmaps : bool
        Use diverging colormaps. Default is ``False``.
    diverging_colors : Optional[Sequence[ColorRepo]]
        Explicit positive colors for diverging colormaps. Default is None.
    diverging_midvalues : Optional[Sequence[float]]
        Midvalues for diverging colormaps. Overrides default of 0. Default is None.
    diverging_symmetric_norm : bool
        When ``True`` (default), symmetric normalization is used. When ``False``,
        ``vmin`` per channel is derived independently from each image's
        ``contrast_lims`` or image minimum.
    img_fname : Optional[str]
        Image filename. Default is None.
    save_path : Optional[str]
        Path to file where to save the plot. Default is None.
    """
    C = img1.shape[0]
    if img1.ndim - 1 == 3:
        assert z_idx is not None, "`z_idx` must be provided for 3D images."
    if contrast_lims1 is not None:
        assert len(contrast_lims1) == C, (
            "Length of `contrast_lims1` must match number of channels."
        )
    if contrast_lims2 is not None:
        assert len(contrast_lims2) == C, (
            "Length of `contrast_lims2` must match number of channels."
        )
    if metrics_per_img is not None:
        assert img_fname is not None, (
            "`img_fname` must be provided to show `metrics_per_img`."
        )
    # crop images
    img1 = _crop_image(img1, z_idx, x_ROI, y_ROI)
    img2 = _crop_image(img2, z_idx, x_ROI, y_ROI)

    # clip images
    if clip1 is not None:
        if isinstance(clip1, (float, int)):
            img1 = img1.clip(min=clip1)
        elif len(clip1) == C:
            for c in range(C):
                img1[c] = img1[c].clip(min=clip1[c])
    if clip2 is not None:
        if isinstance(clip2, (float, int)):
            img2 = img2.clip(min=clip2)
        elif len(clip2) == C:
            for c in range(C):
                img2[c] = img2[c].clip(min=clip2[c])

    # compute pixel-wise differences
    ri_unmixed_img, ri_gt_img = match_range(img1, img2)
    diff_img = np.abs(ri_unmixed_img - ri_gt_img)

    # get colormaps
    _cmap_kwargs = dict(
        cmaps=cmaps,
        multicolor_cmaps=multicolor_cmaps,
        diverging_cmaps=diverging_cmaps,
        diverging_colors=diverging_colors,
        midvalues=diverging_midvalues,
    )
    cmaps1 = _get_multichannel_cmap(
        C, **_cmap_kwargs,
        vmaxs=_derive_vmaxs(img1, contrast_lims1),
        vmins=_derive_vmins(img1, contrast_lims1) if not diverging_symmetric_norm else None,
    )
    cmaps2 = _get_multichannel_cmap(
        C, **_cmap_kwargs,
        vmaxs=_derive_vmaxs(img2, contrast_lims2),
        vmins=_derive_vmins(img2, contrast_lims2) if not diverging_symmetric_norm else None,
    )
    cmaps1, norm1 = zip(*cmaps1)
    cmaps2, norm2 = zip(*cmaps2)

    nrows = 3
    fig, axes = plt.subplots(
        nrows, C, figsize=(6 * C + 1, 6 * nrows + 1), constrained_layout=True
    )
    fig.patch.set_facecolor('black')
    fig.suptitle(suptitle, fontsize=32, color="white")
    
    for i in range(C):
        # plot image 1
        _add_channel_image(
            img1[i], axes[0, i], fig, 
            cmap=cmaps1[i],
            norm=norm1[i] if norm1[i] is not None else None,
            title=title1 if i == 0 else None,
            vlims=contrast_lims1[i] if contrast_lims1 else None,
            add_colorbar=False
        )
        
        # plot image 2
        _add_channel_image(
            img2[i], axes[1, i], fig,
            cmap=cmaps2[i],
            norm=norm2[i] if norm2[i] is not None else None,
            title=title2 if i == 0 else None,
            vlims=contrast_lims2[i] if contrast_lims2 else None,
            add_colorbar=False
        )
        
        # plot pixel-wise differences
        _add_channel_image(
            diff_img[i], axes[2, i], fig,
            cmap="magma",
            title="Pixelwise MAE" if i == 0 else None,
            add_colorbar=True
        )
        # add metrics as text
        if metrics_avg is not None and metrics_per_img is not None:
            _add_metrics_info(axes[2, i], metrics_avg, metrics_per_img, i, img_fname)
    
    # save plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving plot to {save_path}")
        ext = save_path.split('.')[-1]
        plt.savefig(save_path, format=ext)
