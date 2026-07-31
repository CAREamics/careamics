import os
from typing import Literal, Optional, Sequence, Union

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


def plot_multichannel_image_multicomparison(
    imgs: Sequence[NDArray],
    titles: Optional[Sequence[Optional[str]]] = None,
    contrast_lims: Optional[Sequence[Optional[Sequence[Optional[tuple[float, float]]]]]] = None,
    clip_vals: Optional[Sequence[Optional[Union[float, Sequence[float]]]]] = None,
    suptitle: Optional[str] = None,
    annotations: Optional[Sequence[Sequence[Optional[str]]]] = None,
    annot_pos: Literal["topleft", "topright"] = "topright",
    z_idx: Optional[int] = None,
    x_ROI: Optional[tuple[int, int]] = None,
    y_ROI: Optional[tuple[int, int]] = None,
    cmaps: Optional[Union[ColormapRepo, Sequence[ColormapRepo]]] = None,
    multicolor_cmaps: bool = False,
    diverging_cmaps: bool = False,
    diverging_colors: Optional[Sequence[ColorRepo]] = None,
    diverging_midvalues: Optional[Sequence[Optional[Sequence[Optional[float]]]]] = None,
    diverging_symmetric_norm: bool = True,
    show_cbars: bool = False,
    show_hist: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 600,
    transparent: bool = False,
) -> None:
    """Plot multiple multichannel images for comparison.

    This function allows plotting an arbitrary number of multichannel images in rows,
    with each channel displayed in a separate column. Supports custom colormaps,
    intensity histograms, and per-image contrast/clipping controls.

    Parameters
    ----------
    imgs : Sequence[NDArray]
        Sequence of multichannel images. Each image shape is (C, [Z], Y, X),
        where C is the number of channels. All images must have the same shape.
    titles : Optional[Sequence[Optional[str]]]
        Titles for each image (one per row). Default is None.
    contrast_lims : Optional[Sequence[Optional[Sequence[Optional[tuple[float, float]]]]]]
        Contrast limits for each image and each channel. Structure is a sequence where
        each entry corresponds to an image, containing a sequence of contrast limits
        for each channel. Set to None for an image or channel to use default limits.
        Example: [[(0, 100), None], None, [(0, 50), (0, 200)]] for 3 images with 2 channels.
        Default is None.
    clip_vals : Optional[Sequence[Optional[Union[float, Sequence[float]]]]]
        Clip lower bounds for each image. Can be a single float (applied to all channels)
        or a sequence of floats (one per channel). Default is None.
    suptitle : Optional[str]
        Overall title for the entire figure. Default is None.
    annotations : Optional[Sequence[Sequence[Optional[str]]]]
        Per-image, per-channel text labels. Outer sequence has one entry per row
        (image); inner sequence has one entry per column (channel). Set an inner
        entry to ``None`` to skip labelling a particular subplot.
        Example: ``[["GT", "GT"], ["Pred", "Pred"]]`` for 2 images with 2 channels.
        Default is None.
    annot_pos : Literal["topleft", "topright"]
        Corner position for all text overlays. Default is ``"topright"``.
    z_idx : Optional[int]
        Z-slice index to plot in 3D images. Required for 3D images. Default is None.
    x_ROI : Optional[tuple[int, int]]
        X-axis ROI as (start, end). Default is None.
    y_ROI : Optional[tuple[int, int]]
        Y-axis ROI as (start, end). Default is None.
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
    diverging_midvalues : Optional[Sequence[Optional[Sequence[Optional[float]]]]]
        Per-image, per-channel midvalues for diverging colormaps. Outer sequence has
        one entry per image; inner sequence has one entry per channel. Set an outer
        entry to ``None`` to use the default midvalue (0) for all channels of that image,
        or set an inner entry to ``None`` to use the default for a specific channel.
        Example: ``[[0.0, None], None, [1.0, 2.0]]`` for 3 images with 2 channels.
        Default is None.
    diverging_symmetric_norm : bool
        When ``True`` (default), symmetric normalization is used for each image.
        When ``False``, a per-image per-channel ``vmin`` is derived independently
        from ``contrast_lims`` or image minima, enabling asymmetric normalization.
    show_cbars : bool
        Whether to show colorbars. Default is True.
    show_hist : bool
        Whether to show intensity histograms. Default is True.
    save_path : Optional[str]
        Path to file where to save the plot. Default is None.
    dpi : int
        Dots per inch for saved figure. Default is 600.
    transparent : bool
        Whether to save the figure with a transparent background. Default is False.
    """
    # Validate inputs
    assert len(imgs) > 0, "At least one image must be provided."

    # Check all images have the same shape
    first_shape = imgs[0].shape
    assert all(img.shape == first_shape for img in imgs), (
        "All images must have the same shape."
    )

    C = first_shape[0]
    is_3d = len(first_shape) == 4

    if is_3d:
        assert z_idx is not None, "`z_idx` must be provided for 3D images."

    # Validate optional sequences
    if titles is not None:
        assert len(titles) == len(imgs), (
            f"Length of `titles` ({len(titles)}) must match number of images ({len(imgs)})."
        )

    if contrast_lims is not None:
        assert len(contrast_lims) == len(imgs), (
            f"Length of `contrast_lims` ({len(contrast_lims)}) must match number of images ({len(imgs)})."
        )
        # Validate each non-None contrast_lims entry has correct channel count
        for i, clims in enumerate(contrast_lims):
            if clims is not None:
                assert len(clims) == C, (
                    f"Length of `contrast_lims[{i}]` ({len(clims)}) must match "
                    f"number of channels ({C})."
                )

    if clip_vals is not None:
        assert len(clip_vals) == len(imgs), (
            f"Length of `clip_vals` ({len(clip_vals)}) must match number of images ({len(imgs)})."
        )

    if annotations is not None:
        assert len(annotations) == len(imgs), (
            f"Length of `annotations` ({len(annotations)}) must match number of images ({len(imgs)})."
        )
        for i, row_annots in enumerate(annotations):
            if row_annots is not None:
                assert len(row_annots) == C, (
                    f"Length of `annotations[{i}]` ({len(row_annots)}) must match "
                    f"number of channels ({C})."
                )

    if diverging_midvalues is not None:
        assert len(diverging_midvalues) == len(imgs), (
            f"Length of `diverging_midvalues` ({len(diverging_midvalues)}) must match "
            f"number of images ({len(imgs)})."
        )
        for i, mvals in enumerate(diverging_midvalues):
            if mvals is not None:
                assert len(mvals) == C, (
                    f"Length of `diverging_midvalues[{i}]` ({len(mvals)}) must match "
                    f"number of channels ({C})."
                )

    # Preprocess images: crop and clip
    processed_imgs = []
    for i, img in enumerate(imgs):
        # Crop image
        img = _crop_image(img, z_idx, x_ROI, y_ROI)

        # Apply clipping if provided
        if clip_vals is not None and clip_vals[i] is not None:
            clip_val = clip_vals[i]
            if isinstance(clip_val, (float, int)):
                img = img.clip(min=clip_val)
            elif len(clip_val) == C:
                for c in range(C):
                    if clip_val[c] is not None:
                        img[c] = img[c].clip(min=clip_val[c])

        processed_imgs.append(img)

    # Build per-image cmap lists using per-image vmaxs/vmins/midvalues
    per_img_cmap_lists = []
    for i, img in enumerate(processed_imgs):
        img_contrast_lims = contrast_lims[i] if contrast_lims is not None else None
        img_vmaxs = _derive_vmaxs(img, img_contrast_lims)
        img_vmins = (
            _derive_vmins(img, img_contrast_lims)
            if not diverging_symmetric_norm
            else None
        )
        img_midvalues = diverging_midvalues[i] if diverging_midvalues is not None else None
        cmap_list_i = _get_multichannel_cmap(
            n_channels=C,
            cmaps=cmaps,
            multicolor_cmaps=multicolor_cmaps,
            diverging_cmaps=diverging_cmaps,
            diverging_colors=diverging_colors,
            vmaxs=img_vmaxs,
            vmins=img_vmins,
            midvalues=img_midvalues,
        )
        per_img_cmap_lists.append(list(zip(*cmap_list_i)))

    # Create subplot grid
    nrows = len(imgs)
    ncols = C
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols + 1, 6 * nrows + 1),
        constrained_layout=True
    )

    # Handle single image or single channel cases
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Set figure styling
    fig.patch.set_facecolor('black')
    if suptitle:
        fig.suptitle(suptitle, fontsize=32, color="white")

    # Plot each image-channel combination
    for i in range(nrows):
        img = processed_imgs[i]
        cmap_list_i, norms_i = per_img_cmap_lists[i]

        for c in range(ncols):
            ax = axes[i, c]

            # Determine title (only on first channel)
            title = None
            if c == 0 and titles is not None and titles[i] is not None:
                title = titles[i]

            # Determine contrast limits for this channel
            vlims = None
            if contrast_lims is not None and contrast_lims[i] is not None:
                vlims = contrast_lims[i][c]

            # Determine corner text for this channel subplot
            corner_annot = None
            if annotations is not None and annotations[i] is not None and annotations[i][c] is not None:
                corner_annot = annotations[i][c]

            # Plot the channel
            _add_channel_image(
                img[c], ax, fig,
                cmap=cmap_list_i[c],
                norm=norms_i[c] if norms_i[c] is not None else None,
                title=title,
                vlims=vlims,
                add_colorbar=show_cbars,
                add_histogram=show_hist,
                text=corner_annot,
                text_loc=annot_pos,
            )
    
    # Save plot if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving plot to {save_path}")
        ext = save_path.split('.')[-1]
        plt.savefig(
            save_path,
            format=ext,
            dpi=dpi,
            transparent=transparent,
            bbox_inches="tight",
            pad_inches=0.02
        )