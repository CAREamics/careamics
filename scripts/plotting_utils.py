from contextlib import contextmanager
from enum import Enum
from typing import Generator, Literal, Optional, Sequence, Union

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from cmap import Colormap
from matplotlib.axes import Axes
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.typing import NDArray
from skimage.measure import profile_line


@contextmanager
def _set_font(
    font_family: Optional[str] = None,
) -> Generator[None, None, None]:
    """Context manager that temporarily sets the matplotlib font family.

    On exit the previous ``rcParams`` are restored, so this is safe to use
    inside any plotting function without leaking state.

    Parameters
    ----------
    font_family : Optional[str]
        Font family name to use (e.g. ``"LMRoman10"``, ``"serif"``,
        ``"DejaVu Sans"``).  If ``None`` the current default is kept.

    Examples
    --------
    >>> with _set_font("LMRoman10"):
    ...     fig, ax = plt.subplots()
    ...     ax.set_title("LaTeX-friendly title")
    """
    if font_family is None:
        yield
        return

    old_params = mpl.rcParams.copy()
    try:
        mpl.rcParams["font.family"] = font_family
        # Use Computer Modern for math text (LaTeX-style symbols).
        mpl.rcParams["mathtext.fontset"] = "cm"
        # Use TrueType (Type 42) fonts in PDFs so that text remains
        # editable in vector-graphics editors like Adobe Illustrator.
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42
        yield
    finally:
        mpl.rcParams.update(old_params)


def _get_theme_colors(
    mode: Literal["black", "white"] = "black",
) -> dict[str, str]:
    """Return a palette dictionary for the requested theme.

    This is designed to be reusable across all plotting functions in the
    visualization module.

    Parameters
    ----------
    mode : Literal["black", "white"]
        ``"black"`` — dark background, orange accent, white text.
        ``"white"`` — light background, blue accent, black text.

    Returns
    -------
    dict[str, str]
        Keys: ``"bg"``, ``"text"``, ``"accent"``, ``"patch"``,
        ``"spine"``, ``"snr_box"``.
    """
    if mode == "black":
        return {
            "bg": "black",
            "text": "white",
            "accent": "orange",
            "patch": "red",
            "spine": "white",
            "snr_box": "gray",
        }
    elif mode == "white":
        return {
            "bg": "white",
            "text": "black",
            "accent":"orange",
            "patch": "red",
            "spine": "black",
            "snr_box": "lightgray",
        }
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'black' or 'white'.")


class ColormapRepo(Enum):
    """Enum for colormaps used in the visualization."""
    CYAN = "cmap:cyan"
    GREEN = "cmap:green"
    YELLOW = "cmap:yellow"
    ORANGE = "chrisluts:opf_orange"
    RED = "cmap:red"
    MAGENTA = "cmap:magenta"
    BLUE = "chrisluts:bop_blue"
    DARK_ORANGE = "chrisluts:bop_orange"
    PURPLE = "chrisluts:bop_purple"
    WHITE = "cmap:white"


class ColorRepo(Enum):
    """Enum for colors used in the visualization."""
    CYAN = "cyan"
    GREEN = "limegreen"
    YELLOW = "gold"
    ORANGE = "orange"
    RED1 = "orangered"
    RED2 = "firebrick"
    MAGENTA = "magenta"
    PURPLE = "darkviolet"
    DARK_ORANGE = "darkorange"
    VIOLET = "blueviolet"
    WHITE = "white"
    BLUE = "blue"


def _create_custom_diverging_cmap(
    pos_color: Union[str, ColorRepo],
    neg_color: Union[str, ColorRepo],
    midvalue: float = 0.0,
    vmax: Optional[float] = None,
    vmin: Optional[float] = None,
    resolution: int = 512,
) -> tuple[mcolors.Colormap, mcolors.TwoSlopeNorm]:
    """Create a diverging colormap centered at black, ramping up to user-defined
    positive and negative colors.

    The colormap maps ``midvalue`` to black, values above it ramp to
    ``pos_color``, and values below ramp to ``neg_color``.  By default
    normalization is symmetric (both halves span the same absolute distance
    from ``midvalue``); pass an explicit ``vmin`` to make it asymmetric.

    Parameters
    ----------
    pos_color : Union[str, ColorRepo]
        Color for the positive side of the colormap.
    neg_color : Union[str, ColorRepo]
        Color for the negative side of the colormap.
    midvalue : float, optional
        Data value mapped to black (the center of the diverging colormap).
        Default is ``0.0``.
    vmax : float, optional
        Maximum data value. If ``None``, no explicit bounds are set.
    vmin : float, optional
        Minimum data value. If ``None`` (and ``vmax`` is given), derived
        symmetrically as ``2 * midvalue - vmax``. Providing an explicit value
        enables asymmetric normalization.
    resolution : int, optional
        Number of colors in the colormap, by default 512.

    Returns
    -------
    tuple[Colormap, mcolors.TwoSlopeNorm]
        A tuple containing the colormap and a normalization object.
    """
    if vmax is not None:
        assert vmax > midvalue, "`vmax` must be greater than `midvalue`."
    if vmin is not None:
        assert vmin < midvalue, "`vmin` must be less than `midvalue`."
    assert pos_color != neg_color, "Positive and negative colors must be different."
    if isinstance(pos_color, str):
        pos_color = ColorRepo(pos_color)
    if isinstance(neg_color, str):
        neg_color = ColorRepo(neg_color)

    neg_rgb = np.array(mcolors.to_rgb(neg_color.value))
    pos_rgb = np.array(mcolors.to_rgb(pos_color.value))
    t = np.linspace(0, 1, resolution)

    # Interpolate colors for the negative and positive halves
    neg_half = [tuple(t_i * neg_rgb) for t_i in t[::-1]]
    pos_half = [tuple(t_i * pos_rgb) for t_i in t]

    # Create a full colormap with black in the middle
    full_colors = np.vstack([neg_half, [[0, 0, 0]], pos_half])
    positions = np.linspace(0, 1, len(full_colors))
    cmap_dict = {p: tuple(c) for p, c in zip(positions, full_colors)}
    cmap = Colormap(cmap_dict).to_mpl()

    # Derive vmin: explicit (asymmetric) or symmetric fallback
    if vmax is not None and vmin is None:
        vmin = 2 * midvalue - vmax
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=midvalue, vmax=vmax)

    return cmap, norm


def _get_multichannel_cmap(
    n_channels: int,
    cmaps: Optional[Union[ColormapRepo, Sequence[ColormapRepo]]] = None,
    multicolor_cmaps: bool = True,
    diverging_cmaps: bool = False,
    diverging_colors: Optional[Sequence[ColorRepo]] = None,
    vmaxs: Optional[Sequence[float]] = None,
    vmins: Optional[Sequence[float]] = None,
    midvalues: Optional[Sequence[float]] = None,
) -> list[tuple[mcolors.Colormap, Optional[mcolors.TwoSlopeNorm]]]:
    """Get a colormap (and optional norm) for each channel in a multichannel image.

    Parameters
    ----------
    n_channels : int
        Number of channels.
    cmaps : Optional[Union[ColormapRepo, Sequence[ColormapRepo]]]
        Explicit colormap(s) to use. A single entry is broadcast to all channels.
        If provided, overrides ``diverging_cmaps`` and ``multicolor_cmaps``.
    multicolor_cmaps : bool
        If ``cmaps`` is not given, cycle through pre-defined colormaps from
        ``ColormapRepo`` (one distinct color per channel). When ``False`` and
        ``diverging_cmaps`` is also ``False``, grayscale (``"Greys_r"``) is used
        for all channels. Default is ``True``.
    diverging_cmaps : bool
        Build diverging colormaps (black at each channel's midvalue, negative side
        in red). Positive colors are chosen by priority: ``diverging_colors`` >
        ``multicolor_cmaps`` > grayscale. Default is ``False``.
    diverging_colors : Optional[Sequence[ColorRepo]]
        Explicit positive colors for diverging colormaps. Takes full priority over
        ``multicolor_cmaps`` when set. Ignored if ``diverging_cmaps=False``.
    vmaxs : Optional[Sequence[float]]
        Per-channel maximum values for diverging normalization. Ignored when
        ``diverging_cmaps=False``.
    vmins : Optional[Sequence[float]]
        Per-channel minimum values for diverging normalization. When ``None``
        (default), ``vmin`` is derived symmetrically from ``vmax``. Providing
        explicit values enables asymmetric normalization.
    midvalues : Optional[Sequence[float]]
        Per-channel data values mapped to black in diverging colormaps.
        Defaults to ``0.0`` for each channel when ``None``.
    """
    if cmaps is not None:
        if isinstance(cmaps, ColormapRepo):
            cmaps = [cmaps] * n_channels
        else:
            if len(cmaps) != n_channels:
                raise ValueError(
                    f"`cmaps` length ({len(cmaps)}) must match `n_channels` ({n_channels})."
                )
        return [(Colormap(c.value).to_mpl(), None) for c in cmaps]

    if diverging_cmaps:
        if diverging_colors is not None:
            pos_colors = list(diverging_colors)
        elif multicolor_cmaps:
            all_colors = list(ColorRepo)
            pos_colors = [all_colors[i % len(all_colors)] for i in range(n_channels)]
        else:
            pos_colors = [ColorRepo.WHITE] * n_channels
        return [
            _create_custom_diverging_cmap(
                pos_color=pos_colors[i],
                neg_color=ColorRepo.MAGENTA,
                midvalue=(
                    midvalues[i]
                    if midvalues is not None and midvalues[i] is not None
                    else 0.0
                ),
                vmax=vmaxs[i] if vmaxs is not None else None,
                vmin=vmins[i] if vmins is not None else None,
            )
            for i in range(n_channels)
        ]

    if multicolor_cmaps:
        all_cmaps = list(ColormapRepo)
        return [
            (Colormap(all_cmaps[i % len(all_cmaps)].value).to_mpl(), None)
            for i in range(n_channels)
        ]

    return [("Greys_r", None)] * n_channels

def _crop_image(
    img: NDArray,
    z_idx: Optional[int] = None,
    x_lims: Optional[tuple[int, int]] = None,
    y_lims: Optional[tuple[int, int]] = None
) -> NDArray:
    """Crop an image according to the given slice information."""
    if z_idx is not None:
        img = img[:, z_idx, ...]
    if y_lims is not None:
        img = img[..., y_lims[0]:y_lims[1], :]
    if x_lims is not None:
        img = img[..., x_lims[0]:x_lims[1]]

    return img


def _derive_vmaxs(
    img: NDArray,
    contrast_lims: Optional[Sequence[Optional[tuple[float, float]]]] = None,
) -> list[float]:
    """Derive per-channel vmax values from contrast limits or channel image maxima.

    Parameters
    ----------
    img : NDArray
        Multichannel image of shape ``(C, ...)``.
    contrast_lims : Optional[Sequence[Optional[tuple[float, float]]]]
        Per-channel contrast limits ``(vmin, vmax)``. A ``None`` entry falls back
        to the channel's image maximum. If the whole argument is ``None``, image
        maxima are used for every channel.

    Returns
    -------
    list[float]
        Per-channel vmax values, one entry per channel.
    """
    C = img.shape[0]
    img_maxes = [float(np.max(img[c])) for c in range(C)]
    if contrast_lims is None:
        return img_maxes
    return [
        float(c[1]) if c is not None else img_maxes[i]
        for i, c in enumerate(contrast_lims)
    ]


def _derive_vmins(
    img: NDArray,
    contrast_lims: Optional[Sequence[Optional[tuple[float, float]]]] = None,
) -> list[float]:
    """Derive per-channel vmin values from contrast limits or channel image minima.

    Parameters
    ----------
    img : NDArray
        Multichannel image of shape ``(C, ...)``.
    contrast_lims : Optional[Sequence[Optional[tuple[float, float]]]]
        Per-channel contrast limits ``(vmin, vmax)``. A ``None`` entry falls back
        to the channel's image minimum. If the whole argument is ``None``, image
        minima are used for every channel.

    Returns
    -------
    list[float]
        Per-channel vmin values, one entry per channel.
    """
    C = img.shape[0]
    img_mins = [float(np.min(img[c])) for c in range(C)]
    if contrast_lims is None:
        return img_mins
    return [
        float(c[0]) if c is not None else img_mins[i]
        for i, c in enumerate(contrast_lims)
    ]


def _parse_metrics(
    metrics_avg: dict[str, dict[str, float]],
    metrics_per_img: dict[str, dict[str, dict[str, float]]],
    ch_idx: int,
    img_fname: str
) -> dict[str, float]:
    """Parse metrics as a string."""
    # get average metrics
    avg_metrics = {}
    avg_metrics["PSNR"] = metrics_avg["PSNR"][f"RI-PSNR_FP#{ch_idx+1}"]
    avg_metrics["LPIPS"] = metrics_avg["LPIPS"][f"LPIPS_FP#{ch_idx+1}"]
    avg_metrics["MS-SSIM"] = metrics_avg["MSSIM"][f"RI-MSSIM_FP#{ch_idx+1}"]
    
    # get image metrics
    img_metrics = {}
    img_metrics["PSNR"] = metrics_per_img["PSNR"][f"RI-PSNR_FP#{ch_idx+1}"][img_fname]
    img_metrics["LPIPS"] = metrics_per_img["LPIPS"][f"LPIPS_FP#{ch_idx+1}"][img_fname]
    img_metrics["MS-SSIM"] = metrics_per_img["MSSIM"][f"RI-MSSIM_FP#{ch_idx+1}"][img_fname]
    
    return "\n".join([
        f"{avg_kv[0]}: $\\mathbf{{{img_v:.2f}}}$ ({avg_kv[1]:.2f})" 
        for avg_kv, img_v in zip(avg_metrics.items(), img_metrics.values())
    ])


def _add_colorbar(img: torch.Tensor, fig: Figure, ax: Axes) -> None:
    """Add colorbar to a `matplotlib` image."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(img, cax=cax)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.ax.yaxis.label.set_color("white")


def _get_histogram_lims(
    img: NDArray,
    n_bins: int = 200,
    tol: int = 10,
    margin: int = 5,
) -> tuple[float, float]:
    """Get delimiters for image intensity to focus histogram on positive values."""
    # build histogram of values
    img_range = (img.min(), img.max()) 
    hist, bins = np.histogram(img.ravel(), bins=n_bins, range=img_range)
    
    # find the first and last "populated" (i.e., > `tol`) bins
    pos_values = np.where(hist > tol)[0]
    if len(pos_values) > 0:
        min_bin_idx = np.maximum(0, pos_values[0] - margin)
        max_bin_idx = np.minimum(n_bins, pos_values[-1] + 1 + margin)
        I_min, I_max = bins[min_bin_idx], bins[max_bin_idx]
    else:
        I_min, I_max = img_range
    
    return I_min, I_max


def _add_intensity_histogram(
    ax: Axes,
    img: NDArray,
    color: str = "orange",
    size: int = 30
) -> Axes:
    """Add a small histogram in the bottom-left corner of an axis."""
    # Create inset axis
    inset_ax = inset_axes(ax, width=f"{size}%", height=f"{size}%", loc="lower left", borderpad=1)
    inset_ax.patch.set_alpha(0)

    # focus on positive values
    hist_lims = _get_histogram_lims(img)
    img = img[img >= hist_lims[0]]
    img = img[img <= hist_lims[1]]
    
    # plot histogram  
    inset_ax.hist(img, bins=50, color=color, histtype="step", linewidth=3)

    # Customize appearance
    inset_ax.tick_params(axis="both", colors="white", labelsize=8)  # White ticks
    for spine in inset_ax.spines.values():  # White border
        spine.set_color("white")
    # set ticks
    inset_ax.set_xticks([hist_lims[0], 0, hist_lims[1]])  # leave only min and max
    inset_ax.set_xticklabels(
        [f"{int(hist_lims[0])}", "0", f"{int(hist_lims[1])}"], color="white", fontsize=12
    )
    inset_ax.set_yticks([])  # Remove y-axis ticks
    # hide top and right axes
    inset_ax.spines['top'].set_visible(False)
    inset_ax.spines['right'].set_visible(False)

def _add_corner_text(
    ax: Axes,
    text: str,
    loc: Literal["topleft", "topright"] = "topleft",
    fontsize: int = 16,
) -> None:
    """Overlay a plain white text label in a corner of the given axis.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis to annotate.
    text : str
        String to display.
    loc : Literal["topleft", "topright"]
        Corner in which to place the text. Default is ``"topleft"``.
    fontsize : int
        Font size. Default is 16.
    """
    x, ha = (0.03, "left") if loc == "topleft" else (0.97, "right")
    ax.text(
        x, 0.97, text,
        transform=ax.transAxes,
        fontsize=fontsize,
        color="white",
        verticalalignment="top",
        horizontalalignment=ha,
    )


def _add_channel_image(
    img: NDArray,
    ax: Axes,
    fig: Figure,
    cmap: Colormap,
    norm: Optional[mcolors.TwoSlopeNorm] = None,
    title: Optional[str] = None,
    vlims: Optional[tuple[float, float]] = None,
    add_colorbar: bool = False,
    add_histogram: bool = True,
    text: Optional[str] = None,
    text_loc: Literal["topleft", "topright"] = "topleft",
) -> None:
    if title:
        ax.text(
            -0.02, 0.5, title,
            fontsize=28, color="white",
            rotation=90, ha="center", va="center",
            transform=ax.transAxes
        )
    if vlims is None:
        vlims = (None, None)
    plot = ax.imshow(img, cmap=cmap, norm=norm, vmin=vlims[0], vmax=vlims[1])
    if add_histogram:
        _add_intensity_histogram(ax, img)
    if add_colorbar:
        _add_colorbar(plot, fig, ax)
    if text is not None:
        _add_corner_text(ax, text, loc=text_loc)
    ax.axis('off')
    

def _add_metrics_info(
    ax: Axes,
    metrics_avg: dict[str, dict[str, float]],
    metrics_per_img: dict[str, dict[str, dict[str, float]]],
    ch_idx: int,
    img_fname: Optional[str] = None
) -> None:
    """Add metrics information to an axis."""
    metrics_text = _parse_metrics(metrics_avg, metrics_per_img, ch_idx, img_fname)
    ax.text(
        0.97, 0.97, metrics_text,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment='top', 
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.6)
    )


def _min_max_normalize(img: NDArray) -> NDArray:
    """Min-max normalize an image to [0, 1].

    Parameters
    ----------
    img : NDArray
        Input image.

    Returns
    -------
    NDArray
        Normalized image with values in [0, 1].
    """
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return np.zeros_like(img, dtype=np.float64)


def _extract_line_profile(
    img: NDArray,
    point1: tuple[int, int],
    point2: tuple[int, int],
    linewidth: int = 1,
) -> NDArray:
    """Extract intensity profile along a line between two points.

    Uses ``skimage.measure.profile_line`` so the two points need not be
    axis-aligned.  The profile is averaged across *linewidth* pixels
    perpendicular to the line.

    Parameters
    ----------
    img : NDArray
        2-D image array of shape ``(Y, X)``.
    point1 : tuple[int, int]
        ``(y, x)`` coordinates of the first endpoint.
    point2 : tuple[int, int]
        ``(y, x)`` coordinates of the second endpoint.
    linewidth : int
        Width (in pixels) over which the profile is averaged.  Default is 1.

    Returns
    -------
    NDArray
        1-D intensity profile.
    """
    assert img.ndim == 2, "Image must be 2-D."
    assert linewidth >= 1, "`linewidth` must be >= 1."
    profile = profile_line(
        img,
        src=point1,
        dst=point2,
        linewidth=linewidth,
        mode="constant",
    )
    return profile


def _compute_patch_snr(
    img: NDArray,
    patch_origin: tuple[int, int],
    patch_h: int,
    patch_w: int,
    quantile: float = 0.99,
) -> tuple[float, NDArray]:
    """Compute SNR of an image using a background patch.

    SNR is defined as
    ``(quantile(image, q) - mean(patch)) / std(patch)``.

    Parameters
    ----------
    img : NDArray
        2-D image array.
    patch_origin : tuple[int, int]
        ``(y, x)`` top-left corner of the background patch.
    patch_h : int
        Patch height in pixels.
    patch_w : int
        Patch width in pixels.
    quantile : float
        Quantile of the full image used as the signal level.  Default is 0.99.

    Returns
    -------
    tuple[float, NDArray]
        ``(snr_value, patch)`` — the computed SNR and the extracted patch.
    """
    y0, x0 = patch_origin
    patch = img[y0:y0 + patch_h, x0:x0 + patch_w]
    signal = np.quantile(img, quantile)
    snr = (signal - np.mean(patch)) / np.std(patch)
    return float(snr), patch


def _overlay_line_and_patch(
    ax: Axes,
    point1: tuple[int, int],
    point2: tuple[int, int],
    linewidth: int,
    patch_origin: tuple[int, int],
    patch_h: int,
    patch_w: int,
    snr: float,
    line_color: str = "cyan",
    patch_color: str = "red",
    text_color: str = "white",
    snr_box_color: str = "gray",
    snr_fontsize: int = 18,
) -> None:
    """Draw the line segment, line-width band, background patch, and SNR box.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis that already contains the image.
    point1 : tuple[int, int]
        ``(y, x)`` of the first line endpoint.
    point2 : tuple[int, int]
        ``(y, x)`` of the second line endpoint.
    linewidth : int
        Width of the profile band.
    patch_origin : tuple[int, int]
        ``(y, x)`` top-left corner of the background patch.
    patch_h : int
        Height of the background patch.
    patch_w : int
        Width of the background patch.
    snr : float
        Pre-computed SNR value to display.
    line_color : str
        Color for the line and its band.  Default is ``"cyan"``.
    patch_color : str
        Color for the patch rectangle.  Default is ``"red"``.
    text_color : str
        Color for the SNR text.  Default is ``"white"``.
    snr_box_color : str
        Background color of the SNR text box.  Default is ``"gray"``.
    snr_fontsize : int
        Font size for the SNR text box.  Default is 18.
    """
    y1, x1 = point1
    y2, x2 = point2

    # --- line segment ---
    ax.plot([x1, x2], [y1, y2], color=line_color, linewidth=1.5)

    # --- translucent band showing the profile width ---
    dy = y2 - y1
    dx = x2 - x1
    length = np.hypot(dy, dx)
    if length > 0:
        # unit normal to the line
        nx, ny = -dy / length, dx / length
        half_w = linewidth / 2.0
        # four corners of the band rectangle
        corners_x = [
            x1 - half_w * nx, x1 + half_w * nx,
            x2 + half_w * nx, x2 - half_w * nx,
        ]
        corners_y = [
            y1 - half_w * ny, y1 + half_w * ny,
            y2 + half_w * ny, y2 - half_w * ny,
        ]
        band = plt.Polygon(
            list(zip(corners_x, corners_y)),
            closed=True,
            facecolor=line_color,
            edgecolor=line_color,
            alpha=0.25,
            linewidth=0.5,
        )
        ax.add_patch(band)

    # --- endpoints ---
    ax.plot(x1, y1, marker="o", color=line_color, markersize=5)
    ax.plot(x2, y2, marker="o", color=line_color, markersize=5)

    # --- background patch rectangle ---
    py, px = patch_origin
    rect = mpatches.Rectangle(
        (px, py), patch_w, patch_h,
        linewidth=1.5,
        edgecolor=patch_color,
        facecolor=patch_color,
        alpha=0.25,
    )
    ax.add_patch(rect)

    # --- SNR text box (bottom-right) ---
    ax.text(
        0.97, 0.03,
        f"SNR = {snr:.2f}",
        transform=ax.transAxes,
        fontsize=snr_fontsize,
        color=text_color,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(facecolor=snr_box_color, alpha=0.7, boxstyle="round,pad=0.3"),
    )


def match_range(x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """Rescale `x` to match the range of a standardized version of `y`.
    
    This approach is used in the computation of range/scale invariant metrics.
    
    This is done by scaling x by a factor `alpha` a such that the term
    `||y - alpha*x||^2` is minimized, i.e., the LS solution.
    
    Parameters
    ----------
    x : np.ndarray
        The array to rescale. Shape is (C, [Z], Y, X).
    y : np.ndarray
        The reference array. Shape is (C, [Z], Y, X).
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The rescaled x array and the standardized y array.
    """
    axes = tuple(range(1, x.ndim))
    
    # Center x on the mean (so that rescaling only involves variance)
    x_ = x - x.mean(axis=axes, keepdims=True)
    
    # Standardize y (for numerical stability)
    y_ = (y - y.mean(axis=axes, keepdims=True)) / y.std(axis=axes, keepdims=True)
    
    # Compute scaling parameter 
    alpha = (y_ * x_).sum(keepdims=True) / (x_ * x_).sum(keepdims=True)
    return alpha * x_, y_
