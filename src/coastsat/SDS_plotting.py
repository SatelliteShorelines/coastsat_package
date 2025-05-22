import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches as mpatches, lines as mlines
from matplotlib.colors import ListedColormap
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from typing import Dict, Any, Optional
from coastsat import SDS_preprocess, SDS_tools


def plot_detection(
    im_ms: np.ndarray,
    im_labels: np.ndarray,
    shoreline: np.ndarray,
    image_epsg: int,
    georef: np.ndarray,
    settings: Dict[str, Any],
    date: str,
    satname: str,
    cloud_mask: Optional[np.ndarray] = None,
    im_ref_buffer: Optional[np.ndarray] = None,
    output_directory: Optional[str] = None,
    shoreline_extraction_area: Optional[np.ndarray] = None,
    is_sar: bool = False,
):
    """
    Unified function for plotting shoreline detection on SAR and optical images.
    Saves the plot and optionally prompts user input to skip/accept detections.

    Parameters:
    - im_ms: np.array - image array (SAR or multispectral)
    - im_labels: np.array - binary mask or multi-channel classification
    - shoreline: np.array - coordinates of shoreline (X, Y)
    - image_epsg: int - EPSG code of image CRS
    - georef: np.array - georeferencing array
    - settings: dict - config dictionary
    - date: str - date of image
    - satname: str - satellite name
    - cloud_mask: np.array - mask of cloud pixels (optical only)
    - im_ref_buffer: np.array - reference shoreline buffer
    - output_directory: str - path to store outputs
    - shoreline_extraction_area: list[np.array] - area outlines
    - is_sar: bool - whether the input is SAR imagery

    Returns
        bool: True if the user accepted the detections, False otherwise.

    """
    sitename = settings["inputs"]["sitename"]
    filepath_data = settings["inputs"]["filepath"]
    output_path = _prepare_output_dir(output_directory, filepath_data, sitename)

    sl_pix = _transform_shoreline(
        shoreline, settings["output_epsg"], image_epsg, georef
    )

    if is_sar:
        return _plot_sar_detection(
            im_ms,
            im_labels,
            sl_pix,
            date,
            satname,
            im_ref_buffer,
            output_path,
            settings,
        )
    else:
        return _plot_optical_detection(
            im_ms,
            im_labels,
            cloud_mask,
            sl_pix,
            date,
            satname,
            im_ref_buffer,
            output_path,
            settings,
            shoreline_extraction_area,
            sitename,
        )


def _transform_shoreline(shoreline, output_epsg, image_epsg, georef):
    try:
        shoreline_proj = SDS_tools.convert_epsg(shoreline, output_epsg, image_epsg)
        return SDS_tools.convert_world2pix(shoreline_proj[:, :2], georef)
    except Exception:
        return np.array([[np.nan, np.nan], [np.nan, np.nan]])


def _prepare_output_dir(output_dir, base_path, sitename):
    path = os.path.join(
        output_dir or os.path.join(base_path, sitename), "jpg_files", "detection"
    )
    os.makedirs(path, exist_ok=True)
    return path


def _normalize_grayscale(im):
    im = np.nan_to_num(im, nan=0.0)
    im_min, im_max = np.percentile(im, [2, 98])
    return np.clip((im - im_min) / (im_max - im_min), 0, 1)


def _plot_sar_detection(
    im_ms: np.ndarray,
    im_labels: np.ndarray,
    sl_pix: np.ndarray,
    date: str,
    satname: str,
    im_ref_buffer: Optional[np.ndarray],
    output_path: str,
    settings: Dict[str, Any],
) -> bool:
    """
    Plots SAR detection results including grayscale image, shoreline pixels,
    labeled water/land areas, and an optional reference shoreline buffer.

    Args:
        im_ms (np.ndarray): Multispectral or grayscale image to display.
        im_labels (np.ndarray): Labeled image (typically from Otsu thresholding).
        sl_pix (np.ndarray): Array of shoreline pixel coordinates (Nx2).
        date (str): Date string for display and file naming.
        satname (str): Satellite name for display and file naming.
        im_ref_buffer (Optional[np.ndarray]): Boolean mask of the reference shoreline buffer.
        output_path (str): Directory path to save the figure if enabled.
        settings (Dict[str, Any]): Dictionary of settings. Must contain key 'save_figure'.

    Returns:
        bool: Always returns False. Used as a placeholder return value.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"wspace": 0.05})

    im_display = _normalize_grayscale(im_ms)

    # Left panel
    ax1.imshow(im_display, cmap="gray")
    if im_ref_buffer is not None:
        mask = np.ma.masked_where(im_ref_buffer == False, im_ref_buffer)
        ax1.imshow(mask, cmap="PiYG", alpha=0.6)
    ax1.plot(sl_pix[:, 0], sl_pix[:, 1], "r.", markersize=1)
    ax1.set_title(date, fontweight="bold", fontsize=14)
    ax1.axis("off")

    # Right panel
    ax2.imshow(im_display, cmap="gray")
    ax2.imshow(
        im_labels.astype(int), cmap=ListedColormap(["yellow", "blue"]), alpha=0.3
    )
    if im_ref_buffer is not None:
        ax2.imshow(
            np.ma.masked_where(~im_ref_buffer, im_ref_buffer), cmap="PiYG", alpha=0.5
        )
    ax2.plot(sl_pix[:, 0], sl_pix[:, 1], "r.", markersize=1)
    ax2.set_title(satname, fontweight="bold", fontsize=14)
    ax2.axis("off")

    # Add extra space between plots and place legend fully outside
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.2)

    # Legend placed in figure coordinates outside the plot area
    fig.legend(
        handles=[
            mlines.Line2D([], [], color="red", label="Shoreline"),
            mpatches.Patch(color="red", alpha=0.4, label="Reference shoreline buffer"),
            mpatches.Patch(color="yellow", label="Otsu class 1"),
            mpatches.Patch(color="blue", label="Otsu class 2"),
        ],
        loc="center left",
        bbox_to_anchor=(0.44, 0.5),  # Pull it further left to ensure whitespace
        fontsize=9,
        frameon=True,
    )

    if settings.get("save_figure", False):
        fig.savefig(
            os.path.join(output_path, f"{date}_{satname}.jpg"),
            dpi=150,
            bbox_inches="tight",
        )

    plt.close(fig)
    return False


def _plot_optical_detection(
    im_ms: np.ndarray,
    im_labels: np.ndarray,
    cloud_mask: np.ndarray,
    sl_pix: np.ndarray,
    date: str,
    satname: str,
    im_ref_buffer: Optional[np.ndarray],
    output_path: str,
    settings: Dict[str, Any],
    shoreline_extraction_area: Optional[np.ndarray],
    sitename: str,
) -> bool:
    """
    Plots optical shoreline detection including RGB composite, classified output,
    NDWI index, cloud mask, and reference buffer.

    Args:
        im_ms (np.ndarray): Multispectral image array (HxWxBands).
        im_labels (np.ndarray): Labeled output image (e.g., water/land mask).
        cloud_mask (np.ndarray): Boolean array indicating cloud-covered pixels.
        sl_pix (np.ndarray): Nx2 array of detected shoreline pixel coordinates.
        date (str): Acquisition date string.
        satname (str): Name of the satellite (for display and filename).
        im_ref_buffer (Optional[np.ndarray]): Boolean mask for reference shoreline buffer.
        output_path (str): Directory to save the output image.
        settings (Dict[str, Any]): Settings dictionary, should include 'save_figure' flag.
        shoreline_extraction_area (Optional[np.ndarray]): Optional mask indicating area of shoreline extraction.
        sitename (str): Name of the site (for display in title).

    Returns:
        bool: `True` if the image was skipped by user input; otherwise `False`.
    """
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(1, 3)
    ax1, ax2, ax3 = [fig.add_subplot(gs[0, i]) for i in range(3)]

    im_rgb = SDS_preprocess.rescale_image_intensity(
        im_ms[:, :, [2, 1, 0]], cloud_mask, 99.9
    )
    im_class = _get_im_class(im_labels, im_rgb)
    ax1.imshow(np.nan_to_num(im_rgb))
    ax1.plot(sl_pix[:, 0], sl_pix[:, 1], "k.", markersize=1)
    ax1.set_title(sitename, fontweight="bold", fontsize=16)
    ax1.axis("off")

    if im_ref_buffer is not None:
        mask = np.ma.masked_where(im_ref_buffer == False, im_ref_buffer)
        ax2.imshow(mask, cmap="PiYG", alpha=0.6)

    ax2.imshow(im_class)
    # plot the reference shoreline buffer
    if im_ref_buffer is not None:
        mask = np.ma.masked_where(im_ref_buffer == False, im_ref_buffer)
        ax2.imshow(mask, cmap="PiYG", alpha=0.6)
    ax2.plot(sl_pix[:, 0], sl_pix[:, 1], "k.", markersize=1)
    ax2.set_title(date, fontweight="bold", fontsize=16)
    ax2.axis("off")

    _add_optical_legend(ax2, bool(shoreline_extraction_area))

    im_mwi = SDS_tools.nd_index(im_ms[:, :, 4], im_ms[:, :, 1], cloud_mask)
    ax3.imshow(im_mwi, cmap="bwr")
    if im_ref_buffer is not None:
        ax3.imshow(
            np.ma.masked_where(~im_ref_buffer, im_ref_buffer), cmap="PiYG", alpha=0.6
        )
    ax3.plot(sl_pix[:, 0], sl_pix[:, 1], "k.", markersize=1)
    ax3.set_title(satname, fontweight="bold", fontsize=16)
    ax3.axis("off")

    skip_image = _handle_user_input(fig, ax1, settings)

    if settings.get("save_figure", False) and not skip_image:
        fig.savefig(
            os.path.join(output_path, f"{date}_{satname}.jpg"),
            dpi=150,
            bbox_inches="tight",
        )

    plt.close(fig)
    return skip_image


def _get_im_class(im_labels: np.ndarray, im_rgb: np.ndarray) -> np.ndarray:
    """
    Applies color overlays to an RGB image based on label masks.

    Args:
        im_labels (np.ndarray): Boolean label masks of shape (H, W, C), where each channel corresponds to a class.
        im_rgb (np.ndarray): RGB image of shape (H, W, 3) used as the base image.

    Returns:
        np.ndarray: RGB image with overlaid class colors.
    """
    cmap = plt.get_cmap("tab20c")
    colors = np.array(
        [
            cmap(5),  # sand
            [204 / 255, 1, 1, 1],  # whitewater
            [0, 91 / 255, 1, 1],  # water
        ]
    )
    im_class = np.copy(im_rgb)
    for i in range(im_labels.shape[2]):
        im_class[im_labels[:, :, i]] = colors[i][:3]
    return im_class


def _add_optical_legend(
    ax: matplotlib.axes.Axes, include_extraction_area: bool
) -> None:
    """
    Adds a legend to the provided Axes object for optical shoreline classification.

    Args:
        ax (matplotlib.axes.Axes): The axis to which the legend will be added.
        include_extraction_area (bool): Whether to include the shoreline extraction area in the legend.
    """
    handles = [
        mpatches.Patch(color=(0.894, 0.329, 0.188, 1), label="sand"),
        mpatches.Patch(color=(0.8, 1.0, 1.0, 1.0), label="whitewater"),
        mpatches.Patch(color=(0, 0.357, 1.0, 1.0), label="water"),
        mlines.Line2D([], [], color="k", label="shoreline"),
        mpatches.Patch(color="#800000", alpha=0.8, label="reference shoreline buffer"),
    ]
    if include_extraction_area:
        handles.append(
            mlines.Line2D([], [], color="#cb42f5", label="shoreline extraction area")
        )
    ax.legend(handles=handles, bbox_to_anchor=(1, 0.5), fontsize=10)


def _handle_user_input(fig, ax, settings):
    if not settings.get("check_detection", False):
        return False

    key_event = {}

    def press(event):
        key_event["pressed"] = event.key

    fig.canvas.mpl_connect("key_press_event", press)

    while True:
        ax.text(1.1, 0.9, "keep ⇨", size=12, transform=ax.transAxes)
        ax.text(-0.1, 0.9, "⇦ skip", size=12, transform=ax.transAxes)
        ax.text(0.5, 0, "<esc> to quit", size=12, transform=ax.transAxes)
        plt.draw()
        plt.waitforbuttonpress()

        if key_event.get("pressed") == "right":
            return False
        elif key_event.get("pressed") == "left":
            return True
        elif key_event.get("pressed") == "escape":
            plt.close()
            raise StopIteration("User cancelled checking shoreline detection")
