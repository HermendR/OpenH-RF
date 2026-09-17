# SPDX-License-Identifier: Apache-2.0
"""Reconstruction loading, coordinate, plotting, and ground-truth helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def custom_map(file: Any, track_index: int = 0) -> dict[str, np.ndarray]:
    """Return custom elements, resolving track-specific pulse metadata."""

    custom = {element.name: np.asarray(element.data) for element in file.custom}
    prefix = f"track_{track_index}_"
    for key, value in tuple(custom.items()):
        if key.startswith(prefix):
            custom[key.removeprefix(prefix)] = value
    return custom


def t_peak_override(custom: dict[str, np.ndarray]) -> np.ndarray:
    """Return the selected track's finite beamforming peak-time override."""

    if "t_peak" not in custom:
        raise KeyError("HDF5 file is missing required custom field t_peak")
    t_peak = float(np.asarray(custom["t_peak"]).reshape(-1)[0])
    if not np.isfinite(t_peak):
        raise ValueError("HDF5 custom field t_peak must be finite")
    return np.asarray([t_peak], dtype=np.float32)


def load_hdf5(
    path: str | Path,
    num_frames: int = 1,
    track_index: int = 0,
) -> tuple[Any, Any, Any, dict[str, np.ndarray]]:
    """Load one zea acquisition and its custom elements."""

    import zea

    file = zea.File(str(path))
    if track_index < 0 or track_index >= len(file.tracks):
        file.close()
        raise IndexError(
            f"track_index {track_index} is outside [0, {len(file.tracks) - 1}]"
        )
    track = file.tracks[track_index]
    parameters = track.load_parameters()
    data = track.data.raw_data[:num_frames, parameters.selected_transmits, ...]
    return file, parameters, data, custom_map(file, track_index=track_index)


def run_bmode(
    path: str | Path,
    config_path: str | Path,
    num_frames: int = 1,
    track_index: int = 0,
    dynamic_range: tuple[float, float] = (-30.0, 0.0),
    xlims_cm: tuple[float, float] | None = (-1.5, 1.5),
) -> tuple[Any, Any, np.ndarray, dict[str, np.ndarray]]:
    """Run the configured zea pipeline and return the image plus acquisition data.

    ``xlims_cm`` controls the lateral field of view used by the DAS
    beamformer. zea expects these limits in metres, while this public helper
    uses centimetres to match the plotted image axes.
    """

    import keras
    import zea

    file, parameters, data, custom = load_hdf5(
        path, num_frames=num_frames, track_index=track_index
    )
    config = zea.Config.from_path(str(config_path))
    parameters.dynamic_range = tuple(dynamic_range)
    if xlims_cm is not None:
        x_min_cm, x_max_cm = xlims_cm
        if not x_min_cm < x_max_cm:
            raise ValueError("xlims_cm must be ordered as (minimum, maximum)")
        parameters.update(xlims=(x_min_cm * 1e-2, x_max_cm * 1e-2))
    pipeline = zea.Pipeline.from_config(config)
    inputs = pipeline.prepare_parameters(parameters, t_peak=t_peak_override(custom))
    inputs = {pipeline.key: data, **inputs}
    image = keras.ops.convert_to_numpy(pipeline(**inputs)[pipeline.output_key])
    image = np.asarray(keras.ops.squeeze(image))
    image = np.asarray(
        zea.display.to_8bit(image, dynamic_range=parameters.dynamic_range)
    )
    return file, parameters, image, custom


def image_extent_cm(
    parameters: Any,
    custom: dict[str, np.ndarray],
    image_shape: tuple[int, int],
    xlims_cm: tuple[float, float] | None = None,
) -> tuple[float, float, float, float]:
    """Calculate an x/z extent in centimeters for an image-shaped array."""

    height, width = image_shape[-2:]
    domain_width = float(np.asarray(custom.get("domain_width", np.nan)).reshape(-1)[0])
    domain_depth = float(np.asarray(custom.get("domain_depth", np.nan)).reshape(-1)[0])
    if not np.isfinite(domain_width):
        geometry = np.asarray(parameters.probe_geometry)
        domain_width = float(np.max(geometry[:, 0]) - np.min(geometry[:, 0]))
    if not np.isfinite(domain_depth):
        domain_depth = (
            float(parameters.sound_speed)
            * image_shape[-2]
            / (2.0 * float(parameters.sampling_frequency))
        )
    if xlims_cm is None:
        x_extent_cm = (-0.5 * domain_width * 100.0, 0.5 * domain_width * 100.0)
    else:
        x_extent_cm = tuple(float(value) for value in xlims_cm)
        if not x_extent_cm[0] < x_extent_cm[1]:
            raise ValueError("xlims_cm must be ordered as (minimum, maximum)")
    return (*x_extent_cm, 0.0, domain_depth * 100.0)


def bubble_coordinates_cm(
    custom: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite bubble x/z coordinates in centimeters."""

    if "bubble_x" not in custom or "bubble_z" not in custom:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    x = np.asarray(custom["bubble_x"], dtype=np.float32).reshape(-1)
    z = np.asarray(custom["bubble_z"], dtype=np.float32).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(z)
    return x[valid] * 100.0, z[valid] * 100.0


def plot_bmode(
    image: np.ndarray,
    extent_cm: tuple[float, float, float, float],
    output: str | Path,
    bubble_x_cm: np.ndarray | None = None,
    bubble_z_cm: np.ndarray | None = None,
    show_bubbles: bool = True,
    title: str | None = None,
) -> None:
    """Save a B-mode image with centimeter axes and optional GT bubbles."""

    import matplotlib.pyplot as plt

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D B-mode image, got shape {image.shape}")

    x_min, x_max, z_min, z_max = extent_cm
    fig, ax = plt.subplots(figsize=(5, 12), constrained_layout=True)
    ax.imshow(
        image,
        cmap="gray",
        origin="upper",
        aspect="auto",
        extent=(x_min, x_max, z_max, z_min),
        interpolation="nearest",
        vmin=0,
        vmax=255,
    )
    if show_bubbles and bubble_x_cm is not None and bubble_z_cm is not None:
        ax.scatter(
            bubble_x_cm,
            bubble_z_cm,
            facecolors="none",
            edgecolors="red",
            linewidths=0.8,
            s=28,
            label="Bubble ground truth",
        )
        ax.legend(loc="upper right")
    ax.set_xlabel("Lateral position x [cm]")
    ax.set_ylabel("Depth z [cm]")
    if title:
        ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_max, z_min)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_gt_label(
    label: np.ndarray,
    extent_cm: tuple[float, float, float, float],
    output: str | Path,
    title: str | None = None,
) -> None:
    """Save a ground-truth label with the same physical axes as B-mode."""

    import matplotlib.pyplot as plt

    x_min, x_max, z_min, z_max = extent_cm
    fig, ax = plt.subplots(figsize=(5, 12), constrained_layout=True)
    ax.imshow(
        label,
        cmap="gray",
        origin="upper",
        aspect="auto",
        extent=(x_min, x_max, z_max, z_min),
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Lateral position x [cm]")
    ax.set_ylabel("Depth z [cm]")
    if title:
        ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_max, z_min)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def bubble_label(
    image_shape: tuple[int, int],
    extent_cm: tuple[float, float, float, float],
    bubble_x_cm: np.ndarray,
    bubble_z_cm: np.ndarray,
) -> np.ndarray:
    """Create a hard binary bubble label image."""

    height, width = image_shape
    x_min, x_max, z_min, z_max = extent_cm
    label = np.zeros((height, width), dtype=np.float32)
    x_pixels = (bubble_x_cm - x_min) / (x_max - x_min) * (width - 1)
    z_pixels = (bubble_z_cm - z_min) / (z_max - z_min) * (height - 1)
    x_pixels = np.rint(x_pixels).astype(int)
    z_pixels = np.rint(z_pixels).astype(int)
    valid = (x_pixels >= 0) & (x_pixels < width) & (z_pixels >= 0) & (z_pixels < height)
    label[z_pixels[valid], x_pixels[valid]] = 1.0
    return label
