# SPDX-License-Identifier: Apache-2.0
"""Example reconstruction script for the stanford-murine dataset of OpenH-RF.

Dataset link: https://huggingface.co/datasets/nvidia/OpenH-RF/tree/main/stanford-murine

B-mode reconstruction of murine full synthetic aperture, multifocal and
Hadamard-encoded channel data.

Acquisition parameters are built from each track's own metadata, and one
displayed image is saved per track.

Requires zea>=0.1.6 (https://github.com/tue-bmd/zea), the library that does the
ultrasound processing here, together with one of its Keras backends (JAX,
PyTorch or TensorFlow). Installation instructions are at
https://zea.readthedocs.io/en/latest/installation.html.

Usage:
    python reconstruct.py
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

HERE = Path(__file__).parent
CONFIG_BY_TRACK = {
    "multifocal": "pipeline_multifocal.yaml",
    "hadamard": "pipeline_hadamard.yaml",
    "synthetic_aperture": "pipeline_synthetic_aperture.yaml",
}

# --- Inputs -----------------------------------------------------------------
# Defaults stream straight from the published corpus. Swap any of these for a
# local path to run against your own copy.
ZEA_FILE = (
    "hf://nvidia/OpenH-RF/stanford-murine/data/RatExperiments/VerasonicsAcq/"
    "Rat3/ExposedLiver/DATA_Tracks_20190319_115804.hdf5"
)
CONFIG_DIR = "hf://nvidia/OpenH-RF/stanford-murine"  # holds the pipeline_*.yaml configs
OUT_DIR = HERE  # every PNG is written here


def metadata_scalar(group, name: str) -> float:
    """Read one scan scalar as a plain float."""
    return float(getattr(group, name))


def reconstruction_limits(
    file, track, raw_data, np
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Derive the acquired field of view without storing dummy image maps."""
    sampling_frequency = metadata_scalar(track.scan, "sampling_frequency")
    sound_speed = metadata_scalar(track.scan, "sound_speed") or 1540.0
    if sampling_frequency is None or sampling_frequency <= 0:
        raise ValueError(
            "A positive sampling_frequency is required to determine reconstruction depth."
        )

    initial_times = np.asarray(track.scan.initial_times, dtype=np.float64)
    zmin = max(0.0, float(np.min(initial_times) * sound_speed / 2.0))
    zmax = float(
        (np.max(initial_times) + (raw_data.shape[2] - 1) / sampling_frequency) * sound_speed / 2.0
    )

    geometry = np.asarray(file.probe.probe_geometry)
    if file.probe.type != "curved":
        return (
            (float(np.min(geometry[:, 0])), float(np.max(geometry[:, 0]))),
            (zmin, zmax),
        )

    # Curved-array elements lie on a circle whose center is below the array
    # origin. Recover its radius and opening angle from the saved geometry.
    x_positions = geometry[:, 0]
    z_positions = geometry[:, 2]
    curved_elements = np.abs(z_positions) > np.finfo(geometry.dtype).eps
    radii = -(x_positions[curved_elements] ** 2 + z_positions[curved_elements] ** 2) / (
        2.0 * z_positions[curved_elements]
    )
    radius = float(np.median(radii))
    angles = np.arctan2(x_positions, z_positions + radius)
    x_limit = max(
        float(np.max(np.abs(x_positions))),
        float((zmax + radius) * np.sin(np.max(np.abs(angles)))),
    )
    return (-x_limit, x_limit), (zmin, zmax)


def folded_frequency(frequency: float, sampling_frequency: float) -> float:
    """Fold a frequency into the sampled Nyquist interval."""
    return (frequency + sampling_frequency / 2.0) % sampling_frequency - sampling_frequency / 2.0


def demodulation_filter(
    sampling_frequency: float,
    demodulation_frequency: float,
    probe_bandwidth_percent: float,
    np,
    num_taps: int = 127,
) -> tuple[object, float]:
    """Design a baseband FIR whose passband cannot cross a sampled RF image."""
    from scipy.signal import firwin

    nyquist = sampling_frequency / 2.0
    carrier = abs(folded_frequency(demodulation_frequency, sampling_frequency))
    requested_cutoff = abs(demodulation_frequency) * probe_bandwidth_percent / 200.0
    alias_safe_cutoff = min(carrier, nyquist - carrier) * 0.9
    cutoff = min(requested_cutoff, alias_safe_cutoff)
    if not 0.0 < cutoff < nyquist:
        raise ValueError(
            f"Cannot design demodulation filter for fs={sampling_frequency:g} Hz, "
            f"demodulation_frequency={demodulation_frequency:g} Hz"
        )
    return firwin(num_taps, cutoff, fs=sampling_frequency).astype(np.float32), cutoff


def reconstruct_track(path: str, output_dir: Path, track_index: int, config, pipeline) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np
    import zea
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    print(f"Processing {path}")
    with zea.File(str(path)) as file:
        track = file.tracks[track_index]
        track_label = track.label
        extent_data = track.data.raw_data[:1]
        xlims, zlims = reconstruction_limits(file, track, extent_data, np)
        config_params = config.parameters.as_dict()

        # Older Stanford files sometimes marked real RF as already demodulated.
        demodulation_frequency = metadata_scalar(track.scan, "demodulation_frequency")
        center_frequency = metadata_scalar(track.scan, "center_frequency")
        demodulation_override = {}
        if center_frequency is not None and (
            demodulation_frequency is None or abs(demodulation_frequency) < 1.0
        ):
            demodulation_override["demodulation_frequency"] = center_frequency
        parameter_overrides = {
            **config_params,
            **demodulation_override,
            "grid_type": "cartesian",
        }
        parameter_overrides.update({"xlims": xlims, "zlims": zlims})

        dynamic_range = (-50, 0) if file.probe_name == "C5-2v" else (-60, 0)
        parameter_overrides["dynamic_range"] = np.array(dynamic_range, dtype=np.float32)
        parameters = track.load_parameters(**parameter_overrides)
        data = track.data.raw_data[:1, parameters.selected_transmits]

        filter_taps, _ = demodulation_filter(
            float(parameters.sampling_frequency),
            float(parameters.demodulation_frequency),
            float(parameters.probe_bandwidth_percent),
            np,
        )
        pipeline_parameters = pipeline.prepare_parameters(
            parameters,
            fir_filter_taps=filter_taps,
        )

        outputs = pipeline(return_numpy=True, **{pipeline.key: data}, **pipeline_parameters)
        image = np.squeeze(outputs[pipeline.output_key])
        image = np.clip(image, dynamic_range[0], dynamic_range[1])
        demodulation_frequency = float(
            parameter_overrides.get("demodulation_frequency")
            or metadata_scalar(track.scan, "demodulation_frequency")
            or 0.0
        )

    stem = Path(path).stem
    out_path = output_dir / f"{stem}_{track_label}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Axial (mm)")
    im = ax.imshow(
        image,
        cmap="gray",
        vmin=dynamic_range[0],
        vmax=dynamic_range[1],
        extent=parameters.extent_imshow * 1000,
    )
    ax.set_title(f"B-mode: {Path(path).stem} ({track_label})")
    colorbar_ax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)
    fig.colorbar(im, cax=colorbar_ax, label="Amplitude (dB)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def reconstruct_file(
    path: str, output_dir: Path, pipeline_configs: dict[str, tuple[object, object]]
) -> list[Path]:
    import zea

    out_paths = []
    with zea.File(str(path)) as file:
        track_labels = [track.label for track in file.tracks]
    for track_index, track_label in enumerate(track_labels):
        if track_label not in pipeline_configs:
            expected = ", ".join(sorted(pipeline_configs))
            raise ValueError(
                f"No config found for track label {track_label!r}; expected one of {expected}"
            )
        config, pipeline = pipeline_configs[track_label]
        out_paths.append(reconstruct_track(path, output_dir, track_index, config, pipeline))
    return out_paths


def main() -> None:
    import zea

    zea.init_device()
    zea.visualize.set_mpl_style()

    # Build each track pipeline once, then reuse it for every track in the file.
    pipeline_configs = {}
    for track_label, filename in CONFIG_BY_TRACK.items():
        config = zea.Config.from_path(f"{CONFIG_DIR}/{filename}")
        pipeline_configs[track_label] = (config, zea.Pipeline.from_config(config))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reconstruct_file(ZEA_FILE, OUT_DIR, pipeline_configs)


if __name__ == "__main__":
    main()
