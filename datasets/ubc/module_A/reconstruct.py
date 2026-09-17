#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Example reconstruction script for the ubc/module_A dataset of OpenH-RF.

Dataset link: https://huggingface.co/datasets/nvidia/OpenH-RF/tree/main/ubc/module_A

B-mode reconstruction of the corrected synthetic-channel S-WAVE phantom data
on a scanline grid.

Only zea's logical ``raw_data`` field (stored at
``/tracks/track_0/data/raw_data``) and the acquisition parameters are used to
form the B-mode. The stored reference image and the source line RF are never
loaded, so the reconstruction is independent of them.

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

import keras
import matplotlib.pyplot as plt
import numpy as np
import zea
from mpl_toolkits.axes_grid1 import make_axes_locatable
from zea.ops import Beamform, Cast, Demodulate, EnvelopeDetect, LogCompress, Normalize

HERE = Path(__file__).resolve().parent

# --- Inputs -----------------------------------------------------------------
# Defaults stream straight from the published corpus. Swap any of these for a
# local path to run against your own copy.
ZEA_FILE = (
    "hf://nvidia/OpenH-RF/ubc/module_A/acquisitions/case_1.83/"
    "ubc_swave_cirs_1.83_p10_f13.hdf5"
)
OUTPUT = HERE / "results" / "reconstruct_1.83_p10_f13.png"
FRAME = 0

NORMALIZATION_PERCENTILE = 99.5
DISPLAY_FLOOR_DB = -50.0
DISTANCE_TO_APEX_M = 10.0e-3
LINE_DEPTH_MIN_M = 0.1e-3
LINE_DEPTH_MAX_M = 100.023e-3

# Reconstruction grid: one beamformed line per stored transmit, on the polar
# grid the probe acquired. f_number=0 keeps the full aperture on every line.
PARAMETERS = {
    "enable_scanline": True,
    "grid_type": "polar",
    "zlims": [LINE_DEPTH_MIN_M, LINE_DEPTH_MAX_M],
    "grid_size_z": 256,
    "dynamic_range": [DISPLAY_FLOOR_DB, 0.0],
    "f_number": 0.0,
    "apply_lens_correction": False,
}


def build_pipeline() -> zea.Pipeline:
    """Return the zea pipeline used for this reference reconstruction.

    Cast -> Demodulate -> Beamform(...) -> EnvelopeDetect -> Normalize
    -> LogCompress
    """
    return zea.Pipeline(
        operations=[
            # The cast is explicit because the stored channel data is int16.
            Cast(dtype="float32"),
            # Demodulate real RF to complex baseband IQ.
            Demodulate(),
            # Receive DAS with zea's built-in one-transmit-per-scanline mask.
            Beamform(beamformer="delay_and_sum", enable_aligned_apodization=True),
            EnvelopeDetect(),
            Normalize(output_range=[0.0, 1.0], percentile=NORMALIZATION_PERCENTILE),
            LogCompress(),
        ]
    )


def reconstruct(zea_path, frame: int = 0, pipeline=None):
    """Reconstruct one frame, returning the polar B-mode and its parameters."""
    zea.init_device()

    with zea.File(str(zea_path)) as f:
        parameters = f.load_parameters(**PARAMETERS)
        raw_data = f.data.raw_data[frame : frame + 1]

    pipeline = pipeline or build_pipeline()
    inputs = pipeline.prepare_parameters(parameters)
    # return_numpy=True uses keras.ops.convert_to_numpy for multi-backend support.
    outputs = pipeline(**{pipeline.key: raw_data}, **inputs, return_numpy=True)
    return np.squeeze(outputs[pipeline.output_key][0]), parameters


def save_bmode(bmode, parameters, output_path: Path):
    """Scan-convert the scanline B-mode and save a conventional sector PNG."""
    polar_angles = parameters.polar_angles
    sector, scan_parameters = zea.display.scan_convert_2d(
        bmode,
        rho_range=(
            DISTANCE_TO_APEX_M + LINE_DEPTH_MIN_M,
            DISTANCE_TO_APEX_M + LINE_DEPTH_MAX_M,
        ),
        theta_range=(float(polar_angles.min()), float(polar_angles.max())),
        resolution=0.4e-3,
        fill_value=np.nan,
        distance_to_apex=0.0,
    )
    # scan_convert_2d returns backend tensors, so pull them off the device here,
    # at the plotting boundary -- matplotlib needs host arrays.
    sector = keras.ops.convert_to_numpy(sector)
    x_limits_mm = keras.ops.convert_to_numpy(scan_parameters["x_lim"]) * 1e3
    z_limits_mm = (
        keras.ops.convert_to_numpy(scan_parameters["z_lim"]) - DISTANCE_TO_APEX_M
    ) * 1e3

    output_path.parent.mkdir(parents=True, exist_ok=True)
    zea.visualize.set_mpl_style()
    grayscale = plt.get_cmap("gray").copy()
    grayscale.set_bad(color="none")  # samples outside the fan
    fig, axis = plt.subplots(figsize=(7.2, 7.2))
    image = axis.imshow(
        sector,
        cmap=grayscale,
        vmin=DISPLAY_FLOOR_DB,
        vmax=0.0,
        aspect="equal",
        extent=[x_limits_mm[0], x_limits_mm[1], z_limits_mm[1], z_limits_mm[0]],
    )
    axis.set_ylim(LINE_DEPTH_MAX_M * 1e3, 0.0)
    axis.set_title("UBC Module A: zea-native scanline DAS B-mode")
    axis.set_xlabel("Lateral position x (mm)")
    axis.set_ylabel("Depth z from probe surface (mm)")
    cax = make_axes_locatable(axis).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(image, cax=cax, label="Normalized log envelope (dB)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return sector


def main() -> None:
    bmode, parameters = reconstruct(ZEA_FILE, FRAME)
    sector = save_bmode(bmode, parameters, OUTPUT)
    print(f"Polar B-mode   : {bmode.shape}; {bmode.min():.3f}..{bmode.max():.3f} dB")
    print(f"Scan-converted : {sector.shape}")
    print(f"Saved          : {OUTPUT}")


if __name__ == "__main__":
    main()
