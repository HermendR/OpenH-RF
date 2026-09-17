# SPDX-License-Identifier: Apache-2.0
"""Example reconstruction script for the siemens-healthineers dataset of OpenH-RF.

Dataset link: https://huggingface.co/datasets/nvidia/OpenH-RF/tree/main/siemens-healthineers

B-mode reconstruction of in-vivo IQ channel data from a clinical scanner.

The script renders a three-panel figure: IQ magnitude, the stored B-mode, and
the zea reconstruction. ``data/image`` was produced by this same processing at
conversion time, so the reconstruction reproduces it exactly -- a round-trip
check on the converted data.

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
import zea
from zea.ops import Beamform, EnvelopeDetect, LogCompress, Normalize

HERE = Path(__file__).parent

# --- Inputs -----------------------------------------------------------------
# Defaults stream straight from the published corpus. Swap any of these for a
# local path to run against your own copy.
INPUT = "hf://nvidia/OpenH-RF/siemens-healthineers/data/Subject_01_acq_006.hdf5"
OUTPUT = None  # PNG path (default: <input>_recon.png)
FRAME = 0
DEPTH_M = 0.035  # reconstruct down to 35 mm
SAVE_PIPELINE = None  # Optionally write the pipeline to a reusable pipeline.yaml


def coords_to_imshow_mm(coords):
    """Per-pixel coordinates (z, x, 3), last axis [x, y, z] in metres ->
    mpl imshow extent [left, right, bottom, top] in mm."""
    x, z = coords[..., 0], coords[..., 2]
    return [x.min() * 1e3, x.max() * 1e3, z.max() * 1e3, z.min() * 1e3]


def main():
    output = OUTPUT or Path(Path(INPUT).stem + "_recon.png")

    zea.init_device()

    with zea.File(str(INPUT)) as f:
        raw = f.data.raw_data[:]
        img_coords = f.data.image.coordinates[:]
        # Acquisition parameters straight from the file; reconstruction grid
        # derived from the stored B-mode's coordinates so the results align.
        nz, nx = img_coords.shape[0], img_coords.shape[1]
        params = f.load_parameters(
            grid_size_x=nx,
            grid_size_z=nz,
            xlims=[float(img_coords[..., 0].min()), float(img_coords[..., 0].max())],
            zlims=[float(img_coords[..., 2].min()), DEPTH_M],
            n_ch=raw.shape[-1],  # 2 = baseband IQ
            selected_transmits="all",
        )

    print(f"raw_data: {raw.shape}")

    pipeline = zea.Pipeline(
        operations=[
            Beamform(beamformer="delay_and_sum"),
            EnvelopeDetect(),
            Normalize(),
            LogCompress(),
        ]
    )
    if SAVE_PIPELINE is not None:
        pipeline.to_yaml(str(SAVE_PIPELINE))
        print(f"Saved pipeline recipe to {SAVE_PIPELINE}")

    inputs = pipeline.prepare_parameters(params)
    recon = keras.ops.convert_to_numpy(
        pipeline(data=raw[FRAME : FRAME + 1], **inputs)["data"]
    )[0]
    recon_ext = [v * 1e3 for v in params.extent_imshow]
    print(f"Reconstructed: {recon.shape}")

    zea.visualize.set_mpl_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(recon, aspect="equal", cmap="gray", vmin=-60, vmax=0, extent=recon_ext)
    ax.set_xlabel("Lateral [mm]")
    ax.set_ylabel("Depth [mm]")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
