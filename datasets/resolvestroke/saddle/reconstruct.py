# SPDX-License-Identifier: Apache-2.0
"""Example reconstruction script for the resolvestroke/saddle dataset of OpenH-RF.

Dataset link: https://huggingface.co/datasets/nvidia/OpenH-RF/tree/main/resolvestroke/saddle

B-mode reconstruction of a matrix-probe diverging-wave acquisition.

The reconstruction uses a polar (sector) grid: a fan spanning the divergence
angle in the x-z plane (y = 0), with its apex at the virtual source behind the
array. The result is scan-converted and saved as a B-mode PNG showing the
diverging cone.

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

import matplotlib.pyplot as plt
import numpy as np
import zea
from mpl_toolkits.axes_grid1 import make_axes_locatable
from zea import Config, File, Pipeline

HERE = Path(__file__).parent
# Datasets live under data/; fall back to a file sitting next to the script.
_HDF5 = "hf://nvidia/OpenH-RF/resolvestroke/saddle/data/PMP01.hdf5"
DEFAULT_INPUT = _HDF5[0] if _HDF5 else None
CONFIG = HERE / "pipeline.yaml"

# --- Inputs -----------------------------------------------------------------
# Defaults stream straight from the published corpus. Swap any of these for a
# local path to run against your own copy.
INPUT = "hf://nvidia/OpenH-RF/resolvestroke/saddle/data/PMP01.hdf5"
OUTPUT = None  # Output PNG path (default: outputs/<input-stem>_bmode.png)


def main():

    if INPUT is None or not True:
        problem = (
            "no .hdf5 files found under data/"
            if INPUT is None
            else f"input file not found: {INPUT}"
        )
    # Default output goes to outputs/ next to the script; create it if needed.
    out_path = OUTPUT or (HERE / "outputs" / f"{Path(INPUT).stem}_bmode.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    zea.init_device()
    config = Config.from_path(str(CONFIG))

    with File(str(INPUT)) as f:
        parameters = f.load_parameters(**config.parameters)
        if parameters.grid_type == "polar":
            # For a diverging wave the polar-grid apex is the virtual source
            # (|focus_distances|); derive it unless pipeline.yaml pins it.
            apex = config.parameters.get("distance_to_apex")
            if apex is None:
                focus = float(np.abs(np.ravel(parameters.focus_distances)[0]))
                apex = focus if focus > 0 else 0.0
            # zea measures the polar near-bound as radius-from-apex, so shift the
            # near zlim by the apex → the configured zlims are true on-axis depth.
            z0, z1 = (float(v) for v in config.parameters["zlims"])
            overrides = {
                **config.parameters,
                "distance_to_apex": apex,
                "zlims": (z0 + apex, z1),
            }
            parameters = f.load_parameters(**overrides)
        raw = f.data.raw_data[0:1]  # single frame → (1, n_tx, n_ax, n_el, n_ch)

    print(f"raw_data shape : {raw.shape}")
    print(f"grid           : {parameters.grid.shape}  ({parameters.grid_type})")

    pipeline = Pipeline.from_config(config)
    inputs = pipeline.prepare_parameters(parameters)
    outputs = pipeline(**{pipeline.key: raw}, **inputs, return_numpy=True)
    image = np.asarray(outputs[pipeline.output_key])[0]

    # Display dynamic range from pipeline.yaml (default 40 dB).
    dr = config.parameters.get("dynamic_range", [-40, 0])
    vmin, vmax = float(dr[0]), float(dr[1])

    grid = np.asarray(parameters.grid)  # (..., 3), last axis (x, y, z) in metres
    zea.visualize.set_mpl_style()
    fig, ax = plt.subplots(figsize=(7, 7))

    if parameters.grid_type == "polar":
        # Scan-convert: place each (radial, angular) sample at its Cartesian (x, z).
        x_mm, z_mm = grid[..., 0] * 1e3, grid[..., 2] * 1e3
        pm = ax.pcolormesh(
            x_mm, z_mm, image, cmap="gray", vmin=vmin, vmax=vmax, shading="auto"
        )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        title = "Diverging-wave sector B-mode (y=0 plane)"
    else:
        # Cartesian volume: show the central elevation slice.
        iy = image.shape[2] // 2
        image = image[:, :, iy]
        x_mm, z_mm = grid[0, :, iy, 0] * 1e3, grid[:, 0, iy, 2] * 1e3
        extent = [
            float(x_mm.min()),
            float(x_mm.max()),
            float(z_mm.max()),
            float(z_mm.min()),
        ]
        pm = ax.imshow(
            image, cmap="gray", vmin=vmin, vmax=vmax, extent=extent, aspect="auto"
        )
        title = "B-mode (Cartesian, elevation slice y≈0)"

    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(pm, cax=cax, label="dB")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")

    print(f"Reconstructed  : {image.shape}")
    print(f"Saved          : {out_path}")


if __name__ == "__main__":
    main()
