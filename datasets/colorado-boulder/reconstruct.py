# SPDX-License-Identifier: Apache-2.0
"""Example reconstruction script for the colorado-boulder dataset of OpenH-RF.

Dataset link: https://huggingface.co/datasets/nvidia/OpenH-RF/tree/main/colorado-boulder

B-mode reconstruction of tracked swept synthetic aperture phantom channel data.

The probe is swept across the phantom while its pose is tracked, so each frame
is acquired from a different position. A custom ``ApplyProbePose`` operation,
defined in this module and placed before the beamformer, rotates and translates
the probe geometry and transmit origins by the tracked pose of the current
frame, bringing every frame into a common coordinate system before compounding.

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
from keras import ops
from mpl_toolkits.axes_grid1 import make_axes_locatable
from zea import Pipeline
from zea.internal.registry import ops_registry
from zea.ops import (
    Beamform,
    Cast,
    Demodulate,
    EnvelopeDetect,
    LogCompress,
    Map,
    Normalize,
    Operation,
)
from zea.ops.keras_ops import ExpandDims, Squeeze, Sum

HERE = Path(__file__).parent

# --- Inputs -----------------------------------------------------------------
# Defaults stream straight from the published corpus. Swap any of these for a
# local path to run against your own copy.
DATA_FILE = "hf://nvidia/OpenH-RF/colorado-boulder/Sub-dataset-1/sub-phantom2d_01_tracked-ssa.hdf5"
PIPELINE_FILE = HERE / "pipeline.yaml"
OUTPUT_PNG = HERE / "ssa_bmode.png"


# ============================================================
# Custom operation: ApplyProbePose
# ============================================================
# Rotates/translates the probe geometry and transmit origin by the tracked
# probe pose of the current frame. Placed before Beamform in the pipeline.
# The per-frame pose (probe_translation, probe_rotation) is supplied by the
# enclosing Map, which slices them frame-by-frame alongside the data.
# These ops are defined here, not in zea: a pipeline.yaml naming them resolves
# only once this module is imported. See https://github.com/open-h/OpenH-RF
@ops_registry("apply_probe_pose")
class ApplyProbePose(Operation):
    """Apply the tracked probe pose of one frame before beamforming."""

    def __init__(self, **kwargs):
        super().__init__(additional_output_keys=["probe_geometry", "transmit_origins"], **kwargs)

    def call(
        self,
        probe_translation,
        probe_rotation,
        probe_geometry,
        transmit_origins,
        **kwargs,
    ):
        t = ops.reshape(ops.cast(probe_translation, "float32"), (3,))
        q = ops.reshape(ops.cast(probe_rotation, "float32"), (4,))
        q = q / (ops.sqrt(ops.sum(q * q)) + 1e-12)

        x, y, z, w = q[0], q[1], q[2], q[3]
        R = ops.stack(
            [
                ops.stack(
                    [
                        1.0 - 2.0 * (y * y + z * z),
                        2.0 * (x * y - z * w),
                        2.0 * (x * z + y * w),
                    ],
                    axis=0,
                ),
                ops.stack(
                    [
                        2.0 * (x * y + z * w),
                        1.0 - 2.0 * (x * x + z * z),
                        2.0 * (y * z - x * w),
                    ],
                    axis=0,
                ),
                ops.stack(
                    [
                        2.0 * (x * z - y * w),
                        2.0 * (y * z + x * w),
                        1.0 - 2.0 * (x * x + y * y),
                    ],
                    axis=0,
                ),
            ],
            axis=0,
        )

        probe_geometry = ops.matmul(ops.cast(probe_geometry, "float32"), ops.transpose(R)) + t
        transmit_origins = ops.matmul(ops.cast(transmit_origins, "float32"), ops.transpose(R)) + t

        return {
            self.output_key: kwargs[self.key],
            "probe_geometry": probe_geometry,
            "transmit_origins": transmit_origins,
        }


# ============================================================
# One single pipeline
# ============================================================
# Map runs the frame-wise part (pose + beamforming) once per tracked frame,
# slicing data / probe_translation / probe_rotation along the frame axis.
# The stacked beamformed IQ frames are then coherently summed (Sum) before
# envelope detection, as required for SSA.
def create_and_save_pipeline():
    pipeline = Pipeline(
        operations=[
            Map(
                operations=[
                    Squeeze(axis=0),  # (1, n_tx=1, n_ax, n_el, n_ch) -> (n_tx=1, n_ax, n_el, n_ch)
                    Cast(dtype="float32"),
                    Demodulate(),
                    ApplyProbePose(),
                    Beamform(beamformer="delay_and_sum", enable_pfield=False),
                    ExpandDims(axis=0),  # restore the frame axis so Map can stack frames
                ],
                argnames=["data", "probe_translation", "probe_rotation"],
                batch_size=1,  # one tracked frame at a time (each frame has its own pose)
            ),
            Sum(axis=0),  # coherent compounding of the beamformed IQ frames
            EnvelopeDetect(),
            Normalize(),
            LogCompress(),
        ],
        with_batch_dim=False,
    )
    pipeline.to_yaml(str(PIPELINE_FILE))


# ============================================================
# Main reconstruction
# ============================================================
def main():
    zea.init_device(verbose=False)

    create_and_save_pipeline()
    pipeline = Pipeline.from_path(str(PIPELINE_FILE))

    # ------------------------------------------------------------
    # Load zea parameters and raw data
    # ------------------------------------------------------------
    with zea.File(str(DATA_FILE), mode="r") as file:
        parameters = file.load_parameters(
            xlims=(-0.05, 0.05),
            zlims=(0.01, 0.13),
            ylims=(0.0, 0.0),
            f_number=0.75,
            dynamic_range=(-50, 0),
        )
        raw_data = np.array(file.data.raw_data).astype(np.float32)
        probe_pose = file.metadata.probe_pose
        probe_translation = np.array(probe_pose.translation).astype(np.float32)
        probe_rotation = np.array(probe_pose.rotation).astype(np.float32)

    # ------------------------------------------------------------
    # Select frames based on approximately uniform x-spacing
    # ------------------------------------------------------------
    desired_dx_m = 1e-3  # 1 mm spacing along x
    x_motion = probe_translation[:, 0]  # x position for each frame, in meters
    direction = np.sign(x_motion[-1] - x_motion[0])
    dx = desired_dx_m * direction
    desired_x_positions = np.arange(x_motion[0], x_motion[-1] + 0.5 * dx, dx, dtype=np.float32)
    frame_indices = np.array(
        [np.argmin(np.abs(x_motion - x_target)) for x_target in desired_x_positions]
    )

    # Center the selected sweep around the reconstruction grid
    probe_translation = probe_translation - np.mean(probe_translation[frame_indices], axis=0)

    # ------------------------------------------------------------
    # Run the pipeline
    # ------------------------------------------------------------
    inputs = pipeline.prepare_parameters(parameters)
    output = pipeline(
        data=raw_data[frame_indices],
        probe_translation=probe_translation[frame_indices],
        probe_rotation=probe_rotation[frame_indices],
        **inputs,
    )
    bmode = np.squeeze(ops.convert_to_numpy(output["data"]))

    # ------------------------------------------------------------
    # Save B-mode image
    # ------------------------------------------------------------
    flatgrid_np = ops.convert_to_numpy(inputs["flatgrid"])
    x_plot = flatgrid_np[:, 0] * 1e3
    z_plot = flatgrid_np[:, 2] * 1e3

    zea.visualize.set_mpl_style()
    plt.figure(figsize=(6, 7))
    plt.imshow(
        bmode,
        cmap="gray",
        extent=[np.min(x_plot), np.max(x_plot), np.max(z_plot), np.min(z_plot)],
        aspect="auto",
    )
    plt.xlabel("x [mm]")
    plt.ylabel("z [mm]")
    plt.title("Tracked SSA B-mode")
    cax = make_axes_locatable(plt.gca()).append_axes("right", size="5%", pad=0.05)
    plt.gcf().colorbar(plt.gci(), cax=cax, label="dB")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_PNG), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved reconstruction to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
