# SPDX-License-Identifier: Apache-2.0
"""Example reconstruction script for the unc-openpros dataset of OpenH-RF.

Dataset link: https://huggingface.co/datasets/nvidia/OpenH-RF/tree/main/unc-openpros

Speed-of-sound reconstruction of limited-view prostate waveform data with a
pretrained InversionNet.

The waveform data are converted back to the tensor layout used by the OpenPros
models and given the same signed-log and min-max preprocessing as the official
OpenPros implementation. The network's normalized prediction is mapped back to
the physical speed-of-sound range (1300--3600 m/s) and plotted next to the
ground-truth map.

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
from custom_ops import LogTransform, MyRearrange
from mpl_toolkits.axes_grid1 import make_axes_locatable
from network_ops import InversionNetInference
from zea import Config, File, Pipeline
from zea.ops import Normalize

HERE = Path(__file__).parent
INPUT = "hf://nvidia/OpenH-RF/unc-openpros/data/3_04_P_prostate_51.hdf5"
CONFIG = HERE / "pipeline.yaml"
OUTPUT = HERE / "pred_sos.png"

# --- Inputs -----------------------------------------------------------------
# Defaults stream straight from the published corpus. Swap any of these for a
# local path to run against your own copy.
WRITE_CONFIG = False  # Write the pipeline and parameters to a YAML config file
LOAD_CONFIG = False  # Load the pipeline and parameters from a YAML config file
# The file holds 1140 acquisitions; the figure shows the first.
SAMPLES = 1  # acquisitions to run through the network


def plot_comparison(sos, pred, path):
    zea.visualize.set_mpl_style()
    _, ax = plt.subplots(1, 2, figsize=(7, 6))
    im = ax[0].imshow(sos[0, :, :, 0], cmap="gray", vmin=1300, vmax=1700)
    ax[0].set_title("Ground Truth SOS Map")
    ax[1].imshow(
        keras.ops.convert_to_numpy(pred)[0, :, :, 0], cmap="gray", vmin=1300, vmax=1700
    )
    ax[1].set_title("Predicted SOS Map")
    for axis in ax:
        axis.set_xlabel("X (mm)")
        axis.set_xticks(range(0, 161, 40), labels=range(0, 61, 15))
    ax[0].set_ylabel("Z (mm)")
    ax[0].set_yticks(range(0, 401, 80), labels=range(0, 151, 30))
    ax[1].set_yticks(range(0, 401, 80), [])
    cax = make_axes_locatable(ax[1]).append_axes("right", size="5%", pad=0.05)
    plt.gcf().colorbar(im, cax=cax, orientation="vertical", label="SOS (m/s)")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():

    zea.init_device(verbose=False)

    if LOAD_CONFIG:
        config = Config.from_path(str(CONFIG))
        pipeline = Pipeline.from_config(config)
    else:
        # Match the official OpenPros inference preprocessing and postprocessing:
        # restore the original four acquisition blocks (SS, SR, RR, RS), apply
        # the sign-preserving log1p transform with k=1e5, and min-max normalize
        # its configured data range to [-1, 1]. After network inference, undo
        # the label normalization by mapping the prediction from [-1, 1] to the
        # physical SOS range of 1300--3600 m/s.
        pipeline = Pipeline(
            operations=[
                MyRearrange(),  # rearrange data to the layout expected by the network
                LogTransform(data_min=-0.25, data_max=0.45, k=1e5),
                Normalize(output_range=(-1, 1)),
                InversionNetInference(preset="inversionnet-openpros"),
                Normalize(input_range=(-1, 1), output_range=(1300, 3600)),
            ]
        )

    if WRITE_CONFIG:
        config = pipeline.to_config()
        print(config)
        config.to_yaml(str(CONFIG))
        print(f"Wrote pipeline and parameters to {CONFIG}")

    with File(INPUT) as f:
        raw = f.data.raw_data[:SAMPLES]
        sos = f.data.sos_map.values[:SAMPLES]  # gt

    print(f"raw_data shape: {raw.shape}")
    print(f"ground truth shape: {sos.shape}")
    outputs = pipeline(data=raw)["data"]
    print(f"reconstructed shape: {outputs.shape}")
    plot_comparison(sos, outputs, OUTPUT)
    print(f"Saved comparison plot: {OUTPUT}")


if __name__ == "__main__":
    main()
