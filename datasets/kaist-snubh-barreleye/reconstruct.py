# SPDX-License-Identifier: Apache-2.0
"""Example reconstruction script for the kaist-snubh-barreleye dataset of OpenH-RF.

Dataset link: https://huggingface.co/datasets/nvidia/OpenH-RF/tree/main/kaist-snubh-barreleye

B-mode reconstruction of 9-angle plane-wave compounded in-vivo breast channel data.

The reconstruction is written as a single B-mode PNG.

Requires zea>=0.1.6 (https://github.com/tue-bmd/zea), the library that does the
ultrasound processing here, together with one of its Keras backends (JAX,
PyTorch or TensorFlow). Installation instructions are at
https://zea.readthedocs.io/en/latest/installation.html.

Usage:
    python reconstruct.py
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("MPLBACKEND", "Agg")


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zea
from zea import File, Pipeline
from zea.ops import (
    BandPassFilter,
    Beamform,
    Cast,
    Demodulate,
    EnvelopeDetect,
    LogCompress,
    Normalize,
)

_MM = plt.FuncFormatter(lambda v, _: f"{v * 1e3:.0f}")
_HERE = Path(__file__).resolve().parent

# Imaging grid & display parameters (shared by both representations).
PARAMETERS = {
    "grid_size_x": 512,  # lateral pixels
    "grid_size_z": 768,  # axial pixels
    "xlims": [-0.0191, 0.0191],  # metres (lateral_length = 3.82 cm)
    "zlims": [0.002, 0.04],  # metres — skip first 2 mm dead-zone / TX-pulse residue
    "dynamic_range": [-50, 0],
    "apply_lens_correction": False,
    "f_number": 1.5,
}

# --- Inputs -----------------------------------------------------------------
# Defaults stream straight from the published corpus. Swap any of these for a
# local path to run against your own copy.
INPUT = "hf://nvidia/OpenH-RF/kaist-snubh-barreleye/data/S01_D1.hdf5"
SAVE_YAML = False  # write pipeline.yaml / pipeline_simple.yaml from the pipelines below
OUTPUT = _HERE / "reconstruct.png"


def build_pipeline(n_ch: int, filtered: bool = True) -> Pipeline:
    """Pipeline by representation, built directly from zea ops.

    Raw RF (n_ch=1) gets band-pass + demodulate; already-demodulated IQ
    (n_ch=2) goes straight to beamforming.
    """
    ops = [Cast(dtype="float32")]
    if n_ch == 1:
        if filtered:
            # The proposed one-op filter, applied to the raw RF before
            # demodulation. Low edge 1 MHz: the filter ablation
            # (evaluation/*.md) shows the entire benefit of the upstream
            # depth-adaptive filter is rejecting a persistent sub-MHz band
            # before coherent beamforming. High edge 12 MHz: passes the full
            # fundamental around fc = 10 MHz (Nyquist = 31.25 MHz). 255 taps
            # keep the ~0.25 MHz transition well-resolved at the 1 MHz edge
            # (fs = 62.5 MHz).
            ops.append(BandPassFilter(passband=(1e6, 12e6), num_taps=255))
        ops.append(Demodulate())
    ops += [
        Beamform(beamformer="delay_and_sum"),
        EnvelopeDetect(),
        Normalize(),
        LogCompress(),
    ]
    return Pipeline(operations=ops)


def save_yaml() -> None:
    """Export the code-defined pipeline as a shareable YAML recipe."""
    out = _HERE / "pipeline.yaml"
    build_pipeline(n_ch=1).to_yaml(str(out))
    print(f"Saved : {out}")


def reconstruct_file(path: Path, filtered: bool = True) -> dict:
    """Beamform one .hdf5 file. Returns bmode, extent and metadata."""
    with File(str(path)) as f:
        raw = f.data.raw_data[:]
        parameters = f.load_parameters(**PARAMETERS)
        patient = f.metadata.subject.id
        label = f.metadata.annotations.label

    n_ch = raw.shape[-1]
    print(
        f"  {Path(path).parent.name}/{Path(path).name}: n_ch={n_ch} ({'RF' if n_ch == 1 else 'IQ'})"
    )
    pipeline = build_pipeline(n_ch, filtered)
    inputs = pipeline.prepare_parameters(parameters)
    recon = np.array(pipeline(data=raw, **inputs)["data"])
    return {
        "bmode": zea.display.to_8bit(recon[0]),
        "extent": parameters.extent_imshow,
        "patient": patient,
        "label": label,
        "n_ch": n_ch,
    }


def _imshow_recon(ax, bmode, extent, title):
    ax.imshow(bmode, cmap="gray", aspect="auto", extent=extent)
    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.xaxis.set_major_formatter(_MM)
    ax.yaxis.set_major_formatter(_MM)


def run_single() -> int:
    print(f"Input : {INPUT}")
    zea.init_device()
    r = reconstruct_file(INPUT)

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    zea.visualize.set_mpl_style()
    plt.figure(figsize=(5, 7))
    _imshow_recon(
        plt.gca(),
        r["bmode"],
        r["extent"],
        f"DAS reconstruction — patient {r['patient']} ({r['label']})",
    )
    plt.tight_layout()
    plt.savefig(str(OUTPUT), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved : {OUTPUT}")
    return 0


def main() -> int:
    if SAVE_YAML:
        save_yaml()
        return 0
    return run_single()


if __name__ == "__main__":
    raise SystemExit(main())
