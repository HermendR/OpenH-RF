# SPDX-License-Identifier: Apache-2.0
"""Example reconstruction script for the ulmshare dataset of OpenH-RF.

Dataset link: https://huggingface.co/datasets/nvidia/OpenH-RF/tree/main/ulmshare

B-mode, power-Doppler and ULM reconstruction of transcranial plane-wave IQ
channel data.

Alongside the B-mode, a tissue-suppression pass gives the clutter-filtered IQ
movie behind the power-Doppler output and the ULM density map (see :mod:`ulm`).
Everything is written into OUT_DIR: ``bmode.png``, ``power_doppler.png``,
``power_doppler_movie.gif`` and ``ulm_density.png``.

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

import numpy as np
import zea
from ulm import reconstruct_ulm
from zea.ops import (
    Beamform,
    Cast,
    EnvelopeDetect,
    LogCompress,
    Pipeline,
    TissueSuppression,
)

# ULMShare reference beamforming grid (example_script_one_buffer / process.m).
#
# MATLAB's axes are ``startX:deltaGrid:endX-deltaGrid``: an exact 25 um pitch
# stopping one pitch short of the nominal end. zea's ``linspace(lim0, lim1, n)``
# instead spreads n samples over the whole span, which would give a 25.0627 um
# pitch -- a drift of a full pixel by the far edge, misregistering every track
# handed to MATLAB via export_for_matlab. So the limits passed to zea are the
# MATLAB grid's *last sample* (LAST_X/LAST_Z), while END_X/END_Z stay the nominal
# span for matplotlib's extent (which wants outer pixel edges).
START_X, END_X = -0.5e-2, 0.5e-2  # m
START_Z, END_Z = 0.05e-2, 0.85e-2  # m
DELTA_GRID = 2.5e-5  # m
GRID_SIZE_X = int(round((END_X - START_X) / DELTA_GRID))  # 400
GRID_SIZE_Z = int(round((END_Z - START_Z) / DELTA_GRID))  # 320
LAST_X = START_X + (GRID_SIZE_X - 1) * DELTA_GRID  # 4.975 mm
LAST_Z = START_Z + (GRID_SIZE_Z - 1) * DELTA_GRID  # 8.475 mm
EXTENT_MM = [START_X * 1e3, END_X * 1e3, END_Z * 1e3, START_Z * 1e3]
FNUMBER = 1.4

DYNAMIC_RANGE_DB = 50.0  # B-mode display floor
PD_RANGE_DB = 40.0  # Power-Doppler display range
MOVIE_FPS = 10
BATCH_SIZE = 8  # frames beamformed per pipeline call
CLUTTER_FILTER_CUT = 5 / 100  # SVD cutoff as a fraction of frames (process.m)
T_PEAK = 0.0  # MUST adds no pulse-peak offset; zea's default images too deep
# ULMShare records 400-frame buffers, concatenated by convert_acquisition. The
# clutter filter runs per buffer as the reference MATLAB does -- and a
# 1600-frame Casorati SVD does not fit in GPU memory anyway.
FRAMES_PER_BUFFER = 400

# ULM parameters, in beamforming-grid pixel units (see ulm.py). Tuned against the
# MATLAB reference density map for mouse_18/acquisition_3 (buffers 75-78);
# they score NCC ~0.33 (compare_to_matlab.py). This is a like-for-like ULM
# reimplementation, not a port of TAL, so exact agreement is not expected.
ULM_THRESHOLD_SNR = 4.0  # detection floor, as a multiple of the median |IQ|
ULM_MIN_DISTANCE = 1  # local-maximum suppression half-width
ULM_MAX_LINKING_DISTANCE = 2.5  # max frame-to-frame bubble jump
# Bubbles dip below the detection floor constantly; without gap tolerance tracks
# fragment to a median length of 2 and min_track_length rejects nearly all.
ULM_MAX_GAP = 4
ULM_MIN_TRACK_LENGTH = 15  # TAL's config.json uses 20; 15 scored best here
# Density-map up-sampling. MATLAB's wavelength/10 is only ~2.5x finer than the
# 25 um BF grid; 10 is finer than that and costs nothing given track interpolation.
ULM_SUPER_RES = 10

# --- Inputs -----------------------------------------------------------------
# Defaults stream straight from the published corpus. Swap any of these for a
# local path to run against your own copy.
ZEA_FILE = "hf://nvidia/OpenH-RF/ulmshare/data/mouse_40_acquisition_1_zea.hdf5"
OUT_DIR = Path(__file__).parent  # Directory for all outputs (default: next to this script)
FRAME = 0  # B-mode frame index (default: 0)
# An acquisition is ~178 GB / 76000 frames; `None` means all of them, which is
# only sensible on local disk. Four buffers is what the ULM parameters above
# were tuned against.
N_FRAMES = 4 * FRAMES_PER_BUFFER  # frames used for Power-Doppler and ULM


def build_pipeline():
    """Build the ULMShare B-mode pipeline.

    Two departures from zea's default chain, both needed to reproduce ULMShare's
    reference, which takes ``20*log10(|IQ|)`` and subtracts the max: no
    ``Normalize`` (it rescales the envelope, changing the dB values), and
    ``clip=False`` (the default clips to -60 dB, truncating the true range).
    :func:`reconstruct` subtracts the max itself.
    """
    return Pipeline(
        [
            Cast(dtype="float32"),
            Beamform(beamformer="delay_and_sum", enable_pfield=False),
            EnvelopeDetect(),
            LogCompress(clip=False),
        ],
        with_batch_dim=True,
        jit_options="pipeline",
    )


def build_tissue_suppression_pipeline(cutoff=CLUTTER_FILTER_CUT):
    """Build the Power-Doppler pipeline: Cast -> Beamform -> TissueSuppression.

    ``filter_type`` is left to zea, which picks ``svd_complex`` from the channel
    axis for this IQ data -- the Hermitian filter ``process.m`` uses. ``cutoff``
    is only what the YAML records; :func:`clutter_filter` overrides it per buffer.
    """
    return Pipeline(
        [
            Cast(dtype="float32"),
            Beamform(beamformer="delay_and_sum", enable_pfield=False),
            TissueSuppression(cutoff=cutoff),
        ],
        with_batch_dim=True,
        jit_options="pipeline",
    )


def matlab_clutter_cutoff(n_frames, cut=CLUTTER_FILTER_CUT):
    """Component count reproducing MATLAB's SVD clutter cut.

    Both codebases build the Hermitian Gram matrix ``XᴴX`` and project onto the
    orthogonal complement of the leading subspace -- the same filter -- but index
    the kept subspace differently. MATLAB keeps ``eig_vect(:, Ncut:end)``
    (1-based, inclusive) so rejects ``Ncut - 1``; zea keeps ``V[:, cutoff:]``
    (0-based) so rejects ``cutoff``. At 400 frames that is 19 vs 20. With this
    correction the two agree to ~1e-3 relative, versus ~2.4e-1 without.
    """
    return max(int(round(n_frames * cut)) - 1, 0)


def load_pipeline(path):
    """Load a pipeline back from a YAML written by :meth:`Pipeline.to_yaml`."""
    return Pipeline.from_path(str(path))


def split_pipeline(pipeline, op_type):
    """Split ``pipeline`` around the first ``op_type`` op.

    Returns ``(head, op)``: a batched pipeline of everything before ``op_type``,
    and the op itself — letting the beamforming front run per frame while the
    across-frames op is applied to the assembled stack.
    """
    ops = list(pipeline.operations)
    idx = next(i for i, o in enumerate(ops) if isinstance(o, op_type))
    head = Pipeline(ops[:idx], with_batch_dim=True, jit_options="pipeline")
    return head, ops[idx]


def _raw_source(zea_file):
    """Return the object holding the raw-data track of ``zea_file``.

    ``convert_one_buffer`` writes a single-track file whose data hangs off the
    file itself; ``convert_acquisition`` writes a ``raw`` track alongside the
    reference summary image. Both layouts are handled so either converter's
    output can be reconstructed.
    """
    tracks = getattr(zea_file, "tracks", None)
    if not tracks or len(tracks) == 1:
        return zea_file
    for track in tracks:
        if track.label == "raw":
            return track
    raise ValueError(f"No 'raw' track in {zea_file}; found {[t.label for t in tracks]}.")


def _apply_ulmshare_grid(parameters):
    """Set the ULMShare reference BF grid, f-number and ``t_peak``.

    The limits end on the MATLAB grid's last sample, not its nominal end, so
    zea's linspace lands on exactly MATLAB's coordinates -- see the note above.
    """
    parameters.set_transmits("all")
    parameters.xlims = (START_X, LAST_X)
    parameters.zlims = (START_Z, LAST_Z)
    parameters.grid_size_x = GRID_SIZE_X
    parameters.grid_size_z = GRID_SIZE_Z
    parameters.f_number = FNUMBER
    parameters.t_peak = np.full(parameters.n_tx, T_PEAK, dtype=np.float32)
    return parameters


def reconstruct(zea_path, pipeline, frame_index=0):
    """Reconstruct one B-mode frame (2D dB image), normalised to 0 dB max."""
    zea.init_device(verbose=False)
    with zea.File(str(zea_path)) as f:
        source = _raw_source(f)
        parameters = source.load_parameters()
        raw = source.data.raw_data[frame_index : frame_index + 1]  # keep batch dim

    inputs = pipeline.prepare_parameters(_apply_ulmshare_grid(parameters))
    outputs = pipeline(**{pipeline.key: raw}, **inputs, return_numpy=True)
    bmode = np.asarray(outputs[pipeline.output_key])[0]
    return bmode - bmode.max()


def beamform_stack(zea_path, pipeline, n_frames=None, batch_size=BATCH_SIZE):
    """Beamform every frame into an IQ movie ``(n_frames, Nz, Nx, 2)``."""
    zea.init_device(verbose=False)
    with zea.File(str(zea_path)) as f:
        source = _raw_source(f)
        parameters = _apply_ulmshare_grid(source.load_parameters())
        total = source.data.raw_data.shape[0]
        total = total if n_frames is None else min(n_frames, total)
        inputs = pipeline.prepare_parameters(parameters)

        frames = []
        for start in range(0, total, batch_size):
            raw = source.data.raw_data[start : min(start + batch_size, total)]
            out = pipeline(**{pipeline.key: raw}, **inputs, return_numpy=True)
            frames.append(np.asarray(out[pipeline.output_key]))
            print(f"  beamformed {min(start + batch_size, total)}/{total} frames")

    return np.concatenate(frames, axis=0)


def tissue_suppress(iq_bf, op):
    """Run a ``TissueSuppression`` op over one beamformed buffer.

    The op reads axis 0 as the frame axis and takes zea's ``[I, Q]`` channel
    convention directly, so ``(n_frames, Nz, Nx, 2)`` needs no rearranging.
    Returns the complex movie ``(n_frames, Nz, Nx)``.
    """
    import keras

    filtered = op(data=keras.ops.convert_to_tensor(iq_bf))["data"]
    return keras.ops.convert_to_numpy(keras.ops.view_as_complex(filtered))


def clutter_filter(iq_bf, op, frames_per_buffer=FRAMES_PER_BUFFER):
    """SVD clutter-filter a beamformed stack, one acquisition buffer at a time.

    Filtering per buffer is what the reference MATLAB does, and a whole-
    acquisition Casorati SVD would not fit in GPU memory. The rejected-component
    count is resolved per buffer (:func:`matlab_clutter_cutoff`), so a short
    trailing buffer gets its own correct cut.

    Returns one complex movie ``(n_frames, Nz, Nx)`` per buffer.
    """
    total = iq_bf.shape[0]
    step = frames_per_buffer or total
    out = []
    for i, start in enumerate(range(0, total, step), start=1):
        chunk = iq_bf[start : min(start + step, total)]
        op.cutoff = matlab_clutter_cutoff(chunk.shape[0])
        out.append(tissue_suppress(chunk, op))
        print(f"  buffer {i}: {chunk.shape[0]} frames, rejected {op.cutoff} components")
    return out


def power_doppler(iq_cf):
    """Frame-integrated Power-Doppler image in dB (0 dB max)."""
    pd = 20 * np.log10(np.sum(np.abs(iq_cf), axis=0) + 1e-12)
    return pd - pd.max()


def power_doppler_movie(iq_cf):
    """Per-frame Power-Doppler movie in dB.

    As ``powerDopplerMovie`` in the MATLAB: offset by the max over the *whole*
    movie, not per frame, so brightness stays comparable across frames.
    """
    movie = 20 * np.log10(np.abs(iq_cf) + 1e-12)
    return movie - movie.max()


def save_image(image, out_path, title, label, cmap, vmin=None, vmax=None, ticks=None):
    """Render one 2D map on the reconstruction grid to a PNG."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    zea.visualize.set_mpl_style()
    fig, ax = plt.subplots(figsize=(4, 6))
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, extent=EXTENT_MM, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    # Colorbar exactly as tall as the image.
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, label=label, ticks=ticks)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {title} to {out_path}")


def save_movie_gif(movie, out_path, vmin, vmax, cmap="viridis", fps=MOVIE_FPS):
    """Render a ``(n_frames, Nz, Nx)`` dB movie to an animated GIF."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from zea.io_lib import matplotlib_figure_to_numpy, save_to_gif

    fig, ax = plt.subplots(figsize=(4, 6))
    im = ax.imshow(movie[0], cmap=cmap, vmin=vmin, vmax=vmax, extent=EXTENT_MM, aspect="equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, label="dB", ticks=[vmin, vmax])

    frames = []
    for k in range(movie.shape[0]):
        im.set_data(movie[k])
        ax.set_title(f"Power Doppler (dB) — frame {k}")
        frames.append(matplotlib_figure_to_numpy(fig))
    plt.close(fig)

    save_to_gif(np.stack(frames), out_path, fps=fps)
    print(f"Saved Power-Doppler movie ({len(frames)} frames, {fps} fps) to {out_path}")


def render_ulm(density, out_path, super_res=ULM_SUPER_RES):
    """Render a ULM density map to a PNG.

    A light blur makes the sparse splatted counts read as vessels and a gamma < 1
    keeps faint vessels visible; ``vmax`` comes from non-empty cells so the
    mostly-zero background does not collapse the scale. Sigma is specified in BF
    pixels so the look is stable as ``super_res`` changes.
    """
    from scipy.ndimage import gaussian_filter

    disp = gaussian_filter(density, sigma=0.1 * super_res) ** 0.5
    nonzero = disp[disp > 0]
    save_image(
        disp,
        out_path,
        title="ULM density map",
        label="sqrt(count)",
        cmap="hot",
        vmin=0,
        vmax=np.percentile(nonzero, 99) if nonzero.size else 1.0,
    )


def main():
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    bmode_yaml = out_dir / "pipeline_bmode.yaml"
    ts_yaml = out_dir / "pipeline_tissue_suppression.yaml"

    # Persist both pipelines, then run the ones loaded back from disk: the
    # reconstruction is then demonstrably the one the YAMLs describe.
    build_pipeline().to_yaml(str(bmode_yaml))
    build_tissue_suppression_pipeline().to_yaml(str(ts_yaml))
    pipeline = load_pipeline(bmode_yaml)
    ts_pipeline = load_pipeline(ts_yaml)
    print(f"Saved pipelines to {bmode_yaml} and {ts_yaml}")

    # ---- B-mode ------------------------------------------------------------ #
    print(f"Reconstructing frame {FRAME} from {ZEA_FILE} ...")
    bmode = reconstruct(ZEA_FILE, pipeline, frame_index=FRAME)
    print(f"  B-mode {bmode.shape} dB range [{bmode.min():.1f}, {bmode.max():.1f}]")
    save_image(
        bmode,
        out_dir / "bmode.png",
        title=f"B-mode (dB) — frame {FRAME}",
        label="dB",
        cmap="gray",
        vmin=-DYNAMIC_RANGE_DB,
        vmax=0,
        ticks=[-DYNAMIC_RANGE_DB, 0],
    )

    # ---- Power-Doppler ----------------------------------------------------- #
    # TissueSuppression works across frames, so the pipeline is split: its
    # Cast -> Beamform head runs per frame, the op itself over each buffer.
    bf_head, ts_op = split_pipeline(ts_pipeline, TissueSuppression)

    print("Beamforming frame stack ...")
    iq_bf = beamform_stack(ZEA_FILE, bf_head, n_frames=N_FRAMES)

    print("SVD clutter filtering ...")
    iq_cf = np.concatenate(clutter_filter(iq_bf, ts_op), axis=0)

    save_image(
        power_doppler(iq_cf),
        out_dir / "power_doppler.png",
        title="Power Doppler (dB)",
        label="dB",
        cmap="viridis",
        vmin=-PD_RANGE_DB,
        vmax=0,
        ticks=[-PD_RANGE_DB, 0],
    )
    save_movie_gif(
        power_doppler_movie(iq_cf),
        out_dir / "power_doppler_movie.gif",
        vmin=-PD_RANGE_DB,
        vmax=0,
    )

    # ---- ULM (localize -> track -> density map) ---------------------------- #
    print("Running ULM ...")
    density, _ = reconstruct_ulm(
        iq_cf,
        threshold_snr=ULM_THRESHOLD_SNR,
        min_distance=ULM_MIN_DISTANCE,
        max_linking_distance=ULM_MAX_LINKING_DISTANCE,
        min_track_length=ULM_MIN_TRACK_LENGTH,
        max_gap=ULM_MAX_GAP,
        super_res=ULM_SUPER_RES,
    )
    render_ulm(density, out_dir / "ulm_density.png")


if __name__ == "__main__":
    main()
