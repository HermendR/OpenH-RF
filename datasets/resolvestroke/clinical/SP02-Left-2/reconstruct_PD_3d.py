# SPDX-License-Identifier: Apache-2.0
"""Reconstruct power-Doppler 3D volumes for each clip and display as MIP montage.

For each of the 5 clips (4000 frames each), this script:
1. Beamforms N frames onto a real 3D POLAR (sector) volume -> LINEAR envelope.
2. Applies a slow-time high-pass (wall) filter to remove tissue clutter.
3. Integrates power Doppler (sum of squared envelope over slow time).
4. Displays a 5x2 montage: rows = clips, columns = x-z MIP and y-z MIP.

Grid: the acquisition is a matrix-probe DIVERGING wave, so a Cartesian box wastes
compute on the corners that fall outside the insonified cone. Instead we beamform
onto a 3D sector volume (radius x azimuth x elevation) whose apex is the virtual
source (|focus_distances|). Every voxel sits inside the cone, so the same coverage
needs far fewer voxels than a Cartesian box -> a faster beamform. It is still a
real 3D volume, so the MIPs project it (max over elevation for the x-z view, max
over azimuth for the y-z view). The volume's central planes match the two
perpendicular sectors produced by reconstruct.py.

Pipeline: a single pipeline in pipeline_PD_3d.yaml does everything end to end —
cast -> beamform -> envelope_detect -> tissue_highpass -> power_doppler. Steps 2-3
are CUSTOM zea ops (`tissue_highpass`, `power_doppler`) registered here with
`@ops_registry`; they act on the frame (slow-time) axis, so each clip's whole stack
is beamformed in one pass (zea patches the grid to bound memory). Custom
ops need no merged zea PR — they only need to be defined (imported) before a config
referencing them is loaded, which is why the YAML can list them by name.

Usage:
    KERAS_BACKEND=jax uv run --project /path/to/OpenH-RF python reconstruct_PD_3d.py
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zea
from keras import ops as kops
from zea import Config, File, Pipeline
from zea.internal.registry import ops_registry
from zea.ops import Operation

HERE = Path(__file__).parent
_HDF5 = "hf://nvidia/OpenH-RF/resolvestroke/clinical/SP02-Left-2/SP02-Left-2.hdf5"
DEFAULT_INPUT = "hf://nvidia/OpenH-RF/resolvestroke/clinical/SP02-Left-2/SP02-Left-2.hdf5"
CONFIG = HERE / "pipeline_PD_3d.yaml"

# Clip definitions (matching the combined file structure)
CLIPS = [
    "baseline_minus1s",
    "baseline_plus5s",
    "baseline_plus10s",
    "baseline_plus15s",
    "baseline_plus20s",
]
FRAMES_PER_CLIP = 4000
N_FRAMES_BF = 100  # frames to beamform per clip
TGC_DB_PER_CM = 1.5  # display-only linear TGC (dB gain per cm of depth)

# --- Inputs -----------------------------------------------------------------
# Defaults stream straight from the published corpus. Swap any of these for a
# local path to run against your own copy.
INPUT = "hf://nvidia/OpenH-RF/resolvestroke/clinical/SP02-Left-2/SP02-Left-2.hdf5"
N_FRAMES = N_FRAMES_BF
OUTPUT = None

# The 3D sector-volume grid (limits, depth, resolution) and the high-pass params
# all live in pipeline_PD_3d.yaml (`grid:` block and tissue_highpass params).


# --------------------------------------------------------------------------- #
# Custom pipeline operations (registered locally — no zea PR required).
# --------------------------------------------------------------------------- #
# These ops are defined here, not in zea: a pipeline.yaml naming them resolves
# only once this module is imported. See https://github.com/open-h/OpenH-RF
@ops_registry("tissue_highpass")
class TissueHighpass(Operation):
    """Slow-time high-pass (wall) filter to suppress stationary tissue clutter.

    Operates along the frame (slow-time) axis with a frequency-domain filter and a
    raised-cosine taper around the cutoff. Marked non-jittable because it uses the
    NumPy FFT with a data-dependent mask; with the default ``jit_options="ops"``
    the surrounding pipeline still runs the other ops jitted.
    """

    def __init__(
        self,
        cutoff_hz: float = 50.0,
        frame_rate_hz: float = 4000.0,
        transition_hz: float = 10.0,
        time_axis: int = 0,
        **kwargs,
    ):
        super().__init__(jittable=False, **kwargs)
        self.cutoff_hz = cutoff_hz
        self.frame_rate_hz = frame_rate_hz
        self.transition_hz = transition_hz
        self.time_axis = time_axis

    def call(self, **kwargs):
        data = np.asarray(kops.convert_to_numpy(kwargs[self.key]))
        n = data.shape[self.time_axis]
        freqs = np.fft.rfftfreq(n, d=1.0 / self.frame_rate_hz)  # non-negative
        mask = (freqs >= self.cutoff_hz).astype(np.float32)
        # Raised-cosine taper across [cutoff - transition, cutoff) to reduce ringing.
        taper = (freqs >= self.cutoff_hz - self.transition_hz) & (freqs < self.cutoff_hz)
        mask[taper] = 0.5 * (
            1.0 + np.cos(np.pi * (self.cutoff_hz - freqs[taper]) / self.transition_hz)
        )

        spectrum = np.fft.rfft(data, axis=self.time_axis)
        shape = [1] * data.ndim
        shape[self.time_axis] = len(mask)
        spectrum = spectrum * mask.reshape(shape)
        filtered = np.fft.irfft(spectrum, n=n, axis=self.time_axis).astype(np.float32)
        return {self.output_key: filtered}


@ops_registry("power_doppler")
class PowerDoppler(Operation):
    """Power-Doppler integration: sum of squared magnitude over the slow-time axis."""

    def __init__(self, time_axis: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.time_axis = time_axis

    def call(self, **kwargs):
        data = kwargs[self.key]
        return {self.output_key: kops.sum(kops.square(data), axis=self.time_axis)}


def reverse_tgc(raw, parameters):
    """Undo the hardware time-gain compensation baked into ``raw_data``.

    zea stores ``scan/tgc_gain_curve`` (n_ax,) as "the TGC applied to every sample;
    divide by this curve to undo it". A plain per-axial-sample divide here keeps the
    zea pipeline fully standard (no custom op needed).
    """
    tgc = np.asarray(parameters.tgc_gain_curve, dtype=np.float32)  # (n_ax,)
    return np.asarray(raw, dtype=np.float32) / tgc.reshape(1, 1, -1, 1, 1)


# --------------------------------------------------------------------------- #
# Sector-volume grid.
# --------------------------------------------------------------------------- #
def build_sector_volume_grid(azimuth_limits, elevation_limits, zlims, apex, n_r, n_az, n_el):
    """Build a real 3D sector-volume grid for a diverging wave.

    Bi-angular ("pyramidal") parametrization: a ray is steered by an azimuth
    angle alpha (x-z) and an elevation angle beta (y-z), and sampled at radius r
    from the apex at (0, 0, -apex). On the central planes this reduces to zea's
    2D polar fan (beta=0 -> x=r*sin(alpha), z=r*cos(alpha)-apex), so the volume's
    centre slices match reconstruct.py's perpendicular sectors.

    Returns:
        (grid, coords): grid has shape (n_r, n_az, n_el, 3) in Cartesian (x, y, z)
        metres; coords is (r, alpha, beta) 1D arrays for scan-conversion.
    """
    z0, z1 = float(zlims[0]), float(zlims[1])
    # Radius measured from the apex, so offset by apex -> on-axis depth is z0..z1.
    r = np.linspace(z0 + apex, z1 + apex, n_r)
    alpha = np.linspace(azimuth_limits[0], azimuth_limits[1], n_az)
    beta = np.linspace(elevation_limits[0], elevation_limits[1], n_el)

    R, A, B = np.meshgrid(r, alpha, beta, indexing="ij")  # (n_r, n_az, n_el)
    ta, tb = np.tan(A), np.tan(B)
    norm = np.sqrt(ta**2 + tb**2 + 1.0)
    x = R * ta / norm
    y = R * tb / norm
    z = R / norm - apex
    grid = np.stack((x, y, z), axis=-1).astype(np.float32)
    return grid, (r, alpha, beta)


def sector_plane_coords(r, angle, apex):
    """Cartesian (lat, z) mesh for a 2D sector (central-plane MIP), in mm."""
    R, T = np.meshgrid(r, angle, indexing="ij")
    lat = R * np.sin(T) * 1e3
    z = (R * np.cos(T) - apex) * 1e3
    return lat, z


def main():
    out_path = OUTPUT or Path(f"{Path(INPUT).stem}_PD_montage.png")

    n_bf = N_FRAMES
    zea.init_device()
    config = Config.from_path(str(CONFIG))
    g = config["grid"]  # 3D sector-volume spec
    azimuth_limits = tuple(float(v) for v in g["azimuth_limits"])
    elevation_limits = tuple(float(v) for v in g["elevation_limits"])
    zlims = tuple(float(v) for v in g["zlims"])

    # One pipeline, end to end: cast -> beamform -> envelope -> tissue_highpass ->
    # power_doppler. The custom slow-time ops resolve by name because they are
    # registered above at import time.
    pipeline = Pipeline.from_config(config)
    hp_cutoff = pipeline["tissue_highpass"].cutoff_hz

    with File(str(INPUT)) as f:
        parameters = f.load_parameters(**config.parameters)
        total_frames = f.data.raw_data.shape[0]

        # Apex = virtual source behind the array; from the config or |focus_distances|.
        apex = g.get("distance_to_apex")
        if apex is None:
            focus = float(np.abs(np.ravel(parameters.focus_distances)[0]))
            apex = focus if focus > 0 else 0.0

        grid, (r, alpha, beta) = build_sector_volume_grid(
            azimuth_limits,
            elevation_limits,
            zlims,
            apex,
            int(g["n_radial"]),
            int(g["n_azimuth"]),
            int(g["n_elevation"]),
        )
        flatgrid = grid.reshape(-1, 3)
        n_vox = flatgrid.shape[0]
        print(f"Total frames: {total_frames}, beamforming {n_bf} per clip")
        print(
            f"Sector volume: {grid.shape[:-1]} = {n_vox} voxels (polar, apex={apex * 1e3:.1f} mm)"
        )

        inputs = pipeline.prepare_parameters(parameters, grid=grid, flatgrid=flatgrid)

        # Process each clip: the whole frame stack goes through the pipeline in one
        # pass, so the slow-time ops see all frames.
        pd_volumes = []  # list of (n_r, n_az, n_el) power-Doppler volumes
        for clip_idx, clip_name in enumerate(CLIPS):
            clip_start = clip_idx * FRAMES_PER_CLIP
            # Take n_bf frames from the middle of each clip
            frame_offset = (FRAMES_PER_CLIP - n_bf) // 2
            start = clip_start + frame_offset
            end = start + n_bf

            print(f"\n  Clip '{clip_name}': frames {start}-{end}  (HP cutoff={hp_cutoff} Hz)")

            raw = f.data.raw_data[start:end]  # (n_bf, n_tx, n_ax, n_el, n_ch)
            raw = reverse_tgc(raw, parameters)  # undo hardware TGC before beamforming
            outputs = pipeline(**{pipeline.key: raw}, **inputs, return_numpy=True)
            pd = np.asarray(outputs[pipeline.output_key])  # (n_r, n_az, n_el)
            pd_volumes.append(pd)
            print(f"    PD volume: {pd.shape}, linear max={pd.max():.4g}")

    # Normalize all PD volumes to the same global max (common dB scale)
    global_max = max(pv.max() for pv in pd_volumes)
    pd_db_volumes = [
        10.0 * np.log10(np.maximum(pd, global_max * 1e-35) / global_max) for pd in pd_volumes
    ]

    # Linear TGC (display only): a dB gain growing linearly with axial depth,
    # broadcast over the sector's angular axes. On-axis depth per radial sample is
    # (r - apex); the gain is zero at the shallowest sample. Reveals deep flow that
    # attenuation (uncompensated after reverse_tgc) would otherwise bury.
    depth_mm = (r - apex) * 1e3  # (n_r,)
    gain_db = TGC_DB_PER_CM * (depth_mm - depth_mm.min()) / 10.0
    pd_db_volumes = [pd + gain_db[:, None, None] for pd in pd_db_volumes]

    # Scan-conversion coordinates for the two MIP planes (central plane of the fan)
    xz_lat, xz_z = sector_plane_coords(r, alpha, apex)  # (n_r, n_az)
    yz_lat, yz_z = sector_plane_coords(r, beta, apex)  # (n_r, n_el)

    vmin, vmax = -20.0, 0.0  # tight floor so vessels stand out from diffuse perfusion
    zea.visualize.set_mpl_style()
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))

    for col, (clip_name, pd_vol) in enumerate(zip(CLIPS, pd_db_volumes)):
        # x-z MIP: project out elevation (max over the y-fan)
        xz_mip = np.max(pd_vol, axis=2)  # (n_r, n_az)
        ax = axes[0, col]
        ax.pcolormesh(xz_lat, xz_z, xz_mip, cmap="hot", vmin=vmin, vmax=vmax, shading="auto")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(f"{clip_name}\nx-z MIP", fontsize=9)
        ax.set_xlabel("x [mm]")
        if col == 0:
            ax.set_ylabel("z [mm]")

        # y-z MIP: project out azimuth (max over the x-fan)
        yz_mip = np.max(pd_vol, axis=1)  # (n_r, n_el)
        ax = axes[1, col]
        ax.pcolormesh(yz_lat, yz_z, yz_mip, cmap="hot", vmin=vmin, vmax=vmax, shading="auto")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(f"{clip_name}\ny-z MIP", fontsize=9)
        ax.set_xlabel("y [mm]")
        if col == 0:
            ax.set_ylabel("z [mm]")

    fig.suptitle(
        f"Power Doppler 3D polar MIPs — {n_bf} frames/clip, HP cutoff {hp_cutoff} Hz",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved montage: {out_path}")


if __name__ == "__main__":
    main()
