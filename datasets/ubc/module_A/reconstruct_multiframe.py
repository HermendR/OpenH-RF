#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export one Module A temporal sequence as MP4 and NIfTI.

Every frame is reconstructed from zea ``raw_data`` with the same scanline DAS
pipeline embedded in ``reconstruct.py``.  Stored B-mode images and preserved
source line RF are never read.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import struct
import tempfile
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "module_a_ubc_v2_mpl"))
os.environ.setdefault("ZEA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "module_a_ubc_v2_zea"))

import imageio.v2 as imageio
import numpy as np
import zea
from PIL import Image, ImageDraw, ImageFont
from reconstruct import (
    DISPLAY_FLOOR_DB,
    DISTANCE_TO_APEX_M,
    LINE_DEPTH_MAX_M,
    LINE_DEPTH_MIN_M,
    PARAMETERS,
    build_pipeline,
)
from zea import File, Pipeline

HERE = Path(__file__).resolve().parent
FRAME_INTERVAL_S = 1.0 / 250.0
SCAN_RESOLUTION_M = 0.4e-3
FRAMES_PER_SEQUENCE = 25

# --- Inputs ---------------------------------------------------------------
# Defaults stream straight from the published corpus. Swap any of these for a
# local path to run against your own copy.
ACQUISITIONS = "hf://nvidia/OpenH-RF/ubc/module_A/acquisitions"


def find_sequence(acquisitions: str, case_id: str, plane: int) -> list[str]:
    """Return the ordered f00-f24 sequence for one case and motor plane.

    Every acquisition is named from its case, plane and frame index, so the
    sequence is built rather than listed. That keeps it working when
    ``acquisitions`` is an ``hf://`` prefix instead of a local directory.
    """
    return [
        f"{acquisitions}/case_{case_id}/ubc_swave_cirs_{case_id}_p{plane:02d}_f{frame:02d}.hdf5"
        for frame in range(FRAMES_PER_SEQUENCE)
    ]


def reconstruct_one(
    path: str,
    pipeline: Pipeline,
) -> tuple[np.ndarray, object]:
    """Run the submitted zea pipeline on one one-frame HDF5 file."""

    with File(str(path)) as file:
        parameters = file.load_parameters(**PARAMETERS)
        raw_data = file.data.raw_data[0:1]

    if raw_data.ndim != 5 or raw_data.shape[-1] != 1:
        raise ValueError(f"Unexpected raw_data shape in {Path(path).name}: {raw_data.shape}")
    if raw_data.shape[1] != len(parameters.polar_angles):
        raise ValueError(f"Transmit/polar-angle mismatch in {Path(path).name}")

    mask = np.asarray(parameters.flat_aligned_apodization)
    if mask.shape[1] != raw_data.shape[1]:
        raise ValueError(f"Aligned transmit-mask mismatch in {Path(path).name}")
    if not np.all(np.count_nonzero(mask, axis=1) == 1):
        raise ValueError(f"Scanline ownership is not one-hot in {Path(path).name}")

    inputs = pipeline.prepare_parameters(parameters)
    outputs = pipeline(
        **{pipeline.key: raw_data},
        **inputs,
        return_numpy=True,
    )
    bmode = np.squeeze(np.asarray(outputs[pipeline.output_key])[0]).astype(np.float32)
    if bmode.ndim != 2 or not np.isfinite(bmode).all():
        raise ValueError(f"Invalid polar B-mode in {Path(path).name}: {bmode.shape}")
    return bmode, parameters


def scan_convert(
    bmode: np.ndarray,
    parameters: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert polar scanlines to a Cartesian sector and x/z axes in mm."""

    polar_angles = np.asarray(parameters.polar_angles, dtype=np.float64)
    sector, scan_parameters = zea.display.scan_convert_2d(
        bmode,
        rho_range=(
            DISTANCE_TO_APEX_M + LINE_DEPTH_MIN_M,
            DISTANCE_TO_APEX_M + LINE_DEPTH_MAX_M,
        ),
        theta_range=(float(polar_angles.min()), float(polar_angles.max())),
        resolution=SCAN_RESOLUTION_M,
        fill_value=np.nan,
        distance_to_apex=0.0,
    )
    sector = np.asarray(sector, dtype=np.float32)
    x_limits_mm = np.asarray(scan_parameters["x_lim"], dtype=np.float64) * 1e3
    z_limits_mm = (
        np.asarray(scan_parameters["z_lim"], dtype=np.float64) - DISTANCE_TO_APEX_M
    ) * 1e3
    x_mm = np.linspace(x_limits_mm[0], x_limits_mm[1], sector.shape[1])
    z_mm = np.linspace(z_limits_mm[0], z_limits_mm[1], sector.shape[0])
    return sector, x_mm, z_mm


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def display_frame(
    sector: np.ndarray,
    case_id: str,
    plane: int,
    frame_index: int,
) -> np.ndarray:
    """Render a fixed-scale labeled RGB video frame."""

    valid = np.isfinite(sector)
    scaled = np.full(sector.shape, 255, dtype=np.uint8)
    scaled[valid] = np.rint(
        np.clip(
            (sector[valid] - DISPLAY_FLOOR_DB) / (0.0 - DISPLAY_FLOOR_DB),
            0.0,
            1.0,
        )
        * 255.0
    ).astype(np.uint8)

    source = Image.fromarray(scaled, mode="L").convert("RGB")
    source = source.resize((source.width * 2, source.height * 2), Image.Resampling.BILINEAR)
    # 528 image rows + 64 title rows = 592, divisible by 16 for H.264.
    title_height = 64
    canvas = Image.new("RGB", (source.width, source.height + title_height), "white")
    canvas.paste(source, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, title_height), fill=(18, 18, 18))
    source_time_ms = frame_index * FRAME_INTERVAL_S * 1e3
    label = (
        f"Module A | {case_id} kPa | plane {plane:02d} | "
        f"frame f{frame_index:02d} | source t={source_time_ms:.0f} ms"
    )
    draw.text((16, 18), label, fill="white", font=_load_font(24))
    return np.asarray(canvas)


def save_nifti(
    path: Path,
    sectors_zx: np.ndarray,
    x_mm: np.ndarray,
    z_mm: np.ndarray,
) -> None:
    """Write a dependency-free NIfTI-1 file with axes (x, z, 1, time)."""

    if sectors_zx.ndim != 3:
        raise ValueError(f"Expected (time,z,x), got {sectors_zx.shape}")
    data = np.transpose(sectors_zx, (2, 1, 0))[:, :, np.newaxis, :]
    data = np.nan_to_num(data, nan=DISPLAY_FLOOR_DB).astype("<f4", copy=False)
    nx, nz, ny, nt = data.shape
    dx = float(abs(x_mm[1] - x_mm[0]))
    dz = float(abs(z_mm[1] - z_mm[0]))

    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 4, nx, nz, ny, nt, 1, 1, 1)
    struct.pack_into("<h", header, 70, 16)  # NIFTI_TYPE_FLOAT32
    struct.pack_into("<h", header, 72, 32)
    struct.pack_into("<8f", header, 76, 1.0, dx, dz, 1.0, FRAME_INTERVAL_S, 1.0, 1.0, 1.0)
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<f", header, 112, 1.0)
    struct.pack_into("<B", header, 123, 2 | 8)  # millimetres and seconds
    struct.pack_into("<f", header, 124, 0.0)
    struct.pack_into("<f", header, 128, DISPLAY_FLOOR_DB)
    description = b"zea raw-only scanline DAS; normalized log envelope dB"
    header[148 : 148 + len(description)] = description
    struct.pack_into("<h", header, 252, 0)  # qform_code
    struct.pack_into("<h", header, 254, 1)  # sform_code
    struct.pack_into("<4f", header, 280, dx, 0.0, 0.0, float(x_mm[0]))
    struct.pack_into("<4f", header, 296, 0.0, dz, 0.0, float(z_mm[0]))
    struct.pack_into("<4f", header, 312, 0.0, 0.0, 1.0, 0.0)
    header[344:348] = b"n+1\0"

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(path), mode="wb", compresslevel=6, mtime=0) as stream:
        stream.write(header)
        stream.write(b"\0\0\0\0")
        stream.write(data.tobytes(order="F"))


def save_contact_sheet(path: Path, rgb_frames: list[np.ndarray]) -> None:
    """Save nine distributed frames for quick static inspection."""

    chosen = [0, 3, 6, 9, 12, 15, 18, 21, 24]
    thumbs: list[Image.Image] = []
    for index in chosen:
        image = Image.fromarray(rgb_frames[index])
        image.thumbnail((488, 296), Image.Resampling.LANCZOS)
        thumbs.append(image)
    sheet = Image.new("RGB", (488 * 3, 296 * 3), "white")
    for item, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((item % 3) * 488, (item // 3) * 296))
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisitions", default=ACQUISITIONS)
    parser.add_argument("--case-id", default="1.83")
    parser.add_argument("--plane", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    files = find_sequence(args.acquisitions, args.case_id, args.plane)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"A_bmode_reconstructed_case{args.case_id}_p{args.plane:02d}_f00-f24"
    mp4_path = args.output_dir / f"{stem}.mp4"
    nifti_path = args.output_dir / f"{stem}.nii.gz"
    sidecar_path = args.output_dir / f"{stem}.json"
    metrics_path = args.output_dir / f"{stem}_frame_metrics.csv"
    contact_path = args.output_dir / f"{stem}_contact_sheet.png"

    zea.init_device(device=args.device, verbose=False)
    pipeline = build_pipeline()
    sectors: list[np.ndarray] = []
    rgb_frames: list[np.ndarray] = []
    metrics: list[dict[str, object]] = []
    reference_x: np.ndarray | None = None
    reference_z: np.ndarray | None = None

    print(f"Reconstructing {len(files)} frames from zea raw_data only")
    for index, path in enumerate(files):
        bmode, parameters = reconstruct_one(path, pipeline)
        sector, x_mm, z_mm = scan_convert(bmode, parameters)
        if reference_x is None:
            reference_x, reference_z = x_mm, z_mm
        elif not np.allclose(x_mm, reference_x) or not np.allclose(z_mm, reference_z):
            raise ValueError(f"Scan-conversion grid changed at {Path(path).name}")
        sectors.append(sector)
        rgb_frames.append(display_frame(sector, args.case_id, args.plane, index))
        valid = sector[np.isfinite(sector)]
        metrics.append(
            {
                "frame": index,
                "source_time_ms": index * FRAME_INTERVAL_S * 1e3,
                "minimum_db": float(valid.min()),
                "maximum_db": float(valid.max()),
                "mean_db": float(valid.mean()),
                "finite_pixels": int(valid.size),
            }
        )
        print(f"  f{index:02d}: polar={bmode.shape}, sector={sector.shape}")

    if reference_x is None or reference_z is None:
        raise RuntimeError("No frames reconstructed")
    sector_stack = np.stack(sectors, axis=0)

    with imageio.get_writer(
        mp4_path,
        fps=args.fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        ffmpeg_log_level="error",
        output_params=["-movflags", "+faststart"],
    ) as writer:
        for frame in rgb_frames:
            writer.append_data(frame)

    save_nifti(nifti_path, sector_stack, reference_x, reference_z)
    save_contact_sheet(contact_path, rgb_frames)
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    dx_mm = float(abs(reference_x[1] - reference_x[0]))
    dz_mm = float(abs(reference_z[1] - reference_z[0]))
    sidecar = {
        "source": "zea raw_data only; stored image and source line RF were not loaded",
        "reconstruction": "reconstruct.py scanline delay-and-sum pipeline",
        "case_id_kpa": args.case_id,
        "motor_plane": args.plane,
        "temporal_indices": list(range(25)),
        "source_frame_interval_seconds": FRAME_INTERVAL_S,
        "video_playback_fps": args.fps,
        "video_dynamic_range_db": [DISPLAY_FLOOR_DB, 0.0],
        "normalization": "independent 99.5th-percentile normalization per frame",
        "nifti_shape": [int(sector_stack.shape[2]), int(sector_stack.shape[1]), 1, 25],
        "nifti_axis_order": [
            "x",
            "z_from_probe_surface",
            "elevation_singleton",
            "time",
        ],
        "nifti_units": {"space": "mm", "time": "s", "values": "dB"},
        "voxel_spacing": [dx_mm, dz_mm, 1.0, FRAME_INTERVAL_S],
        "x_limits_mm": [float(reference_x[0]), float(reference_x[-1])],
        "z_limits_mm": [float(reference_z[0]), float(reference_z[-1])],
        "outside_sector_value_in_nifti_db": DISPLAY_FLOOR_DB,
        "files": [Path(path).name for path in files],
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    print(f"Saved MP4    : {mp4_path}")
    print(f"Saved NIfTI  : {nifti_path}")
    print(f"Saved sidecar: {sidecar_path}")
    print(f"Saved contact: {contact_path}")


if __name__ == "__main__":
    main()
