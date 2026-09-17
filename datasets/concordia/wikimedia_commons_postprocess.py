# SPDX-License-Identifier: Apache-2.0
"""Post-process a photograph downloaded from Wikimedia Commons into a normalized,
high-contrast grayscale mask. Requires numpy and Pillow.
Usage:  python wikimedia_commons_postprocess.py <image-file>
"""

import os
import sys

import numpy as np
from PIL import Image

OUTPUT_H = 400
OUTPUT_W = 450

# Luma coefficients of the ITU-R BT.601 (NTSC) standard.
_LUMA = np.array([0.298936021293776, 0.587043074451121, 0.114020904255103])


def read_gray(path):
    """Read an image file as one float64 grayscale channel scaled to [0, 1]."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA"):
        im = im.convert(im.mode[:-1])
    arr = np.asarray(im)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr / np.iinfo(arr.dtype).max
    arr = arr.astype(np.float64)
    return arr[..., :3] @ _LUMA if arr.ndim == 3 else arr


def write_gray(img, path):
    """Write a float image, whose values are assumed to lie in [0, 1], as 8-bit."""
    arr = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path, format="PNG")


def _cubic(x):
    absx = np.abs(x)
    absx2 = absx * absx
    absx3 = absx2 * absx
    return np.multiply(1.5 * absx3 - 2.5 * absx2 + 1, absx <= 1) + np.multiply(
        -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2, (absx > 1) & (absx <= 2)
    )


def _contributions(in_length, out_length, scale):
    """Input sample indices and kernel weights for each output sample along one axis."""
    if scale < 1:
        kernel = lambda x: scale * _cubic(scale * x)
        kernel_width = 4.0 / scale
    else:
        kernel = _cubic
        kernel_width = 4.0

    centres = np.arange(1, out_length + 1) / scale + 0.5 * (1 - 1 / scale)
    taps = int(np.ceil(kernel_width)) + 2
    indices = np.floor(centres - kernel_width / 2).astype(np.int64)[:, None] + np.arange(taps) - 1

    weights = kernel(centres[:, None] - indices - 1)
    weights /= weights.sum(axis=1, keepdims=True)

    mirror = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, -1)))
    return weights, mirror[indices % mirror.size]


def _resize_axis0(img, weights, indices):
    return np.sum(weights[:, :, None] * img[indices], axis=1)


# We implemented the resampling here rather than using Pillow's built-in resize: the
# original post-processing was written in MATLAB, and we wanted to ship it in Python
# while reproducing the same output bit-for-bit.


def imresize(img, output_shape):
    """Resample a 2-D image to output_shape = (height, width)."""
    scale = [output_shape[axis] / img.shape[axis] for axis in (0, 1)]
    out = img
    for axis in np.argsort(scale):
        weights, indices = _contributions(img.shape[axis], output_shape[axis], scale[axis])
        if axis == 0:
            out = _resize_axis0(out, weights, indices)
        else:
            out = _resize_axis0(out.T, weights, indices).T
    return out


def histeq(img, levels=64, bins=256):
    """Equalize the histogram of a float image in [0, 1] onto `levels` output levels."""
    counts, _ = np.histogram(img, bins=bins, range=(0.0, 1.0))
    cdf = np.cumsum(counts) / counts.sum()
    target_cdf = np.arange(1, levels + 1) / levels

    nearest = np.argmin(np.abs(cdf[:, None] - target_cdf[None, :]), axis=1)
    transform = nearest / (levels - 1)
    return transform[np.clip((img * bins).astype(np.int64), 0, bins - 1)]


def process_image(input_path):
    """Convert the image at input_path into a normalized grayscale mask and write it to disk.

    The image is resampled to a fixed 400 x 450 canvas, histogram-equalized
    to spread its tonal range, and linearly rescaled to [0, 1]. The extremes
    are then saturated -- values above 0.9 become pure white (1) and values
    below 0.1 become pure black (0) -- leaving the intermediate 0.1-0.9 band
    as a continuous gradient. The result is written as "<input_path>_processed.png".

    Returns the output path.
    """
    image = read_gray(input_path)
    image = imresize(image, (OUTPUT_H, OUTPUT_W))
    image = histeq(image)
    image = image - image.min()
    image /= image.max()
    image[image > 0.9] = 1
    image[image < 0.1] = 0

    output_path = f"{os.path.splitext(input_path)[0]}_processed.png"
    write_gray(image, output_path)
    return output_path


if __name__ == "__main__":
    print(process_image(sys.argv[1]))
