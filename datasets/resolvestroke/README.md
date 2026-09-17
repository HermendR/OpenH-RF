# OpenH-RF - Resolve Stroke datasets

[Resolve Stroke](https://www.resolvestroke.com/) develops SYLVER, a
software-driven ultrasound platform, that supports clinical assessment in
patients suspected of, or at risk for, cerebral perfusion abnormalities by
improving visualization of cerebral vasculature and providing complementary
information on cerebral perfusion.
SYLVER images with a 32×32 matrix array probe at 2 MHz
using diverging-wave transmits. CE and FDA clearances are expected in 2026.
This directory holds the data Resolve Stroke contributes to OpenH-RF: pre-beamformed
RF/IQ channel data from an imaging phantom, a flow phantom, and in-vivo transcranial
acquisitions, all de-identified and released under CC BY 4.0. The in-vivo data comes
from the CPP-approved SCULPT clinical study. Two transmit sequences are included: a
saddle sequence that images a 2D plane for real-time B-mode, and a 4 kHz
single-aperture volume sequence for CEUS and blood-flow measurement.

Each dataset directory is a self-contained data card: an HDF5 file, a
B-mode preview PNG, a `reconstruct.py` + `pipeline.yaml` beamforming recipe, a
`README.md`, and a `LICENCE` (CC BY 4.0).

## OpenH-RF Release Inventory

**Current OpenH-RF release:** 43 HDF5 files; 110.62 GB (110,619,394,048 bytes) stored; root `zea_version` **0.1.6**. Sizes include all HDF5 contents and use decimal units (MB = 10^6 bytes, GB = 10^9 bytes, TB = 10^12 bytes), not decoded-array memory or original-source download sizes.

## Contents

| Path | What's inside |
|------|---------------|
| `clinical/` | 20 clinical transcranial CEUS acquisitions (SCULPT study, subjects `SP01`–`SP10`, Left/Right). Each subdirectory holds one acquisition's HDF5 (five 1 s bolus wash-in clips), reconstruction scripts, and B-mode / power-Doppler / reference montages. |
| `phantom_flow/` | Flow-phantom CEUS (CIRS 769 + ATS523A) with microbubble contrast: five 1 s clips capturing a flow-on/flow-off bolus wash-in. Single combined HDF5 + 3D power-Doppler reconstruction. |
| `phantom_mp/` | Multi-tissue imaging phantom (CIRS 040GSE), pre-beamformed channel data, 1000 volumetric frames of the same static scene (wire targets, cysts, tissue-mimicking background). |
| `saddle/` | Single-frame anatomical reference B-modes (21 files: the 20 clinical acquisitions + 1 imaging phantom) from the wide-angle "saddle" sequence. These are the structural companion to the clinical CEUS clips. |

## Contributors

Aitana Waelbroeck\*, Carl Ferlay\*, Arthur Chavignon\*, Maxence Reberol\*, Vincent Hingot\*

_\*Resolve Stroke, 29 Rue du Faubourg Saint-Jacques, 75014 Paris_


