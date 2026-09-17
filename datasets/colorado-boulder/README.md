---
pretty_name: "OpenH-RF — Tracked Swept Synthetic Aperture 3D Phantom Dataset"
license: cc-by-4.0
task_categories:
  - generalized-reconstruction
tags:
  - ultrasound
  - rf
  - openh-rf
  - 3d
language:
  - en
size_categories:
  - n<1K
---

# Tracked Swept Synthetic Aperture 3D Phantom Ultrasound Dataset

## Dataset Description

This dataset contains tracked swept synthetic aperture (SSA) ultrasound acquisitions of a 3D ultrasound imaging phantom. The data were acquired using a Verasonics Vantage research ultrasound system with a P4-2 phased array transducer.

The dataset includes raw RF channel data, acquisition parameters, probe geometry, transmit information, and frame-wise tracked probe pose metadata. Its purpose is to provide a reproducible example of motion-compensated SSA reconstruction from raw channel data using the zea/OpenH-RF data format.

This dataset contains phantom data only. It does not contain human subject data, animal data, or protected health information (PHI).

## Dataset Contributor(s)

**Contributing organization:** University of Colorado Boulder, Bottenus Lab

**Contributors:**

- Anet Sanchez
- Nick Bottenus

## Dataset Creation Date

06/24/2025

## License / Terms of Use

This dataset is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

The contributed data consist exclusively of phantom ultrasound acquisitions and are cleared for release under CC BY 4.0. Patient consent, clinical data-use agreements, and PHI de-identification are not applicable because the dataset does not contain human subject data.

## Intended Usage

This dataset is intended for research on generalized ultrasound reconstruction, with a particular focus on tracked swept synthetic aperture imaging, motion-compensated beamforming, coherent compounding, and ultrasound image-quality evaluation.

For SSA reconstruction, each raw RF frame is beamformed using its corresponding tracked transducer pose. The resulting beamformed IQ frames are placed on a common reconstruction grid and coherently summed to synthesize a larger effective aperture. Because the reconstruction relies on coherent compounding, summation is performed before envelope detection, normalization, and log compression.

## Dataset Characterization

- **Data Collection Method:** Phantom ultrasound acquisition
- **Labeling Method:** N/A; no manual labels or segmentation masks are provided
- **Acquisition System:** Verasonics Vantage research ultrasound scanner with a Verasonics P4-2 phased array transducer

### Acquisition Details

The transducer was manually swept over the phantom field of view while diverging-wave transmissions were acquired at 400 Hz. All 64 array elements were used on receive.

Diverging waves were generated using a negative virtual source with the 20 central array elements active on transmit.

The transducer was optically tracked using an NDI Polaris Vega® XT optical tracking system manufactured by Northern Digital Inc., Ontario, Canada.

### Probe and Geometry

The acquisition used a Verasonics P4-2 phased array transducer with 64 elements. The center frequency stored in the acquisition and used for reconstruction is 2.5 MHz.

The probe geometry, transmit origins, transmit delays, transmit apodization, and other acquisition parameters are stored in the zea/OpenH-RF file. Frame-wise probe translations and rotations are stored using the native `metadata/probe_pose` structure.

## Dataset Format

The dataset is distributed in the zea/OpenH-RF HDF5 format.

Each file includes:

- Raw RF channel data
- Sampling and center frequencies
- Transmit delays and apodization
- Transmit origins
- Probe geometry
- Sound-speed information
- Frame-wise tracked probe translations
- Frame-wise tracked probe rotations

The stored RF channel data have not been beamformed, envelope detected, normalized, or log compressed. The accompanying reconstruction pipeline performs these processing steps.

## Dataset Quantification

**Current OpenH-RF release:** 62 HDF5 files; 16.04 GB (16,037,117,952 bytes) stored; root `zea_version` **0.1.6**. Sizes include all HDF5 contents and use decimal units (MB = 10^6 bytes, GB = 10^9 bytes, TB = 10^12 bytes), not decoded-array memory or original-source download sizes.

- **Number of phantom objects:** 1
- **Number of acquisitions:** 10
- **Number of RF frames per acquisition:** 1200
- **Number of transmit events per frame:** 1
- **Number of receive elements:** 64
- **Number of active transmit elements:** 20

### Per-File Feature Summary

| Feature | Shape | Data type | Units | Description |
|---|---:|---|---|---|
| Raw RF data | `n_frames × n_tx × n_ax × n_elements × n_channels` | `float32` | acquisition units | Raw RF channel measurements |
| Probe translation | `n_frames × 3` | `float32` | m | Frame-wise tracked probe position |
| Probe rotation | `n_frames × 4` | `float32` | unit quaternion | Frame-wise tracked probe orientation in `xyzw` order |
| Probe geometry | `64 × 3` | `float32` | m | Array-element coordinates |
| Transmit origins | `n_tx × 3` | `float32` | m | Diverging-wave virtual-source coordinates |
| Transmit delays | `n_tx × 64` | `float32` | s | Per-element transmit delays |
| Transmit apodization | `n_tx × 64` | `float32` | unitless | Per-element transmit activation and weighting |

## Subject Metadata

### Metadata Schema Migration

The zea 0.1.6 migration uses these approved metadata locations:

| Legacy location | Canonical location |
|---|---|
| Dataset `metadata/subject_id` | Dataset `metadata/subject/id` |
| Dataset `metadata/subject_type` | Dataset `metadata/subject/type` |
| Dataset `metadata/us_machine` | Root HDF5 attribute `us_machine` |

Read the machine name with `f.attrs["us_machine"]`, not `f["us_machine"]`.
Subject values and their existing attributes are preserved. The machine
string is preserved; migration stops for review if its legacy dataset has
attributes that cannot be represented without loss. No numerical arrays are
rescaled or otherwise changed by these relocations.

The three-field pilot passed full array and metadata parity checks with the
approved description changes and `transmit_only=False` default. Full-release
migration is still pending. Replacement files are uploaded only after
per-file validation; readers supporting both revisions should check the
canonical locations first, then the legacy locations.


This dataset contains one 3D ultrasound imaging phantom.

- **Subject type:** 3D phantom
- **Anatomical region:** Not applicable
- **Human participants:** None
- **Animal subjects:** None
- **Protected health information:** None
- **Scanner:** Verasonics Vantage
- **Probe:** Verasonics P4-2 phased array

## Data Validation

The submission includes a `zea.Pipeline` that reconstructs a representative tracked SSA B-mode image from the raw RF channel data.

The pipeline performs:

1. Frame-wise demodulation
2. Application of the tracked probe pose
3. Delay-and-sum beamforming onto a common reconstruction grid
4. Coherent summation of the beamformed IQ frames
5. Envelope detection
6. Normalization
7. Log compression

The reconstruction is defined in `pipeline.yaml` and executed using `reconstruct.py`. A representative reconstructed B-mode image is included with the dataset.

## Known Issues

- Optical tracking measurements may contain small position and orientation uncertainties.
- Reconstruction quality depends on tracking calibration accuracy and coherent alignment between frames.


## Ethical Considerations

This dataset contains phantom ultrasound data only. It does not contain human participants, animal subjects, personal identifiers, clinical records, or protected health information.

Human-subject consent and institutional review board approval are therefore not applicable.

