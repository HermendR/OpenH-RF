---
pretty_name: "TU/e carotid 2023"
license: cc-by-4.0
zea_repo_id: zeahub/zea-carotid-2023
task_categories:
  - image-classification
tags:
  - ultrasound
  - rf
  - openh-rf
  - carotid
  - in-vivo
language:
  - en
---


# TU/e Carotid 2023

![Longitudinal view of a carotid bifurcation](assets/5_long_bifur_R_0000.gif)

One cardiac cycle of a longitudinal bifurcation scan,
[`data/5_long_bifur_R_0000.hdf5`](https://huggingface.co/datasets/nvidia/OpenH-RF/blob/main/tue-carotid/data/5_long_bifur_R_0000.hdf5).

`zea` renders it straight from the Hub with the
`pipeline.yaml` in this folder. Try it out with the following command:

```bash
zea process \
  --dataset hf://nvidia/OpenH-RF/tue-carotid/data/5_long_bifur_R_0000.hdf5 \
  --config hf://nvidia/OpenH-RF/tue-carotid/pipeline.yaml
```

## Dataset Description

The dataset includes carotid artery scans from 10 subjects.
Each file includes 100 frames, at least one cardiac cycle.
The acquisition scheme it consists of 128 line scanning interleaved with 21 plane waves.

Included views:

- Bifurcation cross section
- Longitudinal bifurcation
- Cross section 2cm from bifurcation
- Cross section 1cm from bifurcation
- Longitudinal section 2cm from bifurcation
- Longitudinal section 1cm from bifurcation
- Longitudinal section

The acquisitions were performed with a Verasonics 256.
The acquisitions were performed with a linear probe (Verasonics L11-5v).
The same operator performed all acquisitions.

## Dataset Contributors(s)

Wessel van Nierop <w.l.v.nierop@tue.nl>
Tristan Stevens
Oisín Nolan
Simon Penninga
Beatrice Federici
Vincent van der Schaft
Ben Luijten
Ruud van Sloun

## Dataset Creation Date

Nov 2023

## License / Terms of Use

CC BY 4.0

## Intended Usage

Image-quality / beamforming has both plane-wave (21) and focused acquisitions (128).

## Dataset Characterization

- Data Collection Method: In-vivo Human Data
- Labeling Method: acquisitions per view
- Acquisition system: see file

## Dataset Format

zea v1.3.0

## Dataset Quantification

**Current OpenH-RF release:** 80 HDF5 files; 435.40 GB (435,403,030,528 bytes) stored; root `zea_version` **0.1.6**. Sizes include all HDF5 contents and use decimal units (MB = 10^6 bytes, GB = 10^9 bytes, TB = 10^12 bytes), not decoded-array memory or original-source download sizes.

- Number of samples / frames / acquisitions
- Train / validation / test split (if applicable)
- Total size on disk
- Per-sample feature table (name, shape, dtype, units, description) — see NV-Raw2Insights-US for reference format

## Subject Metadata

None

## Data Validation

[reconstruct.py](./reconstruct.py)
[pipeline.yaml](./pipeline.yaml)

## Known Issues

None

## Ethical Considerations

Approval was obtained from the Ethical Review Board TU/e (Eindhoven University of Technology). See Approval Letter in this folder.

Reference: ERB2023EE7 Contact details for the Ethical Review Board TU/e: T +31 (0)40 247 6259 <ethics@tue.nl> <intranet.tue.nl/ethics>
