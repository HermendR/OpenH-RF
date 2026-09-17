# phantom_flow.hdf5 — structure & metadata

- **File:** `/media/m2_fast/data-openh-rf/phantom_flow/phantom_flow.hdf5`
- **Size:** 6.397 GB (6,397,249,242 bytes)
- **Root attributes:** `description`='Contrast-enhanced flow phantom acquisition (flow on/off). 5 clips of 1s (4000 frames at 4kHz) extracted at: baseline-1s, +5s, +10s, +15s, +20s relative to end_of_baseline. Source: Internal lab acquisition on flow phantom (CIRS 769 + ATS523A).'; `zea_version`='0.1.1'
- **Top-level groups:** `custom`, `metadata`, `metrics`, `probe`, `tracks`

## Tree

```
custom/
  computed_references/
    coordinates  (192, 171, 171, 3) float32
    mvi  (192, 171, 171) float32
    tube_mask  (192, 171, 171) uint8
    velocity_radial_avg  (192, 171, 171) float32
    velocity_radial_max  (192, 171, 171) float32
    velocity_radial_min  (192, 171, 171) float32
metadata/
  annotations/
    anatomy  () object
    label  (20000,) object
  credit  () object
  subject/
    id  () object
metrics/
probe/
  name  () object
  probe_geometry  (256, 3) float32
tracks/
  track_0/
    data/
      raw_data  (20000, 1, 320, 256, 2) int16
    scan/
      azimuth_angles  (1,) float32
      center_frequency  () float32
      demodulation_frequency  () float32
      focus_distances  (1,) float32
      initial_times  (1,) float32
      polar_angles  (1,) float32
      sampling_frequency  () float32
      sound_speed  () float32
      t0_delays  (1, 256) float32
      tgc_gain_curve  (320,) float32
      time_to_next_transmit  (20000, 1) float32
      transmit_origins  (1, 3) float32
      tx_apodizations  (1, 256) float32
```

## Group attributes

- `/custom/computed_references` — `description`='Computed reference maps on a shared 0.6 mm Cartesian grid (see coordinates). Derived from the SYLVER processed exam.'

## Datasets — shape, dtype, unit, description, value/stats

| Path | Shape | dtype | Unit | Description | Value / stats |
|---|---|---|---|---|---|
| `/custom/computed_references/coordinates` | `(192, 171, 171, 3)` | `float32` | m | Per-voxel Cartesian positions (x, y, z) in metres for all maps. | min=-0.051, max=0.120497, mean=0.0210657 |
| `/custom/computed_references/mvi` | `(192, 171, 171)` | `float32` | a.u. | Microvascular image (contrast/flow intensity, time-integrated). NaN outside sonified volume, 0 inside cone far from tubes, value within ~1 cm of a tube (smooth transition). | min=0, max=1.51211, mean=0.00350104, NaN=3259292 |
| `/custom/computed_references/tube_mask` | `(192, 171, 171)` | `uint8` | – | Ground-truth tube segmentation: 1 = tube (4 mm diameter), 2 = tube (2 mm diameter); 0 = background. | min=0, max=2, mean=0.00129117 |
| `/custom/computed_references/velocity_radial_avg` | `(192, 171, 171)` | `float32` | m/s | Mean radial (along-beam) flow velocity over slow-time; NaN where no flow. Radial component — divide by cos(Doppler angle ~72 deg) for true speed. | min=-1.44523, max=1.17046, mean=-0.152576, NaN=5571770 |
| `/custom/computed_references/velocity_radial_max` | `(192, 171, 171)` | `float32` | m/s | Maximum radial flow velocity over slow-time; NaN where no flow. | min=-1.44523, max=1.43931, mean=-0.0628551, NaN=5571770 |
| `/custom/computed_references/velocity_radial_min` | `(192, 171, 171)` | `float32` | m/s | Minimum radial flow velocity over slow-time; NaN where no flow. | min=-1.44523, max=0.301096, mean=-0.229416, NaN=5571770 |
| `/metadata/annotations/anatomy` | `()` | `object` | – | Anatomy label. | `'phantom'` |
| `/metadata/annotations/label` | `(20000,)` | `object` | – | Per-frame clip label (bolus time-offset relative to end-of-baseline). | counts: baseline_minus1s×4000, baseline_plus10s×4000, baseline_plus15s×4000, baseline_plus20s×4000, baseline_plus5s×4000 |
| `/metadata/credit` | `()` | `object` | – | Credit or attribution for the dataset. | `'Aitana Waelbroeck, Carl Ferlay, Arthur Chavignon, Maxence Reberol, Vincent Hingot - Resolve Stroke'` |
| `/metadata/subject/id` | `()` | `object` | – | Subject ID. Needed for subject-wise splits. | `'phantom_flow'` |
| `/probe/name` | `()` | `object` | – | Probe model name/identifier. | `'SN2652'` |
| `/probe/probe_geometry` | `(256, 3)` | `float32` | m | Element positions (x, y, z) per element, shape (n_el, 3). | min=-0.0082, max=0.0082, mean=0 |
| `/tracks/track_0/data/raw_data` | `(20000, 1, 320, 256, 2)` | `int16` | – | Raw channel data. | min=-32765, max=32766, mean=11.019 |
| `/tracks/track_0/scan/azimuth_angles` | `(1,)` | `float32` | rad | Azimuthal angles of transmit beams. | `[0.]` |
| `/tracks/track_0/scan/center_frequency` | `()` | `float32` | Hz | Center frequency of the transmit pulse. | `2031250.0` |
| `/tracks/track_0/scan/demodulation_frequency` | `()` | `float32` | Hz | Demodulation frequency. | `2031250.0` |
| `/tracks/track_0/scan/focus_distances` | `(1,)` | `float32` | m | Transmit focus distances. | `[-0.022]` |
| `/tracks/track_0/scan/initial_times` | `(1,)` | `float32` | s | A/D converter start times per transmit. | `[1.083077e-05]` |
| `/tracks/track_0/scan/polar_angles` | `(1,)` | `float32` | rad | Polar angles of transmit beams. | `[0.]` |
| `/tracks/track_0/scan/sampling_frequency` | `()` | `float32` | Hz | Sampling frequency. | `2031250.0` |
| `/tracks/track_0/scan/sound_speed` | `()` | `float32` | m/s | Speed of sound. | `1540.0` |
| `/tracks/track_0/scan/t0_delays` | `(1, 256)` | `float32` | s | Transmit delays per element. | min=0, max=1.76589e-06, mean=6.51021e-07 |
| `/tracks/track_0/scan/tgc_gain_curve` | `(320,)` | `float32` | – | Time-gain-compensation curve. | min=15.4516, max=501.187, mean=423.616 |
| `/tracks/track_0/scan/time_to_next_transmit` | `(20000, 1)` | `float32` | s | Time between transmit events. | min=0.00025, max=4, mean=0.00104995 |
| `/tracks/track_0/scan/transmit_origins` | `(1, 3)` | `float32` | m | Transmit beam origins (x, y, z). | `[0. 0. 0.]` |
| `/tracks/track_0/scan/tx_apodizations` | `(1, 256)` | `float32` | – | Transmit apodization per element. | min=1, max=1, mean=1 |
