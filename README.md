# Roman WFI Sparse Ubercalibration Amp Prototype

This repository contains a first toy model for Roman Space Telescope WFI-style
relative photometric calibration with per-amplifier terms.

The fitted model is

```text
m_obs = M_star[star_id]
      + ZP[exposure_id]
      + S_smooth(x, y)
      + A_amp[detector_id, amp_id]
      + noise
```

where `S_smooth` is a low-order detector-coordinate polynomial with basis
`x, y, x^2, x*y, y^2`, and each amplifier is modeled as a vertical stripe.

## Files

- `roman_ubercal_amp_generate_data.py`: generates simulated repeated stellar
  photometry with translational dithers and exposure rotations.
- `roman_ubercal_amp_prototype.py`: reads the saved simulated table and solves
  the sparse weighted least-squares calibration problem.
- `outputs_amp_prototype/simulated_observations.csv`: observation table written
  by the generator.
- `outputs_amp_prototype/truth.npz`: truth arrays used for diagnostics.
- `outputs_amp_prototype/metadata.json`: geometry and simulation settings.
- `outputs_amp_prototype/*.png`: diagnostic plots written by the fitter.

## Run

Use a Python environment with `numpy`, `scipy`, and `matplotlib`. On this
machine, the Anaconda interpreter works:

```bash
/opt/anaconda3/bin/python roman_ubercal_amp_generate_data.py
/opt/anaconda3/bin/python roman_ubercal_amp_prototype.py
```

The fitter expects the generated CSV, truth file, and metadata file to exist in
`outputs_amp_prototype/`.

## Simulated Observation Table

The CSV has one row per stellar observation:

```text
obs_id
star_id
exposure_id
detector_id
amp_id
x_pixel
y_pixel
instrumental_mag
instrumental_mag_uncertainty
```

`star_id` is stable across exposures. The fitter maps the unique IDs in the
table onto compact sparse-matrix parameter columns.

## Geometry

The default detector geometry is

```text
N_DET = 1
NX = 4096
NY = 4096
N_AMP = 32
AMP_WIDTH = NX // N_AMP = 128 columns
```

Amplifiers are vertical stripes:

```text
amp_id = floor(x_pixel / AMP_WIDTH)
```

with clipping to `[0, 31]`.

## Degeneracies

This is a relative calibration problem. The fitter handles the main gauge
freedoms by:

- fixing exposure 0 to `ZP = 0` by omitting its parameter,
- omitting the constant term from the smooth polynomial,
- constraining the mean amplifier offset per detector to zero,
- adding weak Gaussian priors on individual amplifier offsets.

Pure translation dithers leave detector-plane linear smooth gradients degenerate
with fitted star magnitudes and exposure zeropoints. The generator therefore
uses multiple exposure rotation angles by default: `0`, `-5`, and `+5` degrees.
Those rotations break the linear-gradient degeneracy in the simulated data, so
the fitter does not use explicit priors on the smooth `x` or `y` coefficients.

## Default Verification

A default run currently produces about `155,764` observations and `5,076` fitted
parameters. Typical diagnostics are:

```text
RMS residual: 0.004906 mag
Exposure ZP RMS error: 0.000099 mag
Smooth field RMS error, mean removed: 0.000318 mag
Amplifier offset RMS error, per-detector means removed: 0.000466 mag
```

