# Roman WFI Calibration Prototypes

This repository contains small, self-contained Python prototypes for Roman Space
Telescope WFI-style calibration experiments.

- The first prototype fits scalar relative photometric calibration terms:
  stellar magnitudes, exposure zeropoints, smooth star-flat structure, and
  per-amplifier offsets.
- The second prototype fits chromatic calibration terms after scalar
  instrumental calibration: stellar SED parameters, detector-level passband
  shifts/width changes, and an ice optical-depth spectral shape.

Both prototypes use deterministic simulations, sparse linear algebra, and CSV
artifacts intended to be easy to inspect.

## Dependencies

Use a Python environment with `numpy`, `pandas`, `scipy`, and `matplotlib`. On
this machine, the Anaconda interpreter works:

```bash
/opt/anaconda3/bin/python --version
```

## Scalar Amp Ubercal Prototype

This toy model simulates repeated stellar photometry and solves for
per-amplifier sensitivity terms.

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

### Files

- `roman_ubercal_amp_generate_data.py`: generates simulated repeated stellar
  photometry with translational dithers and exposure rotations.
- `roman_ubercal_amp_prototype.py`: reads the saved simulated table and solves
  the sparse weighted least-squares calibration problem.
- `outputs_amp_prototype/simulated_observations.csv`: observation table written
  by the generator.
- `outputs_amp_prototype/truth.npz`: truth arrays used for diagnostics.
- `outputs_amp_prototype/metadata.json`: geometry and simulation settings.
- `outputs_amp_prototype/*.png`: diagnostic plots written by the fitter.

### Run

```bash
/opt/anaconda3/bin/python roman_ubercal_amp_generate_data.py
/opt/anaconda3/bin/python roman_ubercal_amp_prototype.py
```

The fitter expects the generated CSV, truth file, and metadata file to exist in
`outputs_amp_prototype/`.

### Simulated Observation Table

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

`star_id` is stable across exposures. `detector_id` uses NASA-style one-based
numbering, so the default detector IDs are `1..18`. The fitter maps the unique
star and detector IDs in the table onto compact sparse-matrix parameter columns.

### Geometry

The default detector geometry is

```text
N_DET = 18
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

The default `5,000` stars are distributed approximately evenly across the 18
detectors, and each detector has its own independent set of 32 amplifier
offsets. The simulated observation table numbers those detectors `1..18`.

### Degeneracies

This is a relative calibration problem. The fitter handles the main gauge
freedoms by:

- fixing exposure 0 to `ZP = 0` by omitting its parameter,
- omitting the constant term from the smooth polynomial,
- constraining the mean amplifier offset per detector to zero,
- adding weak Gaussian priors on individual amplifier offsets, with default
  width `sigma_amp_prior = 0.02 mag`.

Pure translation dithers leave detector-plane linear smooth gradients degenerate
with fitted star magnitudes and exposure zeropoints. The generator therefore
uses multiple exposure rotation angles by default: `0`, `-5`, and `+5` degrees.
Those rotations break the linear-gradient degeneracy in the simulated data, so
the fitter does not use explicit priors on the smooth `x` or `y` coefficients.

### Default Verification

A default run currently produces about `164,593` observations and `5,620` fitted
parameters. Typical diagnostics are:

```text
RMS residual: 0.004899 mag
Exposure ZP RMS error: 0.000139 mag
Smooth field RMS error, mean removed: 0.000398 mag
Amplifier offset RMS error, per-detector means removed: 0.000432 mag
```

## Passband And Ice Prototype

This prototype assumes the scalar instrumental calibration above has already
removed detector/amplifier throughput terms. It simulates flattened stellar
photometry and fits chromatic calibration parameters.

The broadband flux model is evaluated with linear flux integrals:

```text
F_pred = integral f_s(lambda) * T_b(lambda, d, x, y, e) d_lambda
```

Throughput perturbations are modeled additively in log-throughput /
optical-depth space:

```text
ln T_b(lambda, d, x, y, e)
  = ln T0_b(lambda)
    + delta_lambda_b,d * phi_shift_b(lambda)
    + width_b,d        * phi_width_b(lambda)
    - ice_amount_obs   * tau_ice(lambda)
```

This lets small multiplicative passband/ice changes enter linearly, while the
actual synthetic observations and all response coefficients still come from
linear flux integrals over wavelength.

### Files

- `simulate_roman_passband_data.py`: generates toy broad-filter photometry,
  nominal passbands, passband modes, ice basis functions, and truth files.
- `fit_roman_passband_model.py`: reads the simulator products and runs an
  iterative sparse linearized fit.
- `passband_sim_outputs/measurements.csv`: simulated flattened stellar
  photometry.
- `passband_sim_outputs/nominal_passbands.csv`: long-form nominal passbands.
- `passband_sim_outputs/passband_modes.csv`: shift and width log-throughput
  modes.
- `passband_sim_outputs/ice_basis.csv`: optical-depth basis functions.
- `passband_sim_outputs/true_*params.csv`: simulator truth files.
- `passband_fit_outputs/fit_*params.csv`: recovered stellar, passband, and ice
  parameters.
- `passband_fit_outputs/fit_iteration_summary.csv`: residual RMS by iteration.
- `passband_fit_outputs/fit_residuals.csv`: observation table plus final
  residuals.
- `passband_fit_outputs/*.png`: diagnostic plots.

### Run

```bash
/opt/anaconda3/bin/python simulate_roman_passband_data.py
/opt/anaconda3/bin/python fit_roman_passband_model.py
```

Useful fitter options:

```bash
/opt/anaconda3/bin/python fit_roman_passband_model.py --n-iter 3
/opt/anaconda3/bin/python fit_roman_passband_model.py --max-stars 200
/opt/anaconda3/bin/python fit_roman_passband_model.py --input-dir passband_sim_outputs --output-dir passband_fit_outputs
```

The default simulator uses one detector, four broad toy filters, `2,000` stars,
and `30` exposures. The scripts are structured so the detector axis can be
expanded later.

### Measurement Table

`passband_sim_outputs/measurements.csv` has one row per observation:

```text
obs_id
star_id
exposure_id
epoch_id
filter_id
detector_id
x
y
ice_amount_obs
mag_obs
mag_unc
mag_true_no_noise
true_sed_mag_nominal
true_passband_delta_mag
true_ice_delta_mag
```

`ice_amount_obs` is treated as known input to the fitter. The fitter recovers
the spectral shape coefficients of `tau_ice(lambda)`, not a separate ice amount
per observation.

### Fitting Method

Iteration 0 fits initial stellar SED parameters star-by-star using nominal
passbands and ignoring passband/ice perturbations. Later iterations build a
sparse linearized system for magnitude residuals and solve updates for:

- per-star magnitude normalization,
- per-star temperature-like SED parameter,
- per-star extinction-like parameter,
- per-filter/per-detector passband shift,
- per-filter/per-detector passband width,
- global ice optical-depth basis coefficients.

The response for each passband or ice basis mode is computed from the current
stellar SED and current throughput with the proper broadband flux integral.
Updates are damped before being applied.

### Regularization And Degeneracies

The chromatic problem has real degeneracies among stellar SED colors, passband
color terms, and ice spectral shape. The fitter uses weighted pseudo-observation
rows for weak priors:

- passband shift prior: `sigma_shift_prior = 0.02 um`,
- passband width prior: `sigma_width_prior = 0.05`,
- ice coefficient prior: `sigma_ice_prior = 0.10`,
- stellar temperature update prior: `sigma_temp_update_prior = 400 K`,
- stellar extinction update prior: `sigma_ext_update_prior = 0.05`.

These priors stabilize the linearized fit but do not remove all degeneracy. The
diagnostics intentionally show imperfect passband and stellar-parameter
recovery.

### Default Verification

A default run currently produces `55,074` observations and solves `6,011`
linearized update parameters per iteration. The verified iteration summary is:

```text
iteration  RMS residual [mag]
0          0.013764
1          0.008285
2          0.006217
3          0.005422
4          0.005161
5          0.005089
```

Final default diagnostics:

```text
Passband shift RMS error: 0.006328 um
Passband width RMS error: 0.027010
Ice optical-depth shape RMS error: 0.000609
```
