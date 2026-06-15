# Roman WFI Calibration Prototypes

This repository contains small, self-contained Python prototypes for Roman Space
Telescope WFI-style calibration experiments.

- The first prototype fits scalar relative photometric calibration terms:
  stellar magnitudes, exposure zeropoints, smooth star-flat structure, and
  per-amplifier offsets.
- The second prototype fits chromatic calibration terms after scalar
  instrumental calibration: stellar SED parameters, detector-level passband
  shifts/width changes, and a 2D ice log-throughput surface.

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

Notation:

- `m_obs`: observed instrumental magnitude for one stellar measurement.
- `star_id`: stable simulated star identifier.
- `M_star[star_id]`: fitted relative magnitude of that star.
- `exposure_id`: exposure identifier.
- `ZP[exposure_id]`: fitted scalar zeropoint for that exposure; exposure 0 is
  fixed to zero as the reference.
- `x`, `y`: detector pixel coordinates.
- `S_smooth(x, y)`: fitted smooth detector-coordinate star-flat term.
- `detector_id`: one-based detector identifier in the saved table.
- `amp_id`: zero-based amplifier stripe identifier within a detector.
- `A_amp[detector_id, amp_id]`: fitted additive magnitude offset for that
  detector/amplifier pair.
- `noise`: Gaussian photometric noise in magnitudes.

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

Column meanings:

- `obs_id`: unique row identifier for one measured star in one exposure.
- `star_id`: stable identifier for the same simulated star across exposures.
- `exposure_id`: exposure index; exposure 0 is the fitted zeropoint reference.
- `detector_id`: one-based detector identifier, `1..18` by default.
- `amp_id`: zero-based amplifier stripe identifier, `0..31`.
- `x_pixel`, `y_pixel`: detector pixel coordinates of the observation.
- `instrumental_mag`: simulated observed instrumental magnitude.
- `instrumental_mag_uncertainty`: 1-sigma uncertainty in magnitudes.

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

Throughput perturbations are modeled additively in log-throughput space:

```text
ln T_b(lambda, d, x, y, e)
  = ln T0_b(lambda)
    + delta_lambda_b,d * phi_shift_b(lambda)
    + width_b,d        * phi_width_b(lambda)
    + I(log10(lambda), h_obs)
```

This lets small multiplicative passband/ice changes enter linearly, while the
actual synthetic observations and all response coefficients still come from
linear flux integrals over wavelength.

Notation:

- `lambda`: wavelength in microns.
- `s`: star index.
- `b`: filter/passband index. The default filters are `F062`, `F087`, `F106`,
  `F129`, `F158`, and `F184`.
- `d`: detector index. In the CSV files, detector IDs are one-based; the fitter
  maps them to zero-based internal array indices.
- `x`, `y`: detector pixel coordinates. They are included for future spatial
  models; v1 passband shift/width terms are detector-level constants.
- `e`: epoch/exposure index used for time-dependent ice state.
- `h_obs`: known ice-thickness coordinate for a specific observation.
- `f_s(lambda)`: spectral energy distribution of star `s`.
- `T_b(lambda, d, x, y, e)`: total throughput for filter `b` for that
  detector, position, and epoch.
- `T0_b(lambda)`: nominal throughput curve read from `passbands.txt`.
- `delta_lambda_b,d`: fitted wavelength-shift coefficient for filter `b` and
  detector `d`, in microns.
- `width_b,d`: fitted dimensionless width/stretch coefficient for filter `b`
  and detector `d`.
- `phi_shift_b(lambda)`, `phi_width_b(lambda)`: derivative-based
  log-throughput response modes derived from `T0_b(lambda)`.
- `I(log10(lambda), h_obs)`: fitted ice log-throughput perturbation, evaluated
  from a rectangular spline grid in log10 wavelength and ice thickness.

### Files

- `passbands.txt`: supplied Roman nominal relative throughputs for six filters:
  `F062`, `F087`, `F106`, `F129`, `F158`, and `F184`.
- `ice_loglam_nodes.txt`: default log10 wavelength nodes used for the ice
  spline grid. The simulator can read a different node file.
- `simulate_roman_passband_data.py`: generates six-filter photometry using
  `passbands.txt`, passband modes, a 2D ice spline surface, and truth files.
- `fit_roman_passband_model.py`: reads the simulator products and runs an
  iterative sparse linearized fit.
- `passband_sim_outputs/measurements.csv`: simulated flattened stellar
  photometry.
- `passband_sim_outputs/nominal_passbands.csv`: long-form nominal passbands.
- `passband_sim_outputs/passband_modes.csv`: shift and width log-throughput
  modes.
- `passband_sim_outputs/ice_spline_nodes.csv`: rectangular spline grid nodes in
  log10 wavelength and ice thickness.
- `passband_sim_outputs/true_ice_spline_params.csv`: true ice log-throughput
  values at the spline grid nodes.
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

Useful simulator options:

```bash
/opt/anaconda3/bin/python simulate_roman_passband_data.py --n-ice-thickness-nodes 7
/opt/anaconda3/bin/python simulate_roman_passband_data.py --ice-loglam-nodes-file my_nodes.txt
/opt/anaconda3/bin/python simulate_roman_passband_data.py --ice-thickness-min 0.0 --ice-thickness-max 1.5
```

The default simulator uses one detector, six supplied Roman passbands, `2,000`
stars, and `30` exposures. The scripts are structured so the detector axis can
be expanded later.

### Measurement Table

`passband_sim_outputs/measurements.csv` has one row per observation:

```text
obs_id
star_id
exposure_id
epoch_id
filter_id
filter_name
detector_id
x
y
ice_thickness
ice_amount_obs
mag_obs
mag_unc
mag_true_no_noise
true_sed_mag_nominal
true_passband_delta_mag
true_ice_delta_mag
```

`ice_thickness` is treated as known input to the fitter. `ice_amount_obs` is
kept as a backwards-compatible alias with the same value. The fitter recovers
the node values of the 2D ice log-throughput surface, not a separate ice amount
per observation.

Column meanings:

- `obs_id`: unique row identifier for one measured star in one exposure.
- `star_id`: stable identifier for the same simulated star across exposures.
- `exposure_id`: exposure index. In the default simulation, one exposure uses
  one filter and contains many stellar observations.
- `epoch_id`: time/epoch index used for ice evolution. In v1 this is identical
  to `exposure_id`, but it is kept separate so future simulations can group
  multiple exposures into one epoch.
- `filter_id`: compact numeric filter identifier, `0..5` by default.
- `filter_name`: Roman filter name corresponding to `filter_id`, e.g. `F062`.
- `detector_id`: one-based detector identifier. The default passband simulation
  uses detector `1`.
- `x`, `y`: detector pixel coordinates for this observation.
- `ice_thickness`: known ice-thickness coordinate for this observation,
  including exposure/epoch variation and a small position dependence.
- `ice_amount_obs`: backwards-compatible alias of `ice_thickness`.
- `mag_obs`: noisy simulated observed magnitude.
- `mag_unc`: 1-sigma magnitude uncertainty used for row weighting.
- `mag_true_no_noise`: noiseless simulated magnitude including SED, passband,
  and ice effects.
- `true_sed_mag_nominal`: noiseless magnitude through the nominal passband only.
- `true_passband_delta_mag`: magnitude difference from detector passband shift
  and width perturbations, before ice is applied.
- `true_ice_delta_mag`: magnitude difference caused by ice throughput loss.

### Fitting Method

Iteration 0 fits initial stellar SED parameters star-by-star using nominal
passbands and ignoring passband/ice perturbations. Later iterations build a
sparse linearized system for magnitude residuals and solve updates for:

- per-star magnitude normalization,
- per-star temperature-like SED parameter,
- per-star extinction-like parameter,
- per-filter/per-detector passband shift,
- per-filter/per-detector passband width,
- global ice log-throughput spline node values.

The response for each passband mode or ice spline node is computed from the
current stellar SED and current throughput with the proper broadband flux
integral. Updates are damped before being applied.

### Regularization And Degeneracies

The chromatic problem has real degeneracies among stellar SED colors, passband
color terms, and ice surface modes. The fitter uses weighted pseudo-observation
rows for weak priors:

- passband shift prior: `sigma_shift_prior = 0.02 um`,
- passband width prior: `sigma_width_prior = 0.05`,
- ice spline node prior: `sigma_ice_prior = 0.10`,
- zero-thickness ice-surface prior: `sigma_zero_ice_surface_prior = 1e-4`,
- stellar temperature update prior: `sigma_temp_update_prior = 400 K`,
- stellar extinction update prior: `sigma_ext_update_prior = 0.05`.

These priors stabilize the linearized fit but do not remove all degeneracy. The
diagnostics intentionally show imperfect passband and stellar-parameter
recovery.

The simulator derives `phi_shift` and `phi_width` from derivatives of the
tabulated log-throughput curves. Since the supplied passbands have sharp
sampled edges, the log-throughput derivatives are smoothed over a few wavelength
samples and clipped before simulation. This keeps the injected perturbations in
the small-signal regime needed by the linearized fitter.

The ice model is a rectangular linear tensor-product spline. Log-wavelength
nodes come from `ice_loglam_nodes.txt` or a user-specified file; ice-thickness
nodes are uniformly spaced between `ice_thickness_min` and `ice_thickness_max`.
The default has 9 log-wavelength nodes and 5 thickness nodes, so the ice block
contains 45 fitted node values. The zero-thickness node row is strongly
constrained to zero because zero ice thickness should produce no ice
log-throughput perturbation.

### Default Verification

A default run currently produces `55,071` observations and solves `6,057`
linearized update parameters per iteration. The verified iteration summary is:

```text
iteration  RMS residual [mag]
0          0.009220
1          0.014137
2          0.008400
3          0.005957
4          0.005168
5          0.004950
```

Final default diagnostics:

```text
Passband shift RMS error: 0.003456 um
Passband width RMS error: 0.005506
Ice log-throughput surface RMS error: 0.009032
```
