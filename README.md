# Roman WFI Calibration Prototypes

This repository contains small, self-contained Python prototypes for Roman Space
Telescope WFI-style calibration experiments.

- The first prototype fits scalar relative photometric calibration terms:
  stellar magnitudes, exposure zeropoints, smooth star-flat structure, and
  per-amplifier offsets.
- The second prototype jointly fits scalar and chromatic calibration terms:
  stellar BOSZ EMPCA SED coefficients, detector-level passband shifts/width
  changes, a 2D ice log-throughput surface, filter- and wavelength-dependent
  smooth focal-plane fields, and amplifier gains shared by imaging and prism
  data.

Both prototypes use deterministic simulations, sparse linear algebra, and CSV
artifacts intended to be easy to inspect.

## Dependencies

Use a Python environment with `numpy`, `pandas`, `scipy`, `matplotlib`, and
`tqdm`. On this machine, the Anaconda interpreter works:

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

## Passband, Ice, And Prism Prototype

This prototype combines the scalar focal-plane toy model with broadband
passband/ice calibration and sparse Roman prism spectroscopy. It simulates
dithered stellar photometry plus prism spectra for a fraction of the stars and
fits stellar SED parameters, scalar exposure/focal-plane/amplifier terms,
passband shift/width terms, an ice log-throughput surface, and one prism
sensitivity correction and one smooth focal-plane field for every prism
wavelength pixel.

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

The measured magnitude also includes scalar focal-plane terms from the
ubercalibration toy model:

```text
m_obs = -2.5 log10(F_pred)
      + ZP_e
      + S_imaging,b,d(x, y)
      + A_detector,amp
      + noise
```

The unified fitter estimates stellar SED parameters, chromatic passband/ice
parameters, and scalar focal-plane calibration parameters in the same
linearized iteration.

For prism wavelength pixel `p`, the midpoint photon-counting quadrature is

```text
C_prism,s,p = f_lambda,s(lambda_p)
              * T_prism,0(lambda_p)
              * exp[I(log10(lambda_p), h_obs)]
              * lambda_p * Delta_lambda_p
```

and the instrumental prism magnitude model is

```text
m_prism,obs = -2.5 log10(C_prism,s,p)
              + P_p
              + ZP_e
              + S_prism,p,d(x, y)
              + A_detector,amp
              + noise
```

Here `P_p` is the fitted additive prism sensitivity correction at wavelength
pixel `p`. It is equivalent to multiplying throughput by
`10^(-0.4 P_p)`. The first prism exposure is fixed to `ZP_e = 0`, separately
from imaging exposure 0, to remove the exact additive degeneracy between all
`P_p` values and all prism exposure zeropoints.

The prism wavelength solution is not fitted here. The mapping from
`wavelength_pixel_id` to `lambda_p` is read from `nominal_prism.csv`, which is
generated from the supplied `prism wavelengths` file, and remains fixed during
every iteration. A separate emission-line ubercalibration can determine that
mapping before these spectrophotometric calibration scripts are run. `P_p` is
a flux-sensitivity/zeropoint term, not a wavelength-shift or dispersion term.

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
- `f_s(lambda)`: spectral energy distribution of star `s`. In the current
  prototype this is a BOSZ EMPCA relative SED shape,
  `exp(mean_log_flux + theta_s @ components)`, multiplied by a fitted magnitude
  normalization.
- `theta_s`: low-dimensional BOSZ EMPCA coefficient vector for star `s`.
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
- `ZP_e`: fitted scalar zeropoint for exposure `e`; exposure 0 is fixed to zero
  as the reference.
- `S_imaging,b,d(x, y)`: fitted imaging smooth detector-coordinate polynomial
  for filter `b` and detector `d`, with terms `x, y, x^2, x*y, y^2`.
- `S_prism,p,d(x, y)`: fitted prism smooth detector-coordinate polynomial for
  wavelength pixel `p` and detector `d`, using the same five-term basis. Each
  prism wavelength has its own field; a wavelength second-difference prior
  regularizes the sequence without forcing the fields to be identical.
- `amp`: zero-based amplifier stripe ID computed from detector `x` coordinate.
- `A_detector,amp`: fitted additive magnitude offset for detector/amplifier
  pair, with mean amp offset per detector constrained to zero. This is one
  shared amplifier correction used by all imaging filters and prism
  wavelengths.
- `p`: zero-based prism wavelength-pixel index.
- `lambda_p`, `Delta_lambda_p`: prism pixel center and wavelength-bin width in
  microns, supplied as fixed inputs rather than fitted parameters.
- `T_prism,0(lambda_p)`: toy nominal prism throughput envelope.
- `P_p`: fitted wavelength-dependent prism response correction in magnitudes.

### Files

- `passbands.txt`: supplied Roman nominal relative throughputs for six filters:
  `F062`, `F087`, `F106`, `F129`, `F158`, and `F184`.
- `bosz_logflux_empca_basis.npz`: normalized log-flux BOSZ EMPCA basis used for
  simulated and fitted stellar SED shapes.
- `bosz2024_wave_r500.txt`: wavelength grid for the local BOSZ source spectra.
- `m+0.00/`: local BOSZ source spectra used to build the EMPCA basis; filenames
  encode the stellar effective temperature, but the fitter uses only the EMPCA
  coefficients.
- `ice_loglam_nodes.txt`: default log10 wavelength nodes used for the ice
  spline grid. The simulator can read a different node file.
- `prism wavelengths`: supplied prism pixel-center wavelengths in Angstroms.
  The simulator converts them to microns; the current file has 194 pixels from
  0.750 to about 1.796 microns.
- `simulate_roman_passband_data.py`: generates six-filter photometry using
  `passbands.txt`, BOSZ EMPCA stellar SEDs, passband modes, a 2D ice spline
  surface, scalar focal-plane terms, and truth files.
- `fit_roman_passband_model.py`: reads the simulator products and runs an
  iterative sparse linearized fit.
- `query_calibration.py`: reads the saved simulator/fitter products and queries
  fitted AB zeropoints plus sampled passband curves for new instrumental
  photometry.
- `passband_sim_outputs/measurements.csv`: simulated flattened stellar
  photometry.
- `passband_sim_outputs/prism_measurements.csv`: one row per extracted prism
  wavelength pixel. Only a configurable fraction of stars receive prism
  spectra, but all fixed spectrophotometric standards are included.
- `passband_sim_outputs/exposure_geometry.csv`: imaging and prism exposure
  dithers, rotations, measurement types, and imaging-filter assignments used
  to reconstruct detector footprints on the toy sky tangent plane.
- `passband_sim_outputs/detector_layout.csv`: detector dimensions and display
  offsets for the toy multi-detector sky mosaic.
- `passband_sim_outputs/nominal_prism.csv`: prism wavelength-pixel centers,
  finite bin edges/widths, and nominal toy throughput. This table is the fixed
  wavelength solution used by the fit.
- `passband_sim_outputs/true_prism_response.csv`: true wavelength-dependent
  prism sensitivity correction in magnitudes.
- `passband_sim_outputs/nominal_passbands.csv`: long-form nominal passbands.
- `passband_sim_outputs/passband_modes.csv`: shift and width log-throughput
  modes.
- `passband_sim_outputs/ice_spline_nodes.csv`: rectangular spline grid nodes in
  log10 wavelength and ice thickness.
- `passband_sim_outputs/true_ice_spline_params.csv`: true ice log-throughput
  values at the spline grid nodes.
- `passband_sim_outputs/true_star_params.csv`: true star magnitude
  normalizations, standard-star flags, selected BOSZ model IDs/files, and
  `sed_coeff_*` EMPCA coefficients. This is simulator truth used only for
  diagnostics; it is not a calibration input.
- `passband_sim_outputs/spectrophotometric_standards.csv`: standard-star
  manifest containing only `star_id` and `spectrum_file`.
- `passband_sim_outputs/standard_spectra/*.csv`: one physical spectrum per
  standard, with columns `wavelength_um` and
  `f_lambda_cgs_per_angstrom`.
- `passband_sim_outputs/true_passband_params.csv`: true detector-level
  passband shift and width coefficients.
- `passband_sim_outputs/true_exposure_zeropoints.csv`: true scalar exposure
  zeropoints.
- `passband_sim_outputs/true_smooth_coeffs.csv`: true smooth focal-plane
  polynomial coefficients in long form. `measurement_type` identifies imaging
  or prism rows; imaging rows carry `filter_id`, while prism rows carry
  `wavelength_pixel_id` and `wavelength_um`. Every row also carries the
  one-based `detector_id` and polynomial `basis_name`.
- `passband_sim_outputs/true_amp_offsets.csv`: true per-detector/per-amplifier
  additive magnitude offsets.
- `passband_fit_outputs/fit_*params.csv`: recovered stellar, passband, ice, and
  scalar focal-plane parameters.
- `passband_fit_outputs/fit_star_params.csv`: recovered field-star magnitude
  normalizations and `sed_coeff_*` EMPCA coefficients, with formal
  uncertainties from LSQR. Standard rows identify their physical spectrum
  files and report a derived reference-band AB magnitude; their fitted
  magnitude and EMPCA fields are blank because those parameters do not exist.
- `passband_fit_outputs/fit_ice_spline_params.csv`: recovered ice spline node
  values and formal node uncertainties from the final LSQR variance estimate.
- `passband_fit_outputs/fit_exposure_zeropoints.csv`: recovered scalar
  exposure zeropoints.
- `passband_fit_outputs/fit_smooth_coeffs.csv`: recovered imaging
  filter/detector and prism wavelength/detector smooth polynomial coefficients,
  in the same long-form layout as the truth table, with formal uncertainties.
- `passband_fit_outputs/fit_amp_offsets.csv`: recovered per-detector/per-amp
  offsets.
- `passband_fit_outputs/fit_prism_response.csv`: recovered prism correction
  for every wavelength pixel, with formal LSQR uncertainty.
- `passband_fit_outputs/fit_prism_residuals.csv`: input prism rows plus final
  residual, fitted prism/scalar corrections, AB zeropoint, and calibrated
  narrow-bin AB magnitude.
- `passband_fit_outputs/fit_prism_ab_zeropoints.csv`: compact per-prism-pixel
  AB zeropoint query table.
- `passband_fit_outputs/fit_iteration_summary.csv`: residual RMS by iteration.
- `passband_fit_outputs/fit_residuals.csv`: observation table plus final
  residuals, fitted scalar correction, and fitted AB zeropoint for each
  observation.
- `passband_fit_outputs/fit_ab_zeropoints.csv`: per-observation AB zeropoint,
  `fit_ab_zeropoint_mag`, such that `m_AB = m_inst + fit_ab_zeropoint_mag`.
- `passband_fit_outputs/*.png`: diagnostic plots, including the passband/ice
  recovery plots plus scalar focal-plane plots such as
  `smooth_field_true.png`, `smooth_field_recovered.png`, and
  `smooth_field_residual.png` show one imaging panel per filter for detector 1;
  additional detectors receive detector-suffixed files. Other plots include
  `true_amp_offsets.png`,
  `recovered_amp_offsets.png`, `amp_offset_comparison.png`,
  `residual_vs_x.png`, and `residual_vs_amp.png`.
  Prism diagnostics include `prism_response_true_vs_fit.png`,
  `prism_residual_vs_wavelength.png`, `prism_residual_histogram.png`, and
  `prism_standard_absolute_calibration.png`. `prism_smooth_coefficients.png`
  shows every fitted spatial coefficient versus wavelength and
  `prism_smooth_field_samples.png` compares true, fitted, and residual fields
  at representative wavelengths.
- `passband_sim_outputs/sim_standard_spectra.png`: physical `f_lambda` inputs
  for the simulated spectrophotometric standards.
- `passband_sim_outputs/sky_coverage_F062.png` through
  `sky_coverage_F184.png`, plus `sky_coverage_PRISM.png`: observed source
  positions and detector outlines on the toy sky tangent plane, colored and
  labeled by exposure.

### Run

```bash
/opt/anaconda3/bin/python simulate_roman_passband_data.py
/opt/anaconda3/bin/python fit_roman_passband_model.py
```

The fitter displays a `tqdm` progress bar over the requested nonlinear
iterations. Its compact postfix reports imaging RMS (`i`), prism RMS (`p`), and
the accepted damping factor (`d`) from the latest completed iteration. A
persistent line is also printed after every iteration with the separate imaging
and prism RMS values, damping, and number of LSMR iterations.

The default fit runs 15 iterations. Use `--n-iter` to select a different
iteration count for quick tests or convergence studies.

Useful fitter options:

```bash
/opt/anaconda3/bin/python fit_roman_passband_model.py --n-iter 3
/opt/anaconda3/bin/python fit_roman_passband_model.py --max-stars 200
/opt/anaconda3/bin/python fit_roman_passband_model.py --input-dir passband_sim_outputs --output-dir passband_fit_outputs
/opt/anaconda3/bin/python fit_roman_passband_model.py --sed-basis-path bosz_logflux_empca_basis.npz
```

Useful simulator prism options:

```bash
/opt/anaconda3/bin/python simulate_roman_passband_data.py --prism-wavelength-file "prism wavelengths"
/opt/anaconda3/bin/python simulate_roman_passband_data.py --n-prism-exp 4
/opt/anaconda3/bin/python simulate_roman_passband_data.py --prism-star-fraction 0.01
```

Set `--n-prism-exp 0` for an imaging-only simulation; the simulator still
writes an empty `prism_measurements.csv` with a valid schema, and the fitter
skips prism parameters and diagnostics.

Query a fitted calibration for one instrumental measurement:

```bash
/opt/anaconda3/bin/python query_calibration.py \
  --filter-name F129 \
  --detector-id 1 \
  --exposure-id 2 \
  --x 1024 \
  --y 2048 \
  --ice-thickness 0.3 \
  --instrumental-mag 19.5
```

This writes `calibration_query_results.csv` with `fit_ab_zeropoint_mag` and
`calibrated_ab_mag`, plus `calibration_query_passbands.csv` with the sampled
current passband:

```text
T_current(lambda)
  = T0(lambda) * exp(logt_passband(lambda) + ice_logt(lambda, ice_thickness))
```

For a batch table, provide columns `filter_id` or `filter_name`, `detector_id`,
`exposure_id`, `x`, `y`, and `ice_thickness`. If `amp_id` is absent, it is
computed from `x`; if `instrumental_mag`, `mag_inst`, or `mag_obs` is present,
the script also writes `calibrated_ab_mag`.

```bash
/opt/anaconda3/bin/python query_calibration.py \
  --query-csv my_instrumental_measurements.csv \
  --output-csv my_calibrated_measurements.csv \
  --passband-output-csv my_current_passbands.csv
```

Use `--no-passband-output` for a faster zeropoint-only batch query.

Useful simulator options:

```bash
/opt/anaconda3/bin/python simulate_roman_passband_data.py --n-ice-thickness-nodes 7
/opt/anaconda3/bin/python simulate_roman_passband_data.py --ice-loglam-nodes-file my_nodes.txt
/opt/anaconda3/bin/python simulate_roman_passband_data.py --ice-thickness-min 0.0 --ice-thickness-max 1.5
/opt/anaconda3/bin/python simulate_roman_passband_data.py --sed-basis-path bosz_logflux_empca_basis.npz
/opt/anaconda3/bin/python simulate_roman_passband_data.py --reference-filter-id 3
/opt/anaconda3/bin/python simulate_roman_passband_data.py --n-spectrophotometric-standard 5
```

The default simulator uses one detector, 32 amplifier stripes, six supplied
Roman passbands, `2,000` stars, `5` fixed physical spectrophotometric-standard
spectra, and `30` imaging exposures. The exposure pattern includes translations
and rotations, so stars move across the smooth focal-plane and amplifier terms.

### Stellar SED Basis

The passband prototype uses `bosz_logflux_empca_basis.npz` instead of an
analytic blackbody/extinction family. The basis contains:

```text
wave_micron
mean_log_flux
components
coefficients
model_files
valid_mask
metadata
```

For a star with EMPCA coefficient vector `theta_s`, the relative SED shape is

```text
sed_shape(lambda) = exp(mean_log_flux(lambda) + theta_s @ components(lambda))
```

The EMPCA basis has arbitrary absolute normalization removed. The simulator
therefore draws a separate `mag_norm` for each star, and the fitter solves for
that magnitude normalization independently of the `sed_coeff_*` shape
parameters. The current basis file in this directory has 4 EMPCA components and
56 input BOSZ model coefficient vectors. The simulator draws true stars from
those stored coefficient vectors so the synthetic colors remain on the BOSZ
training manifold.

A small subset is designated as spectrophotometric standards. The simulator
first constructs their physical flux density in
`erg cm^-2 s^-1 Angstrom^-1`, then writes each one as an independent
`f_lambda`-versus-wavelength CSV. The fitter reads those spectra directly and
omits magnitude-normalization and EMPCA update columns for those stars. It
never uses the standards' simulator-truth `mag_norm` or `sed_coeff_*` values.

### Spectrophotometric Standard Inputs

`spectrophotometric_standards.csv` is the only standard-star manifest consumed
by the fitter:

```text
star_id,spectrum_file
228,standard_spectra/standard_star_000228.csv
```

Each referenced spectrum is a CSV with exactly the physical coordinates needed
by synthetic photometry:

```text
wavelength_um,f_lambda_cgs_per_angstrom
0.473,1.200509116574e-16
0.474,1.191627825783e-16
```

Wavelengths must be finite, strictly increasing, and cover both the imaging
passband grid and the fixed prism wavelength grid. Flux densities must be
finite and positive and are interpreted as
`erg cm^-2 s^-1 Angstrom^-1`. Relative paths are resolved from the simulator
input directory. The fitter validates these conditions and also checks that
measurement-table standard flags agree with the manifest.

The simulator still records each standard's generating BOSZ model,
normalization, and EMPCA coordinates in `true_star_params.csv`, but that file
is optional truth for recovery plots only. Deleting it does not change the fit.

### Synthetic Photometry And AB Zeropoints

The measurement table stores instrumental magnitudes. Most stars provide
relative constraints through fitted BOSZ EMPCA shapes and normalizations,
whereas the standards provide physical `f_lambda` spectra and therefore set the
absolute AB scale without any fitted stellar parameters.

The BOSZ EMPCA basis provides relative SED shapes, so the simulator assigns
each star an amplitude by requiring `mag_norm` to be that star's AB magnitude
in a reference filter. By default the reference filter is
`reference_filter_id = 3`, i.e. `F129` for the supplied six-filter file.

The source count integral for one observation is computed on the passband
wavelength grid as

```text
C_source = integral f_lambda,s(lambda)
                  * T0_b(lambda)
                  * exp(logT_passband + logT_ice)
                  * lambda
                  d lambda
```

where `f_lambda,s(lambda)` is the BOSZ EMPCA shape scaled to the star's
reference-filter AB magnitude. The constant `1 / hc` is omitted because it is
common to all photon-counting integrals in this prototype.

The instrumental chromatic magnitude is

```text
m_inst,chromatic = -2.5 log10(C_source)
```

and the scalar calibration terms are added:

```text
m_inst,true = m_inst,chromatic
            + ZP_e
            + S_imaging,b,d(x, y)
            + A_detector,amp
```

Finally Gaussian noise with standard deviation `mag_unc` is added to produce
the instrumental measurement `mag_obs`.

For conversion to AB magnitudes, the fitter evaluates the AB reference count
through the same current passband and ice state:

```text
C_AB = integral f_lambda,AB(lambda) * T_current(lambda) * lambda d lambda
```

where `T_current(lambda) = T0_b(lambda) * exp(logT_passband + logT_ice)`,
`f_nu,AB = 3631 Jy`, and

```text
f_lambda,AB(lambda) = f_nu,AB * c / lambda^2
```

with the unit conversion needed for wavelength in microns. The fitted
per-observation AB zeropoint is

```text
ZP_AB(obs) = 2.5 log10(C_AB)
             - [ZP_e + S_imaging,b,d(x, y) + A_detector,amp]
```

so the calibrated AB magnitude is

```text
m_AB = m_inst + ZP_AB(obs)
```

For prism pixel `p`, the corresponding narrow-bin AB reference count uses the
same midpoint quadrature:

```text
C_AB,prism,p = f_lambda,AB(lambda_p)
               * T_prism,0(lambda_p)
               * exp[I(log10(lambda_p), h_obs)]
               * lambda_p * Delta_lambda_p
```

The fitted prism AB zeropoint is

```text
ZP_AB,prism(obs) = 2.5 log10(C_AB,prism,p)
                   - P_p
                   - [ZP_e + S_prism,p,d(x, y) + A_detector,amp]
```

so `mag_obs + ZP_AB,prism(obs)` is the calibrated narrow-bin AB magnitude.
`fit_prism_residuals.csv` stores both this zeropoint and the resulting
`calibrated_ab_mag` for every simulated spectral pixel.

The same photon-counting AB convention is used for the physical standard-star
files and the EMPCA field stars. For field stars, `mag_norm` supplies the
absolute normalization removed by EMPCA preprocessing. For standards, the
sampled physical `f_lambda` values supply both shape and normalization directly.

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
amp_id
x
y
sky_x
sky_y
ice_thickness
ice_amount_obs
mag_obs
mag_unc
mag_true_no_noise
true_sed_mag_nominal
true_passband_delta_mag
true_ice_delta_mag
true_ab_mag_nominal
true_ab_mag_chromatic
true_zp_delta_mag
true_smooth_delta_mag
true_amp_delta_mag
true_scalar_delta_mag
is_spectrophotometric_standard
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
- `amp_id`: zero-based amplifier stripe ID, computed as
  `floor(x / (nx / n_amp))` and clipped to the available amplifier range.
- `x`, `y`: detector pixel coordinates for this observation.
- `sky_x`, `sky_y`: fixed source position on the simulator's toy sky tangent
  plane, measured in detector-pixel units. Dithers and rotations change `x`
  and `y` between exposures but do not change these sky coordinates.
- `ice_thickness`: known ice-thickness coordinate for this observation,
  including exposure/epoch variation and a small position dependence.
- `ice_amount_obs`: backwards-compatible alias of `ice_thickness`.
- `mag_obs`: noisy simulated instrumental magnitude.
- `mag_unc`: 1-sigma magnitude uncertainty used for row weighting.
- `mag_true_no_noise`: noiseless instrumental magnitude including SED,
  passband, ice, and scalar focal-plane effects.
- `true_sed_mag_nominal`: noiseless instrumental magnitude through the nominal
  passband only.
- `true_passband_delta_mag`: instrumental-magnitude difference from detector
  passband shift and width perturbations, before ice is applied.
- `true_ice_delta_mag`: instrumental-magnitude difference caused by ice
  throughput loss.
- `true_ab_mag_nominal`: noiseless AB magnitude through the nominal passband.
- `true_ab_mag_chromatic`: noiseless AB magnitude through the true perturbed
  passband and ice state, before scalar instrumental terms.
- `true_zp_delta_mag`: true scalar exposure zeropoint contribution.
- `true_smooth_delta_mag`: true smooth focal-plane contribution.
- `true_amp_delta_mag`: true amplifier offset contribution.
- `true_scalar_delta_mag`: sum of true zeropoint, smooth, and amp terms.
- `is_spectrophotometric_standard`: true when `star_id` is listed in
  `spectrophotometric_standards.csv` and its SED is supplied by a physical
  `f_lambda` file.

`passband_sim_outputs/prism_measurements.csv` has one row per wavelength pixel
in one extracted spectrum:

```text
prism_obs_id
spectrum_id
star_id
exposure_id
epoch_id
detector_id
amp_id
wavelength_pixel_id
wavelength_um
bin_width_um
x
y
sky_x
sky_y
ice_thickness
mag_obs
mag_unc
mag_true_no_noise
true_ab_mag
true_prism_response_mag
true_ice_logt
true_zp_delta_mag
true_smooth_delta_mag
true_amp_delta_mag
true_scalar_delta_mag
is_spectrophotometric_standard
```

`prism_obs_id` uniquely identifies one measured spectral pixel, whereas
`spectrum_id` groups all wavelength pixels for one star in one exposure.
`wavelength_pixel_id` selects a row of `nominal_prism.csv` and a fitted `P_p`
term. The prism spectra are vertical: x and `amp_id` are constant within each
`spectrum_id`, while y changes by one detector pixel per wavelength sample.
The source `sky_x` and `sky_y` are consequently constant across the spectrum.
Thus a single spectrum never crosses an amplifier boundary. A dither may move
the whole trace onto a different amplifier in another exposure, which supplies
relative constraints on the same amplifier gains used by all six filters.

### Sky Coverage Geometry

The sky-coverage diagnostics invert the same exposure transform used to create
the measurements. In detector-local coordinates,

```text
[x - center] = Rotation(rotation_deg) [sky - center] + [dither_dx_pix]
[y - center]                                             [dither_dy_pix]
```

`exposure_geometry.csv` stores one row per exposure with `exposure_id`,
`epoch_id`, `measurement_type`, `filter_id`, `filter_name`,
`dither_dx_pix`, `dither_dy_pix`, and `rotation_deg`. The detector corners are
mapped back through the inverse transform to draw each exposure outline.
Imaging plots show every retained stellar observation; the prism plot shows one
source point per `spectrum_id`, rather than repeating the point for every
wavelength pixel.

For one detector, the sky tangent plane is simply the undithered detector frame.
For multiple detectors, `detector_layout.csv` places the simulated detectors in
a six-column display mosaic with an 8% gap. This is only a clear visualization
coordinate system: it is not the physical Roman WFI SCA layout and does not
alter detector-local calibration calculations.

All five default standards receive prism spectra. Their physical `f_lambda`
files are used directly, so they determine the absolute wavelength-dependent
prism spectrophotometric response without fitted stellar magnitudes or EMPCA
coordinates. They do not constrain or update the wavelength assigned to any
detector pixel.
The other prism targets use the same free stellar parameters as their imaging
measurements.

### Fitting Method

Iteration 0 fits initial stellar SED parameters star-by-star using nominal
passbands and ignoring passband/ice perturbations. It searches over the BOSZ
EMPCA coefficient vectors stored in the basis file and solves each star's
magnitude normalization analytically for field stars. Standards bypass this
search: their physical spectra are interpolated once onto the imaging and prism
grids and then held fixed. Later iterations build a sparse linearized system
for magnitude residuals and solve updates for:

- per-star magnitude normalization for non-standard field stars,
- per-star BOSZ EMPCA SED coefficients for non-standard field stars,
- per-filter/per-detector passband shift,
- per-filter/per-detector passband width,
- global ice log-throughput spline node values.
- one global prism sensitivity correction per wavelength pixel,
- per-exposure scalar zeropoints, with exposure 0 fixed to zero,
- per-filter/per-detector imaging smooth focal-plane polynomial coefficients,
- per-wavelength-pixel/per-detector prism smooth focal-plane polynomial
  coefficients,
- per-detector/per-amplifier additive offsets.

The first prism exposure is also fixed to zero. Prism rows contain the same
stellar normalization/EMPCA, ice-spline, exposure, and amplifier columns as
imaging where applicable, plus the smooth-field columns for that prism
wavelength and detector. They do not contain imaging-filter shift or width
columns, and no prism wavelength-solution columns are present. Their amplifier
column points into the exact same parameter block as imaging, which enforces
the shared-gain requirement directly in the sparse system. Imaging filters
likewise have separate smooth-field blocks, so spatial chromatic structure
cannot be forced into a single achromatic star flat.

The prism reference exposure is simulated at zero ice thickness and contains
all fixed standards. The fitter uses those rows to bootstrap `P_p` before the
joint iterations. Subsequent prism exposures span nonzero ice thicknesses, so
the changing spectral pattern constrains the ice surface relative to the
ice-free intrinsic response.

The response for each stellar coefficient, passband mode, or ice spline node is
computed from the current stellar SED and current throughput with the proper
broadband flux integral. Since the BOSZ coefficients are linear in log SED, the
stellar coefficient response uses the same flux-weighted integral form as the
log-throughput response modes. Scalar zeropoint, smooth, and amp terms are
linear additive magnitude responses. Updates are damped before being applied.
Before each sparse solve, columns are scaled to unit norm to condition the
mixed-unit parameter system. A backtracking search then tests the requested
damping and successively halves it until the actual joint weighted residual
improves; rejected nonlinear overshoots are therefore never applied.

After the final iteration, the fitter rebuilds the weighted linearized system
and runs SciPy `lsqr` with `calc_var=True`. This gives a formal diagonal
variance estimate for the fitted update parameters. The ice-surface uncertainty
panel propagates only those diagonal node variances through the rectangular
spline basis; it intentionally ignores off-diagonal covariance, so it should be
read as a quick diagnostic rather than a full posterior uncertainty surface.

### Regularization And Degeneracies

The chromatic problem has real degeneracies among stellar SED coefficients,
passband color terms, and ice surface modes. The fitter uses weighted
pseudo-observation rows for weak priors:

- passband shift prior: `sigma_shift_prior = 0.02 um`,
- passband width prior: `sigma_width_prior = 0.05`,
- ice spline node prior: `sigma_ice_prior = 0.10`,
- zero-thickness ice-surface prior: `sigma_zero_ice_surface_prior = 1e-4`,
- prism response amplitude prior: `sigma_prism_response_prior = 0.20 mag`,
- prism response second-difference prior:
  `sigma_prism_response_smoothness = 0.01 mag`,
- imaging and prism smooth-coefficient amplitude prior:
  `sigma_smooth_prior = 0.02 mag`,
- prism smooth-coefficient wavelength second-difference prior:
  `sigma_prism_smoothness = 0.0001 mag`,
- stellar EMPCA coefficient update prior:
  `sigma_sed_coeff_update_prior_scale = 0.05` times the empirical standard
  deviation of each BOSZ coefficient in the basis file.
- amplifier offset prior: `sigma_amp_prior = 0.02 mag`,
- mean amp offset per detector constraint:
  `sigma_amp_sum_constraint = 1e-4 mag`.

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

A default run currently produces `45,532` imaging observations and `14,125`
prism wavelength-pixel measurements from `75` extracted spectra. The prism
targets are 1% of the 2,000-star sample, including all `5` physical
spectrophotometric standards. The fit solves `11,290` linearized update
parameters per iteration.
The verified iteration summary is:

```text
iteration  combined RMS  imaging RMS  prism RMS  accepted damping
0          0.031690      0.028457     0.040387   0.0000
1          0.029390      0.026048     0.038223   0.0625
2          0.026541      0.023655     0.034226   0.1250
3          0.024325      0.023250     0.027507   0.2500
4          0.020875      0.021466     0.018840   0.5000
5          0.013065      0.012046     0.015913   0.5000
6          0.009751      0.007355     0.015074   0.5000
7          0.008709      0.005569     0.014846   0.5000
8          0.008422      0.005014     0.014783   0.5000
9          0.008342      0.004855     0.014763   0.5000
10         0.008316      0.004803     0.014756   0.5000
11         0.008304      0.004779     0.014752   0.5000
12         0.008297      0.004765     0.014750   0.5000
13         0.008290      0.004749     0.014748   0.5000
14         0.008277      0.004722     0.014747   0.5000
15         0.008261      0.004686     0.014746   0.5000
```

Final default diagnostics:

```text
Passband shift RMS error: 0.008504 um
Passband width RMS error: 0.049688
Ice log-throughput surface RMS error: 0.025392
Prism spectrophotometric-response RMS error: 0.002183 mag
Exposure ZP RMS error: 0.121064 mag
Imaging smooth-coefficient RMS error: 0.002046 mag
Prism smooth-coefficient RMS error: 0.003047 mag
Amp offset RMS error, detector means removed: 0.001177 mag
Field-star EMPCA coefficient RMS error: 0.156358
Median formal ice-node uncertainty: 0.003742
Median formal prism-response uncertainty: 0.000933 mag
```

The imaging and prism residuals reach their injected noise scales, but the
larger exposure-ZP and passband-parameter truth errors make the remaining
chromatic gauge freedom visible: several combinations of exposure offsets,
stellar EMPCA colors, passband modes, and ice structure predict nearly the same
data. This is deliberate prototype behavior rather than a claim of unique
physical parameter recovery.
