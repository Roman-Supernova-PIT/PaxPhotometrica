# PaxPhotometrica

PaxPhotometrica is a sparse calibration prototype for Roman-like WFI imaging
and prism observations. It jointly simulates and fits:

- stellar BOSZ EMPCA SED coefficients and flux normalizations,
- per-exposure zeropoints,
- filter- and detector-dependent smooth focal-plane terms,
- wavelength- and detector-dependent prism focal-plane terms,
- one shared set of detector/amplifier offsets for imaging and prism,
- detector-level passband shifts and width changes,
- a two-dimensional ice log-throughput surface, and
- a wavelength-dependent prism response.

The package is intended for studying calibration leverage, degeneracies, and
survey-design choices. It is an independent research prototype, not an
official NASA or STScI calibration product.

## Installation

PaxPhotometrica currently supports Python 3.9 and newer. Install a development
checkout with:

```bash
python -m pip install -e ".[test]"
```

After the first PyPI release, the installation command will be:

```bash
python -m pip install paxphotometrica
```

Runtime dependencies are NumPy, pandas, SciPy, Matplotlib, and tqdm.

## Quick Start

The package installs one command with three live workflows:

```bash
paxphot simulate
paxphot fit
paxphot query --filter-name F129 \
  --detector-id 1 \
  --exposure-id 2 \
  --x 1024 \
  --y 2048 \
  --ice-thickness 0.3 \
  --instrumental-mag 19.5
```

This creates:

```text
passband_sim_outputs/
passband_fit_outputs/
calibration_query_results.csv
calibration_query_passbands.csv
```

Use command-specific help to see available options:

```bash
paxphot simulate --help
paxphot fit --help
paxphot query --help
```

The default nonlinear fit runs 15 iterations. A smaller smoke test is:

```bash
paxphot fit --n-iter 2 --max-stars 200
```

## Unified Calibration Model

For imaging observation `k`, the simulated instrumental magnitude is

```text
m_obs[k] = m_chromatic[k]
         + ZP[exposure_id[k]]
         + S_filter,detector(x[k], y[k])
         + A_detector,amp[k]
         + noise[k]
```

Definitions:

- `m_obs[k]` is the noisy instrumental magnitude.
- `m_chromatic[k]` is computed from the star SED and the current passband,
  including passband shift, width, and ice effects.
- `ZP[e]` is the scalar zeropoint for exposure `e`.
- `S_filter,detector(x, y)` is a smooth focal-plane polynomial for one imaging
  filter and detector.
- `A_detector,amp` is the additive offset for one detector amplifier.
- `noise[k]` is Gaussian magnitude noise with standard deviation `mag_unc`.

The smooth imaging field uses normalized detector coordinates

```text
xn = 2 * x / (NX - 1) - 1
yn = 2 * y / (NY - 1) - 1
```

and basis

```text
xn, yn, xn^2, xn*yn, yn^2.
```

The constant smooth term is omitted to avoid a zeropoint degeneracy.

Roman-like detector defaults are:

```text
NX = 4096
NY = 4096
N_AMP = 32
AMP_WIDTH = NX / N_AMP = 128 columns
amp_id = floor(x / AMP_WIDTH)
```

Detector IDs are one-based. Amplifier IDs are zero-based. Each amplifier is a
vertical stripe.

There are no separate `amp-simulate` or `amp-fit` commands. Amplifier offsets
are part of every joint simulation and every fit iteration. The same fitted
`A_detector,amp` applies to all six filters and every prism wavelength.

## Synthetic Photometry

The broadband model always begins with a linear photon-counting flux integral:

```text
C_star = integral f_lambda(lambda)
                  * T(lambda)
                  * lambda
                  d_lambda
```

where:

- `lambda` is wavelength in microns,
- `f_lambda` is spectral flux density per micron internally,
- `T(lambda)` is dimensionless throughput, and
- the omitted constant `1/(h*c)` cancels in AB flux ratios.

The corresponding instrumental magnitude is

```text
m_inst = -2.5 log10(C_star).
```

The AB reference spectrum has constant

```text
f_nu = 3631 Jy
```

and is converted to `f_lambda` on the same wavelength grid. A broadband AB
magnitude is

```text
m_AB = -2.5 log10(C_star / C_AB).
```

Input spectrophotometric-standard files use conventional physical units:

```text
erg cm^-2 s^-1 Angstrom^-1.
```

They are converted to per-micron units before numerical integration.

## Stellar SEDs

Field-star SED shapes use a four-component BOSZ EMPCA representation:

```text
log f_shape(lambda)
  = mean_log_flux(lambda)
  + sum_j theta_star,j * component_j(lambda).
```

The shape is exponentiated and assigned a separate magnitude normalization.
The normalization removed while constructing the EMPCA basis is not treated as
a physical BOSZ absolute flux.

Simulated field stars draw coefficient vectors from the empirical BOSZ model
projections. The fitter solves for each field star's normalization and EMPCA
coefficients.

Spectrophotometric standards are different. Their sampled physical
`f_lambda(lambda)` files provide both shape and absolute normalization, so
standards have no fitted stellar SED parameters.

## Passbands And Ice

For filter `b`, detector `d`, detector position `(x, y)`, and exposure epoch
`e`, the photon-counting broadband prediction is

```text
C_pred = integral f_lambda,s(lambda)
                  * T_b(lambda, d, x, y, e)
                  * lambda
                  d_lambda.
```

Here:

- `s` identifies a star,
- `b` identifies one of F062, F087, F106, F129, F158, or F184,
- `d` is the one-based detector ID,
- `(x, y)` are detector pixels, and
- `e` identifies the exposure epoch, and
- the omitted factor `1/(h*c)` cancels when count rates are converted to
  magnitudes.

Multiplicative throughput perturbations are additive in log-throughput:

```text
ln T = ln T0
     + delta_lambda[b,d] * phi_shift[b](lambda)
     + width[b,d] * phi_width[b](lambda)
     + ice_logt(log10(lambda), ice_thickness).
```

`phi_shift` approximates a wavelength shift through `-d ln(T0)/d lambda`.
`phi_width` is a derivative-based stretch mode around the filter center.

The ice term is a rectangular linear tensor-product spline in:

```text
[log10 wavelength, ice thickness].
```

Log-wavelength nodes come from a bundled table or a user file. Ice-thickness
nodes are uniformly spaced and configurable. The zero-thickness row is
strongly constrained to zero.

Although perturbations are represented in log-throughput, all predictions and
linearized response coefficients are computed from linear flux integrals.

## Prism Model

Only a configurable fraction of stars receive prism spectra. Each extracted
spectrum has one row per wavelength pixel.

Prism traces are vertical:

- detector `x` is constant across a spectrum,
- wavelength increases along detector `y`, and
- one spectrum never crosses an amplifier boundary.

A later dither may move the entire spectrum onto a different amplifier. This
links prism and imaging measurements to the same amplifier solution.

For prism wavelength pixel `p`, the model includes:

```text
m_prism = m_source,p
        + P[p]
        + ZP[exposure]
        + S_prism,p,detector(x, y)
        + A_detector,amp
        + noise.
```

`P[p]` is the fitted wavelength-dependent prism response. The prism wavelength
solution is fixed input; wavelength calibration is not fitted.

## Sky Coverage

The parent star catalog is sampled uniformly over the union of all requested
imaging and prism detector footprints. Stars are not confined to the
undithered exposure-0 footprint.

Each measurement stores:

- detector coordinates `x`, `y`, and
- fixed toy tangent-plane coordinates `sky_x`, `sky_y`.

The simulator writes one coverage plot per filter and one for the prism. Each
plot shows observed source positions and exposure-colored detector outlines.

For multiple detectors, the diagnostic uses a documented six-column display
mosaic. It is not the physical Roman SCA layout and does not alter calibration
calculations.

## Iterative Sparse Fit

The fitter first selects an initial BOSZ template for each field star using the
nominal passbands. It then repeatedly:

1. evaluates current imaging and prism predictions,
2. computes magnitude derivatives from flux integrals,
3. builds one weighted SciPy sparse matrix,
4. solves for parameter updates with LSMR,
5. applies a damped update, and
6. recomputes the nonlinear model.

Each data row is divided by its measurement uncertainty. Gaussian priors and
gauge constraints are implemented as weighted pseudo-observation rows.

The update vector contains:

```text
star normalizations
star EMPCA coefficients
passband shifts
passband widths
ice spline nodes
prism response values
exposure zeropoints
imaging smooth coefficients
prism smooth coefficients
shared amplifier offsets
```

Formal parameter uncertainties are estimated from the final LSQR variance
diagnostic. They are useful local linearized diagnostics, not a full posterior.

## Remaining Degeneracies

This is a relative, linearized calibration problem. Important controls include:

- fixed reference exposure zeropoints,
- no constant smooth-field term,
- zero mean amplifier offset per detector,
- weak amplifier, passband, ice, prism, and stellar-update priors,
- zero ice response at zero thickness, and
- exposure rotations in addition to translations.

Stellar colors, passband perturbations, and ice structure remain partially
degenerate. Stars observed in only one filter have no empirical color leverage.
The simulator and diagnostics intentionally expose these limitations.

## Outputs

The simulator writes CSV truth and measurement products to
`passband_sim_outputs/`. Important files include:

- `measurements.csv`
- `prism_measurements.csv`
- `spectrophotometric_standards.csv`
- `standard_spectra/*.csv`
- `nominal_passbands.csv`
- `nominal_prism.csv`
- `passband_modes.csv`
- `ice_spline_nodes.csv`
- `exposure_geometry.csv`
- `detector_layout.csv`
- `true_*params.csv`
- `true_amp_offsets.csv`
- `simulation_metadata.json`
- `sky_coverage_*.png`

The principal table identifiers have distinct roles:

- `star_id` uniquely identifies a simulated astrophysical source across all
  imaging and prism tables.
- `obs_id` uniquely identifies one row of `measurements.csv`.
- `prism_obs_id` uniquely identifies one wavelength-pixel row of
  `prism_measurements.csv`.
- `spectrum_id` groups all wavelength-pixel rows from one star in one prism
  exposure.
- `exposure_id` groups measurements from one detector pointing/readout and
  selects its fitted exposure zeropoint. Imaging and prism exposure IDs occupy
  one shared, non-overlapping sequence.
- `epoch_id` identifies the time state used for time-dependent calibration.
  The toy simulator currently assigns one epoch per exposure, so the two IDs
  have the same numeric value, but they are kept separate for future
  many-exposures-per-epoch models.
- `detector_id` is the one-based Roman-style SCA number; `amp_id` is the
  zero-based vertical amplifier stripe within that detector.

The fitter writes recovered products to `passband_fit_outputs/`, including:

- `fit_star_params.csv`
- `fit_passband_params.csv`
- `fit_ice_spline_params.csv`
- `fit_prism_response.csv`
- `fit_exposure_zeropoints.csv`
- `fit_smooth_coeffs.csv`
- `fit_amp_offsets.csv`
- `fit_ab_zeropoints.csv`
- `fit_prism_ab_zeropoints.csv`
- `fit_residuals.csv`
- `fit_prism_residuals.csv`
- `fit_iteration_summary.csv`
- diagnostic PNG files

## Querying A Calibration

For imaging, the fitted row-level AB zeropoint is defined so that

```text
m_AB = m_inst + fit_ab_zeropoint_mag.
```

Query one measurement:

```bash
paxphot query \
  --filter-name F129 \
  --detector-id 1 \
  --exposure-id 2 \
  --x 1024 \
  --y 2048 \
  --ice-thickness 0.3 \
  --instrumental-mag 19.5
```

Query a CSV table:

```bash
paxphot query \
  --query-csv my_instrumental_measurements.csv \
  --output-csv my_calibrated_measurements.csv \
  --passband-output-csv my_current_passbands.csv
```

Input rows require `filter_id` or `filter_name`, `detector_id`, `exposure_id`,
`x`, `y`, and `ice_thickness`. If `amp_id` is absent, it is computed from `x`.

## Python API

The command implementations are also importable:

```python
from paxphotometrica.simulate import SimConfig, simulate_data
from paxphotometrica.fit import FitConfig, load_data, run_fit
from paxphotometrica.query import QueryConfig, load_calibration_products
```

For example:

```python
from paxphotometrica.simulate import SimConfig, simulate_data

simulate_data(
    SimConfig(
        n_star=500,
        n_exp=12,
        n_det=1,
        output_dir="my_simulation",
    )
)
```

## Bundled Data

Small runtime reference files are installed under `paxphotometrica.data`.
Callers can override every bundled simulator input with a command-line path.

The BOSZ EMPCA basis is derived from the BOSZ High-Level Science Products,
which MAST distributes under CC BY 4.0:

- Bohlin et al. 2017, AJ, 153, 234
- Meszaros et al. 2024, A&A, 688, A197
- DOI `10.17909/T95G68`

See `src/paxphotometrica/data/README.md` for details. The raw BOSZ `m+0.00/`
source grid is not included in the package.

The supplied passband and prism tables are included as the current reference
inputs for this advanced prototype.

## Development

Run tests:

```bash
python -m pytest
```

Build a wheel and source distribution:

```bash
python -m pip install ".[release]"
python -m build
python -m twine check dist/*
```

The GitHub Actions test matrix covers Python 3.9 through 3.13. Publishing uses
PyPI Trusted Publishing from `.github/workflows/release.yml` when a GitHub
release is published.

Version `0.1.0` is an alpha prototype intended for research and calibration
experiments rather than production operations.

## License

PaxPhotometrica is released under the MIT License. See `LICENSE`.
