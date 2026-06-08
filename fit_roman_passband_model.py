#!/usr/bin/env python3
"""Fit a sparse linearized Roman-like WFI passband and ice calibration model.

The simulator generates broadband magnitudes from linear flux integrals. This
fitter iteratively linearizes those magnitudes around the current stellar SED,
passband, and ice model, solves a sparse weighted least-squares system for small
updates, damps the update, and repeats.

The model is intentionally not production-grade. It is a compact prototype for
studying identifiability and degeneracies among stellar SEDs, detector-level
passband shifts/widths, and wavelength-dependent ice optical depth.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
import warnings

_MPL_CACHE = Path(tempfile.gettempdir()) / "roman_passband_mpl_cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_CACHE.resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr


MAG_FACTOR = 2.5 / np.log(10.0)


@dataclass
class FitConfig:
    input_dir: str = "passband_sim_outputs"
    output_dir: str = "passband_fit_outputs"
    n_iter: int = 5
    damping: float = 0.5
    max_stars: int | None = None
    sigma_shift_prior: float = 0.02
    sigma_width_prior: float = 0.05
    sigma_ice_prior: float = 0.10
    sigma_temp_update_prior: float = 400.0
    sigma_ext_update_prior: float = 0.05
    temp_fd_step: float = 20.0
    ext_fd_step: float = 0.002
    min_temperature: float = 2800.0
    max_temperature: float = 12000.0
    min_extinction: float = 0.0
    max_extinction: float = 1.0
    chunk_size: int = 512
    random_seed: int = 12345


@dataclass
class DataBundle:
    measurements: pd.DataFrame
    wave: np.ndarray
    filter_ids: np.ndarray
    filter_names: list[str]
    detector_ids: np.ndarray
    star_ids: np.ndarray
    filter_param_id: np.ndarray
    detector_param_id: np.ndarray
    star_param_id: np.ndarray
    passbands: np.ndarray
    phi_shift: np.ndarray
    phi_width: np.ndarray
    ice_basis: np.ndarray
    filter_effective_wavelength: np.ndarray
    true_star_params: pd.DataFrame | None
    true_passband_params: pd.DataFrame | None
    true_ice_params: pd.DataFrame | None
    sim_metadata: dict


@dataclass
class ModelState:
    mag_norm: np.ndarray
    temperature: np.ndarray
    extinction: np.ndarray
    shift: np.ndarray
    width: np.ndarray
    ice_coeff: np.ndarray


@dataclass
class Linearization:
    mag_model: np.ndarray
    d_norm: np.ndarray
    d_temp: np.ndarray
    d_ext: np.ndarray
    r_shift: np.ndarray
    r_width: np.ndarray
    r_ice: np.ndarray


def trapz_integral(y: np.ndarray, wave: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compatibility wrapper for NumPy's trapezoid integration."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, wave, axis=axis)
    return np.trapz(y, wave, axis=axis)


def extinction_curve(wave: np.ndarray) -> np.ndarray:
    return (wave / 1.0) ** (-1.2)


def planck_like_lambda(wave_um: np.ndarray, temperature_k: np.ndarray) -> np.ndarray:
    """Same Planck-like SED family used by the simulator."""
    wave = np.asarray(wave_um)
    temp = np.asarray(temperature_k)
    c2_um_k = 14387.76877
    x = c2_um_k / (temp[..., None] * wave[None, :])
    x = np.clip(x, 1e-3, 700.0)
    b_lambda = 1.0 / (wave[None, :] ** 5 * np.expm1(x))
    x_ref = np.clip(c2_um_k / (temp * 1.0), 1e-3, 700.0)
    b_ref = 1.0 / np.expm1(x_ref)
    return b_lambda / b_ref[..., None]


def stellar_sed(
    wave: np.ndarray, mag_norm: np.ndarray, temperature: np.ndarray, extinction: np.ndarray
) -> np.ndarray:
    scale = 10.0 ** (-0.4 * np.asarray(mag_norm))
    shape = planck_like_lambda(wave, np.asarray(temperature))
    extinct = np.exp(-np.asarray(extinction)[..., None] * extinction_curve(wave)[None, :])
    return scale[..., None] * shape * extinct


def flux_to_mag(flux: np.ndarray) -> np.ndarray:
    return -2.5 * np.log10(np.maximum(flux, 1e-300))


def load_long_grid_csv(path: Path, value_columns: list[str], id_column: str) -> tuple[np.ndarray, dict]:
    """Load long-form wavelength data into arrays indexed by the id column."""
    table = pd.read_csv(path)
    wave = np.sort(table["wavelength_um"].unique())
    ids = np.sort(table[id_column].unique())
    values = {col: np.zeros((ids.size, wave.size), dtype=float) for col in value_columns}

    for i, item_id in enumerate(ids):
        sub = table.loc[table[id_column] == item_id].sort_values("wavelength_um")
        if not np.allclose(sub["wavelength_um"].to_numpy(), wave):
            raise ValueError(f"Inconsistent wavelength grid in {path}")
        for col in value_columns:
            values[col][i] = sub[col].to_numpy()

    values[id_column + "s"] = ids
    return wave, values


def load_data(config: FitConfig) -> DataBundle:
    input_dir = Path(config.input_dir)
    measurements = pd.read_csv(input_dir / "measurements.csv")
    if config.max_stars is not None:
        keep_star_ids = np.sort(measurements["star_id"].unique())[: config.max_stars]
        measurements = measurements.loc[measurements["star_id"].isin(keep_star_ids)].copy()
        measurements.reset_index(drop=True, inplace=True)

    wave, pass_data = load_long_grid_csv(
        input_dir / "nominal_passbands.csv", ["throughput"], "filter_id"
    )
    wave_modes, mode_data = load_long_grid_csv(
        input_dir / "passband_modes.csv", ["phi_shift", "phi_width"], "filter_id"
    )
    wave_ice, ice_data = load_long_grid_csv(input_dir / "ice_basis.csv", ["psi"], "basis_id")
    if not np.allclose(wave, wave_modes) or not np.allclose(wave, wave_ice):
        raise ValueError("Passband, mode, and ice-basis wavelength grids do not match")

    filter_ids = np.sort(measurements["filter_id"].unique())
    detector_ids = np.sort(measurements["detector_id"].unique())
    star_ids, star_param_id = np.unique(measurements["star_id"].to_numpy(int), return_inverse=True)
    filter_lookup = {value: i for i, value in enumerate(filter_ids)}
    detector_lookup = {value: i for i, value in enumerate(detector_ids)}
    filter_param_id = measurements["filter_id"].map(filter_lookup).to_numpy(int)
    detector_param_id = measurements["detector_id"].map(detector_lookup).to_numpy(int)

    all_pass_filter_ids = pass_data["filter_ids"]
    pass_lookup = {value: i for i, value in enumerate(all_pass_filter_ids)}
    pass_indices = np.array([pass_lookup[value] for value in filter_ids], dtype=int)

    passband_table = pd.read_csv(input_dir / "nominal_passbands.csv")
    if "filter_name" in passband_table.columns:
        filter_name_lookup = (
            passband_table[["filter_id", "filter_name"]]
            .drop_duplicates()
            .set_index("filter_id")["filter_name"]
            .to_dict()
        )
        filter_names = [str(filter_name_lookup.get(value, f"filter {value}")) for value in filter_ids]
    else:
        filter_names = [f"filter {value}" for value in filter_ids]

    passbands = pass_data["throughput"][pass_indices]
    phi_shift = mode_data["phi_shift"][pass_indices]
    phi_width = mode_data["phi_width"][pass_indices]
    ice_basis = ice_data["psi"]

    denom = trapz_integral(passbands, wave, axis=1)
    filter_eff = trapz_integral(passbands * wave[None, :], wave, axis=1) / denom

    def read_optional(name: str) -> pd.DataFrame | None:
        path = input_dir / name
        return pd.read_csv(path) if path.exists() else None

    metadata_path = input_dir / "simulation_metadata.json"
    sim_metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as handle:
            sim_metadata = json.load(handle)

    return DataBundle(
        measurements=measurements,
        wave=wave,
        filter_ids=filter_ids,
        filter_names=filter_names,
        detector_ids=detector_ids,
        star_ids=star_ids,
        filter_param_id=filter_param_id,
        detector_param_id=detector_param_id,
        star_param_id=star_param_id,
        passbands=passbands,
        phi_shift=phi_shift,
        phi_width=phi_width,
        ice_basis=ice_basis,
        filter_effective_wavelength=filter_eff,
        true_star_params=read_optional("true_star_params.csv"),
        true_passband_params=read_optional("true_passband_params.csv"),
        true_ice_params=read_optional("true_ice_params.csv"),
        sim_metadata=sim_metadata,
    )


def make_initial_sed_grid(data: DataBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute nominal-passband colors for a temperature/extinction grid."""
    temps = np.linspace(3200.0, 9500.0, 31)
    exts = np.linspace(0.0, 0.45, 19)
    grid_temp, grid_ext = np.meshgrid(temps, exts, indexing="ij")
    flat_temp = grid_temp.ravel()
    flat_ext = grid_ext.ravel()
    sed_shape = stellar_sed(data.wave, np.zeros_like(flat_temp), flat_temp, flat_ext)

    shape_mag = np.zeros((flat_temp.size, data.filter_ids.size))
    for filt in range(data.filter_ids.size):
        flux = trapz_integral(sed_shape * data.passbands[filt][None, :], data.wave, axis=1)
        shape_mag[:, filt] = flux_to_mag(flux)
    return flat_temp, flat_ext, shape_mag


def fit_initial_stellar_seds(data: DataBundle) -> ModelState:
    """Initial star-by-star grid search with nominal passbands and no ice.

    For each grid point in temperature/extinction, the magnitude normalization is
    analytic: it is the weighted mean of observed magnitude minus model color.
    """
    grid_temp, grid_ext, shape_mag = make_initial_sed_grid(data)
    mag = data.measurements["mag_obs"].to_numpy(float)
    sigma = data.measurements["mag_unc"].to_numpy(float)
    weight = 1.0 / sigma**2

    mag_norm = np.zeros(data.star_ids.size)
    temperature = np.zeros(data.star_ids.size)
    extinction = np.zeros(data.star_ids.size)

    for star_index in range(data.star_ids.size):
        obs = np.nonzero(data.star_param_id == star_index)[0]
        filt = data.filter_param_id[obs]
        y = mag[obs]
        w = weight[obs]
        offsets = shape_mag[:, filt]
        norm_grid = np.sum(w[None, :] * (y[None, :] - offsets), axis=1) / np.sum(w)
        resid = y[None, :] - (norm_grid[:, None] + offsets)
        chi2 = np.sum(w[None, :] * resid**2, axis=1)
        best = int(np.argmin(chi2))
        mag_norm[star_index] = norm_grid[best]
        temperature[star_index] = grid_temp[best]
        extinction[star_index] = grid_ext[best]

    shift = np.zeros((data.filter_ids.size, data.detector_ids.size))
    width = np.zeros_like(shift)
    ice_coeff = np.zeros(data.ice_basis.shape[0])
    return ModelState(mag_norm, temperature, extinction, shift, width, ice_coeff)


def evaluate_model_and_responses(
    data: DataBundle, state: ModelState, config: FitConfig, need_responses: bool = True
) -> Linearization:
    """Evaluate current model magnitudes and, optionally, linear response columns.

    The response coefficients are derivatives of broadband magnitudes, but each
    derivative is computed from the correct linear flux integral over wavelength.
    """
    n_obs = len(data.measurements)
    n_ice = data.ice_basis.shape[0]
    mag_model = np.zeros(n_obs)
    d_norm = np.ones(n_obs)
    d_temp = np.zeros(n_obs)
    d_ext = np.zeros(n_obs)
    r_shift = np.zeros(n_obs)
    r_width = np.zeros(n_obs)
    r_ice = np.zeros((n_obs, n_ice))

    ice_amount = data.measurements["ice_amount_obs"].to_numpy(float)
    tau = state.ice_coeff @ data.ice_basis

    for start in range(0, n_obs, config.chunk_size):
        end = min(n_obs, start + config.chunk_size)
        sl = slice(start, end)
        star = data.star_param_id[sl]
        filt = data.filter_param_id[sl]
        det = data.detector_param_id[sl]
        ice = ice_amount[sl]

        sed = stellar_sed(
            data.wave,
            state.mag_norm[star],
            state.temperature[star],
            state.extinction[star],
        )
        logt = (
            state.shift[filt, det][:, None] * data.phi_shift[filt]
            + state.width[filt, det][:, None] * data.phi_width[filt]
            - ice[:, None] * tau[None, :]
        )
        t_current = data.passbands[filt] * np.exp(logt)
        weighted = sed * t_current
        denom = trapz_integral(weighted, data.wave, axis=1)
        mag_model[sl] = flux_to_mag(denom)

        if not need_responses:
            continue

        temp_step = config.temp_fd_step
        temp_plus = np.clip(
            state.temperature[star] + temp_step,
            config.min_temperature,
            config.max_temperature,
        )
        temp_minus = np.clip(
            state.temperature[star] - temp_step,
            config.min_temperature,
            config.max_temperature,
        )
        sed_t_plus = stellar_sed(data.wave, state.mag_norm[star], temp_plus, state.extinction[star])
        sed_t_minus = stellar_sed(
            data.wave, state.mag_norm[star], temp_minus, state.extinction[star]
        )
        mag_t_plus = flux_to_mag(trapz_integral(sed_t_plus * t_current, data.wave, axis=1))
        mag_t_minus = flux_to_mag(trapz_integral(sed_t_minus * t_current, data.wave, axis=1))
        d_temp[sl] = (mag_t_plus - mag_t_minus) / np.maximum(temp_plus - temp_minus, 1e-6)

        ext_step = config.ext_fd_step
        ext_plus = np.clip(
            state.extinction[star] + ext_step,
            config.min_extinction,
            config.max_extinction,
        )
        ext_minus = np.clip(
            state.extinction[star] - ext_step,
            config.min_extinction,
            config.max_extinction,
        )
        sed_e_plus = stellar_sed(
            data.wave, state.mag_norm[star], state.temperature[star], ext_plus
        )
        sed_e_minus = stellar_sed(
            data.wave, state.mag_norm[star], state.temperature[star], ext_minus
        )
        mag_e_plus = flux_to_mag(trapz_integral(sed_e_plus * t_current, data.wave, axis=1))
        mag_e_minus = flux_to_mag(trapz_integral(sed_e_minus * t_current, data.wave, axis=1))
        d_ext[sl] = (mag_e_plus - mag_e_minus) / np.maximum(ext_plus - ext_minus, 1e-9)

        r_shift[sl] = -MAG_FACTOR * (
            trapz_integral(weighted * data.phi_shift[filt], data.wave, axis=1) / denom
        )
        r_width[sl] = -MAG_FACTOR * (
            trapz_integral(weighted * data.phi_width[filt], data.wave, axis=1) / denom
        )
        for basis_id in range(n_ice):
            numerator = trapz_integral(weighted * data.ice_basis[basis_id][None, :], data.wave, axis=1)
            r_ice[sl, basis_id] = MAG_FACTOR * ice * numerator / denom

    return Linearization(mag_model, d_norm, d_temp, d_ext, r_shift, r_width, r_ice)


def parameter_slices(data: DataBundle) -> dict[str, slice]:
    n_star = data.star_ids.size
    n_pass = data.filter_ids.size * data.detector_ids.size
    n_ice = data.ice_basis.shape[0]
    start = 0
    slices = {}
    slices["norm"] = slice(start, start + n_star)
    start += n_star
    slices["temp"] = slice(start, start + n_star)
    start += n_star
    slices["ext"] = slice(start, start + n_star)
    start += n_star
    slices["shift"] = slice(start, start + n_pass)
    start += n_pass
    slices["width"] = slice(start, start + n_pass)
    start += n_pass
    slices["ice"] = slice(start, start + n_ice)
    return slices


def build_sparse_system(
    data: DataBundle, state: ModelState, lin: Linearization, config: FitConfig
) -> tuple[coo_matrix, np.ndarray, dict[str, slice]]:
    """Build weighted sparse system for one linearized update."""
    slices = parameter_slices(data)
    n_params = slices["ice"].stop
    rows = []
    cols = []
    vals = []
    rhs = []
    row = 0

    obs_mag = data.measurements["mag_obs"].to_numpy(float)
    obs_unc = data.measurements["mag_unc"].to_numpy(float)
    residual = obs_mag - lin.mag_model

    n_det = data.detector_ids.size
    for obs_index in range(len(data.measurements)):
        weight = 1.0 / obs_unc[obs_index]
        star = data.star_param_id[obs_index]
        filt = data.filter_param_id[obs_index]
        det = data.detector_param_id[obs_index]
        pass_index = filt * n_det + det

        entries = [
            (slices["norm"].start + star, lin.d_norm[obs_index]),
            (slices["temp"].start + star, lin.d_temp[obs_index]),
            (slices["ext"].start + star, lin.d_ext[obs_index]),
            (slices["shift"].start + pass_index, lin.r_shift[obs_index]),
            (slices["width"].start + pass_index, lin.r_width[obs_index]),
        ]
        for basis_id in range(data.ice_basis.shape[0]):
            entries.append((slices["ice"].start + basis_id, lin.r_ice[obs_index, basis_id]))

        for col, value in entries:
            rows.append(row)
            cols.append(col)
            vals.append(value * weight)
        rhs.append(residual[obs_index] * weight)
        row += 1

    # Priors on current+update for calibration parameters choose a conservative
    # gauge in this degenerate chromatic problem.
    for flat_index, current in enumerate(state.shift.ravel()):
        rows.append(row)
        cols.append(slices["shift"].start + flat_index)
        vals.append(1.0 / config.sigma_shift_prior)
        rhs.append(-current / config.sigma_shift_prior)
        row += 1

    for flat_index, current in enumerate(state.width.ravel()):
        rows.append(row)
        cols.append(slices["width"].start + flat_index)
        vals.append(1.0 / config.sigma_width_prior)
        rhs.append(-current / config.sigma_width_prior)
        row += 1

    for basis_id, current in enumerate(state.ice_coeff):
        rows.append(row)
        cols.append(slices["ice"].start + basis_id)
        vals.append(1.0 / config.sigma_ice_prior)
        rhs.append(-current / config.sigma_ice_prior)
        row += 1

    # SED color/extinction update priors prevent each star from freely absorbing
    # global passband and ice structure in every iteration.
    for star in range(data.star_ids.size):
        rows.append(row)
        cols.append(slices["temp"].start + star)
        vals.append(1.0 / config.sigma_temp_update_prior)
        rhs.append(0.0)
        row += 1

        rows.append(row)
        cols.append(slices["ext"].start + star)
        vals.append(1.0 / config.sigma_ext_update_prior)
        rhs.append(0.0)
        row += 1

    A = coo_matrix((vals, (rows, cols)), shape=(row, n_params)).tocsr()
    return A, np.asarray(rhs), slices


def apply_update(
    state: ModelState, theta: np.ndarray, slices: dict[str, slice], data: DataBundle, config: FitConfig
) -> None:
    n_star = data.star_ids.size
    n_filter = data.filter_ids.size
    n_det = data.detector_ids.size
    damping = config.damping

    state.mag_norm += damping * theta[slices["norm"]]
    state.temperature += damping * theta[slices["temp"]]
    state.extinction += damping * theta[slices["ext"]]
    state.temperature = np.clip(state.temperature, config.min_temperature, config.max_temperature)
    state.extinction = np.clip(state.extinction, config.min_extinction, config.max_extinction)
    state.shift += damping * theta[slices["shift"]].reshape(n_filter, n_det)
    state.width += damping * theta[slices["width"]].reshape(n_filter, n_det)
    state.ice_coeff += damping * theta[slices["ice"]]
    _ = n_star


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def run_fit(data: DataBundle, config: FitConfig) -> tuple[ModelState, pd.DataFrame, np.ndarray]:
    state = fit_initial_stellar_seds(data)
    output_rows = []

    initial_lin = evaluate_model_and_responses(data, state, config, need_responses=False)
    obs_mag = data.measurements["mag_obs"].to_numpy(float)
    initial_resid = obs_mag - initial_lin.mag_model
    output_rows.append({"iteration": 0, "rms_residual": rms(initial_resid), "lsmr_iters": 0})
    print(f"Initial RMS residual: {rms(initial_resid):.6f} mag")

    for iteration in range(1, config.n_iter + 1):
        lin = evaluate_model_and_responses(data, state, config, need_responses=True)
        A, b, slices = build_sparse_system(data, state, lin, config)
        result = lsmr(A, b, atol=1e-9, btol=1e-9, maxiter=2000, show=False)
        theta = result[0]
        apply_update(state, theta, slices, data, config)

        updated = evaluate_model_and_responses(data, state, config, need_responses=False)
        resid = obs_mag - updated.mag_model
        output_rows.append(
            {
                "iteration": iteration,
                "rms_residual": rms(resid),
                "lsmr_iters": result[2],
                "lsmr_stop_code": result[1],
                "update_norm": float(np.linalg.norm(theta)),
            }
        )
        print(
            f"Iteration {iteration}: RMS residual = {rms(resid):.6f} mag, "
            f"LSMR iters = {result[2]}"
        )

    final_lin = evaluate_model_and_responses(data, state, config, need_responses=False)
    final_resid = obs_mag - final_lin.mag_model
    return state, pd.DataFrame(output_rows), final_resid


def make_truth_arrays(data: DataBundle) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    true_shift = None
    true_width = None
    true_ice = None

    if data.true_passband_params is not None:
        true_shift = np.zeros((data.filter_ids.size, data.detector_ids.size))
        true_width = np.zeros_like(true_shift)
        for filt_i, filt in enumerate(data.filter_ids):
            for det_i, det in enumerate(data.detector_ids):
                row = data.true_passband_params.loc[
                    (data.true_passband_params["filter_id"] == filt)
                    & (data.true_passband_params["detector_id"] == det)
                ]
                if not row.empty:
                    true_shift[filt_i, det_i] = row.iloc[0]["delta_lambda_um"]
                    true_width[filt_i, det_i] = row.iloc[0]["width"]

    if data.true_ice_params is not None:
        true_ice = np.zeros(data.ice_basis.shape[0])
        for _, row in data.true_ice_params.iterrows():
            basis_id = int(row["basis_id"])
            if 0 <= basis_id < true_ice.size:
                true_ice[basis_id] = row["ice_coeff"]

    return true_shift, true_width, true_ice


def save_outputs(
    data: DataBundle,
    state: ModelState,
    summary: pd.DataFrame,
    final_resid: np.ndarray,
    config: FitConfig,
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "star_id": data.star_ids,
            "mag_norm": state.mag_norm,
            "temperature_k": state.temperature,
            "extinction": state.extinction,
        }
    ).to_csv(output_dir / "fit_star_params.csv", index=False)

    pass_rows = []
    for filt_i, filt in enumerate(data.filter_ids):
        for det_i, det in enumerate(data.detector_ids):
            pass_rows.append(
                {
                    "filter_id": filt,
                    "filter_name": data.filter_names[filt_i],
                    "detector_id": det,
                    "delta_lambda_um": state.shift[filt_i, det_i],
                    "width": state.width[filt_i, det_i],
                }
            )
    pd.DataFrame(pass_rows).to_csv(output_dir / "fit_passband_params.csv", index=False)

    pd.DataFrame(
        {"basis_id": np.arange(state.ice_coeff.size), "ice_coeff": state.ice_coeff}
    ).to_csv(output_dir / "fit_ice_params.csv", index=False)

    summary.to_csv(output_dir / "fit_iteration_summary.csv", index=False)

    residuals = data.measurements.copy()
    residuals["mag_residual"] = final_resid
    residuals.to_csv(output_dir / "fit_residuals.csv", index=False)

    with open(output_dir / "fit_config.json", "w", encoding="utf-8") as handle:
        json.dump(config.__dict__, handle, indent=2, sort_keys=True)

    # Copy metadata for convenience when fit outputs are inspected standalone.
    meta_path = Path(config.input_dir) / "simulation_metadata.json"
    if meta_path.exists():
        shutil.copy(meta_path, output_dir / "simulation_metadata.json")


def print_diagnostics(
    data: DataBundle, state: ModelState, summary: pd.DataFrame, final_resid: np.ndarray
) -> None:
    true_shift, true_width, true_ice = make_truth_arrays(data)

    print("\nRoman passband / ice sparse fit diagnostics")
    print("-------------------------------------------")
    print(f"Number of observations: {len(data.measurements)}")
    print(f"Number of stars: {data.star_ids.size}")
    print(f"Number of filters: {data.filter_ids.size}")
    print(f"Number of detectors: {data.detector_ids.size}")
    n_params = 3 * data.star_ids.size + 2 * data.filter_ids.size * data.detector_ids.size
    n_params += data.ice_basis.shape[0]
    print(f"Number of fitted linearized parameters per iteration: {n_params}")
    print(f"Initial RMS residual: {summary.iloc[0]['rms_residual']:.6f} mag")
    print(f"Final RMS residual: {rms(final_resid):.6f} mag")

    residual_frame = data.measurements[["filter_id", "detector_id"]].copy()
    residual_frame["residual"] = final_resid
    print("RMS residual by filter:")
    for filt, sub in residual_frame.groupby("filter_id"):
        filt_index = int(np.nonzero(data.filter_ids == filt)[0][0])
        print(f"  {data.filter_names[filt_index]}: {rms(sub['residual'].to_numpy()):.6f} mag")
    print("RMS residual by detector:")
    for det, sub in residual_frame.groupby("detector_id"):
        print(f"  detector {int(det):02d}: {rms(sub['residual'].to_numpy()):.6f} mag")

    if true_shift is not None:
        print(f"Passband shift RMS error: {rms(state.shift - true_shift):.6f} um")
    if true_width is not None:
        print(f"Passband width RMS error: {rms(state.width - true_width):.6f}")
    if true_ice is not None:
        true_tau = true_ice @ data.ice_basis
        fit_tau = state.ice_coeff @ data.ice_basis
        print(f"Ice optical-depth shape RMS error: {rms(fit_tau - true_tau):.6f}")

    fit_tau = state.ice_coeff @ data.ice_basis
    negative_fraction = np.mean(fit_tau < -1e-3)
    if negative_fraction > 0.1:
        print(
            "WARNING: recovered tau_ice(lambda) is negative over "
            f"{negative_fraction:.1%} of the wavelength grid."
        )
    print(
        "Note: stellar extinction, passband color terms, and ice shape remain "
        "partially degenerate; recovery depends on priors and ice-amount leverage."
    )


def make_plots(
    data: DataBundle,
    state: ModelState,
    summary: pd.DataFrame,
    final_resid: np.ndarray,
    config: FitConfig,
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    true_shift, true_width, true_ice = make_truth_arrays(data)

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(summary["iteration"], summary["rms_residual"], marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("RMS residual [mag]")
    plt.title("Residuals by iteration")
    plt.tight_layout()
    plt.savefig(output_dir / "residuals_by_iteration.png", dpi=160)
    plt.close()

    if true_shift is not None:
        lim = 1.1 * np.max(np.abs(np.r_[true_shift.ravel(), state.shift.ravel()]))
        plt.figure(figsize=(5.5, 5.5))
        plt.scatter(true_shift.ravel(), state.shift.ravel(), s=45)
        plt.plot([-lim, lim], [-lim, lim], color="0.2", linewidth=1)
        plt.xlabel("True shift [um]")
        plt.ylabel("Fitted shift [um]")
        plt.title("Passband shift recovery")
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.savefig(output_dir / "passband_shift_true_vs_fit.png", dpi=160)
        plt.close()

    if true_width is not None:
        lim = 1.1 * np.max(np.abs(np.r_[true_width.ravel(), state.width.ravel()]))
        plt.figure(figsize=(5.5, 5.5))
        plt.scatter(true_width.ravel(), state.width.ravel(), s=45)
        plt.plot([-lim, lim], [-lim, lim], color="0.2", linewidth=1)
        plt.xlabel("True width coefficient")
        plt.ylabel("Fitted width coefficient")
        plt.title("Passband width recovery")
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.savefig(output_dir / "passband_width_true_vs_fit.png", dpi=160)
        plt.close()

    if true_ice is not None:
        plt.figure(figsize=(7, 4.5))
        plt.plot(data.wave, true_ice @ data.ice_basis, label="true")
        plt.plot(data.wave, state.ice_coeff @ data.ice_basis, label="fit")
        plt.xlabel("Wavelength [um]")
        plt.ylabel("Optical depth per ice amount")
        plt.title("Ice optical-depth shape")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "ice_tau_true_vs_fit.png", dpi=160)
        plt.close()

    filt = data.filter_param_id
    eff_wave = data.filter_effective_wavelength[filt]
    plt.figure(figsize=(7, 4.5))
    plt.scatter(eff_wave, final_resid, s=4, alpha=0.25)
    grouped = pd.DataFrame({"wave": eff_wave, "resid": final_resid}).groupby("wave")
    med = grouped.median().reset_index()
    plt.plot(med["wave"], med["resid"], color="black", marker="o", linewidth=1)
    plt.axhline(0.0, color="0.3", linewidth=1)
    plt.xlabel("Filter effective wavelength [um]")
    plt.ylabel("Residual [mag]")
    plt.title("Residual versus filter wavelength")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_vs_wavelength_or_filter.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.scatter(data.measurements["ice_amount_obs"], final_resid, s=4, alpha=0.25)
    plt.axhline(0.0, color="0.3", linewidth=1)
    plt.xlabel("ice_amount_obs")
    plt.ylabel("Residual [mag]")
    plt.title("Residual versus ice amount")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_vs_ice_amount.png", dpi=160)
    plt.close()

    if data.true_star_params is not None:
        truth = data.true_star_params.set_index("star_id").loc[data.star_ids]
        plt.figure(figsize=(10, 4.5))
        plt.subplot(1, 2, 1)
        plt.scatter(truth["temperature_k"], state.temperature, s=5, alpha=0.35)
        tmin = min(truth["temperature_k"].min(), state.temperature.min())
        tmax = max(truth["temperature_k"].max(), state.temperature.max())
        plt.plot([tmin, tmax], [tmin, tmax], color="0.2", linewidth=1)
        plt.xlabel("True temperature [K]")
        plt.ylabel("Fitted temperature [K]")
        plt.subplot(1, 2, 2)
        plt.scatter(truth["extinction"], state.extinction, s=5, alpha=0.35)
        emax = max(truth["extinction"].max(), state.extinction.max())
        plt.plot([0, emax], [0, emax], color="0.2", linewidth=1)
        plt.xlabel("True extinction")
        plt.ylabel("Fitted extinction")
        plt.suptitle("Stellar parameter recovery")
        plt.tight_layout()
        plt.savefig(output_dir / "stellar_param_recovery.png", dpi=160)
        plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.hist(final_resid, bins=60, histtype="stepfilled", alpha=0.75)
    plt.xlabel("Residual [mag]")
    plt.ylabel("Count")
    plt.title("Residual histogram")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_histogram.png", dpi=160)
    plt.close()


def parse_args() -> FitConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=FitConfig.input_dir)
    parser.add_argument("--output-dir", default=FitConfig.output_dir)
    parser.add_argument("--n-iter", type=int, default=FitConfig.n_iter)
    parser.add_argument("--damping", type=float, default=FitConfig.damping)
    parser.add_argument("--max-stars", type=int, default=None)
    args = parser.parse_args()
    return FitConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_iter=args.n_iter,
        damping=args.damping,
        max_stars=args.max_stars,
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    config = parse_args()
    data = load_data(config)
    state, summary, final_resid = run_fit(data, config)
    save_outputs(data, state, summary, final_resid, config)
    print_diagnostics(data, state, summary, final_resid)
    make_plots(data, state, summary, final_resid, config)
    print(f"Saved fit outputs to {Path(config.output_dir).resolve()}")


if __name__ == "__main__":
    main()
