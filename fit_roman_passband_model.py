#!/usr/bin/env python3
"""Fit a sparse Roman-like WFI imaging, passband, ice, and prism model.

The simulator generates broadband magnitudes from linear flux integrals. This
fitter iteratively linearizes those magnitudes around the current stellar SED,
passband, and ice model, solves a sparse weighted least-squares system for small
updates, damps the update, and repeats.

The model is intentionally not production-grade. It is a compact prototype for
studying identifiability and degeneracies among stellar SEDs, detector-level
passband shifts/widths, and a wavelength/thickness-dependent ice
log-throughput surface. Sparse prism spectra add one sensitivity parameter per
wavelength pixel while sharing the imaging amplifier gains.
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
from scipy.sparse.linalg import lsmr, lsqr


MAG_FACTOR = 2.5 / np.log(10.0)
SPEED_OF_LIGHT_CM_S = 2.99792458e10
MICRON_TO_CM = 1.0e-4
AB_FNU_CGS = 3631.0e-23  # erg / s / cm^2 / Hz


class BOSZEMPCASEDLibrary:
    """BOSZ EMPCA stellar SED basis evaluated in normalized log-flux space."""

    def __init__(self, basis_path: str | Path):
        self.path = Path(basis_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Missing BOSZ EMPCA basis file: {self.path}")
        data = np.load(self.path, allow_pickle=False)
        self.wave_micron = data["wave_micron"].astype(float)
        self.source_wave_micron = self.wave_micron
        self.mean_log_flux = data["mean_log_flux"].astype(float)
        self.components = data["components"].astype(float)
        self.coefficients = data["coefficients"].astype(float)
        self.model_files = data["model_files"].astype(str)
        self.metadata = json.loads(data["metadata"].item()) if "metadata" in data.files else {}

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    @property
    def coefficient_scales(self) -> np.ndarray:
        scales = np.std(self.coefficients, axis=0)
        return np.maximum(scales, 1e-3)

    def resampled_to(self, wave: np.ndarray) -> "BOSZEMPCASEDLibrary":
        """Interpolate normalized log-flux basis vectors to the passband grid."""
        if wave.min() < self.wave_micron.min() or wave.max() > self.wave_micron.max():
            print(
                "Warning: passband wavelength grid extends slightly outside the BOSZ "
                "basis; endpoint log-flux values will be used for extrapolation."
            )
        clone = object.__new__(BOSZEMPCASEDLibrary)
        clone.path = self.path
        clone.wave_micron = np.asarray(wave, dtype=float)
        clone.source_wave_micron = self.source_wave_micron
        clone.mean_log_flux = np.interp(wave, self.wave_micron, self.mean_log_flux)
        clone.components = np.vstack(
            [np.interp(wave, self.wave_micron, component) for component in self.components]
        )
        clone.coefficients = self.coefficients
        clone.model_files = self.model_files
        clone.metadata = self.metadata
        return clone

    def sed_shape_from_coefficients(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate unitless relative SED shapes for one or many coefficient vectors."""
        theta = np.asarray(theta, dtype=float)
        log_sed = self.mean_log_flux + theta @ self.components
        return np.exp(log_sed)

    def sed_from_coefficients(
        self, theta: np.ndarray, mag_norm: np.ndarray, reference_passband: np.ndarray
    ) -> np.ndarray:
        """Evaluate physical-scale SEDs with ``mag_norm`` as reference AB mag."""
        shape = self.sed_shape_from_coefficients(theta)
        mag_norm = np.asarray(mag_norm, dtype=float)
        shape_count = photon_count_integral(
            shape * reference_passband[None, :], self.wave_micron, axis=1
        )
        ab_count = ab_reference_count(self.wave_micron, reference_passband)
        scale = ab_count * 10.0 ** (-0.4 * mag_norm) / np.maximum(shape_count, 1e-300)
        return scale[..., None] * shape


@dataclass
class FitConfig:
    input_dir: str = "passband_sim_outputs"
    output_dir: str = "passband_fit_outputs"
    sed_basis_path: str = ""
    n_iter: int = 8
    damping: float = 0.5
    min_damping: float = 0.0078125
    max_stars: int | None = None
    sigma_shift_prior: float = 0.02
    sigma_width_prior: float = 0.05
    sigma_ice_prior: float = 0.10
    sigma_zero_ice_surface_prior: float = 1e-4
    sigma_prism_response_prior: float = 0.20
    sigma_prism_response_smoothness: float = 0.01
    sigma_sed_coeff_update_prior_scale: float = 0.05
    sigma_amp_prior: float = 0.02
    sigma_amp_sum_constraint: float = 1e-4
    nx: int = 4096
    ny: int = 4096
    n_amp: int = 32
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
    exposure_id: np.ndarray
    filter_param_id: np.ndarray
    detector_param_id: np.ndarray
    star_param_id: np.ndarray
    amp_id: np.ndarray
    prism_measurements: pd.DataFrame
    prism_wave: np.ndarray
    prism_bin_width: np.ndarray
    prism_nominal_throughput: np.ndarray
    prism_star_param_id: np.ndarray
    prism_detector_param_id: np.ndarray
    prism_exposure_id: np.ndarray
    prism_amp_id: np.ndarray
    prism_pixel_id: np.ndarray
    prism_mean_log_flux: np.ndarray
    prism_components: np.ndarray
    prism_ice_loglam_basis: np.ndarray
    free_exposure_ids: np.ndarray
    exposure_free_index: np.ndarray
    star_is_calibrator: np.ndarray
    free_star_indices: np.ndarray
    free_star_index: np.ndarray
    passbands: np.ndarray
    reference_filter_index: int
    phi_shift: np.ndarray
    phi_width: np.ndarray
    sed_library: BOSZEMPCASEDLibrary
    ice_loglam_nodes: np.ndarray
    ice_thickness_nodes: np.ndarray
    ice_loglam_basis: np.ndarray
    filter_effective_wavelength: np.ndarray
    absolute_calibrators: pd.DataFrame | None
    true_star_params: pd.DataFrame | None
    true_passband_params: pd.DataFrame | None
    true_ice_spline_params: pd.DataFrame | None
    true_exposure_zeropoints: pd.DataFrame | None
    true_smooth_coeffs: pd.DataFrame | None
    true_amp_offsets: pd.DataFrame | None
    true_prism_response: pd.DataFrame | None
    sim_metadata: dict


@dataclass
class ModelState:
    mag_norm: np.ndarray
    sed_coeff: np.ndarray
    shift: np.ndarray
    width: np.ndarray
    ice_coeff: np.ndarray
    prism_response: np.ndarray
    zp: np.ndarray
    smooth_coeff: np.ndarray
    amp_offset: np.ndarray


@dataclass
class Linearization:
    mag_model: np.ndarray
    d_norm: np.ndarray
    d_sed_coeff: np.ndarray
    r_shift: np.ndarray
    r_width: np.ndarray
    r_ice: np.ndarray


@dataclass
class PrismLinearization:
    mag_model: np.ndarray
    d_norm: np.ndarray
    d_sed_coeff: np.ndarray
    r_ice: np.ndarray


def trapz_integral(y: np.ndarray, wave: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compatibility wrapper for NumPy's trapezoid integration."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, wave, axis=axis)
    return np.trapz(y, wave, axis=axis)


def ab_f_lambda_per_micron(wave_um: np.ndarray) -> np.ndarray:
    """AB reference spectrum, flat f_nu=3631 Jy, as f_lambda per micron."""
    wave_cm = np.asarray(wave_um, dtype=float) * MICRON_TO_CM
    return AB_FNU_CGS * SPEED_OF_LIGHT_CM_S / wave_cm**2 * MICRON_TO_CM


def photon_count_integral(y: np.ndarray, wave_um: np.ndarray, axis: int = -1) -> np.ndarray:
    """Photon-counting integral up to the constant 1/(hc)."""
    return trapz_integral(y * np.asarray(wave_um), wave_um, axis=axis)


def ab_reference_count(wave_um: np.ndarray, throughput: np.ndarray) -> float:
    """Photon-counting AB reference integral for one passband."""
    return float(photon_count_integral(ab_f_lambda_per_micron(wave_um) * throughput, wave_um))


def flux_to_abmag(count_flux: np.ndarray, wave_um: np.ndarray, throughput: np.ndarray) -> np.ndarray:
    """Convert photon-counting flux integral to AB magnitude."""
    ref = ab_reference_count(wave_um, throughput)
    return -2.5 * np.log10(np.maximum(count_flux, 1e-300) / ref)


def counts_to_instrumental_mag(count_flux: np.ndarray) -> np.ndarray:
    """Convert source count integral to an instrumental magnitude."""
    return -2.5 * np.log10(np.maximum(count_flux, 1e-300))


def amp_id_from_x(x: np.ndarray, nx: int = 4096, n_amp: int = 32) -> np.ndarray:
    """Return amplifier stripe id for detector x pixel coordinate."""
    amp_width = nx // n_amp
    amp_id = np.floor(np.asarray(x) / amp_width).astype(int)
    return np.clip(amp_id, 0, n_amp - 1)


def normalized_xy(
    x: np.ndarray, y: np.ndarray, nx: int = 4096, ny: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """Map detector pixels to [-1, 1] normalized coordinates."""
    xn = 2.0 * (x / (nx - 1.0)) - 1.0
    yn = 2.0 * (y / (ny - 1.0)) - 1.0
    return xn, yn


def poly_basis(xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
    """Smooth star-flat polynomial terms: x, y, x^2, x*y, y^2."""
    xn = np.asarray(xn)
    yn = np.asarray(yn)
    return np.column_stack((xn, yn, xn**2, xn * yn, yn**2))


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


def load_ice_spline_nodes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load rectangular ice spline grid nodes from simulator output."""
    table = pd.read_csv(path)
    loglam_nodes = np.sort(table["log10_wavelength"].unique())
    thickness_nodes = np.sort(table["ice_thickness"].unique())
    expected = loglam_nodes.size * thickness_nodes.size
    if len(table.drop_duplicates(["log10_wavelength", "ice_thickness"])) != expected:
        raise ValueError("Ice spline node file is not a complete rectangular grid")
    return loglam_nodes, thickness_nodes


def make_loglam_basis(wave: np.ndarray, loglam_nodes: np.ndarray) -> np.ndarray:
    """Linear spline basis functions evaluated on the wavelength grid."""
    log_wave = np.log10(wave)
    basis = np.zeros((loglam_nodes.size, wave.size))
    for node_id in range(loglam_nodes.size):
        unit = np.zeros(loglam_nodes.size)
        unit[node_id] = 1.0
        basis[node_id] = np.interp(log_wave, loglam_nodes, unit)
    return basis


def thickness_brackets(
    thickness: np.ndarray, thickness_nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return lower/upper node indices and linear interpolation weights."""
    t = np.clip(thickness, thickness_nodes[0], thickness_nodes[-1])
    hi = np.searchsorted(thickness_nodes, t, side="right")
    hi = np.clip(hi, 1, thickness_nodes.size - 1)
    lo = hi - 1
    denom = thickness_nodes[hi] - thickness_nodes[lo]
    w_hi = np.divide(t - thickness_nodes[lo], denom, out=np.zeros_like(t), where=denom != 0.0)
    w_lo = 1.0 - w_hi
    return lo, hi, w_lo, w_hi


def evaluate_ice_surface_chunk(
    ice_node_values: np.ndarray,
    loglam_basis: np.ndarray,
    thickness_nodes: np.ndarray,
    thickness: np.ndarray,
) -> np.ndarray:
    """Evaluate the linear tensor-product spline for a chunk of observations."""
    n_thick = thickness_nodes.size
    n_loglam = loglam_basis.shape[0]
    values = ice_node_values.reshape(n_thick, n_loglam)
    lo, hi, w_lo, w_hi = thickness_brackets(thickness, thickness_nodes)
    surface_lo = values[lo] @ loglam_basis
    surface_hi = values[hi] @ loglam_basis
    return w_lo[:, None] * surface_lo + w_hi[:, None] * surface_hi


def resolve_sed_basis_path(config: FitConfig, sim_metadata: dict, input_dir: Path) -> Path:
    """Find the BOSZ EMPCA basis, honoring CLI, metadata, and local defaults."""
    candidates = []
    if config.sed_basis_path:
        candidates.append(Path(config.sed_basis_path))
    metadata_path = sim_metadata.get("sed_basis_path")
    if metadata_path:
        candidates.extend([Path(metadata_path), input_dir / metadata_path])
    candidates.append(Path("bosz_logflux_empca_basis.npz"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find BOSZ EMPCA basis. Pass --sed-basis-path or keep "
        "bosz_logflux_empca_basis.npz in the working directory."
    )


def load_data(config: FitConfig) -> DataBundle:
    input_dir = Path(config.input_dir)
    measurements = pd.read_csv(input_dir / "measurements.csv")
    prism_path = input_dir / "prism_measurements.csv"
    prism_measurements = pd.read_csv(prism_path) if prism_path.exists() else pd.DataFrame()
    if config.max_stars is not None:
        keep_star_ids = np.sort(measurements["star_id"].unique())[: config.max_stars]
        calibrator_path = input_dir / "stellar_calibrators.csv"
        if calibrator_path.exists():
            calibrator_ids = pd.read_csv(calibrator_path, usecols=["star_id"])[
                "star_id"
            ].to_numpy(int)
            keep_star_ids = np.unique(np.r_[keep_star_ids, calibrator_ids])
        measurements = measurements.loc[measurements["star_id"].isin(keep_star_ids)].copy()
        measurements.reset_index(drop=True, inplace=True)
        if not prism_measurements.empty:
            prism_measurements = prism_measurements.loc[
                prism_measurements["star_id"].isin(keep_star_ids)
            ].copy()
            prism_measurements.reset_index(drop=True, inplace=True)

    wave, pass_data = load_long_grid_csv(
        input_dir / "nominal_passbands.csv", ["throughput"], "filter_id"
    )
    wave_modes, mode_data = load_long_grid_csv(
        input_dir / "passband_modes.csv", ["phi_shift", "phi_width"], "filter_id"
    )
    if not np.allclose(wave, wave_modes):
        raise ValueError("Passband and mode wavelength grids do not match")

    metadata_path = input_dir / "simulation_metadata.json"
    sim_metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as handle:
            sim_metadata = json.load(handle)
    for key in ("nx", "ny", "n_amp"):
        if key in sim_metadata:
            setattr(config, key, int(sim_metadata[key]))
    sed_basis_path = resolve_sed_basis_path(config, sim_metadata, input_dir)
    config.sed_basis_path = str(sed_basis_path)
    sed_library = BOSZEMPCASEDLibrary(sed_basis_path).resampled_to(wave)

    ice_loglam_nodes, ice_thickness_nodes = load_ice_spline_nodes(
        input_dir / "ice_spline_nodes.csv"
    )
    ice_loglam_basis = make_loglam_basis(wave, ice_loglam_nodes)

    filter_ids = np.sort(measurements["filter_id"].unique())
    detector_ids = np.sort(measurements["detector_id"].unique())
    star_ids, star_param_id = np.unique(measurements["star_id"].to_numpy(int), return_inverse=True)
    exposure_id = measurements["exposure_id"].to_numpy(int)
    filter_lookup = {value: i for i, value in enumerate(filter_ids)}
    detector_lookup = {value: i for i, value in enumerate(detector_ids)}
    filter_param_id = measurements["filter_id"].map(filter_lookup).to_numpy(int)
    detector_param_id = measurements["detector_id"].map(detector_lookup).to_numpy(int)
    if "amp_id" in measurements.columns:
        amp_id = measurements["amp_id"].to_numpy(int)
    else:
        amp_id = amp_id_from_x(measurements["x"].to_numpy(float), nx=config.nx, n_amp=config.n_amp)

    star_lookup = {value: i for i, value in enumerate(star_ids)}
    if not prism_measurements.empty:
        missing_prism_stars = sorted(
            set(prism_measurements["star_id"].to_numpy(int)).difference(star_lookup)
        )
        if missing_prism_stars:
            raise ValueError("Prism table contains stars absent from imaging measurements")
        prism_star_param_id = prism_measurements["star_id"].map(star_lookup).to_numpy(int)
        prism_detector_param_id = (
            prism_measurements["detector_id"].map(detector_lookup).to_numpy(int)
        )
        prism_exposure_id = prism_measurements["exposure_id"].to_numpy(int)
        prism_pixel_id = prism_measurements["wavelength_pixel_id"].to_numpy(int)
        if "amp_id" in prism_measurements.columns:
            prism_amp_id = prism_measurements["amp_id"].to_numpy(int)
        else:
            prism_amp_id = amp_id_from_x(
                prism_measurements["x"].to_numpy(float), nx=config.nx, n_amp=config.n_amp
            )
    else:
        prism_star_param_id = np.asarray([], dtype=int)
        prism_detector_param_id = np.asarray([], dtype=int)
        prism_exposure_id = np.asarray([], dtype=int)
        prism_pixel_id = np.asarray([], dtype=int)
        prism_amp_id = np.asarray([], dtype=int)

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
    reference_filter_id = int(sim_metadata.get("reference_filter_id", filter_ids[min(3, filter_ids.size - 1)]))
    if reference_filter_id not in filter_lookup:
        raise ValueError("Reference filter is not present in the fitted measurement subset")
    reference_filter_index = filter_lookup[reference_filter_id]

    nominal_prism_path = input_dir / "nominal_prism.csv"
    if not prism_measurements.empty:
        if not nominal_prism_path.exists():
            raise FileNotFoundError("prism_measurements.csv requires nominal_prism.csv")
        nominal_prism = pd.read_csv(nominal_prism_path).sort_values("wavelength_pixel_id")
        expected_pixel = np.arange(len(nominal_prism), dtype=int)
        if not np.array_equal(
            nominal_prism["wavelength_pixel_id"].to_numpy(int), expected_pixel
        ):
            raise ValueError("nominal_prism.csv wavelength_pixel_id must be contiguous from zero")
        prism_wave = nominal_prism["wavelength_um"].to_numpy(float)
        prism_bin_width = nominal_prism["bin_width_um"].to_numpy(float)
        prism_nominal_throughput = nominal_prism["nominal_throughput"].to_numpy(float)
        if prism_pixel_id.min() < 0 or prism_pixel_id.max() >= prism_wave.size:
            raise ValueError("Prism measurement wavelength_pixel_id is out of range")
        prism_mean_log_flux = np.interp(
            prism_wave, wave, sed_library.mean_log_flux
        )
        prism_components = np.vstack(
            [np.interp(prism_wave, wave, component) for component in sed_library.components]
        )
        prism_ice_loglam_basis = make_loglam_basis(prism_wave, ice_loglam_nodes)
    else:
        prism_wave = np.asarray([], dtype=float)
        prism_bin_width = np.asarray([], dtype=float)
        prism_nominal_throughput = np.asarray([], dtype=float)
        prism_mean_log_flux = np.asarray([], dtype=float)
        prism_components = np.zeros((sed_library.n_components, 0))
        prism_ice_loglam_basis = np.zeros((ice_loglam_nodes.size, 0))

    denom = trapz_integral(passbands, wave, axis=1)
    filter_eff = trapz_integral(passbands * wave[None, :], wave, axis=1) / denom

    def read_optional(name: str) -> pd.DataFrame | None:
        path = input_dir / name
        return pd.read_csv(path) if path.exists() else None

    absolute_calibrators = read_optional("stellar_calibrators.csv")
    star_is_calibrator = np.zeros(star_ids.size, dtype=bool)
    if absolute_calibrators is not None and not absolute_calibrators.empty:
        required = {"star_id", "mag_norm"}
        required.update(f"sed_coeff_{i}" for i in range(sed_library.n_components))
        missing = sorted(required.difference(absolute_calibrators.columns))
        if missing:
            raise ValueError(
                "stellar_calibrators.csv is missing required columns: "
                + ", ".join(missing)
            )
        calibrator_star_ids = set(absolute_calibrators["star_id"].to_numpy(int))
        star_is_calibrator = np.array(
            [int(star_id) in calibrator_star_ids for star_id in star_ids],
            dtype=bool,
        )
        absolute_calibrators = absolute_calibrators.loc[
            absolute_calibrators["star_id"].isin(star_ids)
        ].copy()

    free_star_indices = np.nonzero(~star_is_calibrator)[0]
    free_star_index = np.full(star_ids.size, -1, dtype=int)
    free_star_index[free_star_indices] = np.arange(free_star_indices.size, dtype=int)

    all_exposure_ids = np.unique(
        np.r_[exposure_id, prism_exposure_id]
    ).astype(int)
    reference_exposure_ids = {0}
    prism_reference = sim_metadata.get("prism_reference_exposure_id")
    if prism_reference is not None and int(prism_reference) in all_exposure_ids:
        reference_exposure_ids.add(int(prism_reference))
    free_exposure_ids = np.asarray(
        [exp_id for exp_id in all_exposure_ids if exp_id not in reference_exposure_ids],
        dtype=int,
    )
    exposure_free_index = np.full(int(all_exposure_ids.max()) + 1, -1, dtype=int)
    exposure_free_index[free_exposure_ids] = np.arange(free_exposure_ids.size, dtype=int)

    return DataBundle(
        measurements=measurements,
        wave=wave,
        filter_ids=filter_ids,
        filter_names=filter_names,
        detector_ids=detector_ids,
        star_ids=star_ids,
        exposure_id=exposure_id,
        filter_param_id=filter_param_id,
        detector_param_id=detector_param_id,
        star_param_id=star_param_id,
        amp_id=amp_id,
        prism_measurements=prism_measurements,
        prism_wave=prism_wave,
        prism_bin_width=prism_bin_width,
        prism_nominal_throughput=prism_nominal_throughput,
        prism_star_param_id=prism_star_param_id,
        prism_detector_param_id=prism_detector_param_id,
        prism_exposure_id=prism_exposure_id,
        prism_amp_id=prism_amp_id,
        prism_pixel_id=prism_pixel_id,
        prism_mean_log_flux=prism_mean_log_flux,
        prism_components=prism_components,
        prism_ice_loglam_basis=prism_ice_loglam_basis,
        free_exposure_ids=free_exposure_ids,
        exposure_free_index=exposure_free_index,
        star_is_calibrator=star_is_calibrator,
        free_star_indices=free_star_indices,
        free_star_index=free_star_index,
        passbands=passbands,
        reference_filter_index=reference_filter_index,
        phi_shift=phi_shift,
        phi_width=phi_width,
        sed_library=sed_library,
        ice_loglam_nodes=ice_loglam_nodes,
        ice_thickness_nodes=ice_thickness_nodes,
        ice_loglam_basis=ice_loglam_basis,
        filter_effective_wavelength=filter_eff,
        absolute_calibrators=absolute_calibrators,
        true_star_params=read_optional("true_star_params.csv"),
        true_passband_params=read_optional("true_passband_params.csv"),
        true_ice_spline_params=read_optional("true_ice_spline_params.csv"),
        true_exposure_zeropoints=read_optional("true_exposure_zeropoints.csv"),
        true_smooth_coeffs=read_optional("true_smooth_coeffs.csv"),
        true_amp_offsets=read_optional("true_amp_offsets.csv"),
        true_prism_response=read_optional("true_prism_response.csv"),
        sim_metadata=sim_metadata,
    )


def make_initial_sed_grid(data: DataBundle) -> tuple[np.ndarray, np.ndarray]:
    """Precompute nominal-passband colors for each BOSZ EMPCA template."""
    grid_coeff = data.sed_library.coefficients
    reference_passband = data.passbands[data.reference_filter_index]
    sed_shape = data.sed_library.sed_from_coefficients(
        grid_coeff, np.zeros(grid_coeff.shape[0]), reference_passband
    )

    shape_mag = np.zeros((grid_coeff.shape[0], data.filter_ids.size))
    for filt in range(data.filter_ids.size):
        flux = photon_count_integral(
            sed_shape * data.passbands[filt][None, :], data.wave, axis=1
        )
        shape_mag[:, filt] = counts_to_instrumental_mag(flux)
    return grid_coeff, shape_mag


def fit_initial_stellar_seds(data: DataBundle, config: FitConfig) -> ModelState:
    """Initial star-by-star BOSZ template search with nominal passbands and no ice.

    For each BOSZ template coefficient vector, the magnitude normalization is
    analytic: it is the weighted mean of observed magnitude minus template color.
    """
    grid_coeff, shape_mag = make_initial_sed_grid(data)
    mag = data.measurements["mag_obs"].to_numpy(float)
    sigma = data.measurements["mag_unc"].to_numpy(float)
    weight = 1.0 / sigma**2

    mag_norm = np.zeros(data.star_ids.size)
    sed_coeff = np.zeros((data.star_ids.size, data.sed_library.n_components))

    if data.absolute_calibrators is not None and not data.absolute_calibrators.empty:
        coeff_cols = [f"sed_coeff_{i}" for i in range(data.sed_library.n_components)]
        calibrators = data.absolute_calibrators.set_index("star_id")
        for star_index, star_id in enumerate(data.star_ids):
            if not data.star_is_calibrator[star_index]:
                continue
            row = calibrators.loc[star_id]
            mag_norm[star_index] = float(row["mag_norm"])
            sed_coeff[star_index] = row[coeff_cols].to_numpy(float)

    for star_index in range(data.star_ids.size):
        if data.star_is_calibrator[star_index]:
            continue
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
        sed_coeff[star_index] = grid_coeff[best]

    shift = np.zeros((data.filter_ids.size, data.detector_ids.size))
    width = np.zeros_like(shift)
    ice_coeff = np.zeros(data.ice_thickness_nodes.size * data.ice_loglam_nodes.size)
    prism_response = np.zeros(data.prism_wave.size)
    n_exp = data.exposure_free_index.size
    zp = np.zeros(n_exp)
    smooth_coeff = np.zeros(5)
    amp_offset = np.zeros((data.detector_ids.size, int(data.sim_metadata.get("n_amp", 32))))
    state = ModelState(
        mag_norm,
        sed_coeff,
        shift,
        width,
        ice_coeff,
        prism_response,
        zp,
        smooth_coeff,
        amp_offset,
    )
    if not data.prism_measurements.empty and np.any(data.star_is_calibrator):
        # Bootstrap the wavelength response from fixed standards in the prism
        # reference exposure. The later joint iterations separate this initial
        # response estimate from ice and focal-plane terms using all dithers.
        prism_reference = data.sim_metadata.get("prism_reference_exposure_id")
        standard = data.star_is_calibrator[data.prism_star_param_id]
        if prism_reference is not None:
            standard &= data.prism_exposure_id == int(prism_reference)
        initial_prism = evaluate_prism_model_and_responses(
            data, state, config, need_responses=False
        )
        bootstrap_residual = (
            data.prism_measurements["mag_obs"].to_numpy(float)
            - initial_prism.mag_model
        )
        for pixel_id in range(data.prism_wave.size):
            selected = standard & (data.prism_pixel_id == pixel_id)
            if np.any(selected):
                state.prism_response[pixel_id] = np.median(
                    bootstrap_residual[selected]
                )
    return state


def evaluate_model_and_responses(
    data: DataBundle, state: ModelState, config: FitConfig, need_responses: bool = True
) -> Linearization:
    """Evaluate current model magnitudes and, optionally, linear response columns.

    The response coefficients are derivatives of broadband magnitudes, but each
    derivative is computed from the correct linear flux integral over wavelength.
    """
    n_obs = len(data.measurements)
    n_ice = data.ice_thickness_nodes.size * data.ice_loglam_nodes.size
    n_sed_coeff = data.sed_library.n_components
    mag_model = np.zeros(n_obs)
    d_norm = np.ones(n_obs)
    d_sed_coeff = np.zeros((n_obs, n_sed_coeff))
    r_shift = np.zeros(n_obs)
    r_width = np.zeros(n_obs)
    r_ice = np.zeros((n_obs, n_ice))

    if "ice_thickness" in data.measurements.columns:
        ice_thickness = data.measurements["ice_thickness"].to_numpy(float)
    else:
        ice_thickness = data.measurements["ice_amount_obs"].to_numpy(float)
    x_pix = data.measurements["x"].to_numpy(float)
    y_pix = data.measurements["y"].to_numpy(float)

    for start in range(0, n_obs, config.chunk_size):
        end = min(n_obs, start + config.chunk_size)
        sl = slice(start, end)
        star = data.star_param_id[sl]
        filt = data.filter_param_id[sl]
        det = data.detector_param_id[sl]
        exp_id = data.exposure_id[sl]
        amp = data.amp_id[sl]
        ice = ice_thickness[sl]

        reference_passband = data.passbands[data.reference_filter_index]
        sed = data.sed_library.sed_from_coefficients(
            state.sed_coeff[star],
            state.mag_norm[star],
            reference_passband,
        )
        logt = (
            state.shift[filt, det][:, None] * data.phi_shift[filt]
            + state.width[filt, det][:, None] * data.phi_width[filt]
            + evaluate_ice_surface_chunk(
                state.ice_coeff,
                data.ice_loglam_basis,
                data.ice_thickness_nodes,
                ice,
            )
        )
        t_current = data.passbands[filt] * np.exp(logt)
        weighted = sed * t_current
        denom = photon_count_integral(weighted, data.wave, axis=1)
        xn, yn = normalized_xy(x_pix[sl], y_pix[sl], nx=config.nx, ny=config.ny)
        smooth = poly_basis(xn, yn) @ state.smooth_coeff
        scalar = state.zp[exp_id] + smooth + state.amp_offset[det, amp]
        mag_model[sl] = counts_to_instrumental_mag(denom) + scalar

        if not need_responses:
            continue

        # The BOSZ EMPCA coefficients perturb normalized log flux:
        # log f_s(lambda) = mean_log_flux + theta @ components + log amplitude.
        # Because mag_norm is defined as the AB magnitude in a reference
        # passband, coefficient updates change colors at fixed reference-band
        # magnitude. The derivative therefore subtracts the reference-passband
        # component average.
        ref_weighted = sed * reference_passband[None, :]
        ref_denom = photon_count_integral(ref_weighted, data.wave, axis=1)
        for component_id in range(n_sed_coeff):
            obs_mean = (
                photon_count_integral(
                    weighted * data.sed_library.components[component_id][None, :],
                    data.wave,
                    axis=1,
                )
                / denom
            )
            ref_mean = (
                photon_count_integral(
                    ref_weighted * data.sed_library.components[component_id][None, :],
                    data.wave,
                    axis=1,
                )
                / ref_denom
            )
            d_sed_coeff[sl, component_id] = -MAG_FACTOR * (obs_mean - ref_mean)

        r_shift[sl] = -MAG_FACTOR * (
            photon_count_integral(weighted * data.phi_shift[filt], data.wave, axis=1) / denom
        )
        r_width[sl] = -MAG_FACTOR * (
            photon_count_integral(weighted * data.phi_width[filt], data.wave, axis=1) / denom
        )
        n_loglam = data.ice_loglam_nodes.size
        lo, hi, w_lo, w_hi = thickness_brackets(ice, data.ice_thickness_nodes)
        loglam_integrals = np.zeros((end - start, n_loglam))
        for loglam_id in range(n_loglam):
            loglam_integrals[:, loglam_id] = trapz_integral(
                weighted * data.ice_loglam_basis[loglam_id][None, :] * data.wave[None, :],
                data.wave,
                axis=1,
            )
        # The ice spline enters as ln T += node_value * basis, so a positive
        # node value increases flux and decreases magnitude. The sign is
        # therefore negative.
        for local_row in range(end - start):
            for loglam_id in range(n_loglam):
                r_ice[start + local_row, lo[local_row] * n_loglam + loglam_id] += (
                    -MAG_FACTOR * w_lo[local_row] * loglam_integrals[local_row, loglam_id]
                    / denom[local_row]
                )
                r_ice[start + local_row, hi[local_row] * n_loglam + loglam_id] += (
                    -MAG_FACTOR * w_hi[local_row] * loglam_integrals[local_row, loglam_id]
                    / denom[local_row]
                )

    return Linearization(mag_model, d_norm, d_sed_coeff, r_shift, r_width, r_ice)


def evaluate_prism_model_and_responses(
    data: DataBundle, state: ModelState, config: FitConfig, need_responses: bool = True
) -> PrismLinearization:
    """Evaluate prism-pixel magnitudes and their linearized response columns.

    Each prism row is a finite wavelength-bin photon-counting measurement. The
    midpoint quadrature used by the simulator is linear in physical flux. A
    fitted additive magnitude response at every wavelength pixel represents the
    prism's wavelength-dependent sensitivity calibration. Ice, stellar SED,
    focal-plane, and amplifier terms are evaluated at that same detector pixel.
    """
    n_obs = len(data.prism_measurements)
    n_component = data.sed_library.n_components
    n_ice = data.ice_thickness_nodes.size * data.ice_loglam_nodes.size
    if n_obs == 0:
        return PrismLinearization(
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.zeros((0, n_component)),
            np.zeros((0, n_ice)),
        )

    reference_passband = data.passbands[data.reference_filter_index]
    shape = data.sed_library.sed_shape_from_coefficients(state.sed_coeff)
    reference_weighted = shape * reference_passband[None, :]
    reference_count = photon_count_integral(reference_weighted, data.wave, axis=1)
    reference_ab_count = ab_reference_count(data.wave, reference_passband)
    amplitude = (
        reference_ab_count
        * 10.0 ** (-0.4 * state.mag_norm)
        / np.maximum(reference_count, 1e-300)
    )

    reference_component_mean = np.zeros((data.star_ids.size, n_component))
    if need_responses:
        for component_id in range(n_component):
            reference_component_mean[:, component_id] = photon_count_integral(
                reference_weighted * data.sed_library.components[component_id][None, :],
                data.wave,
                axis=1,
            ) / np.maximum(reference_count, 1e-300)

    star = data.prism_star_param_id
    pixel = data.prism_pixel_id
    detector = data.prism_detector_param_id
    log_shape = data.prism_mean_log_flux[pixel]
    log_shape += np.sum(
        state.sed_coeff[star] * data.prism_components[:, pixel].T,
        axis=1,
    )
    sed_flux_density = amplitude[star] * np.exp(log_shape)

    thickness = data.prism_measurements["ice_thickness"].to_numpy(float)
    lo, hi, w_lo, w_hi = thickness_brackets(thickness, data.ice_thickness_nodes)
    loglam_basis = data.prism_ice_loglam_basis[:, pixel].T
    n_loglam = data.ice_loglam_nodes.size
    ice_grid = state.ice_coeff.reshape(data.ice_thickness_nodes.size, n_loglam)
    ice_logt = w_lo * np.sum(ice_grid[lo] * loglam_basis, axis=1)
    ice_logt += w_hi * np.sum(ice_grid[hi] * loglam_basis, axis=1)

    count_flux = (
        sed_flux_density
        * data.prism_nominal_throughput[pixel]
        * np.exp(ice_logt)
        * data.prism_wave[pixel]
        * data.prism_bin_width[pixel]
    )
    x_pix = data.prism_measurements["x"].to_numpy(float)
    y_pix = data.prism_measurements["y"].to_numpy(float)
    xn, yn = normalized_xy(x_pix, y_pix, nx=config.nx, ny=config.ny)
    smooth = poly_basis(xn, yn) @ state.smooth_coeff
    scalar = (
        state.zp[data.prism_exposure_id]
        + smooth
        + state.amp_offset[detector, data.prism_amp_id]
    )
    mag_model = (
        counts_to_instrumental_mag(count_flux)
        + state.prism_response[pixel]
        + scalar
    )

    d_sed_coeff = np.zeros((n_obs, n_component))
    r_ice = np.zeros((n_obs, n_ice))
    if need_responses:
        d_sed_coeff = -MAG_FACTOR * (
            data.prism_components[:, pixel].T - reference_component_mean[star]
        )
        for obs_index in range(n_obs):
            for loglam_id in np.nonzero(loglam_basis[obs_index])[0]:
                r_ice[obs_index, lo[obs_index] * n_loglam + loglam_id] += (
                    -MAG_FACTOR * w_lo[obs_index] * loglam_basis[obs_index, loglam_id]
                )
                r_ice[obs_index, hi[obs_index] * n_loglam + loglam_id] += (
                    -MAG_FACTOR * w_hi[obs_index] * loglam_basis[obs_index, loglam_id]
                )

    return PrismLinearization(
        mag_model=mag_model,
        d_norm=np.ones(n_obs),
        d_sed_coeff=d_sed_coeff,
        r_ice=r_ice,
    )


def parameter_slices(data: DataBundle) -> dict[str, slice]:
    n_free_star = data.free_star_indices.size
    n_sed = n_free_star * data.sed_library.n_components
    n_pass = data.filter_ids.size * data.detector_ids.size
    n_ice = data.ice_thickness_nodes.size * data.ice_loglam_nodes.size
    n_prism = data.prism_wave.size
    n_zp = data.free_exposure_ids.size
    n_smooth = 5
    n_amp = data.detector_ids.size * int(data.sim_metadata.get("n_amp", 32))
    start = 0
    slices = {}
    slices["norm"] = slice(start, start + n_free_star)
    start += n_free_star
    slices["sed_coeff"] = slice(start, start + n_sed)
    start += n_sed
    slices["shift"] = slice(start, start + n_pass)
    start += n_pass
    slices["width"] = slice(start, start + n_pass)
    start += n_pass
    slices["ice"] = slice(start, start + n_ice)
    start += n_ice
    slices["prism_response"] = slice(start, start + n_prism)
    start += n_prism
    slices["zp"] = slice(start, start + n_zp)
    start += n_zp
    slices["smooth"] = slice(start, start + n_smooth)
    start += n_smooth
    slices["amp"] = slice(start, start + n_amp)
    return slices


def build_sparse_system(
    data: DataBundle,
    state: ModelState,
    lin: Linearization,
    prism_lin: PrismLinearization,
    config: FitConfig,
) -> tuple[coo_matrix, np.ndarray, dict[str, slice]]:
    """Build one weighted sparse update system from imaging and prism rows."""
    slices = parameter_slices(data)
    n_params = slices["amp"].stop
    n_components = data.sed_library.n_components
    rows = []
    cols = []
    vals = []
    rhs = []
    row = 0

    obs_mag = data.measurements["mag_obs"].to_numpy(float)
    obs_unc = data.measurements["mag_unc"].to_numpy(float)
    residual = obs_mag - lin.mag_model
    x_pix = data.measurements["x"].to_numpy(float)
    y_pix = data.measurements["y"].to_numpy(float)
    xn, yn = normalized_xy(x_pix, y_pix, nx=config.nx, ny=config.ny)
    smooth_basis = poly_basis(xn, yn)

    n_det = data.detector_ids.size
    for obs_index in range(len(data.measurements)):
        weight = 1.0 / obs_unc[obs_index]
        star = data.star_param_id[obs_index]
        filt = data.filter_param_id[obs_index]
        det = data.detector_param_id[obs_index]
        pass_index = filt * n_det + det
        free_star = data.free_star_index[star]
        exp_id = data.exposure_id[obs_index]

        entries = [
            (slices["shift"].start + pass_index, lin.r_shift[obs_index]),
            (slices["width"].start + pass_index, lin.r_width[obs_index]),
        ]
        if free_star >= 0:
            entries.append((slices["norm"].start + free_star, lin.d_norm[obs_index]))
            for component_id in range(n_components):
                coeff_index = free_star * n_components + component_id
                entries.append(
                    (
                        slices["sed_coeff"].start + coeff_index,
                        lin.d_sed_coeff[obs_index, component_id],
                    )
                )
        for basis_id in range(data.ice_thickness_nodes.size * data.ice_loglam_nodes.size):
            entries.append((slices["ice"].start + basis_id, lin.r_ice[obs_index, basis_id]))
        exposure_param = data.exposure_free_index[exp_id]
        if exposure_param >= 0:
            entries.append((slices["zp"].start + exposure_param, 1.0))
        for smooth_id in range(5):
            entries.append((slices["smooth"].start + smooth_id, smooth_basis[obs_index, smooth_id]))
        amp_index = det * config.n_amp + data.amp_id[obs_index]
        entries.append((slices["amp"].start + amp_index, 1.0))

        for col, value in entries:
            rows.append(row)
            cols.append(col)
            vals.append(value * weight)
        rhs.append(residual[obs_index] * weight)
        row += 1

    if not data.prism_measurements.empty:
        prism_mag = data.prism_measurements["mag_obs"].to_numpy(float)
        prism_unc = data.prism_measurements["mag_unc"].to_numpy(float)
        prism_residual = prism_mag - prism_lin.mag_model
        prism_x = data.prism_measurements["x"].to_numpy(float)
        prism_y = data.prism_measurements["y"].to_numpy(float)
        prism_xn, prism_yn = normalized_xy(
            prism_x, prism_y, nx=config.nx, ny=config.ny
        )
        prism_smooth_basis = poly_basis(prism_xn, prism_yn)
        for obs_index in range(len(data.prism_measurements)):
            weight = 1.0 / prism_unc[obs_index]
            star = data.prism_star_param_id[obs_index]
            detector = data.prism_detector_param_id[obs_index]
            pixel = data.prism_pixel_id[obs_index]
            free_star = data.free_star_index[star]
            exp_id = data.prism_exposure_id[obs_index]
            entries = [
                (slices["prism_response"].start + pixel, 1.0),
            ]
            if free_star >= 0:
                entries.append((slices["norm"].start + free_star, 1.0))
                for component_id in range(n_components):
                    coeff_index = free_star * n_components + component_id
                    entries.append(
                        (
                            slices["sed_coeff"].start + coeff_index,
                            prism_lin.d_sed_coeff[obs_index, component_id],
                        )
                    )
            for basis_id in np.nonzero(prism_lin.r_ice[obs_index])[0]:
                entries.append(
                    (
                        slices["ice"].start + basis_id,
                        prism_lin.r_ice[obs_index, basis_id],
                    )
                )
            exposure_param = data.exposure_free_index[exp_id]
            if exposure_param >= 0:
                entries.append((slices["zp"].start + exposure_param, 1.0))
            for smooth_id in range(5):
                entries.append(
                    (
                        slices["smooth"].start + smooth_id,
                        prism_smooth_basis[obs_index, smooth_id],
                    )
                )
            amp_index = detector * config.n_amp + data.prism_amp_id[obs_index]
            entries.append((slices["amp"].start + amp_index, 1.0))

            for col, value in entries:
                rows.append(row)
                cols.append(col)
                vals.append(value * weight)
            rhs.append(prism_residual[obs_index] * weight)
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

    # The spectrophotometric standards anchor the absolute prism response. A
    # weak amplitude prior and second-difference smoothness prior stabilize
    # wavelengths with missing or truncated spectra without erasing real shape.
    for pixel_id, current in enumerate(state.prism_response):
        rows.append(row)
        cols.append(slices["prism_response"].start + pixel_id)
        vals.append(1.0 / config.sigma_prism_response_prior)
        rhs.append(-current / config.sigma_prism_response_prior)
        row += 1

    for pixel_id in range(1, state.prism_response.size - 1):
        for offset, coefficient in ((-1, 1.0), (0, -2.0), (1, 1.0)):
            rows.append(row)
            cols.append(slices["prism_response"].start + pixel_id + offset)
            vals.append(coefficient / config.sigma_prism_response_smoothness)
        current_second_difference = (
            state.prism_response[pixel_id - 1]
            - 2.0 * state.prism_response[pixel_id]
            + state.prism_response[pixel_id + 1]
        )
        rhs.append(-current_second_difference / config.sigma_prism_response_smoothness)
        row += 1

    # At exactly zero ice thickness, the ice perturbation is physically zero.
    # This removes a gauge freedom between the ice surface and stellar/passband
    # color terms without constraining nonzero-thickness interference structure.
    n_loglam = data.ice_loglam_nodes.size
    for loglam_id in range(n_loglam):
        current = state.ice_coeff[loglam_id]
        rows.append(row)
        cols.append(slices["ice"].start + loglam_id)
        vals.append(1.0 / config.sigma_zero_ice_surface_prior)
        rhs.append(-current / config.sigma_zero_ice_surface_prior)
        row += 1

    # SED coefficient update priors prevent each star from freely absorbing
    # global passband and ice structure in every iteration. The prior scale is
    # based on the empirical BOSZ coefficient scatter for each EMPCA component.
    coeff_sigma = (
        data.sed_library.coefficient_scales * config.sigma_sed_coeff_update_prior_scale
    )
    for free_star in range(data.free_star_indices.size):
        for component_id, sigma in enumerate(coeff_sigma):
            coeff_index = free_star * n_components + component_id
            rows.append(row)
            cols.append(slices["sed_coeff"].start + coeff_index)
            vals.append(1.0 / sigma)
            rhs.append(0.0)
            row += 1

    for flat_index, current in enumerate(state.amp_offset.ravel()):
        rows.append(row)
        cols.append(slices["amp"].start + flat_index)
        vals.append(1.0 / config.sigma_amp_prior)
        rhs.append(-current / config.sigma_amp_prior)
        row += 1

    for det_id in range(n_det):
        for amp_id in range(config.n_amp):
            flat_index = det_id * config.n_amp + amp_id
            rows.append(row)
            cols.append(slices["amp"].start + flat_index)
            vals.append((1.0 / config.n_amp) / config.sigma_amp_sum_constraint)
        current_mean = state.amp_offset[det_id].mean()
        rhs.append(-current_mean / config.sigma_amp_sum_constraint)
        row += 1

    A = coo_matrix((vals, (rows, cols)), shape=(row, n_params)).tocsr()
    return A, np.asarray(rhs), slices


def apply_update(
    state: ModelState,
    theta: np.ndarray,
    slices: dict[str, slice],
    data: DataBundle,
    config: FitConfig,
    damping: float | None = None,
) -> None:
    n_components = data.sed_library.n_components
    n_filter = data.filter_ids.size
    n_det = data.detector_ids.size
    damping = config.damping if damping is None else damping

    free = data.free_star_indices
    state.mag_norm[free] += damping * theta[slices["norm"]]
    state.sed_coeff[free] += damping * theta[slices["sed_coeff"]].reshape(
        free.size, n_components
    )
    state.shift += damping * theta[slices["shift"]].reshape(n_filter, n_det)
    state.width += damping * theta[slices["width"]].reshape(n_filter, n_det)
    state.ice_coeff += damping * theta[slices["ice"]]
    state.prism_response += damping * theta[slices["prism_response"]]
    state.zp[data.free_exposure_ids] += damping * theta[slices["zp"]]
    state.smooth_coeff += damping * theta[slices["smooth"]]
    state.amp_offset += damping * theta[slices["amp"]].reshape(n_det, config.n_amp)


def copy_model_state(state: ModelState) -> ModelState:
    """Deep-copy the numerical arrays in a model state for line-search trials."""
    return ModelState(
        mag_norm=state.mag_norm.copy(),
        sed_coeff=state.sed_coeff.copy(),
        shift=state.shift.copy(),
        width=state.width.copy(),
        ice_coeff=state.ice_coeff.copy(),
        prism_response=state.prism_response.copy(),
        zp=state.zp.copy(),
        smooth_coeff=state.smooth_coeff.copy(),
        amp_offset=state.amp_offset.copy(),
    )


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def evaluate_smooth_field(
    coeffs: np.ndarray, x: np.ndarray, y: np.ndarray, config: FitConfig
) -> np.ndarray:
    """Evaluate the smooth focal-plane polynomial at detector pixel positions."""
    xn, yn = normalized_xy(np.asarray(x), np.asarray(y), nx=config.nx, ny=config.ny)
    flat_xn = np.ravel(xn)
    flat_yn = np.ravel(yn)
    values = poly_basis(flat_xn, flat_yn) @ coeffs
    return values.reshape(np.shape(xn))


def smooth_fields_on_grid(
    data: DataBundle, state: ModelState, config: FitConfig, n_grid: int = 120
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Evaluate true, fitted, and residual smooth fields on a detector grid."""
    if data.true_smooth_coeffs is None:
        return None
    true_coeff = data.true_smooth_coeffs["coefficient_mag"].to_numpy(float)
    if true_coeff.size != state.smooth_coeff.size:
        return None
    x_grid = np.linspace(0.0, config.nx - 1.0, n_grid)
    y_grid = np.linspace(0.0, config.ny - 1.0, n_grid)
    xx, yy = np.meshgrid(x_grid, y_grid)
    true_field = evaluate_smooth_field(true_coeff, xx, yy, config)
    fit_field = evaluate_smooth_field(state.smooth_coeff, xx, yy, config)
    residual_field = fit_field - true_field
    residual_field -= np.mean(residual_field)
    return xx, yy, true_field, fit_field, residual_field


def evaluate_scalar_terms(data: DataBundle, state: ModelState, config: FitConfig) -> np.ndarray:
    """Evaluate fitted additive scalar terms for each observation."""
    x_pix = data.measurements["x"].to_numpy(float)
    y_pix = data.measurements["y"].to_numpy(float)
    xn, yn = normalized_xy(x_pix, y_pix, nx=config.nx, ny=config.ny)
    smooth = poly_basis(xn, yn) @ state.smooth_coeff
    return (
        state.zp[data.exposure_id]
        + smooth
        + state.amp_offset[data.detector_param_id, data.amp_id]
    )


def evaluate_ab_zeropoint_observations(
    data: DataBundle, state: ModelState, config: FitConfig
) -> np.ndarray:
    """AB zeropoint to add to instrumental magnitudes for each observation.

    For the current throughput model, m_AB = m_inst + ZP_AB, where m_inst is
    -2.5 log10(source counts). The scalar focal-plane terms are already present
    in the instrumental magnitude model, so they are subtracted here.
    """
    if "ice_thickness" in data.measurements.columns:
        ice_thickness = data.measurements["ice_thickness"].to_numpy(float)
    else:
        ice_thickness = data.measurements["ice_amount_obs"].to_numpy(float)

    zp_ab = np.zeros(len(data.measurements))
    scalar = evaluate_scalar_terms(data, state, config)
    for start in range(0, len(data.measurements), config.chunk_size):
        end = min(len(data.measurements), start + config.chunk_size)
        sl = slice(start, end)
        filt = data.filter_param_id[sl]
        det = data.detector_param_id[sl]
        ice = ice_thickness[sl]
        logt = (
            state.shift[filt, det][:, None] * data.phi_shift[filt]
            + state.width[filt, det][:, None] * data.phi_width[filt]
            + evaluate_ice_surface_chunk(
                state.ice_coeff,
                data.ice_loglam_basis,
                data.ice_thickness_nodes,
                ice,
            )
        )
        t_current = data.passbands[filt] * np.exp(logt)
        ab_flux = ab_f_lambda_per_micron(data.wave)[None, :] * t_current
        ab_count = photon_count_integral(ab_flux, data.wave, axis=1)
        zp_ab[sl] = 2.5 * np.log10(np.maximum(ab_count, 1e-300)) - scalar[sl]
    return zp_ab


def evaluate_prism_scalar_terms(
    data: DataBundle, state: ModelState, config: FitConfig
) -> np.ndarray:
    """Evaluate scalar exposure, smooth-field, and shared amp terms for prism rows."""
    if data.prism_measurements.empty:
        return np.asarray([], dtype=float)
    x_pix = data.prism_measurements["x"].to_numpy(float)
    y_pix = data.prism_measurements["y"].to_numpy(float)
    xn, yn = normalized_xy(x_pix, y_pix, nx=config.nx, ny=config.ny)
    smooth = poly_basis(xn, yn) @ state.smooth_coeff
    return (
        state.zp[data.prism_exposure_id]
        + smooth
        + state.amp_offset[data.prism_detector_param_id, data.prism_amp_id]
    )


def evaluate_prism_ab_zeropoints(
    data: DataBundle, state: ModelState, config: FitConfig
) -> np.ndarray:
    """Return the fitted narrow-bin AB zeropoint for every prism measurement."""
    if data.prism_measurements.empty:
        return np.asarray([], dtype=float)
    pixel = data.prism_pixel_id
    thickness = data.prism_measurements["ice_thickness"].to_numpy(float)
    lo, hi, w_lo, w_hi = thickness_brackets(thickness, data.ice_thickness_nodes)
    n_loglam = data.ice_loglam_nodes.size
    ice_grid = state.ice_coeff.reshape(data.ice_thickness_nodes.size, n_loglam)
    loglam_basis = data.prism_ice_loglam_basis[:, pixel].T
    ice_logt = w_lo * np.sum(ice_grid[lo] * loglam_basis, axis=1)
    ice_logt += w_hi * np.sum(ice_grid[hi] * loglam_basis, axis=1)
    ab_count = (
        ab_f_lambda_per_micron(data.prism_wave[pixel])
        * data.prism_nominal_throughput[pixel]
        * np.exp(ice_logt)
        * data.prism_wave[pixel]
        * data.prism_bin_width[pixel]
    )
    return (
        2.5 * np.log10(np.maximum(ab_count, 1e-300))
        - state.prism_response[pixel]
        - evaluate_prism_scalar_terms(data, state, config)
    )


def ice_surface_on_grid(data: DataBundle, ice_node_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ice log-throughput surface on a dense thickness/wavelength grid."""
    thickness_grid = np.linspace(data.ice_thickness_nodes.min(), data.ice_thickness_nodes.max(), 80)
    surface = evaluate_ice_surface_chunk(
        ice_node_values,
        data.ice_loglam_basis,
        data.ice_thickness_nodes,
        thickness_grid,
    )
    return np.log10(data.wave), thickness_grid, surface


def ice_surface_uncertainty_on_grid(
    data: DataBundle, ice_node_sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate diagonal node variances to the dense ice surface grid.

    This intentionally ignores off-diagonal covariance. It is a useful formal
    1-sigma diagnostic from LSQR's diagonal variance estimate, not a complete
    posterior uncertainty surface.
    """
    thickness_grid = np.linspace(data.ice_thickness_nodes.min(), data.ice_thickness_nodes.max(), 80)
    n_loglam = data.ice_loglam_nodes.size
    sigma_grid = ice_node_sigma.reshape(data.ice_thickness_nodes.size, n_loglam)
    lo, hi, w_lo, w_hi = thickness_brackets(thickness_grid, data.ice_thickness_nodes)
    basis2 = data.ice_loglam_basis**2
    sigma_surface = np.zeros((thickness_grid.size, data.wave.size))
    for row in range(thickness_grid.size):
        node_var = (
            w_lo[row] ** 2 * sigma_grid[lo[row]] ** 2
            + w_hi[row] ** 2 * sigma_grid[hi[row]] ** 2
        )
        sigma_surface[row] = np.sqrt(np.maximum(node_var @ basis2, 0.0))
    return np.log10(data.wave), thickness_grid, sigma_surface


def joint_weighted_rms(
    data: DataBundle, imaging_residual: np.ndarray, prism_residual: np.ndarray
) -> float:
    """RMS of residual/sigma over both measurement types."""
    imaging_sigma = data.measurements["mag_unc"].to_numpy(float)
    weighted = [imaging_residual / imaging_sigma]
    if prism_residual.size:
        prism_sigma = data.prism_measurements["mag_unc"].to_numpy(float)
        weighted.append(prism_residual / prism_sigma)
    return rms(np.concatenate(weighted))


def equilibrate_columns(A):
    """Scale sparse columns to unit norm and return the inverse transformation.

    The calibration matrix mixes parameters with very different natural units.
    Solving ``(A D) z = b`` with unit-norm columns is algebraically equivalent
    to the original system when ``theta = D z``, but is much better conditioned
    for Krylov solvers.
    """
    column_norm = np.sqrt(np.asarray(A.power(2).sum(axis=0)).ravel())
    parameter_scale = np.ones_like(column_norm)
    nonzero = column_norm > 0.0
    parameter_scale[nonzero] = 1.0 / column_norm[nonzero]
    return A.multiply(parameter_scale).tocsr(), parameter_scale


def run_fit(
    data: DataBundle, config: FitConfig
) -> tuple[ModelState, pd.DataFrame, np.ndarray, np.ndarray]:
    state = fit_initial_stellar_seds(data, config)
    output_rows = []

    initial_lin = evaluate_model_and_responses(data, state, config, need_responses=False)
    initial_prism_lin = evaluate_prism_model_and_responses(
        data, state, config, need_responses=False
    )
    obs_mag = data.measurements["mag_obs"].to_numpy(float)
    prism_mag = (
        data.prism_measurements["mag_obs"].to_numpy(float)
        if not data.prism_measurements.empty
        else np.asarray([], dtype=float)
    )
    initial_resid = obs_mag - initial_lin.mag_model
    initial_prism_resid = prism_mag - initial_prism_lin.mag_model
    initial_all_resid = np.r_[initial_resid, initial_prism_resid]
    output_rows.append(
        {
            "iteration": 0,
            "rms_residual": rms(initial_all_resid),
            "imaging_rms_residual": rms(initial_resid),
            "prism_rms_residual": (
                rms(initial_prism_resid) if initial_prism_resid.size else np.nan
            ),
            "weighted_rms_residual": joint_weighted_rms(
                data, initial_resid, initial_prism_resid
            ),
            "accepted_damping": 0.0,
            "lsmr_iters": 0,
        }
    )
    print(f"Initial combined RMS residual: {rms(initial_all_resid):.6f} mag")

    for iteration in range(1, config.n_iter + 1):
        lin = evaluate_model_and_responses(data, state, config, need_responses=True)
        prism_lin = evaluate_prism_model_and_responses(
            data, state, config, need_responses=True
        )
        A, b, slices = build_sparse_system(data, state, lin, prism_lin, config)
        A_scaled, parameter_scale = equilibrate_columns(A)
        result = lsmr(
            A_scaled, b, atol=1e-9, btol=1e-9, maxiter=2000, show=False
        )
        theta = parameter_scale * result[0]
        current_resid = obs_mag - lin.mag_model
        current_prism_resid = prism_mag - prism_lin.mag_model
        current_objective = joint_weighted_rms(
            data, current_resid, current_prism_resid
        )

        accepted_damping = 0.0
        trial_damping = config.damping
        resid = current_resid
        prism_resid = current_prism_resid
        while trial_damping >= config.min_damping:
            candidate = copy_model_state(state)
            apply_update(
                candidate,
                theta,
                slices,
                data,
                config,
                damping=trial_damping,
            )
            updated = evaluate_model_and_responses(
                data, candidate, config, need_responses=False
            )
            updated_prism = evaluate_prism_model_and_responses(
                data, candidate, config, need_responses=False
            )
            trial_resid = obs_mag - updated.mag_model
            trial_prism_resid = prism_mag - updated_prism.mag_model
            trial_objective = joint_weighted_rms(
                data, trial_resid, trial_prism_resid
            )
            if np.isfinite(trial_objective) and trial_objective < current_objective:
                state = candidate
                resid = trial_resid
                prism_resid = trial_prism_resid
                accepted_damping = trial_damping
                break
            trial_damping *= 0.5

        if accepted_damping == 0.0:
            print(
                f"Iteration {iteration}: no tested damping improved the joint "
                "weighted objective; update rejected."
            )
        all_resid = np.r_[resid, prism_resid]
        output_rows.append(
            {
                "iteration": iteration,
                "rms_residual": rms(all_resid),
                "imaging_rms_residual": rms(resid),
                "prism_rms_residual": rms(prism_resid) if prism_resid.size else np.nan,
                "lsmr_iters": result[2],
                "lsmr_stop_code": result[1],
                "update_norm": float(np.linalg.norm(theta)),
                "accepted_damping": accepted_damping,
                "weighted_rms_residual": joint_weighted_rms(
                    data, resid, prism_resid
                ),
            }
        )
        prism_message = (
            f"prism = {rms(prism_resid):.6f} mag, " if prism_resid.size else ""
        )
        print(
            f"Iteration {iteration}: combined RMS = {rms(all_resid):.6f} mag, "
            f"imaging = {rms(resid):.6f} mag, "
            f"{prism_message}damping = {accepted_damping:.5f}, "
            f"LSMR iters = {result[2]}"
        )

    final_lin = evaluate_model_and_responses(data, state, config, need_responses=False)
    final_prism_lin = evaluate_prism_model_and_responses(
        data, state, config, need_responses=False
    )
    final_resid = obs_mag - final_lin.mag_model
    final_prism_resid = prism_mag - final_prism_lin.mag_model
    return state, pd.DataFrame(output_rows), final_resid, final_prism_resid


def estimate_parameter_uncertainties(
    data: DataBundle, state: ModelState, config: FitConfig
) -> tuple[np.ndarray, dict[str, slice], dict[str, float]]:
    """Estimate formal 1-sigma parameter uncertainties with LSQR.

    SciPy's LSMR solver does not expose variance estimates. LSQR can estimate
    the diagonal of ``inv(A.T @ A)`` when ``calc_var=True``. Because all rows in
    this prototype are already weighted by their data or prior sigma, the square
    root of that diagonal is a formal linearized 1-sigma uncertainty estimate.
    We scale by the reduced chi-square of the final linearized solve so the
    values are not overconfident if the weighted residuals are larger than one.
    """
    print("Estimating formal parameter uncertainties with LSQR...")
    lin = evaluate_model_and_responses(data, state, config, need_responses=True)
    prism_lin = evaluate_prism_model_and_responses(
        data, state, config, need_responses=True
    )
    A, b, slices = build_sparse_system(data, state, lin, prism_lin, config)
    A_scaled, parameter_scale = equilibrate_columns(A)
    result = lsqr(
        A_scaled,
        b,
        atol=1e-9,
        btol=1e-9,
        iter_lim=4000,
        show=False,
        calc_var=True,
    )
    var = np.maximum(result[-1], 0.0) * parameter_scale**2
    dof = max(A.shape[0] - A.shape[1], 1)
    reduced_chi2 = (result[3] ** 2) / dof
    sigma = np.sqrt(var * max(reduced_chi2, 1e-12))
    info = {
        "lsqr_uncertainty_stop_code": float(result[1]),
        "lsqr_uncertainty_iterations": float(result[2]),
        "lsqr_uncertainty_reduced_chi2": float(reduced_chi2),
    }
    return sigma, slices, info


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

    if data.true_ice_spline_params is not None:
        n_thick = data.ice_thickness_nodes.size
        n_loglam = data.ice_loglam_nodes.size
        true_ice = np.zeros(n_thick * n_loglam)
        for _, row in data.true_ice_spline_params.iterrows():
            thick_id = int(row["ice_thickness_node_id"])
            loglam_id = int(row["loglam_node_id"])
            index = thick_id * n_loglam + loglam_id
            if 0 <= index < true_ice.size:
                true_ice[index] = row["ice_logt_node_value"]

    return true_shift, true_width, true_ice


def save_outputs(
    data: DataBundle,
    state: ModelState,
    summary: pd.DataFrame,
    final_resid: np.ndarray,
    final_prism_resid: np.ndarray,
    config: FitConfig,
    param_sigma: np.ndarray | None = None,
    slices: dict[str, slice] | None = None,
    uncertainty_info: dict[str, float] | None = None,
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    star_payload = {
        "star_id": data.star_ids,
        "is_absolute_calibrator": data.star_is_calibrator,
        "mag_norm": state.mag_norm,
    }
    if param_sigma is not None and slices is not None:
        mag_norm_sigma = np.zeros(data.star_ids.size)
        mag_norm_sigma[data.free_star_indices] = param_sigma[slices["norm"]]
        star_payload["mag_norm_sigma"] = mag_norm_sigma
    for component_id in range(data.sed_library.n_components):
        star_payload[f"sed_coeff_{component_id}"] = state.sed_coeff[:, component_id]
        if param_sigma is not None and slices is not None:
            sigma_start = slices["sed_coeff"].start + component_id
            coeff_sigma = np.zeros(data.star_ids.size)
            coeff_sigma[data.free_star_indices] = param_sigma[
                sigma_start : slices["sed_coeff"].stop : data.sed_library.n_components
            ]
            star_payload[f"sed_coeff_{component_id}_sigma"] = coeff_sigma
    pd.DataFrame(star_payload).to_csv(output_dir / "fit_star_params.csv", index=False)

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
            if param_sigma is not None and slices is not None:
                pass_index = filt_i * data.detector_ids.size + det_i
                pass_rows[-1]["delta_lambda_um_sigma"] = param_sigma[
                    slices["shift"].start + pass_index
                ]
                pass_rows[-1]["width_sigma"] = param_sigma[slices["width"].start + pass_index]
    pd.DataFrame(pass_rows).to_csv(output_dir / "fit_passband_params.csv", index=False)

    zp_rows = []
    prism_reference = data.sim_metadata.get("prism_reference_exposure_id")
    for exp_id, zp_value in enumerate(state.zp):
        exposure_param = data.exposure_free_index[exp_id]
        row = {
            "exposure_id": exp_id,
            "measurement_type": (
                "prism"
                if prism_reference is not None and exp_id >= int(prism_reference)
                else "imaging"
            ),
            "is_reference": exposure_param < 0,
            "zp_mag": zp_value,
        }
        if param_sigma is not None and slices is not None:
            row["zp_mag_sigma"] = (
                0.0
                if exposure_param < 0
                else param_sigma[slices["zp"].start + exposure_param]
            )
        zp_rows.append(row)
    pd.DataFrame(zp_rows).to_csv(output_dir / "fit_exposure_zeropoints.csv", index=False)

    smooth_rows = []
    for smooth_id, name in enumerate(["x", "y", "x2", "xy", "y2"]):
        row = {"basis_name": name, "coefficient_mag": state.smooth_coeff[smooth_id]}
        if param_sigma is not None and slices is not None:
            row["coefficient_mag_sigma"] = param_sigma[slices["smooth"].start + smooth_id]
        smooth_rows.append(row)
    pd.DataFrame(smooth_rows).to_csv(output_dir / "fit_smooth_coeffs.csv", index=False)

    amp_rows = []
    for det_i, det in enumerate(data.detector_ids):
        for amp_id in range(config.n_amp):
            index = det_i * config.n_amp + amp_id
            row = {
                "detector_id": det,
                "amp_id": amp_id,
                "amp_offset_mag": state.amp_offset[det_i, amp_id],
            }
            if param_sigma is not None and slices is not None:
                row["amp_offset_mag_sigma"] = param_sigma[slices["amp"].start + index]
            amp_rows.append(row)
    pd.DataFrame(amp_rows).to_csv(output_dir / "fit_amp_offsets.csv", index=False)

    ice_rows = []
    n_loglam = data.ice_loglam_nodes.size
    for thick_id, thickness in enumerate(data.ice_thickness_nodes):
        for loglam_id, loglam in enumerate(data.ice_loglam_nodes):
            index = thick_id * n_loglam + loglam_id
            ice_rows.append(
                {
                    "ice_thickness_node_id": thick_id,
                    "ice_thickness": thickness,
                    "loglam_node_id": loglam_id,
                    "log10_wavelength": loglam,
                    "ice_logt_node_value": state.ice_coeff[index],
                }
            )
            if param_sigma is not None and slices is not None:
                ice_rows[-1]["ice_logt_node_sigma"] = param_sigma[slices["ice"].start + index]
    pd.DataFrame(ice_rows).to_csv(output_dir / "fit_ice_spline_params.csv", index=False)

    prism_response_rows = pd.DataFrame(
        {
            "wavelength_pixel_id": np.arange(data.prism_wave.size, dtype=int),
            "wavelength_um": data.prism_wave,
            "prism_response_mag": state.prism_response,
        }
    )
    if param_sigma is not None and slices is not None and data.prism_wave.size:
        prism_response_rows["prism_response_mag_sigma"] = param_sigma[
            slices["prism_response"]
        ]
    prism_response_rows.to_csv(output_dir / "fit_prism_response.csv", index=False)

    summary.to_csv(output_dir / "fit_iteration_summary.csv", index=False)

    residuals = data.measurements.copy()
    residuals["mag_residual"] = final_resid
    residuals["fit_scalar_delta_mag"] = evaluate_scalar_terms(data, state, config)
    residuals["fit_ab_zeropoint_mag"] = evaluate_ab_zeropoint_observations(data, state, config)
    residuals.to_csv(output_dir / "fit_residuals.csv", index=False)

    residuals[
        [
            "obs_id",
            "exposure_id",
            "filter_id",
            "filter_name",
            "detector_id",
            "amp_id",
            "x",
            "y",
            "ice_thickness",
            "fit_ab_zeropoint_mag",
        ]
    ].to_csv(output_dir / "fit_ab_zeropoints.csv", index=False)

    if not data.prism_measurements.empty:
        prism_residuals = data.prism_measurements.copy()
        prism_residuals["mag_residual"] = final_prism_resid
        prism_residuals["fit_prism_response_mag"] = state.prism_response[
            data.prism_pixel_id
        ]
        prism_residuals["fit_scalar_delta_mag"] = evaluate_prism_scalar_terms(
            data, state, config
        )
        prism_residuals["fit_ab_zeropoint_mag"] = evaluate_prism_ab_zeropoints(
            data, state, config
        )
        prism_residuals["calibrated_ab_mag"] = (
            prism_residuals["mag_obs"] + prism_residuals["fit_ab_zeropoint_mag"]
        )
        prism_residuals.to_csv(output_dir / "fit_prism_residuals.csv", index=False)
        prism_residuals[
            [
                "prism_obs_id",
                "spectrum_id",
                "exposure_id",
                "star_id",
                "detector_id",
                "amp_id",
                "wavelength_pixel_id",
                "wavelength_um",
                "x",
                "y",
                "ice_thickness",
                "fit_ab_zeropoint_mag",
            ]
        ].to_csv(output_dir / "fit_prism_ab_zeropoints.csv", index=False)

    with open(output_dir / "fit_config.json", "w", encoding="utf-8") as handle:
        payload = dict(config.__dict__)
        if uncertainty_info is not None:
            payload.update(uncertainty_info)
        json.dump(payload, handle, indent=2, sort_keys=True)

    # Copy metadata for convenience when fit outputs are inspected standalone.
    meta_path = Path(config.input_dir) / "simulation_metadata.json"
    if meta_path.exists():
        shutil.copy(meta_path, output_dir / "simulation_metadata.json")


def print_diagnostics(
    data: DataBundle,
    state: ModelState,
    summary: pd.DataFrame,
    final_resid: np.ndarray,
    final_prism_resid: np.ndarray,
    param_sigma: np.ndarray | None = None,
    slices: dict[str, slice] | None = None,
) -> None:
    true_shift, true_width, true_ice = make_truth_arrays(data)

    print("\nRoman imaging / prism sparse fit diagnostics")
    print("---------------------------------------------")
    print(f"Number of imaging observations: {len(data.measurements)}")
    print(f"Number of prism wavelength pixels: {len(data.prism_measurements)}")
    print(f"Number of stars: {data.star_ids.size}")
    print(f"Number of fixed absolute calibrator stars: {int(data.star_is_calibrator.sum())}")
    print(f"Number of fitted stars: {data.free_star_indices.size}")
    print(f"Number of filters: {data.filter_ids.size}")
    print(f"Number of detectors: {data.detector_ids.size}")
    print(f"Number of BOSZ EMPCA SED components: {data.sed_library.n_components}")
    n_params = parameter_slices(data)["amp"].stop
    n_amp = int(data.sim_metadata.get("n_amp", 32))
    print(f"Number of fitted linearized parameters per iteration: {n_params}")
    print(f"Initial combined RMS residual: {summary.iloc[0]['rms_residual']:.6f} mag")
    print(f"Final imaging RMS residual: {rms(final_resid):.6f} mag")
    if final_prism_resid.size:
        print(f"Final prism RMS residual: {rms(final_prism_resid):.6f} mag")

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
        _, _, true_surface = ice_surface_on_grid(data, true_ice)
        _, _, fit_surface = ice_surface_on_grid(data, state.ice_coeff)
        print(f"Ice log-throughput surface RMS error: {rms(fit_surface - true_surface):.6f}")
    if data.true_prism_response is not None and state.prism_response.size:
        truth = data.true_prism_response.sort_values("wavelength_pixel_id")
        true_response = truth["prism_response_mag"].to_numpy(float)
        if true_response.size == state.prism_response.size:
            print(
                "Prism wavelength-response RMS error: "
                f"{rms(state.prism_response - true_response):.6f} mag"
            )
    if data.true_exposure_zeropoints is not None:
        true_zp = np.zeros_like(state.zp)
        for _, row in data.true_exposure_zeropoints.iterrows():
            exp_id = int(row["exposure_id"])
            if 0 <= exp_id < true_zp.size:
                true_zp[exp_id] = row["zp_mag"]
        print(f"Exposure ZP RMS error: {rms(state.zp - true_zp):.6f} mag")
    if data.true_smooth_coeffs is not None:
        true_smooth = data.true_smooth_coeffs["coefficient_mag"].to_numpy(float)
        if true_smooth.size == state.smooth_coeff.size:
            print(f"Smooth coefficient RMS error: {rms(state.smooth_coeff - true_smooth):.6f} mag")
    if data.true_amp_offsets is not None:
        true_amp = np.zeros_like(state.amp_offset)
        for _, row in data.true_amp_offsets.iterrows():
            det_matches = np.nonzero(data.detector_ids == int(row["detector_id"]))[0]
            amp_id = int(row["amp_id"])
            if det_matches.size and 0 <= amp_id < n_amp:
                true_amp[det_matches[0], amp_id] = row["amp_offset_mag"]
        fit_centered = state.amp_offset - state.amp_offset.mean(axis=1, keepdims=True)
        true_centered = true_amp - true_amp.mean(axis=1, keepdims=True)
        print(f"Amp offset RMS error, detector means removed: {rms(fit_centered - true_centered):.6f} mag")
    if data.true_star_params is not None:
        coeff_cols = [f"sed_coeff_{i}" for i in range(data.sed_library.n_components)]
        if all(col in data.true_star_params.columns for col in coeff_cols):
            truth = data.true_star_params.set_index("star_id").loc[data.star_ids]
            true_coeff = truth[coeff_cols].to_numpy(float)
            print(f"Stellar EMPCA coefficient RMS error: {rms(state.sed_coeff - true_coeff):.6f}")
    if param_sigma is not None and slices is not None:
        ice_sigma = param_sigma[slices["ice"]]
        print(f"Median formal ice-node uncertainty: {np.median(ice_sigma):.6f}")
        if state.prism_response.size:
            print(
                "Median formal prism-response uncertainty: "
                f"{np.median(param_sigma[slices['prism_response']]):.6f} mag"
            )
    print(
        "Note: stellar EMPCA coefficients, passband color terms, and ice surface "
        "modes remain partially degenerate; recovery depends on priors and "
        "ice-thickness leverage."
    )


def make_plots(
    data: DataBundle,
    state: ModelState,
    summary: pd.DataFrame,
    final_resid: np.ndarray,
    final_prism_resid: np.ndarray,
    config: FitConfig,
    param_sigma: np.ndarray | None = None,
    slices: dict[str, slice] | None = None,
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    true_shift, true_width, true_ice = make_truth_arrays(data)

    def plot_passband_recovery(true_values, fit_values, xlabel, ylabel, title, filename):
        lim = 1.1 * np.max(np.abs(np.r_[true_values.ravel(), fit_values.ravel()]))
        lim = max(lim, 1e-6)
        plt.figure(figsize=(6.2, 5.8))
        cmap = plt.get_cmap("tab10")
        for filt_i, filter_name in enumerate(data.filter_names):
            color = cmap(filt_i % 10)
            plt.scatter(
                true_values[filt_i],
                fit_values[filt_i],
                s=52,
                color=color,
                label=filter_name,
                alpha=0.9,
            )
            x_med = np.median(true_values[filt_i])
            y_med = np.median(fit_values[filt_i])
            plt.annotate(
                filter_name,
                (x_med, y_med),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )
        plt.plot([-lim, lim], [-lim, lim], color="0.2", linewidth=1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.gca().set_aspect("equal", adjustable="box")
        if data.detector_ids.size > 1:
            plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(
        summary["iteration"], summary["rms_residual"], marker="o", label="combined"
    )
    plt.plot(
        summary["iteration"],
        summary["imaging_rms_residual"],
        marker="s",
        label="imaging",
    )
    if summary["prism_rms_residual"].notna().any():
        plt.plot(
            summary["iteration"],
            summary["prism_rms_residual"],
            marker="^",
            label="prism",
        )
    plt.xlabel("Iteration")
    plt.ylabel("RMS residual [mag]")
    plt.title("Residuals by iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "residuals_by_iteration.png", dpi=160)
    plt.close()

    if true_shift is not None:
        plot_passband_recovery(
            true_shift,
            state.shift,
            "True shift [um]",
            "Fitted shift [um]",
            "Passband shift recovery",
            "passband_shift_true_vs_fit.png",
        )

    if true_width is not None:
        plot_passband_recovery(
            true_width,
            state.width,
            "True width coefficient",
            "Fitted width coefficient",
            "Passband width recovery",
            "passband_width_true_vs_fit.png",
        )

    if data.true_prism_response is not None and state.prism_response.size:
        truth = data.true_prism_response.sort_values("wavelength_pixel_id")
        true_response = truth["prism_response_mag"].to_numpy(float)
        response_sigma = (
            param_sigma[slices["prism_response"]]
            if param_sigma is not None and slices is not None
            else None
        )
        response_residual = state.prism_response - true_response
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(9, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0]},
        )
        axes[0].plot(data.prism_wave, true_response, label="true", linewidth=1.8)
        axes[0].plot(data.prism_wave, state.prism_response, label="fit", linewidth=1.3)
        if response_sigma is not None:
            axes[0].fill_between(
                data.prism_wave,
                state.prism_response - response_sigma,
                state.prism_response + response_sigma,
                alpha=0.22,
                label="formal 1-sigma",
            )
        axes[0].set_ylabel("Prism response [mag]")
        axes[0].set_title("Prism wavelength calibration recovery")
        axes[0].legend()
        axes[1].plot(data.prism_wave, response_residual, color="tab:red")
        axes[1].axhline(0.0, color="0.3", linewidth=1)
        axes[1].set_xlabel("Wavelength [um]")
        axes[1].set_ylabel("Fit - true [mag]")
        fig.tight_layout()
        fig.savefig(output_dir / "prism_response_true_vs_fit.png", dpi=160)
        plt.close(fig)

    if final_prism_resid.size:
        prism_frame = pd.DataFrame(
            {
                "wavelength_um": data.prism_measurements["wavelength_um"].to_numpy(float),
                "pixel_id": data.prism_pixel_id,
                "residual": final_prism_resid,
            }
        )
        grouped = prism_frame.groupby("pixel_id")
        wave_by_pixel = grouped["wavelength_um"].median()
        median_by_pixel = grouped["residual"].median()
        rms_by_pixel = grouped["residual"].apply(lambda values: rms(values.to_numpy()))
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        axes[0].scatter(
            prism_frame["wavelength_um"],
            prism_frame["residual"],
            s=2,
            alpha=0.08,
        )
        axes[0].plot(wave_by_pixel, median_by_pixel, color="black", linewidth=1.2)
        axes[0].axhline(0.0, color="0.3", linewidth=1)
        axes[0].set_ylabel("Residual [mag]")
        axes[0].set_title("Prism residual versus wavelength")
        axes[1].plot(wave_by_pixel, rms_by_pixel, color="tab:orange")
        axes[1].set_xlabel("Wavelength [um]")
        axes[1].set_ylabel("RMS residual [mag]")
        fig.tight_layout()
        fig.savefig(output_dir / "prism_residual_vs_wavelength.png", dpi=160)
        plt.close(fig)

        plt.figure(figsize=(8, 4.8))
        standard = data.prism_measurements[
            "is_spectrophotometric_standard"
        ].to_numpy(bool)
        bins = np.linspace(
            np.percentile(final_prism_resid, 0.2),
            np.percentile(final_prism_resid, 99.8),
            55,
        )
        plt.hist(
            final_prism_resid[~standard], bins=bins, histtype="step", label="field stars"
        )
        if np.any(standard):
            plt.hist(
                final_prism_resid[standard],
                bins=bins,
                histtype="step",
                linewidth=1.6,
                label="spectrophotometric standards",
            )
        plt.axvline(0.0, color="0.3", linewidth=1)
        plt.xlabel("Prism residual [mag]")
        plt.ylabel("Count")
        plt.title("Prism residual distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "prism_residual_histogram.png", dpi=160)
        plt.close()

        if np.any(standard):
            prism_ab_zp = evaluate_prism_ab_zeropoints(data, state, config)
            calibrated_ab = (
                data.prism_measurements["mag_obs"].to_numpy(float) + prism_ab_zp
            )
            standard_ids, standard_counts = np.unique(
                data.prism_measurements.loc[standard, "star_id"].to_numpy(int),
                return_counts=True,
            )
            selected_star = standard_ids[np.argmax(standard_counts)]
            selected = standard & (
                data.prism_measurements["star_id"].to_numpy(int) == selected_star
            )
            standard_frame = pd.DataFrame(
                {
                    "pixel_id": data.prism_pixel_id[selected],
                    "wavelength_um": data.prism_measurements.loc[
                        selected, "wavelength_um"
                    ].to_numpy(float),
                    "fit_ab": calibrated_ab[selected],
                    "true_ab": data.prism_measurements.loc[
                        selected, "true_ab_mag"
                    ].to_numpy(float),
                }
            )
            med = standard_frame.groupby("pixel_id").median(numeric_only=True)
            standard_residual = med["fit_ab"] - med["true_ab"]
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(9, 7),
                sharex=True,
                gridspec_kw={"height_ratios": [2.0, 1.0]},
            )
            axes[0].plot(med["wavelength_um"], med["true_ab"], label="known standard")
            axes[0].plot(med["wavelength_um"], med["fit_ab"], label="calibrated prism")
            axes[0].invert_yaxis()
            axes[0].set_ylabel("Narrow-bin AB magnitude")
            axes[0].set_title(f"Spectrophotometric standard star {selected_star}")
            axes[0].legend()
            axes[1].plot(med["wavelength_um"], standard_residual, color="tab:red")
            axes[1].axhline(0.0, color="0.3", linewidth=1)
            axes[1].set_xlabel("Wavelength [um]")
            axes[1].set_ylabel("Fit - known [mag]")
            fig.tight_layout()
            fig.savefig(output_dir / "prism_standard_absolute_calibration.png", dpi=160)
            plt.close(fig)

    amps = np.arange(config.n_amp)
    dets = range(data.detector_ids.size)
    true_amp_offsets = None
    if data.true_amp_offsets is not None:
        true_amp_offsets = np.zeros_like(state.amp_offset)
        for _, row in data.true_amp_offsets.iterrows():
            det_matches = np.nonzero(data.detector_ids == int(row["detector_id"]))[0]
            amp_id = int(row["amp_id"])
            if det_matches.size and 0 <= amp_id < config.n_amp:
                true_amp_offsets[det_matches[0], amp_id] = row["amp_offset_mag"]

        plt.figure(figsize=(10, 5))
        for det_i in dets:
            plt.plot(
                amps,
                true_amp_offsets[det_i],
                marker="o",
                label=f"det {int(data.detector_ids[det_i]):02d}",
            )
        plt.axhline(0.0, color="0.3", linewidth=1)
        plt.xlabel("Amplifier ID")
        plt.ylabel("True amp offset [mag]")
        plt.title("True amplifier offsets")
        if data.detector_ids.size > 1:
            plt.legend(ncol=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / "true_amp_offsets.png", dpi=160)
        plt.close()

    plt.figure(figsize=(10, 5))
    for det_i in dets:
        plt.plot(
            amps,
            state.amp_offset[det_i],
            marker="o",
            label=f"det {int(data.detector_ids[det_i]):02d}",
        )
    plt.axhline(0.0, color="0.3", linewidth=1)
    plt.xlabel("Amplifier ID")
    plt.ylabel("Recovered amp offset [mag]")
    plt.title("Recovered amplifier offsets")
    if data.detector_ids.size > 1:
        plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "recovered_amp_offsets.png", dpi=160)
    plt.close()

    if true_amp_offsets is not None:
        true_centered = true_amp_offsets - true_amp_offsets.mean(axis=1, keepdims=True)
        fit_centered = state.amp_offset - state.amp_offset.mean(axis=1, keepdims=True)
        lim = 1.1 * np.max(np.abs(np.r_[true_centered.ravel(), fit_centered.ravel()]))
        lim = max(lim, 1e-6)
        plt.figure(figsize=(6, 6))
        plt.scatter(true_centered.ravel(), fit_centered.ravel(), s=28, alpha=0.8)
        plt.plot([-lim, lim], [-lim, lim], color="0.2", linewidth=1)
        plt.xlabel("True amp offset, detector mean removed [mag]")
        plt.ylabel("Recovered amp offset, detector mean removed [mag]")
        plt.title("Amplifier offset recovery")
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.savefig(output_dir / "amp_offset_comparison.png", dpi=160)
        plt.close()

    smooth_grid = smooth_fields_on_grid(data, state, config, n_grid=120)
    if smooth_grid is not None:
        _, _, true_field, fit_field, residual_field = smooth_grid
        image_kwargs = {
            "origin": "lower",
            "extent": [0, config.nx - 1, 0, config.ny - 1],
            "aspect": "equal",
        }
        plt.figure(figsize=(6.5, 5.5))
        im = plt.imshow(true_field, **image_kwargs)
        plt.colorbar(im, label="mag")
        plt.xlabel("x pixel")
        plt.ylabel("y pixel")
        plt.title("True smooth field")
        plt.tight_layout()
        plt.savefig(output_dir / "smooth_field_true.png", dpi=160)
        plt.close()

        plt.figure(figsize=(6.5, 5.5))
        im = plt.imshow(fit_field, **image_kwargs)
        plt.colorbar(im, label="mag")
        plt.xlabel("x pixel")
        plt.ylabel("y pixel")
        plt.title("Recovered smooth field")
        plt.tight_layout()
        plt.savefig(output_dir / "smooth_field_recovered.png", dpi=160)
        plt.close()

        plt.figure(figsize=(6.5, 5.5))
        vmax = np.max(np.abs(residual_field))
        vmax = max(vmax, 1e-9)
        im = plt.imshow(
            residual_field, vmin=-vmax, vmax=vmax, cmap="coolwarm", **image_kwargs
        )
        plt.colorbar(im, label="mag")
        plt.xlabel("x pixel")
        plt.ylabel("y pixel")
        plt.title("Recovered - true smooth field, mean removed")
        plt.tight_layout()
        plt.savefig(output_dir / "smooth_field_residual.png", dpi=160)
        plt.close()

    if true_ice is not None:
        log_wave, thickness_grid, true_surface = ice_surface_on_grid(data, true_ice)
        _, _, fit_surface = ice_surface_on_grid(data, state.ice_coeff)
        resid_surface = fit_surface - true_surface
        vmax = np.max(np.abs(np.r_[true_surface.ravel(), fit_surface.ravel()]))
        rvmax = np.max(np.abs(resid_surface))
        extent = [log_wave.min(), log_wave.max(), thickness_grid.min(), thickness_grid.max()]

        uncertainty_surface = None
        if param_sigma is not None and slices is not None:
            _, _, uncertainty_surface = ice_surface_uncertainty_on_grid(
                data, param_sigma[slices["ice"]]
            )

        plt.figure(figsize=(11, 8.2))
        panels = [
            (1, true_surface, "True", "coolwarm", -vmax, vmax),
            (2, fit_surface, "Fit", "coolwarm", -vmax, vmax),
            (3, resid_surface, "Fit - true", "coolwarm", -rvmax, rvmax),
        ]
        if uncertainty_surface is not None:
            panels.append(
                (
                    4,
                    uncertainty_surface,
                    "Formal 1-sigma uncertainty",
                    "viridis",
                    0.0,
                    np.max(uncertainty_surface),
                )
            )
        for panel, image, title, cmap, vmin, vmax_panel in panels:
            plt.subplot(2, 2, panel)
            im = plt.imshow(
                image,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax_panel,
            )
            plt.title(title)
            plt.xlabel("log10 wavelength [um]")
            if panel in (1, 3):
                plt.ylabel("Ice thickness")
            label = "1-sigma log-throughput" if panel == 4 else "log-throughput"
            plt.colorbar(im, fraction=0.046, pad=0.04, label=label)
        plt.suptitle("Ice log-throughput surface")
        plt.tight_layout()
        plt.savefig(output_dir / "ice_surface_true_vs_fit.png", dpi=160)
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
    if "ice_thickness" in data.measurements.columns:
        ice_x = data.measurements["ice_thickness"]
        ice_label = "ice_thickness"
    else:
        ice_x = data.measurements["ice_amount_obs"]
        ice_label = "ice_amount_obs"
    plt.scatter(ice_x, final_resid, s=4, alpha=0.25)
    plt.axhline(0.0, color="0.3", linewidth=1)
    plt.xlabel(ice_label)
    plt.ylabel("Residual [mag]")
    plt.title("Residual versus ice thickness")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_vs_ice_thickness.png", dpi=160)
    plt.close()

    x_pix = data.measurements["x"].to_numpy(float)
    rng = np.random.default_rng(config.random_seed + 1)
    max_points = 40000
    if final_resid.size > max_points:
        sample = rng.choice(final_resid.size, size=max_points, replace=False)
    else:
        sample = np.arange(final_resid.size)
    plt.figure(figsize=(9, 5))
    plt.scatter(x_pix[sample], final_resid[sample], s=3, alpha=0.25)
    plt.axhline(0.0, color="0.2", linewidth=1)
    plt.xlabel("x pixel")
    plt.ylabel("Photometric residual [mag]")
    plt.title("Residual versus x")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_vs_x.png", dpi=160)
    plt.close()

    amp_median = np.full((data.detector_ids.size, config.n_amp), np.nan)
    prism_amp_median = np.full_like(amp_median, np.nan)
    for det_i in range(data.detector_ids.size):
        for amp_id in range(config.n_amp):
            mask = (data.detector_param_id == det_i) & (data.amp_id == amp_id)
            if np.any(mask):
                amp_median[det_i, amp_id] = np.median(final_resid[mask])
            if final_prism_resid.size:
                prism_mask = (
                    (data.prism_detector_param_id == det_i)
                    & (data.prism_amp_id == amp_id)
                )
                if np.any(prism_mask):
                    prism_amp_median[det_i, amp_id] = np.median(
                        final_prism_resid[prism_mask]
                    )
    plt.figure(figsize=(10, 5))
    for det_i in range(data.detector_ids.size):
        plt.plot(
            amps,
            amp_median[det_i],
            marker="o",
            label=f"imaging det {int(data.detector_ids[det_i]):02d}",
        )
        if final_prism_resid.size:
            plt.plot(
                amps,
                prism_amp_median[det_i],
                marker="s",
                linestyle="--",
                label=f"prism det {int(data.detector_ids[det_i]):02d}",
            )
    plt.axhline(0.0, color="0.2", linewidth=1)
    plt.xlabel("Amplifier ID")
    plt.ylabel("Median residual [mag]")
    plt.title("Median residual per amplifier")
    if data.detector_ids.size > 1 or final_prism_resid.size:
        plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "residual_vs_amp.png", dpi=160)
    plt.close()

    if data.true_star_params is not None:
        truth = data.true_star_params.set_index("star_id").loc[data.star_ids]
        coeff_cols = [f"sed_coeff_{i}" for i in range(data.sed_library.n_components)]
        if all(col in truth.columns for col in coeff_cols):
            parameters = [
                ("mag_norm", truth["mag_norm"].to_numpy(float), state.mag_norm)
            ]
            for component_id, col in enumerate(coeff_cols):
                parameters.append(
                    (
                        f"coeff {component_id}",
                        truth[col].to_numpy(float),
                        state.sed_coeff[:, component_id],
                    )
                )

            n_cols = len(parameters)
            fig, axes = plt.subplots(
                2,
                n_cols,
                figsize=(3.5 * n_cols, 7.0),
                squeeze=False,
            )
            free_mask = ~data.star_is_calibrator
            calib_mask = data.star_is_calibrator
            for col_index, (label, true_values, fit_values) in enumerate(parameters):
                recovery_ax = axes[0, col_index]
                residual_ax = axes[1, col_index]
                residual = fit_values - true_values

                recovery_ax.scatter(true_values[free_mask], fit_values[free_mask], s=5, alpha=0.35)
                if np.any(calib_mask):
                    recovery_ax.scatter(
                        true_values[calib_mask],
                        fit_values[calib_mask],
                        s=24,
                        facecolors="none",
                        edgecolors="crimson",
                        linewidths=0.8,
                    )
                vmin = min(true_values.min(), fit_values.min())
                vmax = max(true_values.max(), fit_values.max())
                recovery_ax.plot([vmin, vmax], [vmin, vmax], color="0.2", linewidth=1)
                recovery_ax.set_title(label)
                recovery_ax.set_xlabel(f"True {label}")
                recovery_ax.set_ylabel(f"Fitted {label}")

                residual_ax.scatter(true_values[free_mask], residual[free_mask], s=5, alpha=0.35)
                if np.any(calib_mask):
                    residual_ax.scatter(
                        true_values[calib_mask],
                        residual[calib_mask],
                        s=24,
                        facecolors="none",
                        edgecolors="crimson",
                        linewidths=0.8,
                    )
                residual_ax.axhline(0.0, color="0.2", linewidth=1)
                residual_ax.set_xlabel(f"True {label}")
                residual_ax.set_ylabel("Fit - true")
            plt.suptitle("Stellar BOSZ EMPCA parameter recovery")
            plt.tight_layout()
            plt.savefig(output_dir / "stellar_param_recovery.png", dpi=160)
            plt.close()

    bins = np.linspace(
        np.percentile(final_resid, 0.2),
        np.percentile(final_resid, 99.8),
        50,
    )
    plt.figure(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    for filt_i, filter_name in enumerate(data.filter_names):
        mask = data.filter_param_id == filt_i
        plt.hist(
            final_resid[mask],
            bins=bins,
            histtype="step",
            linewidth=1.4,
            color=cmap(filt_i % 10),
            label=filter_name,
        )
    plt.axvline(0.0, color="0.3", linewidth=1)
    plt.xlabel("Residual [mag]")
    plt.ylabel("Count")
    plt.title("Residual histogram by filter")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "residual_histogram.png", dpi=160)
    plt.close()


def parse_args() -> FitConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=FitConfig.input_dir)
    parser.add_argument("--output-dir", default=FitConfig.output_dir)
    parser.add_argument("--sed-basis-path", default=FitConfig.sed_basis_path)
    parser.add_argument("--n-iter", type=int, default=FitConfig.n_iter)
    parser.add_argument("--damping", type=float, default=FitConfig.damping)
    parser.add_argument("--max-stars", type=int, default=None)
    args = parser.parse_args()
    return FitConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sed_basis_path=args.sed_basis_path,
        n_iter=args.n_iter,
        damping=args.damping,
        max_stars=args.max_stars,
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    config = parse_args()
    data = load_data(config)
    state, summary, final_resid, final_prism_resid = run_fit(data, config)
    param_sigma, slices, uncertainty_info = estimate_parameter_uncertainties(data, state, config)
    save_outputs(
        data,
        state,
        summary,
        final_resid,
        final_prism_resid,
        config,
        param_sigma=param_sigma,
        slices=slices,
        uncertainty_info=uncertainty_info,
    )
    print_diagnostics(
        data,
        state,
        summary,
        final_resid,
        final_prism_resid,
        param_sigma=param_sigma,
        slices=slices,
    )
    make_plots(
        data,
        state,
        summary,
        final_resid,
        final_prism_resid,
        config,
        param_sigma=param_sigma,
        slices=slices,
    )
    print(f"Saved fit outputs to {Path(config.output_dir).resolve()}")


if __name__ == "__main__":
    main()
