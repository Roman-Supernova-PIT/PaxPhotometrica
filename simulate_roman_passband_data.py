#!/usr/bin/env python3
"""Simulate Roman-like WFI scalar and chromatic calibration photometry.

This simulator combines the scalar ubercalibration toy model with chromatic
calibration: exposure zeropoints, smooth focal-plane terms, amplifier offsets,
small passband shifts, passband width changes, and ice-induced
wavelength/thickness-dependent throughput changes.

Throughput perturbations are modeled in log-throughput space because small
multiplicative throughput changes then add linearly. Observations are still
generated from linear broadband flux integrals, which is the physical
measurement made by the detector.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Tuple

_MPL_CACHE = Path(tempfile.gettempdir()) / "roman_passband_mpl_cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_CACHE.resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPEED_OF_LIGHT_CM_S = 2.99792458e10
MICRON_TO_CM = 1.0e-4
AB_FNU_CGS = 3631.0e-23  # erg / s / cm^2 / Hz


@dataclass
class SimConfig:
    random_seed: int = 12345
    n_star: int = 2000
    n_exp: int = 30
    n_det: int = 1
    nx: int = 4096
    ny: int = 4096
    n_amp: int = 32
    n_filter: int = 6
    reference_filter_id: int = 3
    n_absolute_calibrator: int = 5
    wave_min: float = 0.45
    wave_max: float = 2.30
    n_wave: int = 2000
    passband_file: str = "passbands.txt"
    sed_basis_path: str = "bosz_logflux_empca_basis.npz"
    ice_loglam_nodes_file: str = "ice_loglam_nodes.txt"
    n_ice_thickness_nodes: int = 5
    ice_thickness_min: float = 0.0
    ice_thickness_max: float = 1.2
    phot_sigma_mag: float = 0.005
    output_dir: str = "passband_sim_outputs"
    detection_fraction: float = 0.92
    dither_sigma_pix: float = 500.0
    zp_sigma_mag: float = 0.01
    amp_sigma_mag: float = 0.003
    shift_sigma_um: float = 0.001
    width_sigma: float = 0.01
    mode_smooth_sigma_pix: float = 4.0
    max_abs_phi_shift: float = 250.0
    max_abs_phi_width: float = 80.0
    rotation_angles_deg: Tuple[float, ...] = (0.0, -5.0, 5.0)
    true_smooth_coeffs: Tuple[float, float, float, float, float] = (
        0.0040,
        -0.0030,
        0.0060,
        -0.0020,
        -0.0040,
    )


class BOSZEMPCASEDLibrary:
    """Low-dimensional BOSZ EMPCA stellar SED library.

    The basis stores normalized log flux. The simulator draws one of the input
    BOSZ coefficient vectors for each star and assigns a separate magnitude
    normalization, so photometry uses realistic relative colors without
    pretending the BOSZ absolute flux scale survived the EMPCA normalization.
    """

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

    def resampled_to(self, wave: np.ndarray) -> "BOSZEMPCASEDLibrary":
        """Return a lightweight copy with log-flux basis interpolated to ``wave``."""
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
        """Evaluate physical-scale SEDs with ``mag_norm`` as reference AB mag.

        The EMPCA basis stores only relative log-flux shapes. We assign an
        amplitude by requiring each SED to have AB magnitude ``mag_norm`` in the
        configured reference passband. Flux density is returned per micron in
        arbitrary-but-AB-consistent cgs units.
        """
        shape = self.sed_shape_from_coefficients(theta)
        mag_norm = np.asarray(mag_norm, dtype=float)
        shape_count = photon_count_integral(
            shape * reference_passband[None, :], self.wave_micron, axis=1
        )
        ab_count = ab_reference_count(self.wave_micron, reference_passband)
        scale = ab_count * 10.0 ** (-0.4 * mag_norm) / np.maximum(shape_count, 1e-300)
        return scale[..., None] * shape


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
    """Photon-counting integral up to the constant 1/(hc).

    For wavelength-grid flux density per micron, counts are proportional to
    integral f_lambda(lambda) T(lambda) lambda dlambda. The missing constant
    cancels in AB ratios.
    """
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


def normalized_xy(x: np.ndarray, y: np.ndarray, config: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    """Map detector pixels to [-1, 1] normalized coordinates."""
    xn = 2.0 * (x / (config.nx - 1.0)) - 1.0
    yn = 2.0 * (y / (config.ny - 1.0)) - 1.0
    return xn, yn


def poly_basis(xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
    """Smooth star-flat polynomial terms: x, y, x^2, x*y, y^2."""
    xn = np.asarray(xn)
    yn = np.asarray(yn)
    return np.column_stack((xn, yn, xn**2, xn * yn, yn**2))


def exposure_rotation_sequence(config: SimConfig) -> np.ndarray:
    """Create a deterministic sequence with at least two rotation angles."""
    angles = np.asarray(config.rotation_angles_deg, dtype=float)
    if angles.size < 2:
        raise ValueError("rotation_angles_deg must contain at least two angles")
    return angles[np.arange(config.n_exp) % angles.size]


def apply_exposure_transform(
    x0: np.ndarray, y0: np.ndarray, dx: float, dy: float, rotation_deg: float, config: SimConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate positions about detector center, then apply translational dither."""
    cx = 0.5 * (config.nx - 1.0)
    cy = 0.5 * (config.ny - 1.0)
    theta = np.deg2rad(rotation_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_centered = x0 - cx
    y_centered = y0 - cy
    x = cx + cos_t * x_centered - sin_t * y_centered + dx
    y = cy + sin_t * x_centered + cos_t * y_centered + dy
    return x, y


def make_wavelength_grid(config: SimConfig) -> np.ndarray:
    return np.linspace(config.wave_min, config.wave_max, config.n_wave)


def read_nominal_passbands(
    config: SimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Read Roman nominal passbands from a whitespace-delimited text file.

    The file is expected to have a wavelength column in microns followed by one
    throughput column per filter, for example: Wave F062 F087 ... F184. The
    throughputs only need to be relative; broadband magnitudes absorb the
    arbitrary normalization into each star's fitted magnitude normalization.
    """
    path = Path(config.passband_file)
    if not path.exists():
        raise FileNotFoundError(f"Missing passband file: {path}")

    table = pd.read_csv(path, sep=r"\s+", engine="python")
    if "Wave" in table.columns:
        wave_col = "Wave"
    else:
        wave_col = table.columns[0]

    wave = table[wave_col].to_numpy(float)
    filter_names = [col.strip() for col in table.columns if col != wave_col]
    passbands = table[filter_names].to_numpy(float).T
    passbands = np.clip(passbands, 0.0, None)

    if passbands.shape[0] != config.n_filter:
        raise ValueError(
            f"Config n_filter={config.n_filter}, but {path} has "
            f"{passbands.shape[0]} filter columns"
        )
    if np.any(np.diff(wave) <= 0.0):
        raise ValueError(f"Wavelength grid in {path} must be strictly increasing")

    denom = trapz_integral(passbands, wave, axis=1)
    if np.any(denom <= 0.0):
        raise ValueError("Every passband must have positive integrated throughput")
    centers = trapz_integral(passbands * wave[None, :], wave, axis=1) / denom
    return wave, passbands, centers, filter_names


def gaussian_smooth_1d(values: np.ndarray, sigma_pix: float) -> np.ndarray:
    """Small dependency-free Gaussian smoothing for tabulated passband modes."""
    if sigma_pix <= 0.0:
        return values.copy()
    radius = max(1, int(np.ceil(4.0 * sigma_pix)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_pix) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def make_passband_modes(
    wave: np.ndarray, passbands: np.ndarray, centers: np.ndarray, config: SimConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Derivative-based log-throughput modes for shift and width changes."""
    phi_shift = np.zeros_like(passbands)
    phi_width = np.zeros_like(passbands)

    for filt in range(passbands.shape[0]):
        t0 = passbands[filt]
        t_floor = np.maximum(t0, 1e-6 * np.max(t0))
        # The delivered passbands are finely sampled and can have very sharp
        # numerical edges. Smoothing the floored log-throughput before taking a
        # derivative keeps the toy perturbation modes in the small-signal regime
        # appropriate for the linearized fitter.
        logt_smooth = gaussian_smooth_1d(np.log(t_floor), config.mode_smooth_sigma_pix)
        dlogt_dwave = np.gradient(logt_smooth, wave)
        phi_shift[filt] = np.clip(
            -dlogt_dwave, -config.max_abs_phi_shift, config.max_abs_phi_shift
        )
        phi_width[filt] = np.clip(
            -(wave - centers[filt]) * dlogt_dwave,
            -config.max_abs_phi_width,
            config.max_abs_phi_width,
        )

    return phi_shift, phi_width


def read_ice_loglam_nodes(config: SimConfig, wave: np.ndarray) -> np.ndarray:
    """Read log10 wavelength spline nodes from a user-specified file.

    The file can be whitespace or comma separated. If it has a named column
    `log10_wavelength`, that column is used; otherwise all numeric values in
    the file are flattened and sorted.
    """
    path = Path(config.ice_loglam_nodes_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ice log-wavelength node file: {path}. "
            "Pass --ice-loglam-nodes-file or create the default file."
        )

    try:
        nodes = np.loadtxt(path, comments="#", ndmin=1)
    except Exception:
        table = pd.read_csv(path, sep=None, engine="python", comment="#")
        if "log10_wavelength" in table.columns:
            nodes = table["log10_wavelength"].to_numpy(float)
        else:
            nodes = table.select_dtypes(include=[np.number]).to_numpy().ravel()

    nodes = np.asarray(nodes, dtype=float)
    nodes = np.unique(nodes[np.isfinite(nodes)])
    if nodes.size < 2:
        raise ValueError("At least two log10 wavelength nodes are required")

    log_wave = np.log10(wave)
    if nodes[0] > log_wave.min() or nodes[-1] < log_wave.max():
        raise ValueError(
            "Ice log-wavelength nodes must cover the passband wavelength grid: "
            f"{log_wave.min():.6f} .. {log_wave.max():.6f}"
        )
    return nodes


def make_ice_thickness_nodes(config: SimConfig) -> np.ndarray:
    """Uniform ice-thickness nodes for the rectangular spline grid."""
    if config.n_ice_thickness_nodes < 2:
        raise ValueError("n_ice_thickness_nodes must be at least 2")
    if config.ice_thickness_max <= config.ice_thickness_min:
        raise ValueError("ice_thickness_max must be greater than ice_thickness_min")
    return np.linspace(
        config.ice_thickness_min,
        config.ice_thickness_max,
        config.n_ice_thickness_nodes,
    )


def make_true_ice_spline_values(
    loglam_nodes: np.ndarray, thickness_nodes: np.ndarray
) -> np.ndarray:
    """Create a small oscillatory AR/interference-like log-throughput surface.

    The values are log-throughput perturbations on the rectangular node grid.
    A linear tensor-product spline interpolates between nodes. The surface is
    zero at zero ice thickness and contains both absorption-like and oscillatory
    terms, so it can represent interference effects rather than a separable
    wavelength-only attenuation curve.
    """
    u = (loglam_nodes - loglam_nodes.min()) / (loglam_nodes.max() - loglam_nodes.min())
    t = thickness_nodes / thickness_nodes.max()
    uu, tt = np.meshgrid(u, t)
    absorption = -0.010 * tt * (0.35 + 0.65 * uu)
    ripple_1 = 0.06 * tt * np.sin(2.0 * np.pi * (2.2 * uu + 1.3 * tt))
    ripple_2 = 0.06 * tt**1.4 * np.cos(2.0 * np.pi * (5.0 * uu - 0.7 * tt))
    return absorption + ripple_1 + ripple_2


def interpolate_ice_surface(
    node_values: np.ndarray,
    loglam_nodes: np.ndarray,
    thickness_nodes: np.ndarray,
    wave: np.ndarray,
    thickness: float,
) -> np.ndarray:
    """Evaluate the linear tensor-product spline at one ice thickness."""
    log_wave = np.log10(wave)
    t = np.clip(thickness, thickness_nodes[0], thickness_nodes[-1])
    hi = int(np.searchsorted(thickness_nodes, t, side="right"))
    hi = np.clip(hi, 1, thickness_nodes.size - 1)
    lo = hi - 1
    denom = thickness_nodes[hi] - thickness_nodes[lo]
    w_hi = 0.0 if denom == 0.0 else (t - thickness_nodes[lo]) / denom
    w_lo = 1.0 - w_hi
    surface_lo = np.interp(log_wave, loglam_nodes, node_values[lo])
    surface_hi = np.interp(log_wave, loglam_nodes, node_values[hi])
    return w_lo * surface_lo + w_hi * surface_hi


def write_passband_files(
    output_dir: Path,
    wave: np.ndarray,
    passbands: np.ndarray,
    phi_shift: np.ndarray,
    phi_width: np.ndarray,
    filter_names: list[str],
) -> None:
    passband_rows = []
    mode_rows = []
    for filt in range(passbands.shape[0]):
        passband_rows.append(
            pd.DataFrame(
                {
                    "wavelength_um": wave,
                    "filter_id": filt,
                    "filter_name": filter_names[filt],
                    "throughput": passbands[filt],
                }
            )
        )
        mode_rows.append(
            pd.DataFrame(
                {
                    "wavelength_um": wave,
                    "filter_id": filt,
                    "filter_name": filter_names[filt],
                    "phi_shift": phi_shift[filt],
                    "phi_width": phi_width[filt],
                }
            )
        )
    pd.concat(passband_rows, ignore_index=True).to_csv(
        output_dir / "nominal_passbands.csv", index=False
    )
    pd.concat(mode_rows, ignore_index=True).to_csv(output_dir / "passband_modes.csv", index=False)


def write_ice_spline_files(
    output_dir: Path,
    loglam_nodes: np.ndarray,
    thickness_nodes: np.ndarray,
    true_node_values: np.ndarray,
) -> None:
    """Write rectangular spline grid geometry and true log-throughput values."""
    rows = []
    for thick_id, thickness in enumerate(thickness_nodes):
        for loglam_id, loglam in enumerate(loglam_nodes):
            rows.append(
                {
                    "ice_thickness_node_id": thick_id,
                    "ice_thickness": thickness,
                    "loglam_node_id": loglam_id,
                    "log10_wavelength": loglam,
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "ice_spline_nodes.csv", index=False)

    true_rows = []
    for thick_id, thickness in enumerate(thickness_nodes):
        for loglam_id, loglam in enumerate(loglam_nodes):
            true_rows.append(
                {
                    "ice_thickness_node_id": thick_id,
                    "ice_thickness": thickness,
                    "loglam_node_id": loglam_id,
                    "log10_wavelength": loglam,
                    "ice_logt_node_value": true_node_values[thick_id, loglam_id],
                }
            )
    pd.DataFrame(true_rows).to_csv(output_dir / "true_ice_spline_params.csv", index=False)


def simulate_data(config: SimConfig) -> None:
    rng = np.random.default_rng(config.random_seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wave, passbands, filter_centers, filter_names = read_nominal_passbands(config)
    if not 0 <= config.reference_filter_id < passbands.shape[0]:
        raise ValueError("reference_filter_id must select one of the loaded passbands")
    reference_passband = passbands[config.reference_filter_id]
    sed_library = BOSZEMPCASEDLibrary(config.sed_basis_path).resampled_to(wave)
    phi_shift, phi_width = make_passband_modes(wave, passbands, filter_centers, config)
    loglam_nodes = read_ice_loglam_nodes(config, wave)
    thickness_nodes = make_ice_thickness_nodes(config)
    true_ice_node_values = make_true_ice_spline_values(loglam_nodes, thickness_nodes)

    write_passband_files(
        output_dir, wave, passbands, phi_shift, phi_width, filter_names
    )
    write_ice_spline_files(output_dir, loglam_nodes, thickness_nodes, true_ice_node_values)

    detector_ids = np.arange(1, config.n_det + 1, dtype=int)
    star_detector_index = np.arange(config.n_star, dtype=int) % config.n_det
    rng.shuffle(star_detector_index)
    star_detector_id = detector_ids[star_detector_index]
    star_base_x = rng.uniform(0.0, config.nx, size=config.n_star)
    star_base_y = rng.uniform(0.0, config.ny, size=config.n_star)

    true_mag_norm = rng.uniform(18.0, 22.0, size=config.n_star)
    true_model_index = rng.integers(sed_library.coefficients.shape[0], size=config.n_star)
    true_sed_coeff = sed_library.coefficients[true_model_index].copy()
    n_calibrator = min(config.n_absolute_calibrator, config.n_star)
    calibrator_ids = np.sort(rng.choice(config.n_star, size=n_calibrator, replace=False))
    is_calibrator = np.zeros(config.n_star, dtype=bool)
    is_calibrator[calibrator_ids] = True

    true_shift = rng.normal(
        0.0, config.shift_sigma_um, size=(config.n_filter, config.n_det)
    )
    true_width = rng.normal(0.0, config.width_sigma, size=(config.n_filter, config.n_det))

    star_param_payload = {
        "star_id": np.arange(config.n_star, dtype=int),
        "detector_id": star_detector_id,
        "x": star_base_x,
        "y": star_base_y,
        "mag_norm": true_mag_norm,
        "is_absolute_calibrator": is_calibrator,
        "bosz_model_index": true_model_index,
        "bosz_model_file": sed_library.model_files[true_model_index],
    }
    for component_id in range(sed_library.n_components):
        star_param_payload[f"sed_coeff_{component_id}"] = true_sed_coeff[:, component_id]
    star_params = pd.DataFrame(star_param_payload)
    star_params.to_csv(output_dir / "true_star_params.csv", index=False)
    star_params.loc[is_calibrator].to_csv(output_dir / "stellar_calibrators.csv", index=False)

    pass_rows = []
    for filt in range(config.n_filter):
        for det_index, det_id in enumerate(detector_ids):
            pass_rows.append(
                {
                    "filter_id": filt,
                    "filter_name": filter_names[filt],
                    "detector_id": det_id,
                    "delta_lambda_um": true_shift[filt, det_index],
                    "width": true_width[filt, det_index],
                }
            )
    pd.DataFrame(pass_rows).to_csv(output_dir / "true_passband_params.csv", index=False)

    true_zp = rng.normal(0.0, config.zp_sigma_mag, size=config.n_exp)
    true_zp[0] = 0.0
    true_smooth_coeffs = np.asarray(config.true_smooth_coeffs, dtype=float)
    true_amp_offsets = rng.normal(
        0.0, config.amp_sigma_mag, size=(config.n_det, config.n_amp)
    )
    true_amp_offsets -= true_amp_offsets.mean(axis=1, keepdims=True)

    pd.DataFrame(
        {"exposure_id": np.arange(config.n_exp, dtype=int), "zp_mag": true_zp}
    ).to_csv(output_dir / "true_exposure_zeropoints.csv", index=False)
    pd.DataFrame(
        {
            "basis_name": ["x", "y", "x2", "xy", "y2"],
            "coefficient_mag": true_smooth_coeffs,
        }
    ).to_csv(output_dir / "true_smooth_coeffs.csv", index=False)
    amp_rows = []
    for det_index, det_id in enumerate(detector_ids):
        for amp_id in range(config.n_amp):
            amp_rows.append(
                {
                    "detector_id": det_id,
                    "amp_id": amp_id,
                    "amp_offset_mag": true_amp_offsets[det_index, amp_id],
                }
            )
    pd.DataFrame(amp_rows).to_csv(output_dir / "true_amp_offsets.csv", index=False)

    # One exposure uses one filter. The known ice amount has an epoch component
    # plus a weak detector-position component, resembling an RCS-derived scalar.
    exposure_filter = np.arange(config.n_exp, dtype=int) % config.n_filter
    epoch_id = np.arange(config.n_exp, dtype=int)
    dither_dx = rng.normal(0.0, config.dither_sigma_pix, size=config.n_exp)
    dither_dy = rng.normal(0.0, config.dither_sigma_pix, size=config.n_exp)
    dither_dx[0] = 0.0
    dither_dy[0] = 0.0
    rotation_deg = exposure_rotation_sequence(config)
    rotation_deg[0] = 0.0
    slow_phase = np.linspace(0.0, 2.0 * np.pi, config.n_exp)
    exposure_ice = 0.55 + 0.35 * np.sin(slow_phase) + rng.normal(0.0, 0.07, config.n_exp)
    exposure_ice = np.clip(exposure_ice, 0.02, 1.20)

    sed_all = sed_library.sed_from_coefficients(
        true_sed_coeff, true_mag_norm, reference_passband
    )

    rows = []
    obs_id = 0
    for exp_id in range(config.n_exp):
        filt = exposure_filter[exp_id]
        x_exp, y_exp = apply_exposure_transform(
            star_base_x,
            star_base_y,
            dither_dx[exp_id],
            dither_dy[exp_id],
            rotation_deg[exp_id],
            config,
        )
        in_bounds = (
            (x_exp >= 0.0)
            & (x_exp < config.nx)
            & (y_exp >= 0.0)
            & (y_exp < config.ny)
        )
        keep = (rng.random(config.n_star) < config.detection_fraction) & in_bounds
        star_indices = np.nonzero(keep)[0]

        for star_id in star_indices:
            det_index = star_detector_index[star_id]
            det_id = star_detector_id[star_id]
            x = x_exp[star_id]
            y = y_exp[star_id]
            amp_id = int(amp_id_from_x(x, nx=config.nx, n_amp=config.n_amp))
            xn, yn = normalized_xy(np.asarray([x]), np.asarray([y]), config)
            smooth_delta = float((poly_basis(xn, yn) @ true_smooth_coeffs)[0])
            amp_delta = true_amp_offsets[det_index, amp_id]
            scalar_delta = true_zp[exp_id] + smooth_delta + amp_delta
            position_term = 1.0 + 0.12 * (x / (config.nx - 1.0) - 0.5)
            position_term += 0.08 * (y / (config.ny - 1.0) - 0.5)
            ice_thickness = np.clip(
                exposure_ice[exp_id] * position_term,
                config.ice_thickness_min,
                config.ice_thickness_max,
            )

            t0 = passbands[filt]
            sed = sed_all[star_id]
            logt_pass = (
                true_shift[filt, det_index] * phi_shift[filt]
                + true_width[filt, det_index] * phi_width[filt]
            )
            logt_ice = interpolate_ice_surface(
                true_ice_node_values, loglam_nodes, thickness_nodes, wave, ice_thickness
            )
            logt_true = logt_pass + logt_ice
            t_pass = t0 * np.exp(logt_pass)
            t_true = t0 * np.exp(logt_true)

            flux_nominal = photon_count_integral(sed * t0, wave)
            flux_pass = photon_count_integral(sed * t_pass, wave)
            flux_true = photon_count_integral(sed * t_true, wave)
            inst_mag_nominal = counts_to_instrumental_mag(flux_nominal)
            inst_mag_pass = counts_to_instrumental_mag(flux_pass)
            inst_mag_chromatic = counts_to_instrumental_mag(flux_true)
            ab_mag_nominal = flux_to_abmag(flux_nominal, wave, t0)
            ab_mag_chromatic = flux_to_abmag(flux_true, wave, t_true)
            mag_true = inst_mag_chromatic + scalar_delta
            mag_obs = mag_true + rng.normal(0.0, config.phot_sigma_mag)

            rows.append(
                {
                    "obs_id": obs_id,
                    "star_id": star_id,
                    "exposure_id": exp_id,
                    "epoch_id": epoch_id[exp_id],
                    "filter_id": filt,
                    "filter_name": filter_names[filt],
                    "detector_id": det_id,
                    "amp_id": amp_id,
                    "x": x,
                    "y": y,
                    "ice_thickness": ice_thickness,
                    "ice_amount_obs": ice_thickness,
                    "mag_obs": mag_obs,
                    "mag_unc": config.phot_sigma_mag,
                    "mag_true_no_noise": mag_true,
                    "true_sed_mag_nominal": inst_mag_nominal,
                    "true_passband_delta_mag": inst_mag_pass - inst_mag_nominal,
                    "true_ice_delta_mag": inst_mag_chromatic - inst_mag_pass,
                    "true_ab_mag_nominal": ab_mag_nominal,
                    "true_ab_mag_chromatic": ab_mag_chromatic,
                    "true_zp_delta_mag": true_zp[exp_id],
                    "true_smooth_delta_mag": smooth_delta,
                    "true_amp_delta_mag": amp_delta,
                    "true_scalar_delta_mag": scalar_delta,
                }
            )
            obs_id += 1

    pd.DataFrame(rows).to_csv(output_dir / "measurements.csv", index=False)

    metadata = asdict(config)
    metadata["wave_min"] = float(wave.min())
    metadata["wave_max"] = float(wave.max())
    metadata["n_wave"] = int(wave.size)
    metadata["sed_basis_path"] = str(Path(config.sed_basis_path))
    metadata["sed_basis_n_components"] = sed_library.n_components
    metadata["sed_basis_n_models"] = int(sed_library.coefficients.shape[0])
    metadata["sed_basis_wave_min"] = float(sed_library.source_wave_micron.min())
    metadata["sed_basis_wave_max"] = float(sed_library.source_wave_micron.max())
    metadata["detector_ids"] = detector_ids.tolist()
    metadata["filter_names"] = filter_names
    metadata["filter_centers_um"] = filter_centers.tolist()
    metadata["reference_filter_id"] = int(config.reference_filter_id)
    metadata["reference_filter_name"] = filter_names[config.reference_filter_id]
    metadata["measurement_magnitude_system"] = "instrumental"
    metadata["standard_star_magnitude_system"] = "AB"
    metadata["magnitude_system"] = "instrumental_measurements_ab_standards"
    metadata["synthetic_photometry"] = "photon_counting_f_lambda_T_lambda_lambda_dlambda"
    metadata["ice_loglam_nodes_file"] = config.ice_loglam_nodes_file
    metadata["n_ice_loglam_nodes"] = int(loglam_nodes.size)
    metadata["n_ice_thickness_nodes"] = int(thickness_nodes.size)
    metadata["n_absolute_calibrator"] = int(n_calibrator)
    metadata["n_obs"] = len(rows)
    with open(output_dir / "simulation_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    make_diagnostic_plots(
        output_dir,
        wave,
        passbands,
        filter_names,
        loglam_nodes,
        thickness_nodes,
        true_ice_node_values,
    )

    print(f"Saved {len(rows)} observations to {output_dir / 'measurements.csv'}")
    print(f"Saved simulator products to {output_dir.resolve()}")


def make_diagnostic_plots(
    output_dir: Path,
    wave: np.ndarray,
    passbands: np.ndarray,
    filter_names: list[str],
    loglam_nodes: np.ndarray,
    thickness_nodes: np.ndarray,
    true_ice_node_values: np.ndarray,
) -> None:
    plt.figure(figsize=(8, 4.5))
    for filt in range(passbands.shape[0]):
        plt.plot(wave, passbands[filt], label=filter_names[filt])
    plt.xlabel("Wavelength [um]")
    plt.ylabel("Nominal throughput")
    plt.title("Nominal passbands")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "sim_nominal_passbands.png", dpi=160)
    plt.close()

    log_wave = np.log10(wave)
    thick_grid = np.linspace(thickness_nodes.min(), thickness_nodes.max(), 80)
    surface = np.vstack(
        [
            interpolate_ice_surface(
                true_ice_node_values, loglam_nodes, thickness_nodes, wave, thickness
            )
            for thickness in thick_grid
        ]
    )
    vmax = np.max(np.abs(surface))
    plt.figure(figsize=(8, 4.8))
    im = plt.imshow(
        surface,
        origin="lower",
        aspect="auto",
        extent=[log_wave.min(), log_wave.max(), thick_grid.min(), thick_grid.max()],
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    plt.colorbar(im, label="log-throughput perturbation")
    plt.scatter(
        np.tile(loglam_nodes, thickness_nodes.size),
        np.repeat(thickness_nodes, loglam_nodes.size),
        s=8,
        color="black",
        alpha=0.55,
    )
    plt.xlabel("log10 wavelength [um]")
    plt.ylabel("Ice thickness coordinate")
    plt.title("True ice log-throughput surface")
    plt.tight_layout()
    plt.savefig(output_dir / "sim_true_ice_surface.png", dpi=160)
    plt.close()


def parse_args() -> SimConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=SimConfig.output_dir)
    parser.add_argument("--passband-file", default=SimConfig.passband_file)
    parser.add_argument("--sed-basis-path", default=SimConfig.sed_basis_path)
    parser.add_argument("--reference-filter-id", type=int, default=SimConfig.reference_filter_id)
    parser.add_argument("--n-absolute-calibrator", type=int, default=SimConfig.n_absolute_calibrator)
    parser.add_argument("--ice-loglam-nodes-file", default=SimConfig.ice_loglam_nodes_file)
    parser.add_argument("--n-ice-thickness-nodes", type=int, default=SimConfig.n_ice_thickness_nodes)
    parser.add_argument("--ice-thickness-min", type=float, default=SimConfig.ice_thickness_min)
    parser.add_argument("--ice-thickness-max", type=float, default=SimConfig.ice_thickness_max)
    args = parser.parse_args()
    return SimConfig(
        output_dir=args.output_dir,
        passband_file=args.passband_file,
        sed_basis_path=args.sed_basis_path,
        reference_filter_id=args.reference_filter_id,
        n_absolute_calibrator=args.n_absolute_calibrator,
        ice_loglam_nodes_file=args.ice_loglam_nodes_file,
        n_ice_thickness_nodes=args.n_ice_thickness_nodes,
        ice_thickness_min=args.ice_thickness_min,
        ice_thickness_max=args.ice_thickness_max,
    )


def main() -> None:
    simulate_data(parse_args())


if __name__ == "__main__":
    main()
