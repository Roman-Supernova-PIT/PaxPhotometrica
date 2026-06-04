#!/usr/bin/env python3
"""Fit a sparse Roman-like ubercalibration / star-flat toy model with amp terms.

This script reads a simulated observation table produced by
roman_ubercal_amp_generate_data.py and solves the relative calibration model:

    m_obs = M_star + ZP_exposure + S_smooth(x, y) + A_detector_amp + noise

The fit is deliberately relative. Exposure 0 is the zeropoint reference, the
constant smooth term is omitted, and each detector's mean amplifier correction is
weakly constrained to zero. The simulator includes multiple rotation angles so
the fitted x/y smooth gradients are constrained by data rather than by explicit
smooth-gradient priors.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
import os
from pathlib import Path
import tempfile

_MPL_CACHE = Path(tempfile.gettempdir()) / "roman_ubercal_amp_mpl_cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_CACHE.resolve()))

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr


@dataclass
class Config:
    n_det: int = 1
    n_exp: int = 40
    nx: int = 4096
    ny: int = 4096
    n_amp: int = 32
    sigma_amp_prior: float = 0.02
    sigma_amp_sum_constraint: float = 1e-4
    random_seed: int = 12345
    output_dir: str = "outputs_amp_prototype"
    observation_table: str = "simulated_observations.csv"
    truth_file: str = "truth.npz"
    metadata_file: str = "metadata.json"


@dataclass
class ObservationData:
    star_id: np.ndarray
    star_param_id: np.ndarray
    unique_star_ids: np.ndarray
    exposure_id: np.ndarray
    detector_id: np.ndarray
    detector_param_id: np.ndarray
    unique_detector_ids: np.ndarray
    amp_id: np.ndarray
    x: np.ndarray
    y: np.ndarray
    m_obs: np.ndarray
    sigma: np.ndarray
    true_star_mag: object
    true_zp: object
    true_smooth_coeffs: object
    true_amp_offsets: object
    n_obs: int
    n_star_params: int
    n_zp_params: int
    n_smooth_params: int
    n_amp_params: int
    n_params: int
    idx_star: int
    idx_zp: int
    idx_smooth: int
    idx_amp: int


@dataclass
class Solution:
    theta: np.ndarray
    star_mag: np.ndarray
    zp: np.ndarray
    smooth_coeffs: np.ndarray
    amp_offsets: np.ndarray
    model_m_obs: np.ndarray
    residual: np.ndarray
    lsqr_info: tuple


def amp_id_from_x(x, nx=4096, n_amp=32):
    """Return amplifier stripe id for pixel coordinate x."""
    amp_width = nx // n_amp
    amp_id = np.floor(np.asarray(x) / amp_width).astype(int)
    return np.clip(amp_id, 0, n_amp - 1)


def normalized_xy(x, y, config):
    """Map detector pixels to [-1, 1] normalized coordinates."""
    xn = 2.0 * (x / (config.nx - 1.0)) - 1.0
    yn = 2.0 * (y / (config.ny - 1.0)) - 1.0
    return xn, yn


def poly_basis(xn, yn):
    """Smooth star-flat polynomial terms: x, y, x^2, x*y, y^2.

    The constant term is intentionally omitted because it is degenerate with the
    relative magnitude, zeropoint, and amplifier-offset scales.
    """
    xn = np.asarray(xn)
    yn = np.asarray(yn)
    return np.column_stack((xn, yn, xn**2, xn * yn, yn**2))


def load_config():
    """Load fitter configuration, using simulator metadata when available."""
    config = Config()
    metadata_path = Path(config.output_dir) / config.metadata_file
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {metadata_path}. Run roman_ubercal_amp_generate_data.py first."
        )

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    allowed = {field.name for field in fields(Config)}
    for key, value in metadata.items():
        if key in allowed:
            setattr(config, key, value)
    return config


def _load_truth(config):
    truth_path = Path(config.output_dir) / config.truth_file
    if not truth_path.exists():
        return None
    return np.load(truth_path)


def load_simulated_data(config):
    """Read the simulated observation table and optional truth arrays."""
    table_path = Path(config.output_dir) / config.observation_table
    if not table_path.exists():
        raise FileNotFoundError(
            f"Missing {table_path}. Run roman_ubercal_amp_generate_data.py first."
        )

    table = np.genfromtxt(table_path, delimiter=",", names=True)
    table = np.atleast_1d(table)

    star_id = table["star_id"].astype(int)
    unique_star_ids, star_param_id = np.unique(star_id, return_inverse=True)
    exposure_id = table["exposure_id"].astype(int)
    detector_id = table["detector_id"].astype(int)
    unique_detector_ids, detector_param_id = np.unique(detector_id, return_inverse=True)
    amp_id = table["amp_id"].astype(int)
    x = table["x_pixel"].astype(float)
    y = table["y_pixel"].astype(float)
    m_obs = table["instrumental_mag"].astype(float)
    sigma = table["instrumental_mag_uncertainty"].astype(float)

    config.n_exp = max(config.n_exp, int(exposure_id.max()) + 1)
    config.n_det = unique_detector_ids.size

    truth = _load_truth(config)
    true_star_mag = None
    true_zp = None
    true_smooth_coeffs = None
    true_amp_offsets = None
    if truth is not None:
        true_star_mag = truth["true_star_mag"]
        true_zp = truth["true_zp"]
        true_smooth_coeffs = truth["true_smooth_coeffs"]
        true_amp_offsets = truth["true_amp_offsets"]

    n_star_params = unique_star_ids.size
    n_zp_params = config.n_exp - 1
    n_smooth_params = 5
    n_amp_params = config.n_det * config.n_amp
    idx_star = 0
    idx_zp = idx_star + n_star_params
    idx_smooth = idx_zp + n_zp_params
    idx_amp = idx_smooth + n_smooth_params
    n_params = idx_amp + n_amp_params

    return ObservationData(
        star_id=star_id,
        star_param_id=star_param_id,
        unique_star_ids=unique_star_ids,
        exposure_id=exposure_id,
        detector_id=detector_id,
        detector_param_id=detector_param_id,
        unique_detector_ids=unique_detector_ids,
        amp_id=amp_id,
        x=x,
        y=y,
        m_obs=m_obs,
        sigma=sigma,
        true_star_mag=true_star_mag,
        true_zp=true_zp,
        true_smooth_coeffs=true_smooth_coeffs,
        true_amp_offsets=true_amp_offsets,
        n_obs=m_obs.size,
        n_star_params=n_star_params,
        n_zp_params=n_zp_params,
        n_smooth_params=n_smooth_params,
        n_amp_params=n_amp_params,
        n_params=n_params,
        idx_star=idx_star,
        idx_zp=idx_zp,
        idx_smooth=idx_smooth,
        idx_amp=idx_amp,
    )


def build_design_matrix(data, config):
    """Build weighted sparse rows for observations, amp priors, and constraints."""
    rows = []
    cols = []
    vals = []
    rhs = []
    row = 0

    xn, yn = normalized_xy(data.x, data.y, config)
    basis = poly_basis(xn, yn)

    for k in range(data.n_obs):
        weight = 1.0 / data.sigma[k]

        rows.append(row)
        cols.append(data.idx_star + data.star_param_id[k])
        vals.append(weight)

        exp_id = data.exposure_id[k]
        if exp_id != 0:
            rows.append(row)
            cols.append(data.idx_zp + exp_id - 1)
            vals.append(weight)

        for p in range(data.n_smooth_params):
            rows.append(row)
            cols.append(data.idx_smooth + p)
            vals.append(basis[k, p] * weight)

        amp_col = data.idx_amp + data.detector_param_id[k] * config.n_amp + data.amp_id[k]
        rows.append(row)
        cols.append(amp_col)
        vals.append(weight)

        rhs.append(data.m_obs[k] * weight)
        row += 1

    # Weak Gaussian priors stabilize sparsely sampled amp columns without
    # dominating well-sampled amplifier solutions.
    for det_id in range(config.n_det):
        for amp_id in range(config.n_amp):
            rows.append(row)
            cols.append(data.idx_amp + det_id * config.n_amp + amp_id)
            vals.append(1.0 / config.sigma_amp_prior)
            rhs.append(0.0)
            row += 1

    # Mean amp offset per detector is constrained to zero. This removes the
    # remaining amp/star magnitude degeneracy while preserving relative amp
    # structure within each detector.
    for det_id in range(config.n_det):
        for amp_id in range(config.n_amp):
            rows.append(row)
            cols.append(data.idx_amp + det_id * config.n_amp + amp_id)
            vals.append((1.0 / config.n_amp) / config.sigma_amp_sum_constraint)
        rhs.append(0.0)
        row += 1

    A = coo_matrix((vals, (rows, cols)), shape=(row, data.n_params)).tocsr()
    b = np.asarray(rhs)
    return A, b


def solve_system(A, b):
    """Solve weighted sparse least squares with LSQR."""
    result = lsqr(A, b, atol=1e-10, btol=1e-10, iter_lim=5000, show=False)
    theta = result[0]
    return theta, result


def unpack_solution(theta, data, config):
    """Reconstruct fitted physical parameters from the concatenated vector."""
    star_mag = theta[data.idx_star : data.idx_zp]

    zp = np.zeros(config.n_exp)
    zp[1:] = theta[data.idx_zp : data.idx_smooth]

    smooth_coeffs = theta[data.idx_smooth : data.idx_amp]
    amp_offsets = theta[data.idx_amp :].reshape(config.n_det, config.n_amp)

    smooth = evaluate_smooth(smooth_coeffs, data.x, data.y, config)
    model_m_obs = (
        star_mag[data.star_param_id]
        + zp[data.exposure_id]
        + smooth
        + amp_offsets[data.detector_param_id, data.amp_id]
    )
    residual = data.m_obs - model_m_obs

    return Solution(
        theta=theta,
        star_mag=star_mag,
        zp=zp,
        smooth_coeffs=smooth_coeffs,
        amp_offsets=amp_offsets,
        model_m_obs=model_m_obs,
        residual=residual,
        lsqr_info=(),
    )


def evaluate_smooth(coeffs, x, y, config):
    """Evaluate the smooth detector response polynomial at pixel coordinates."""
    xn, yn = normalized_xy(np.asarray(x), np.asarray(y), config)
    flat_xn = np.ravel(xn)
    flat_yn = np.ravel(yn)
    values = poly_basis(flat_xn, flat_yn) @ coeffs
    return values.reshape(np.shape(xn))


def _aligned_zp_error(solution, data):
    """Compare ZPs after applying the exposure-0 reference convention."""
    true_rel = data.true_zp - data.true_zp[0]
    recovered_rel = solution.zp - solution.zp[0]
    return recovered_rel - true_rel


def _grid_smooth_fields(solution, data, config, n_grid=80):
    x_grid = np.linspace(0.0, config.nx - 1.0, n_grid)
    y_grid = np.linspace(0.0, config.ny - 1.0, n_grid)
    xx, yy = np.meshgrid(x_grid, y_grid)
    true_field = evaluate_smooth(data.true_smooth_coeffs, xx, yy, config)
    recovered_field = evaluate_smooth(solution.smooth_coeffs, xx, yy, config)
    residual_field = recovered_field - true_field
    residual_field -= np.mean(residual_field)
    return xx, yy, true_field, recovered_field, residual_field


def make_diagnostics(data, solution, config):
    """Print scalar diagnostics for the sparse calibration fit."""
    rms_resid = np.sqrt(np.mean(solution.residual**2))
    med_abs_resid = np.median(np.abs(solution.residual))
    amp_counts = np.bincount(
        data.detector_param_id * config.n_amp + data.amp_id,
        minlength=config.n_det * config.n_amp,
    ).reshape(config.n_det, config.n_amp)

    print("Sparse Roman-like ubercalibration / amp prototype diagnostics")
    print("-------------------------------------------------------------")
    print(f"Number of observations: {data.n_obs}")
    print(f"Number of unique fitted stars: {data.n_star_params}")
    print(f"Number of fitted parameters: {data.n_params}")
    print(f"LSQR iterations: {solution.lsqr_info[2]}")
    print(f"LSQR stop code: {solution.lsqr_info[1]}")
    print(f"RMS residual: {rms_resid:.6f} mag")
    print(f"Median absolute residual: {med_abs_resid:.6f} mag")

    if data.true_zp is not None:
        zp_rms = np.sqrt(np.mean(_aligned_zp_error(solution, data) ** 2))
        print(f"Exposure ZP RMS error: {zp_rms:.6f} mag")

    if data.true_smooth_coeffs is not None:
        _, _, true_field, recovered_field, smooth_resid = _grid_smooth_fields(
            solution, data, config
        )
        smooth_rms = np.sqrt(np.mean(smooth_resid**2))
        print(f"Smooth field RMS error, mean removed: {smooth_rms:.6f} mag")
        _ = true_field, recovered_field

    if data.true_amp_offsets is not None:
        true_amp = data.true_amp_offsets - data.true_amp_offsets.mean(axis=1, keepdims=True)
        recovered_amp = solution.amp_offsets - solution.amp_offsets.mean(axis=1, keepdims=True)
        amp_rms = np.sqrt(np.mean((recovered_amp - true_amp) ** 2))
        print(f"Amplifier offset RMS error, per-detector means removed: {amp_rms:.6f} mag")

    print("Number of observations per amplifier:")
    for det_id in range(config.n_det):
        counts = " ".join(f"{count:6d}" for count in amp_counts[det_id])
        print(f"  det {data.unique_detector_ids[det_id]:02d}: {counts}")


def _detectors_to_plot(config):
    return range(config.n_det)


def _detector_label(data, det_index):
    return f"det {data.unique_detector_ids[det_index]}"


def make_plots(data, solution, config):
    """Create diagnostic PNG plots using matplotlib only."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    amps = np.arange(config.n_amp)
    dets = _detectors_to_plot(config)

    if data.true_amp_offsets is not None:
        plt.figure(figsize=(10, 5))
        for det_id in dets:
            plt.plot(
                amps,
                data.true_amp_offsets[det_id],
                marker="o",
                label=_detector_label(data, det_id),
            )
        plt.axhline(0.0, color="0.3", linewidth=1)
        plt.xlabel("Amplifier ID")
        plt.ylabel("True amp offset [mag]")
        plt.title("True amplifier offsets")
        if config.n_det > 1:
            plt.legend(ncol=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / "true_amp_offsets.png", dpi=160)
        plt.close()

    plt.figure(figsize=(10, 5))
    for det_id in dets:
        plt.plot(
            amps,
            solution.amp_offsets[det_id],
            marker="o",
            label=_detector_label(data, det_id),
        )
    plt.axhline(0.0, color="0.3", linewidth=1)
    plt.xlabel("Amplifier ID")
    plt.ylabel("Recovered amp offset [mag]")
    plt.title("Recovered amplifier offsets")
    if config.n_det > 1:
        plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "recovered_amp_offsets.png", dpi=160)
    plt.close()

    if data.true_amp_offsets is not None:
        true_amp = data.true_amp_offsets - data.true_amp_offsets.mean(axis=1, keepdims=True)
        recovered_amp = solution.amp_offsets - solution.amp_offsets.mean(axis=1, keepdims=True)
        lim = 1.1 * np.max(np.abs(np.r_[true_amp.ravel(), recovered_amp.ravel()]))
        plt.figure(figsize=(6, 6))
        plt.scatter(true_amp.ravel(), recovered_amp.ravel(), s=28, alpha=0.8)
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

    image_kwargs = {
        "origin": "lower",
        "extent": [0, config.nx - 1, 0, config.ny - 1],
        "aspect": "equal",
    }
    if data.true_smooth_coeffs is not None:
        _, _, true_field, recovered_field, residual_field = _grid_smooth_fields(
            solution, data, config, n_grid=120
        )

        plt.figure(figsize=(6.5, 5.5))
        im = plt.imshow(true_field, **image_kwargs)
        plt.colorbar(im, label="mag")
        plt.xlabel("x pixel")
        plt.ylabel("y pixel")
        plt.title("True smooth field, detector 0")
        plt.tight_layout()
        plt.savefig(output_dir / "smooth_field_true.png", dpi=160)
        plt.close()

        plt.figure(figsize=(6.5, 5.5))
        im = plt.imshow(recovered_field, **image_kwargs)
        plt.colorbar(im, label="mag")
        plt.xlabel("x pixel")
        plt.ylabel("y pixel")
        plt.title("Recovered smooth field, detector 0")
        plt.tight_layout()
        plt.savefig(output_dir / "smooth_field_recovered.png", dpi=160)
        plt.close()

        plt.figure(figsize=(6.5, 5.5))
        vmax = np.max(np.abs(residual_field))
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

    plt.figure(figsize=(9, 5))
    rng = np.random.default_rng(config.random_seed + 1)
    max_points = 40000
    if data.n_obs > max_points:
        sample = rng.choice(data.n_obs, size=max_points, replace=False)
    else:
        sample = np.arange(data.n_obs)
    plt.scatter(data.x[sample], solution.residual[sample], s=3, alpha=0.25)
    plt.axhline(0.0, color="0.2", linewidth=1)
    plt.xlabel("x pixel")
    plt.ylabel("Photometric residual [mag]")
    plt.title("Residual versus x")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_vs_x.png", dpi=160)
    plt.close()

    amp_median = np.full((config.n_det, config.n_amp), np.nan)
    for det_id in range(config.n_det):
        for amp_id in range(config.n_amp):
            mask = (data.detector_param_id == det_id) & (data.amp_id == amp_id)
            if np.any(mask):
                amp_median[det_id, amp_id] = np.median(solution.residual[mask])

    plt.figure(figsize=(10, 5))
    for det_id in dets:
        plt.plot(amps, amp_median[det_id], marker="o", label=_detector_label(data, det_id))
    plt.axhline(0.0, color="0.2", linewidth=1)
    plt.xlabel("Amplifier ID")
    plt.ylabel("Median residual [mag]")
    plt.title("Median residual per amplifier")
    if config.n_det > 1:
        plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "residual_vs_amp.png", dpi=160)
    plt.close()


def main():
    config = load_config()
    data = load_simulated_data(config)
    A, b = build_design_matrix(data, config)
    theta, lsqr_info = solve_system(A, b)
    solution = unpack_solution(theta, data, config)
    solution.lsqr_info = lsqr_info
    make_diagnostics(data, solution, config)
    make_plots(data, solution, config)
    print(f"Saved plots to: {Path(config.output_dir).resolve()}")


if __name__ == "__main__":
    main()
