#!/usr/bin/env python3
"""Generate simulated Roman-like WFI photometry for sparse ubercalibration.

The output is intentionally simple and portable:

* simulated_observations.csv: one row per stellar observation
* truth.npz: simulation truth used only for diagnostics
* metadata.json: geometry and simulation settings needed by the fitter

The exposure pattern includes translations and multiple rotation angles. The
rotations break the otherwise exact degeneracy between detector-plane linear
smooth terms and a linear trend in fitted star magnitudes plus exposure ZPs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class Config:
    n_det: int = 18
    n_star: int = 5000
    n_exp: int = 40
    nx: int = 4096
    ny: int = 4096
    n_amp: int = 32
    dither_sigma_pix: float = 500.0
    phot_sigma_mag: float = 0.005
    zp_sigma_mag: float = 0.01
    amp_sigma_mag: float = 0.003
    random_seed: int = 12345
    output_dir: str = "outputs_amp_prototype"
    observation_table: str = "simulated_observations.csv"
    truth_file: str = "truth.npz"
    metadata_file: str = "metadata.json"
    rotation_angles_deg: Tuple[float, ...] = (0.0, -5.0, 5.0)
    true_smooth_coeffs: Tuple[float, float, float, float, float] = (
        0.0040,
        -0.0030,
        0.0060,
        -0.0020,
        -0.0040,
    )


@dataclass
class SimulatedData:
    star_id: np.ndarray
    exposure_id: np.ndarray
    detector_id: np.ndarray
    amp_id: np.ndarray
    x: np.ndarray
    y: np.ndarray
    m_obs: np.ndarray
    sigma: np.ndarray
    true_star_mag: np.ndarray
    true_zp: np.ndarray
    true_smooth_coeffs: np.ndarray
    true_amp_offsets: np.ndarray
    star_base_x: np.ndarray
    star_base_y: np.ndarray
    star_detector_id: np.ndarray
    dither_dx: np.ndarray
    dither_dy: np.ndarray
    rotation_deg: np.ndarray


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
    """Smooth star-flat polynomial terms: x, y, x^2, x*y, y^2."""
    xn = np.asarray(xn)
    yn = np.asarray(yn)
    return np.column_stack((xn, yn, xn**2, xn * yn, yn**2))


def exposure_rotation_sequence(config):
    """Create a deterministic sequence using at least two rotation angles."""
    angles = np.asarray(config.rotation_angles_deg, dtype=float)
    if angles.size < 2:
        raise ValueError("rotation_angles_deg must contain at least two angles")
    return angles[np.arange(config.n_exp) % angles.size]


def apply_exposure_transform(x0, y0, dx, dy, rotation_deg, config):
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


def simulate_data(config):
    """Simulate dithered and rotated stellar photometry."""
    rng = np.random.default_rng(config.random_seed)

    true_star_mag = rng.uniform(18.0, 23.0, size=config.n_star)
    star_base_x = rng.uniform(0.0, config.nx, size=config.n_star)
    star_base_y = rng.uniform(0.0, config.ny, size=config.n_star)
    detector_ids = np.arange(1, config.n_det + 1, dtype=int)
    star_detector_index = np.arange(config.n_star, dtype=int) % config.n_det
    rng.shuffle(star_detector_index)
    star_detector_id = detector_ids[star_detector_index]

    dither_dx = rng.normal(0.0, config.dither_sigma_pix, size=config.n_exp)
    dither_dy = rng.normal(0.0, config.dither_sigma_pix, size=config.n_exp)
    dither_dx[0] = 0.0
    dither_dy[0] = 0.0
    rotation_deg = exposure_rotation_sequence(config)
    rotation_deg[0] = 0.0

    true_zp = rng.normal(0.0, config.zp_sigma_mag, size=config.n_exp)
    true_zp[0] = 0.0
    true_smooth_coeffs = np.asarray(config.true_smooth_coeffs, dtype=float)

    true_amp_offsets = rng.normal(
        0.0, config.amp_sigma_mag, size=(config.n_det, config.n_amp)
    )
    true_amp_offsets -= true_amp_offsets.mean(axis=1, keepdims=True)

    obs_star_id = []
    obs_exposure_id = []
    obs_detector_id = []
    obs_amp_id = []
    obs_x = []
    obs_y = []
    obs_m = []

    for exp_id in range(config.n_exp):
        x, y = apply_exposure_transform(
            star_base_x,
            star_base_y,
            dither_dx[exp_id],
            dither_dy[exp_id],
            rotation_deg[exp_id],
            config,
        )
        in_bounds = (x >= 0.0) & (x < config.nx) & (y >= 0.0) & (y < config.ny)
        if not np.any(in_bounds):
            continue

        sid = np.nonzero(in_bounds)[0]
        x_obs = x[in_bounds]
        y_obs = y[in_bounds]
        det_index_obs = star_detector_index[in_bounds]
        det_obs = star_detector_id[in_bounds]
        amp_obs = amp_id_from_x(x_obs, nx=config.nx, n_amp=config.n_amp)
        xn, yn = normalized_xy(x_obs, y_obs, config)
        smooth = poly_basis(xn, yn) @ true_smooth_coeffs
        amp = true_amp_offsets[det_index_obs, amp_obs]
        noise = rng.normal(0.0, config.phot_sigma_mag, size=sid.size)
        m_obs = true_star_mag[sid] + true_zp[exp_id] + smooth + amp + noise

        obs_star_id.append(sid)
        obs_exposure_id.append(np.full(sid.size, exp_id, dtype=int))
        obs_detector_id.append(det_obs)
        obs_amp_id.append(amp_obs)
        obs_x.append(x_obs)
        obs_y.append(y_obs)
        obs_m.append(m_obs)

    star_id = np.concatenate(obs_star_id)
    exposure_id = np.concatenate(obs_exposure_id)
    detector_id = np.concatenate(obs_detector_id)
    amp_id = np.concatenate(obs_amp_id)
    x = np.concatenate(obs_x)
    y = np.concatenate(obs_y)
    m_obs = np.concatenate(obs_m)
    sigma = np.full(m_obs.size, config.phot_sigma_mag)

    return SimulatedData(
        star_id=star_id,
        exposure_id=exposure_id,
        detector_id=detector_id,
        amp_id=amp_id,
        x=x,
        y=y,
        m_obs=m_obs,
        sigma=sigma,
        true_star_mag=true_star_mag,
        true_zp=true_zp,
        true_smooth_coeffs=true_smooth_coeffs,
        true_amp_offsets=true_amp_offsets,
        star_base_x=star_base_x,
        star_base_y=star_base_y,
        star_detector_id=star_detector_id,
        dither_dx=dither_dx,
        dither_dy=dither_dy,
        rotation_deg=rotation_deg,
    )


def save_simulated_data(data, config):
    """Save the observation table, truth arrays, and metadata."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_id = np.arange(data.m_obs.size, dtype=int)
    table = np.column_stack(
        (
            obs_id,
            data.star_id,
            data.exposure_id,
            data.detector_id,
            data.amp_id,
            data.x,
            data.y,
            data.m_obs,
            data.sigma,
        )
    )
    header = ",".join(
        (
            "obs_id",
            "star_id",
            "exposure_id",
            "detector_id",
            "amp_id",
            "x_pixel",
            "y_pixel",
            "instrumental_mag",
            "instrumental_mag_uncertainty",
        )
    )
    np.savetxt(
        output_dir / config.observation_table,
        table,
        delimiter=",",
        header=header,
        comments="",
        fmt=["%d", "%d", "%d", "%d", "%d", "%.8f", "%.8f", "%.8f", "%.8f"],
    )

    np.savez(
        output_dir / config.truth_file,
        true_star_mag=data.true_star_mag,
        true_zp=data.true_zp,
        true_smooth_coeffs=data.true_smooth_coeffs,
        true_amp_offsets=data.true_amp_offsets,
        star_base_x=data.star_base_x,
        star_base_y=data.star_base_y,
        star_detector_id=data.star_detector_id,
        dither_dx=data.dither_dx,
        dither_dy=data.dither_dy,
        rotation_deg=data.rotation_deg,
    )

    metadata = asdict(config)
    metadata["n_obs"] = int(data.m_obs.size)
    metadata["detector_ids"] = list(range(1, config.n_det + 1))
    metadata["rotation_angles_deg"] = list(config.rotation_angles_deg)
    metadata["true_smooth_coeffs"] = list(config.true_smooth_coeffs)
    with open(output_dir / config.metadata_file, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def main():
    config = Config()
    data = simulate_data(config)
    save_simulated_data(data, config)
    output_dir = Path(config.output_dir).resolve()
    print(f"Saved {data.m_obs.size} observations to {output_dir / config.observation_table}")
    print(f"Saved truth arrays to {output_dir / config.truth_file}")
    print(f"Saved metadata to {output_dir / config.metadata_file}")


if __name__ == "__main__":
    main()
