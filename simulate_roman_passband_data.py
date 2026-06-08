#!/usr/bin/env python3
"""Simulate Roman-like WFI passband and ice calibration photometry.

This simulator assumes scalar instrumental calibration has already been applied.
It focuses on chromatic calibration: small passband shifts, passband width
changes, and wavelength-dependent ice throughput loss.

Throughput perturbations are modeled in log-throughput / optical-depth space,
because small multiplicative throughput changes then add linearly. Observations
are still generated from linear broadband flux integrals, which is the physical
measurement made by the detector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile

_MPL_CACHE = Path(tempfile.gettempdir()) / "roman_passband_mpl_cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_CACHE.resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class SimConfig:
    random_seed: int = 12345
    n_star: int = 2000
    n_exp: int = 30
    n_det: int = 1
    nx: int = 4096
    ny: int = 4096
    n_filter: int = 6
    wave_min: float = 0.45
    wave_max: float = 2.30
    n_wave: int = 2000
    passband_file: str = "passbands.txt"
    phot_sigma_mag: float = 0.005
    output_dir: str = "passband_sim_outputs"
    detection_fraction: float = 0.92
    shift_sigma_um: float = 0.001
    width_sigma: float = 0.01
    mode_smooth_sigma_pix: float = 4.0
    max_abs_phi_shift: float = 250.0
    max_abs_phi_width: float = 80.0


def trapz_integral(y: np.ndarray, wave: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compatibility wrapper for NumPy's trapezoid integration."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, wave, axis=axis)
    return np.trapz(y, wave, axis=axis)


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


def make_ice_basis(wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Three broad Gaussian optical-depth basis functions."""
    centers = np.array([1.0, 1.5, 2.0])
    widths = np.array([0.16, 0.22, 0.26])
    basis = np.exp(-0.5 * ((wave[None, :] - centers[:, None]) / widths[:, None]) ** 2)
    return basis, centers


def extinction_curve(wave: np.ndarray) -> np.ndarray:
    """Simple decreasing extinction law in arbitrary optical-depth units."""
    return (wave / 1.0) ** (-1.2)


def planck_like_lambda(wave_um: np.ndarray, temperature_k: np.ndarray) -> np.ndarray:
    """A stable Planck-like SED family in arbitrary units.

    The result is normalized near 1 micron so the stellar normalization parameter
    controls the overall magnitude scale.
    """
    wave = np.asarray(wave_um)
    temp = np.asarray(temperature_k)
    c2_um_k = 14387.76877
    x = c2_um_k / (temp[..., None] * wave[None, :])
    x = np.clip(x, 1e-3, 700.0)
    b_lambda = 1.0 / (wave[None, :] ** 5 * np.expm1(x))

    ref_wave = 1.0
    x_ref = np.clip(c2_um_k / (temp * ref_wave), 1e-3, 700.0)
    b_ref = 1.0 / (ref_wave**5 * np.expm1(x_ref))
    return b_lambda / b_ref[..., None]


def stellar_sed(
    wave: np.ndarray, mag_norm: np.ndarray, temperature: np.ndarray, extinction: np.ndarray
) -> np.ndarray:
    """Return flux-density SEDs for one or many stars."""
    mag_norm = np.asarray(mag_norm)
    temperature = np.asarray(temperature)
    extinction = np.asarray(extinction)
    scale = 10.0 ** (-0.4 * mag_norm)
    shape = planck_like_lambda(wave, temperature)
    extinct = np.exp(-extinction[..., None] * extinction_curve(wave)[None, :])
    return scale[..., None] * shape * extinct


def flux_to_mag(flux: np.ndarray) -> np.ndarray:
    return -2.5 * np.log10(np.maximum(flux, 1e-300))


def write_passband_files(
    output_dir: Path,
    wave: np.ndarray,
    passbands: np.ndarray,
    phi_shift: np.ndarray,
    phi_width: np.ndarray,
    ice_basis: np.ndarray,
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

    ice_rows = []
    for basis_id in range(ice_basis.shape[0]):
        ice_rows.append(
            pd.DataFrame(
                {
                    "wavelength_um": wave,
                    "basis_id": basis_id,
                    "psi": ice_basis[basis_id],
                }
            )
        )
    pd.concat(ice_rows, ignore_index=True).to_csv(output_dir / "ice_basis.csv", index=False)


def simulate_data(config: SimConfig) -> None:
    rng = np.random.default_rng(config.random_seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wave, passbands, filter_centers, filter_names = read_nominal_passbands(config)
    phi_shift, phi_width = make_passband_modes(wave, passbands, filter_centers, config)
    ice_basis, _ = make_ice_basis(wave)
    true_ice_coeff = np.array([0.010, 0.018, 0.026])
    tau_ice_true = true_ice_coeff @ ice_basis

    write_passband_files(
        output_dir, wave, passbands, phi_shift, phi_width, ice_basis, filter_names
    )

    detector_ids = np.arange(1, config.n_det + 1, dtype=int)
    star_detector_index = np.arange(config.n_star, dtype=int) % config.n_det
    rng.shuffle(star_detector_index)
    star_detector_id = detector_ids[star_detector_index]
    star_x = rng.uniform(0.0, config.nx, size=config.n_star)
    star_y = rng.uniform(0.0, config.ny, size=config.n_star)

    true_mag_norm = rng.uniform(18.0, 22.0, size=config.n_star)
    true_temperature = rng.uniform(3600.0, 8500.0, size=config.n_star)
    true_extinction = rng.uniform(0.0, 0.35, size=config.n_star)

    true_shift = rng.normal(
        0.0, config.shift_sigma_um, size=(config.n_filter, config.n_det)
    )
    true_width = rng.normal(0.0, config.width_sigma, size=(config.n_filter, config.n_det))

    star_params = pd.DataFrame(
        {
            "star_id": np.arange(config.n_star, dtype=int),
            "detector_id": star_detector_id,
            "x": star_x,
            "y": star_y,
            "mag_norm": true_mag_norm,
            "temperature_k": true_temperature,
            "extinction": true_extinction,
        }
    )
    star_params.to_csv(output_dir / "true_star_params.csv", index=False)

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

    pd.DataFrame(
        {"basis_id": np.arange(true_ice_coeff.size), "ice_coeff": true_ice_coeff}
    ).to_csv(output_dir / "true_ice_params.csv", index=False)

    # One exposure uses one filter. The known ice amount has an epoch component
    # plus a weak detector-position component, resembling an RCS-derived scalar.
    exposure_filter = np.arange(config.n_exp, dtype=int) % config.n_filter
    epoch_id = np.arange(config.n_exp, dtype=int)
    slow_phase = np.linspace(0.0, 2.0 * np.pi, config.n_exp)
    exposure_ice = 0.55 + 0.35 * np.sin(slow_phase) + rng.normal(0.0, 0.07, config.n_exp)
    exposure_ice = np.clip(exposure_ice, 0.02, 1.20)

    sed_all = stellar_sed(wave, true_mag_norm, true_temperature, true_extinction)

    rows = []
    obs_id = 0
    for exp_id in range(config.n_exp):
        filt = exposure_filter[exp_id]
        keep = rng.random(config.n_star) < config.detection_fraction
        star_indices = np.nonzero(keep)[0]

        for star_id in star_indices:
            det_index = star_detector_index[star_id]
            det_id = star_detector_id[star_id]
            x = star_x[star_id]
            y = star_y[star_id]
            position_term = 1.0 + 0.12 * (x / (config.nx - 1.0) - 0.5)
            position_term += 0.08 * (y / (config.ny - 1.0) - 0.5)
            ice_amount = max(0.0, exposure_ice[exp_id] * position_term)

            t0 = passbands[filt]
            sed = sed_all[star_id]
            logt_pass = (
                true_shift[filt, det_index] * phi_shift[filt]
                + true_width[filt, det_index] * phi_width[filt]
            )
            logt_true = logt_pass - ice_amount * tau_ice_true
            t_pass = t0 * np.exp(logt_pass)
            t_true = t0 * np.exp(logt_true)

            flux_nominal = trapz_integral(sed * t0, wave)
            flux_pass = trapz_integral(sed * t_pass, wave)
            flux_true = trapz_integral(sed * t_true, wave)
            mag_nominal = flux_to_mag(flux_nominal)
            mag_pass = flux_to_mag(flux_pass)
            mag_true = flux_to_mag(flux_true)
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
                    "x": x,
                    "y": y,
                    "ice_amount_obs": ice_amount,
                    "mag_obs": mag_obs,
                    "mag_unc": config.phot_sigma_mag,
                    "mag_true_no_noise": mag_true,
                    "true_sed_mag_nominal": mag_nominal,
                    "true_passband_delta_mag": mag_pass - mag_nominal,
                    "true_ice_delta_mag": mag_true - mag_pass,
                }
            )
            obs_id += 1

    pd.DataFrame(rows).to_csv(output_dir / "measurements.csv", index=False)

    metadata = asdict(config)
    metadata["wave_min"] = float(wave.min())
    metadata["wave_max"] = float(wave.max())
    metadata["n_wave"] = int(wave.size)
    metadata["detector_ids"] = detector_ids.tolist()
    metadata["filter_names"] = filter_names
    metadata["filter_centers_um"] = filter_centers.tolist()
    metadata["n_obs"] = len(rows)
    with open(output_dir / "simulation_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    make_diagnostic_plots(output_dir, wave, passbands, tau_ice_true, filter_names)

    print(f"Saved {len(rows)} observations to {output_dir / 'measurements.csv'}")
    print(f"Saved simulator products to {output_dir.resolve()}")


def make_diagnostic_plots(
    output_dir: Path,
    wave: np.ndarray,
    passbands: np.ndarray,
    tau_ice_true: np.ndarray,
    filter_names: list[str],
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

    plt.figure(figsize=(8, 4.5))
    plt.plot(wave, tau_ice_true, color="tab:blue")
    plt.xlabel("Wavelength [um]")
    plt.ylabel("Optical depth per ice amount")
    plt.title("True ice optical-depth shape")
    plt.tight_layout()
    plt.savefig(output_dir / "sim_true_ice_tau.png", dpi=160)
    plt.close()


def main() -> None:
    simulate_data(SimConfig())


if __name__ == "__main__":
    main()
