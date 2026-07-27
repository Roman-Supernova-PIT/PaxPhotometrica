#!/usr/bin/env python3
"""Query fitted Roman-like calibration products for new instrumental photometry.

This utility reads the CSV products written by ``paxphot fit``
and evaluates, for arbitrary observations:

* the scalar instrumental calibration term
  ``ZP_exposure + S_imaging,filter,detector(x, y) + A_detector,amp``;
* the fitted chromatic passband, including detector shift/width and ice;
* the AB zeropoint that converts instrumental magnitudes to AB magnitudes.

The conversion convention is

    m_AB = m_inst + fit_ab_zeropoint_mag

where ``m_inst = -2.5 log10(counts)`` in the same arbitrary count units used by
the simulator. The passband is evaluated on the simulator wavelength grid.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPEED_OF_LIGHT_CM_S = 2.99792458e10
MICRON_TO_CM = 1.0e-4
AB_FNU_CGS = 3631.0e-23  # erg / s / cm^2 / Hz


@dataclass
class QueryConfig:
    sim_dir: str = "passband_sim_outputs"
    fit_dir: str = "passband_fit_outputs"
    query_csv: str | None = None
    output_csv: str = "calibration_query_results.csv"
    passband_output_csv: str = "calibration_query_passbands.csv"
    no_passband_output: bool = False
    print_max_rows: int = 20
    nx: int = 4096
    ny: int = 4096
    n_amp: int = 32


@dataclass
class CalibrationProducts:
    wave: np.ndarray
    filter_ids: np.ndarray
    filter_names: list[str]
    passbands: np.ndarray
    phi_shift: np.ndarray
    phi_width: np.ndarray
    shift_width: pd.DataFrame
    exposure_zp: dict[int, float]
    imaging_smooth_coeff: dict[tuple[int, int], np.ndarray]
    amp_offsets: dict[tuple[int, int], float]
    ice_loglam_nodes: np.ndarray
    ice_thickness_nodes: np.ndarray
    ice_node_values: np.ndarray
    loglam_basis: np.ndarray
    nx: int
    ny: int
    n_amp: int


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


def amp_id_from_x(x: np.ndarray, nx: int = 4096, n_amp: int = 32) -> np.ndarray:
    """Return amplifier stripe id for detector x pixel coordinate."""
    amp_width = nx // n_amp
    amp_id = np.floor(np.asarray(x) / amp_width).astype(int)
    return np.clip(amp_id, 0, n_amp - 1)


def normalized_xy(
    x: np.ndarray, y: np.ndarray, nx: int = 4096, ny: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """Map detector pixels to [-1, 1] normalized coordinates."""
    xn = 2.0 * (np.asarray(x) / (nx - 1.0)) - 1.0
    yn = 2.0 * (np.asarray(y) / (ny - 1.0)) - 1.0
    return xn, yn


def poly_basis(xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
    """Smooth star-flat polynomial terms: x, y, x^2, x*y, y^2."""
    xn = np.asarray(xn)
    yn = np.asarray(yn)
    return np.column_stack((xn, yn, xn**2, xn * yn, yn**2))


def load_long_grid_csv(path: Path, value_columns: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, np.ndarray]]:
    """Load long-form wavelength tables indexed by filter_id."""
    table = pd.read_csv(path)
    wave = np.sort(table["wavelength_um"].unique())
    filter_ids = np.sort(table["filter_id"].unique())
    filter_names = []
    values = {col: np.zeros((filter_ids.size, wave.size), dtype=float) for col in value_columns}

    for i, filter_id in enumerate(filter_ids):
        sub = table.loc[table["filter_id"] == filter_id].sort_values("wavelength_um")
        if not np.allclose(sub["wavelength_um"].to_numpy(float), wave):
            raise ValueError(f"Inconsistent wavelength grid in {path}")
        if "filter_name" in sub.columns:
            filter_names.append(str(sub["filter_name"].iloc[0]))
        else:
            filter_names.append(str(filter_id))
        for col in value_columns:
            values[col][i] = sub[col].to_numpy(float)
    return wave, filter_ids, filter_names, values


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
    t = np.clip(np.asarray(thickness, dtype=float), thickness_nodes[0], thickness_nodes[-1])
    hi = np.searchsorted(thickness_nodes, t, side="right")
    hi = np.clip(hi, 1, thickness_nodes.size - 1)
    lo = hi - 1
    denom = thickness_nodes[hi] - thickness_nodes[lo]
    w_hi = np.divide(t - thickness_nodes[lo], denom, out=np.zeros_like(t), where=denom != 0.0)
    w_lo = 1.0 - w_hi
    return lo, hi, w_lo, w_hi


def evaluate_ice_logt(products: CalibrationProducts, ice_thickness: np.ndarray) -> np.ndarray:
    """Evaluate the fitted ice log-throughput surface for each query row."""
    n_thick = products.ice_thickness_nodes.size
    n_loglam = products.ice_loglam_nodes.size
    values = products.ice_node_values.reshape(n_thick, n_loglam)
    lo, hi, w_lo, w_hi = thickness_brackets(ice_thickness, products.ice_thickness_nodes)
    surface_lo = values[lo] @ products.loglam_basis
    surface_hi = values[hi] @ products.loglam_basis
    return w_lo[:, None] * surface_lo + w_hi[:, None] * surface_hi


def load_calibration_products(config: QueryConfig) -> CalibrationProducts:
    """Read simulator wavelength products and fitted coefficient tables."""
    sim_dir = Path(config.sim_dir)
    fit_dir = Path(config.fit_dir)

    metadata_path = sim_dir / "simulation_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        config.nx = int(metadata.get("nx", config.nx))
        config.ny = int(metadata.get("ny", config.ny))
        config.n_amp = int(metadata.get("n_amp", config.n_amp))

    wave, filter_ids, filter_names, pass_data = load_long_grid_csv(
        sim_dir / "nominal_passbands.csv", ["throughput"]
    )
    wave_modes, mode_filter_ids, mode_names, mode_data = load_long_grid_csv(
        sim_dir / "passband_modes.csv", ["phi_shift", "phi_width"]
    )
    if not np.allclose(wave, wave_modes) or not np.array_equal(filter_ids, mode_filter_ids):
        raise ValueError("Nominal passbands and passband modes do not share the same grid")

    shift_width = pd.read_csv(fit_dir / "fit_passband_params.csv")
    exposure_table = pd.read_csv(fit_dir / "fit_exposure_zeropoints.csv")
    smooth_table = pd.read_csv(fit_dir / "fit_smooth_coeffs.csv")
    amp_table = pd.read_csv(fit_dir / "fit_amp_offsets.csv")
    ice_table = pd.read_csv(fit_dir / "fit_ice_spline_params.csv")

    exposure_zp = {
        int(row.exposure_id): float(row.zp_mag) for row in exposure_table.itertuples(index=False)
    }
    basis_names = ["x", "y", "x2", "xy", "y2"]
    imaging_smooth_coeff = {}
    if "measurement_type" in smooth_table.columns:
        imaging_rows = smooth_table.loc[
            smooth_table["measurement_type"] == "imaging"
        ]
        for (filter_id, detector_id), group in imaging_rows.groupby(
            ["filter_id", "detector_id"]
        ):
            smooth_map = {
                str(row.basis_name): float(row.coefficient_mag)
                for row in group.itertuples(index=False)
            }
            imaging_smooth_coeff[(int(filter_id), int(detector_id))] = np.array(
                [smooth_map[name] for name in basis_names]
            )
    else:
        smooth_map = {
            str(row.basis_name): float(row.coefficient_mag)
            for row in smooth_table.itertuples(index=False)
        }
        shared_coeff = np.array([smooth_map[name] for name in basis_names])
        for filter_id in filter_ids:
            for detector_id in shift_width["detector_id"].unique():
                imaging_smooth_coeff[(int(filter_id), int(detector_id))] = shared_coeff
    amp_offsets = {
        (int(row.detector_id), int(row.amp_id)): float(row.amp_offset_mag)
        for row in amp_table.itertuples(index=False)
    }

    ice_loglam_nodes = np.sort(ice_table["log10_wavelength"].unique())
    ice_thickness_nodes = np.sort(ice_table["ice_thickness"].unique())
    ice_node_values = np.zeros(ice_loglam_nodes.size * ice_thickness_nodes.size, dtype=float)
    for row in ice_table.itertuples(index=False):
        thick_id = np.where(np.isclose(ice_thickness_nodes, float(row.ice_thickness)))[0][0]
        loglam_id = np.where(np.isclose(ice_loglam_nodes, float(row.log10_wavelength)))[0][0]
        ice_node_values[thick_id * ice_loglam_nodes.size + loglam_id] = float(
            row.ice_logt_node_value
        )

    return CalibrationProducts(
        wave=wave,
        filter_ids=filter_ids,
        filter_names=filter_names,
        passbands=pass_data["throughput"],
        phi_shift=mode_data["phi_shift"],
        phi_width=mode_data["phi_width"],
        shift_width=shift_width,
        exposure_zp=exposure_zp,
        imaging_smooth_coeff=imaging_smooth_coeff,
        amp_offsets=amp_offsets,
        ice_loglam_nodes=ice_loglam_nodes,
        ice_thickness_nodes=ice_thickness_nodes,
        ice_node_values=ice_node_values,
        loglam_basis=make_loglam_basis(wave, ice_loglam_nodes),
        nx=config.nx,
        ny=config.ny,
        n_amp=config.n_amp,
    )


def build_manual_query(args: argparse.Namespace) -> pd.DataFrame:
    """Build a one-row query table from command-line scalar arguments."""
    required = {
        "detector_id": args.detector_id,
        "exposure_id": args.exposure_id,
        "x": args.x,
        "y": args.y,
        "ice_thickness": args.ice_thickness,
    }
    missing = [name for name, value in required.items() if value is None]
    if args.filter_id is None and args.filter_name is None:
        missing.append("filter_id or filter_name")
    if missing:
        raise SystemExit(
            "Either pass --query-csv or provide manual query arguments: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    row = dict(required)
    if args.filter_id is not None:
        row["filter_id"] = args.filter_id
    if args.instrumental_mag is not None:
        row["instrumental_mag"] = args.instrumental_mag
    if args.amp_id is not None:
        row["amp_id"] = args.amp_id
    if args.filter_name is not None:
        row["filter_name"] = args.filter_name
    return pd.DataFrame([row])


def normalize_query_table(query: pd.DataFrame, products: CalibrationProducts) -> pd.DataFrame:
    """Validate and complete a query table."""
    query = query.copy()
    if "filter_id" not in query.columns:
        if "filter_name" not in query.columns:
            raise ValueError("Query table needs either filter_id or filter_name")
        name_to_id = dict(zip(products.filter_names, products.filter_ids))
        query["filter_id"] = query["filter_name"].map(name_to_id)
    if query["filter_id"].isna().any():
        raise ValueError("Some query rows have unknown filter_id/filter_name")

    if "filter_name" not in query.columns:
        id_to_name = dict(zip(products.filter_ids, products.filter_names))
        query["filter_name"] = query["filter_id"].astype(int).map(id_to_name)

    for col in ["detector_id", "exposure_id", "x", "y", "ice_thickness"]:
        if col not in query.columns:
            raise ValueError(f"Query table is missing required column: {col}")
    if "amp_id" not in query.columns:
        query["amp_id"] = amp_id_from_x(query["x"].to_numpy(float), products.nx, products.n_amp)

    if "query_id" not in query.columns:
        query.insert(0, "query_id", np.arange(len(query), dtype=int))

    integer_cols = ["query_id", "filter_id", "detector_id", "exposure_id", "amp_id"]
    for col in integer_cols:
        query[col] = query[col].astype(int)
    for col in ["x", "y", "ice_thickness"]:
        query[col] = query[col].astype(float)
    return query


def current_passbands_for_queries(
    query: pd.DataFrame, products: CalibrationProducts
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate fitted throughput and its log-throughput components."""
    filter_to_index = {int(filter_id): i for i, filter_id in enumerate(products.filter_ids)}
    filt_index = np.array([filter_to_index[int(fid)] for fid in query["filter_id"]], dtype=int)
    detector_ids = query["detector_id"].to_numpy(int)
    ice_thickness = query["ice_thickness"].to_numpy(float)

    shift_values = np.zeros(len(query), dtype=float)
    width_values = np.zeros(len(query), dtype=float)
    for row_index, (filter_id, detector_id) in enumerate(
        zip(query["filter_id"].to_numpy(int), detector_ids)
    ):
        match = products.shift_width.loc[
            (products.shift_width["filter_id"].astype(int) == int(filter_id))
            & (products.shift_width["detector_id"].astype(int) == int(detector_id))
        ]
        if match.empty:
            raise ValueError(
                f"No fitted passband parameters for filter_id={filter_id}, "
                f"detector_id={detector_id}"
            )
        shift_values[row_index] = float(match["delta_lambda_um"].iloc[0])
        width_values[row_index] = float(match["width"].iloc[0])

    logt_passband = (
        shift_values[:, None] * products.phi_shift[filt_index]
        + width_values[:, None] * products.phi_width[filt_index]
    )
    ice_logt = evaluate_ice_logt(products, ice_thickness)
    nominal = products.passbands[filt_index]
    throughput = nominal * np.exp(logt_passband + ice_logt)
    return throughput, nominal, logt_passband, ice_logt, filt_index


def evaluate_scalar_terms(query: pd.DataFrame, products: CalibrationProducts) -> np.ndarray:
    """Evaluate fitted scalar instrumental terms for each query row."""
    xn, yn = normalized_xy(query["x"].to_numpy(float), query["y"].to_numpy(float), products.nx, products.ny)
    basis = poly_basis(xn, yn)
    smooth_coeff = []
    missing_smooth = []
    for filter_id, detector_id in zip(
        query["filter_id"].to_numpy(int), query["detector_id"].to_numpy(int)
    ):
        coeff = products.imaging_smooth_coeff.get((int(filter_id), int(detector_id)))
        if coeff is None:
            missing_smooth.append((int(filter_id), int(detector_id)))
            coeff = np.full(5, np.nan)
        smooth_coeff.append(coeff)
    if missing_smooth:
        raise ValueError(
            "No fitted imaging smooth field for filter/detector pairs: "
            f"{sorted(set(missing_smooth))}"
        )
    smooth = np.einsum("ij,ij->i", basis, np.asarray(smooth_coeff))

    zp = np.array(
        [
            products.exposure_zp.get(int(exposure_id), np.nan)
            for exposure_id in query["exposure_id"].to_numpy(int)
        ]
    )
    if np.isnan(zp).any():
        bad = sorted(query.loc[np.isnan(zp), "exposure_id"].astype(int).unique())
        raise ValueError(f"No fitted exposure zeropoint for exposure_id values: {bad}")

    amp = np.array(
        [
            products.amp_offsets.get((int(detector_id), int(amp_id)), np.nan)
            for detector_id, amp_id in zip(
                query["detector_id"].to_numpy(int), query["amp_id"].to_numpy(int)
            )
        ]
    )
    if np.isnan(amp).any():
        bad_rows = query.loc[np.isnan(amp), ["detector_id", "amp_id"]].drop_duplicates()
        raise ValueError(f"No fitted amp offset for rows:\n{bad_rows}")

    return zp + smooth + amp


def query_calibration(
    query: pd.DataFrame, products: CalibrationProducts, include_passband_table: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate AB zeropoints and passband curves for a query table."""
    query = normalize_query_table(query, products)
    throughput, nominal, logt_passband, ice_logt, _ = current_passbands_for_queries(query, products)
    scalar = evaluate_scalar_terms(query, products)
    ab_count = photon_count_integral(
        ab_f_lambda_per_micron(products.wave)[None, :] * throughput,
        products.wave,
        axis=1,
    )
    fit_ab_zeropoint = 2.5 * np.log10(np.maximum(ab_count, 1e-300)) - scalar

    result = query.copy()
    result["fit_scalar_delta_mag"] = scalar
    result["ab_reference_count"] = ab_count
    result["fit_ab_zeropoint_mag"] = fit_ab_zeropoint

    mag_col = None
    for candidate in ("instrumental_mag", "mag_inst", "mag_obs"):
        if candidate in result.columns:
            mag_col = candidate
            break
    if mag_col is not None:
        result["calibrated_ab_mag"] = result[mag_col].to_numpy(float) + fit_ab_zeropoint

    passband_table = pd.DataFrame()
    if include_passband_table:
        passband_rows = []
        for row_index, row in enumerate(result.itertuples(index=False)):
            query_id = int(getattr(row, "query_id"))
            frame = pd.DataFrame(
                {
                    "query_id": query_id,
                    "wavelength_um": products.wave,
                    "filter_id": int(getattr(row, "filter_id")),
                    "filter_name": str(getattr(row, "filter_name")),
                    "detector_id": int(getattr(row, "detector_id")),
                    "ice_thickness": float(getattr(row, "ice_thickness")),
                    "nominal_throughput": nominal[row_index],
                    "logt_passband": logt_passband[row_index],
                    "ice_logt": ice_logt[row_index],
                    "current_throughput": throughput[row_index],
                }
            )
            passband_rows.append(frame)
        passband_table = pd.concat(passband_rows, ignore_index=True)
    return result, passband_table


def print_summary(result: pd.DataFrame, max_rows: int) -> None:
    """Print a compact human-readable summary for command-line use."""
    cols = [
        "query_id",
        "filter_name",
        "detector_id",
        "exposure_id",
        "amp_id",
        "x",
        "y",
        "ice_thickness",
        "fit_scalar_delta_mag",
        "fit_ab_zeropoint_mag",
    ]
    if "calibrated_ab_mag" in result.columns:
        cols.append("calibrated_ab_mag")
    shown = result[cols].head(max_rows)
    print(shown.to_string(index=False))
    if len(result) > max_rows:
        print(f"... {len(result) - max_rows} more rows written to the output CSV")


def parse_args(
    argv: Sequence[str] | None = None, prog: str | None = None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query fitted Roman calibration products for AB zeropoints and passbands.",
        prog=prog,
    )
    parser.add_argument("--sim-dir", default=QueryConfig.sim_dir)
    parser.add_argument("--fit-dir", default=QueryConfig.fit_dir)
    parser.add_argument("--query-csv", default=None, help="CSV with query rows.")
    parser.add_argument("--output-csv", default=QueryConfig.output_csv)
    parser.add_argument("--passband-output-csv", default=QueryConfig.passband_output_csv)
    parser.add_argument(
        "--no-passband-output",
        action="store_true",
        help="Only write the row-level zeropoint query table.",
    )
    parser.add_argument(
        "--print-max-rows",
        type=int,
        default=QueryConfig.print_max_rows,
        help="Maximum number of queried rows to print to stdout.",
    )

    parser.add_argument("--filter-id", type=int, default=None)
    parser.add_argument("--filter-name", default=None)
    parser.add_argument("--detector-id", type=int, default=None)
    parser.add_argument("--exposure-id", type=int, default=None)
    parser.add_argument("--x", type=float, default=None)
    parser.add_argument("--y", type=float, default=None)
    parser.add_argument("--amp-id", type=int, default=None)
    parser.add_argument("--ice-thickness", type=float, default=None)
    parser.add_argument(
        "--instrumental-mag",
        type=float,
        default=None,
        help="Optional instrumental magnitude to calibrate to AB for a manual one-row query.",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None, prog: str | None = None
) -> None:
    args = parse_args(argv, prog)
    config = QueryConfig(
        sim_dir=args.sim_dir,
        fit_dir=args.fit_dir,
        query_csv=args.query_csv,
        output_csv=args.output_csv,
        passband_output_csv=args.passband_output_csv,
        no_passband_output=args.no_passband_output,
        print_max_rows=args.print_max_rows,
    )
    products = load_calibration_products(config)

    if args.query_csv:
        query = pd.read_csv(args.query_csv)
    else:
        query = build_manual_query(args)

    result, passband_table = query_calibration(
        query, products, include_passband_table=not config.no_passband_output
    )
    result.to_csv(config.output_csv, index=False)
    if not config.no_passband_output:
        passband_table.to_csv(config.passband_output_csv, index=False)

    print_summary(result, config.print_max_rows)
    print(f"\nWrote row-level calibration query results to {config.output_csv}")
    if not config.no_passband_output:
        print(f"Wrote sampled current passbands to {config.passband_output_csv}")


if __name__ == "__main__":
    main()
