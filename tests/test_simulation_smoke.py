"""Small end-to-end simulator and fitter-input smoke test."""

from __future__ import annotations

import pandas as pd

from paxphotometrica.fit import FitConfig, load_data
from paxphotometrica.simulate import SimConfig, simulate_data


def test_simulation_outputs_feed_unified_fit(tmp_path) -> None:
    output_dir = tmp_path / "simulation"
    simulate_data(
        SimConfig(
            output_dir=str(output_dir),
            n_star=36,
            n_exp=6,
            n_det=1,
            n_absolute_calibrator=2,
            n_prism_exp=1,
            prism_star_fraction=0.10,
            detection_fraction=1.0,
            prism_detection_fraction=1.0,
            dither_sigma_pix=150.0,
        )
    )

    imaging = pd.read_csv(output_dir / "measurements.csv")
    prism = pd.read_csv(output_dir / "prism_measurements.csv")
    amp_truth = pd.read_csv(output_dir / "true_amp_offsets.csv")

    assert not imaging.empty
    assert not prism.empty
    assert {"sky_x", "sky_y", "amp_id"} <= set(imaging.columns)
    assert {"sky_x", "sky_y", "amp_id"} <= set(prism.columns)
    assert len(amp_truth) == 32
    assert set(imaging["amp_id"]).issubset(set(amp_truth["amp_id"]))
    assert set(prism["amp_id"]).issubset(set(amp_truth["amp_id"]))
    assert (
        prism.groupby("spectrum_id")[["x", "amp_id", "sky_x", "sky_y"]]
        .nunique()
        .to_numpy()
        .max()
        == 1
    )
    assert (
        prism.loc[
            (prism["exposure_id"] == 6)
            & prism["is_spectrophotometric_standard"],
            "star_id",
        ].nunique()
        == 2
    )

    loaded = load_data(FitConfig(input_dir=str(output_dir), max_stars=12))
    assert loaded.measurements.size
    assert loaded.prism_measurements.size
    assert loaded.amp_id.min() >= 0
    assert loaded.amp_id.max() < 32

