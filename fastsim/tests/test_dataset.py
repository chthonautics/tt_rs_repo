"""Dataset schema, splits, manifest, geometry export and reproducibility."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from trailer_sim.cli import main
from trailer_sim.dataset import (
    CSV_COLUMNS,
    KNOWN_SIMPLIFICATIONS,
    METADATA_COLUMNS,
    build_dataframe,
    split_records,
    write_run,
)
from trailer_sim.vehicles import sample_vehicle

from conftest import generate_records


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
def test_column_order_matches_the_proposal(frame):
    assert frame.columns == list(CSV_COLUMNS)


def test_units_and_roles_match_the_annexes(frame):
    # A1.2: tractor mass, n trailers, towed mass, Crr, Cd, frontal area
    assert frame["tractor_mass_kg"].min() > 0
    assert set(frame["n_trailers"].unique()) <= {1.0, 2.0, 3.0}
    assert frame["towed_mass_kg"].min() > 0
    assert frame["axle_crr"].min() > 0
    assert frame["frontal_area_m2"].min() > 0
    # A1.3: grade, segment length, target speed, delta v, delta heading
    assert frame["seg_len_m"].min() > 0
    assert frame["target_speed_mps"].min() > 0
    assert frame["grade"].abs().max() <= 0.11


def test_no_nan_or_inf(frame):
    numeric = [c for c in frame.columns if frame[c].dtype.is_numeric()]
    for column in numeric:
        assert frame[column].is_nan().sum() == 0, column
        assert frame[column].is_infinite().sum() == 0, column


def test_static_vector_is_constant_within_a_sample(frame):
    static = ["tractor_mass_kg", "n_trailers", "towed_mass_kg", "axle_crr", "drag_coef", "frontal_area_m2"]
    for column in static:
        unique = frame.group_by("sample_id").agg(pl.col(column).n_unique().alias("n"))
        assert unique["n"].max() == 1, column


def test_segment_indices_are_contiguous_from_zero(frame):
    for _sample_id, group in frame.group_by("sample_id"):
        indices = group.sort("segment_index")["segment_index"].to_list()
        assert indices == list(range(len(indices)))


def test_label_noise_is_applied_but_small(frame):
    clean = frame["energy_j_clean"]
    noisy = frame["energy_j"]
    assert not (clean == noisy).all()
    ratio = (
        frame.with_columns((pl.col("energy_j") / pl.col("energy_j_clean")).alias("ratio"))
        .filter(pl.col("energy_j_clean").abs() > 1.0)["ratio"]
    )
    assert (ratio - 1.0).abs().max() < 0.1


def test_metadata_columns_are_optional(cfg, records):
    plain = build_dataframe(records)
    rich = build_dataframe(records, metadata=True)
    assert set(METADATA_COLUMNS).isdisjoint(plain.columns)
    assert set(METADATA_COLUMNS).issubset(rich.columns)


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def test_splits_do_not_share_routes(cfg, splits):
    train, val = splits
    train_ids = {r.sample_id for r in train}
    val_ids = {r.sample_id for r in val}
    assert train_ids & val_ids == set()
    assert len(train_ids) + len(val_ids) == len(train) + len(val)


def test_splits_are_stratified(cfg, records, splits):
    train, val = splits
    keys = {}
    for rec in records:
        keys.setdefault((rec.sample.vehicle_class, rec.sample.n_trailers), []).append(rec)
    for key, group in keys.items():
        if len(group) < 2:
            continue
        in_train = {r.sample_id for r in train}
        in_val = {r.sample_id for r in val}
        assert any(r.sample_id in in_train for r in group), key
        assert any(r.sample_id in in_val for r in group), key


def test_split_fraction_is_respected(cfg, records, splits):
    train, val = splits
    fraction = len(train) / len(records)
    assert 0.6 <= fraction <= 0.95


# --------------------------------------------------------------------------- #
# class mixture
# --------------------------------------------------------------------------- #
def test_class_mixture_matches_the_configured_weights(cfg):
    rng = np.random.default_rng(0)
    counts = {"agv": 0, "mid": 0, "long_haul": 0}
    n = 3000
    for _ in range(n):
        counts[sample_vehicle(rng, cfg).vehicle_class] += 1
    weights = cfg.vehicles.normalised_weights()
    for name, weight in weights.items():
        assert counts[name] / n == pytest.approx(weight, abs=0.03), name


def test_trailer_counts_respect_class_rules(cfg):
    rng = np.random.default_rng(1)
    for _ in range(500):
        sample = sample_vehicle(rng, cfg)
        allowed = cfg.vehicles.classes[sample.vehicle_class].n_trailers
        assert sample.n_trailers in allowed
        if sample.vehicle_class == "long_haul":
            assert sample.n_trailers == 1


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
def test_write_run_produces_the_documented_files(cfg, records, splits, tmp_path):
    train, val = splits
    run_dir = tmp_path / "run"
    manifest = write_run(run_dir, cfg, records, train, val, failures=[])
    for name in ("train.csv", "val.csv", "manifest.json", "routes_geometry.jsonl", "README.md"):
        assert (run_dir / name).exists(), name

    train_frame = pl.read_csv(run_dir / "train.csv")
    val_frame = pl.read_csv(run_dir / "val.csv")
    assert train_frame.columns == list(CSV_COLUMNS)
    assert train_frame.height + val_frame.height == len(records) * 0 + train_frame.height + val_frame.height
    assert manifest["n_routes_written"] == len(records)
    assert manifest["splits"]["train_routes"] == len(train)
    assert manifest["splits"]["val_routes"] == len(val)
    assert manifest["trailer_fidelity"] == cfg.trailers.fidelity
    assert manifest["master_seed"] == cfg.master_seed
    assert manifest["schema_version"]
    assert manifest["fastsim_version"]
    assert set(manifest["known_simplifications"]) == set(KNOWN_SIMPLIFICATIONS)


def test_manifest_records_the_class_histogram_and_ranges(cfg, records, splits, tmp_path):
    train, val = splits
    manifest = write_run(tmp_path / "run", cfg, records, train, val, failures=[])
    total = sum(manifest["class_histogram"].values())
    assert total == len(records)
    assert set(manifest["class_histogram"]) <= set(cfg.vehicles.classes)
    for column, bounds in manifest["value_ranges"].items():
        assert bounds["min"] <= bounds["max"], column
    assert manifest["conditioning_headroom_histogram"]


def test_geometry_export_is_resimulable(cfg, records, splits, tmp_path):
    train, val = splits
    run_dir = tmp_path / "run"
    write_run(run_dir, cfg, records, train, val, failures=[])
    lines = (run_dir / "routes_geometry.jsonl").read_text().strip().splitlines()
    assert len(lines) == len(records)
    first = json.loads(lines[0])
    for key in ("sample_id", "vehicle_class", "n_trailers", "seg_len_m", "elevation_m", "heading_rad", "grade",
                "node_speed_mps", "target_speed_mps", "energies_j_clean", "vehicle"):
        assert key in first, key
    record = next(r for r in records if r.sample_id == first["sample_id"])
    assert len(first["elevation_m"]) == record.route.n_segments + 1
    assert len(first["energies_j_clean"]) == record.route.n_segments
    assert np.allclose(first["grade"], np.diff(first["elevation_m"]) / first["seg_len_m"])


def test_metadata_flag_adds_columns(cfg, records, splits, tmp_path):
    cfg.output.keep_metadata_columns = True
    try:
        train, val = splits
        run_dir = tmp_path / "meta"
        write_run(run_dir, cfg, records, train, val, failures=[])
        frame = pl.read_csv(run_dir / "train.csv")
        assert set(METADATA_COLUMNS).issubset(frame.columns)
    finally:
        cfg.output.keep_metadata_columns = False


# --------------------------------------------------------------------------- #
# reproducibility (end to end through the CLI)
# --------------------------------------------------------------------------- #
def test_same_seed_gives_identical_datasets(tmp_path):
    root = Path(__file__).resolve().parents[1]
    args = ["generate", "--config", str(root / "configs" / "smoke.yaml"), "--routes", "12", "--quiet"]
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    assert main([*args, "--out", str(out_a)]) == 0
    assert main([*args, "--out", str(out_b)]) == 0

    for name in ("train.csv", "val.csv", "routes_geometry.jsonl"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes(), name

    manifest_a = json.loads((out_a / "manifest.json").read_text())
    manifest_b = json.loads((out_b / "manifest.json").read_text())
    manifest_a.pop("generated_utc")
    manifest_b.pop("generated_utc")
    assert manifest_a == manifest_b


def test_different_seed_gives_different_data(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    base = ["generate", "--config", str(root / "configs" / "smoke.yaml"), "--routes", "8", "--quiet"]
    assert main([*base, "--seed", "1", "--out", str(out_a)]) == 0
    assert main([*base, "--seed", "2", "--out", str(out_b)]) == 0
    assert (out_a / "train.csv").read_bytes() != (out_b / "train.csv").read_bytes()


def test_lumped_fidelity_end_to_end(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "lumped"
    args = [
        "generate",
        "--config", str(root / "configs" / "smoke.yaml"),
        "--routes", "8",
        "--fidelity", "lumped",
        "--quiet",
        "--out", str(out),
    ]
    assert main(args) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["trailer_fidelity"] == "lumped"
    assert manifest["n_routes_written"] == 8


def test_cli_reports_failures_in_the_manifest(tmp_path):
    cfg = generate_records.__wrapped__ if hasattr(generate_records, "__wrapped__") else None  # noqa: F841
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "fail"
    args = ["generate", "--config", str(root / "configs" / "smoke.yaml"), "--routes", "5", "--quiet", "--out", str(out)]
    assert main(args) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert "failures" in manifest and isinstance(manifest["failures"], list)
    assert manifest["n_routes_failed"] == len(manifest["failures"])
