"""Configuration loading, defaults and validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from trailer_sim.config import Config, ConfigError, default_config, load_config


def test_default_mixture_matches_the_proposal():
    cfg = default_config()
    weights = cfg.vehicles.normalised_weights()
    assert set(weights) == {"agv", "mid", "long_haul"}
    assert weights["mid"] == pytest.approx(0.60)
    assert weights["agv"] == pytest.approx(0.20)
    assert weights["long_haul"] == pytest.approx(0.20)


def test_class_rules_from_the_proposal():
    cfg = default_config()
    # long-haul is single-trailer only; multi-trailer appears in agv/mid
    assert cfg.vehicles.classes["long_haul"].n_trailers == [1]
    assert max(cfg.vehicles.classes["agv"].n_trailers) == 3
    assert cfg.vehicles.classes["mid"].n_trailers == [1, 2]
    # long-haul is the fastest, agv the slowest
    assert cfg.vehicles.classes["long_haul"].speed_mps[1] > cfg.vehicles.classes["mid"].speed_mps[1]
    assert cfg.vehicles.classes["agv"].speed_mps[1] < cfg.vehicles.classes["mid"].speed_mps[1]


def test_yaml_overrides_are_merged(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("run_name: mine\nn_routes: 7\nnoise:\n  output:\n    sigma: 0.05\n")
    cfg = load_config(path)
    assert cfg.run_name == "mine"
    assert cfg.n_routes == 7
    assert cfg.noise.output.sigma == pytest.approx(0.05)
    # untouched sections keep their defaults
    assert cfg.trailers.fidelity == "quasi_static"


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict({"n_routes": 1, "not_a_key": 2})


def test_invalid_fidelity_is_rejected():
    with pytest.raises(ConfigError, match="fidelity"):
        load_config(overrides={"trailers": {"fidelity": "wizardry"}})


def test_dynamic_fidelity_is_rejected():
    with pytest.raises(ConfigError, match="stretch goal"):
        load_config(overrides={"trailers": {"fidelity": "dynamic"}})


def test_inconsistent_speed_range_is_rejected():
    data = {"vehicles": {"classes": {"x": {"weight": 1.0, "speed_mps": [0.2, 1.0], "creep_speed_mps": 0.5}}}}
    with pytest.raises(ConfigError, match="creep"):
        Config.from_dict(data)


def test_snapshot_is_json_serialisable():
    import json

    json.dumps(default_config().snapshot())


def test_unknown_stratification_key_is_rejected():
    with pytest.raises(ConfigError, match="stratify"):
        load_config(overrides={"output": {"stratify": ["vehicle_class", "nonsense"]}})


def test_smoke_and_dev_configs_load():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("smoke.yaml", "dev.yaml"):
        cfg = load_config(root / "configs" / name)
        assert cfg.n_routes > 0
        assert cfg.trailers.fidelity == "quasi_static"
    dev = load_config(root / "configs" / "dev.yaml")
    smoke = load_config(root / "configs" / "smoke.yaml")
    assert dev.n_routes > smoke.n_routes
    assert yaml.safe_load((root / "configs" / "smoke.yaml").read_text())["run_name"] == "smoke_v0"


# --------------------------------------------------------------------------- #
# vehicle-class presets (one YAML file per class)
# --------------------------------------------------------------------------- #
def test_every_default_class_has_a_preset_file():
    from trailer_sim.config import DEFAULT_CLASS_PRESETS, PRESETS_DIR

    assert DEFAULT_CLASS_PRESETS == ("agv", "mid", "long_haul")
    for name in DEFAULT_CLASS_PRESETS:
        path = PRESETS_DIR / f"{name}.yaml"
        assert path.is_file(), path
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict)
        assert "weight" in data and "n_trailers" in data


def test_default_config_comes_from_the_preset_files():
    """Editing a preset must change the built-in mixture (no values in Python)."""
    from trailer_sim import config as config_module

    preset = {
        "weight": 1.0,
        "n_trailers": [2],
        "tractor_mass_kg": [1234.0, 4321.0],
        "speed_mps": [11.0, 22.0],
    }
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "custom.yaml").write_text(yaml.safe_dump(preset))
        original_dir, original_names = config_module.PRESETS_DIR, config_module.DEFAULT_CLASS_PRESETS
        config_module.PRESETS_DIR = directory
        config_module.DEFAULT_CLASS_PRESETS = ("custom",)
        try:
            cfg = config_module.default_config()
        finally:
            config_module.PRESETS_DIR = original_dir
            config_module.DEFAULT_CLASS_PRESETS = original_names

    assert list(cfg.vehicles.classes) == ["custom"]
    klass = cfg.vehicles.classes["custom"]
    assert klass.tractor_mass_kg == (1234.0, 4321.0)
    assert klass.n_trailers == [2]
    assert cfg.vehicles.normalised_weights() == {"custom": 1.0}


def test_missing_preset_is_reported():
    from trailer_sim.config import PresetError, load_class_preset

    with pytest.raises(PresetError, match="not found"):
        load_class_preset("does_not_exist")


def test_preset_with_unknown_key_is_rejected():
    from trailer_sim import config as config_module
    from trailer_sim.config import PresetError

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "broken.yaml").write_text("weight: 1.0\nnot_a_field: 3\n")
        original = config_module.PRESETS_DIR
        config_module.PRESETS_DIR = directory
        try:
            with pytest.raises(ConfigError, match="unknown key"):
                config_module.load_class_preset("broken")
        finally:
            config_module.PRESETS_DIR = original


def test_preset_with_reversed_range_is_rejected():
    from trailer_sim import config as config_module
    from trailer_sim.config import ConfigError

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "reversed.yaml").write_text("weight: 1.0\ntractor_mass_kg: [900.0, 100.0]\n")
        original = config_module.PRESETS_DIR
        config_module.PRESETS_DIR = directory
        try:
            with pytest.raises(ConfigError, match="low bound"):
                config_module.load_class_preset("reversed")
        finally:
            config_module.PRESETS_DIR = original
