"""Shared fixtures: a small generated dataset reused by most tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trailer_sim.config import default_config
from trailer_sim.dataset import SampleRecord, build_dataframe, split_records
from trailer_sim.noise import add_output_noise, perturb_sample, sample_rng
from trailer_sim.routes import synthesize_route
from trailer_sim.simulate import simulate_route
from trailer_sim.trailers import apply_corrections, correction_terms, effective_params
from trailer_sim.vehicles import build_vehicle, sample_vehicle

N_ROUTES = 24


@pytest.fixture(scope="session")
def cfg():
    config = default_config()
    config.n_routes = N_ROUTES
    config.master_seed = 4242
    return config


def generate_records(config, n_routes: int = N_ROUTES) -> list[SampleRecord]:
    """Deterministic mini-pipeline (mirrors ``trailer_sim.cli.generate``)."""
    records: list[SampleRecord] = []
    for index in range(n_routes):
        geo_rng = sample_rng(config.master_seed, index, stream=1)
        noise_rng = sample_rng(config.master_seed, index, stream=2)
        label_rng = sample_rng(config.master_seed, index, stream=3)

        nominal = sample_vehicle(geo_rng, config)
        eff_nominal = effective_params(nominal, config.trailers)
        for _attempt in range(config.sim.max_route_retries + 1):
            route = synthesize_route(geo_rng, nominal.spec, sample=nominal)
            route.vehicle_class = nominal.vehicle_class
            route.n_trailers = nominal.n_trailers
            sim_sample = perturb_sample(noise_rng, nominal, config)
            eff_sim = effective_params(sim_sample, config.trailers)
            corrections = correction_terms(sim_sample, route, eff_sim, config.trailers)
            result = None
            for multiplier in config.conditioning.ladder:
                headroom = multiplier * nominal.spec.conditioning_headroom_j
                vehicle = build_vehicle(sim_sample, eff_sim)
                try:
                    result = simulate_route(route, sim_sample, vehicle, config, headroom_j=headroom)
                    break
                except Exception:
                    result = None
            if result is not None:
                break
        assert result is not None, f"route {index} infeasible in fixture"

        clean, _ = apply_corrections(result.energies_j, corrections, config.trailers)
        noisy = add_output_noise(label_rng, clean, config)
        records.append(
            SampleRecord(
                sample_id=index,
                sample=sim_sample,
                route=route,
                static_vector=nominal.static_vector(eff_nominal),
                energies_j=noisy,
                energies_clean_j=clean,
                conditioning_headroom_j=result.conditioning_headroom_j,
                trace_miss_used=result.trace_miss_used,
            )
        )
    return records


@pytest.fixture(scope="session")
def records(cfg) -> list[SampleRecord]:
    return generate_records(cfg)


@pytest.fixture(scope="session")
def frame(cfg, records):
    return build_dataframe(records)


@pytest.fixture(scope="session")
def splits(cfg, records):
    return split_records(records, cfg)


@pytest.fixture()
def flat_vehicle():
    """A 36 t Class-8-like combination for the physics-anchor tests."""
    config = default_config()
    spec = replace(
        config.vehicles.classes["long_haul"],
        tractor_mass_kg=(11000.0, 11000.0),
        trailer_mass_kg=(25000.0, 25000.0),
        drag_coef=(0.55, 0.55),
        frontal_area_m2=(10.0, 10.0),
        axle_crr=(0.006, 0.006),
        em_pwr_kw=(450.0, 450.0),
    )
    config.vehicles.classes["long_haul"] = spec
    rng = np.random.default_rng(7)
    sample = sample_vehicle(rng, config, force_class="long_haul")
    return config, sample


class SimFrame:
    """A FASTSim history frame plus the index where the steady phase begins."""

    def __init__(self, frame, ramp_steps: int):
        self.frame = frame
        self.ramp_steps = ramp_steps

    def final(self, column: str) -> float:
        return float(self.frame[column][-1])

    def steady(self, column: str) -> float:
        """Value accumulated after the spool-up ramp (the steady phase only)."""
        return float(self.frame[column][-1]) - float(self.frame[column][self.ramp_steps - 1])


def constant_speed_frame(config, sample, speed_mps: float, grade: float, seconds: int = 120) -> SimFrame:
    """Run FASTSim on a flat/constant-speed cycle and return the history frame."""
    import fastsim

    from trailer_sim.simulate import _sim_params, _spool_up

    eff = effective_params(sample, config.trailers)
    vehicle = build_vehicle(sample, eff)
    ramp_speeds, ramp_grades, ramp_dts = _spool_up(sample, speed_mps, 1.0)
    speeds = ramp_speeds + [speed_mps] * seconds
    grades = ramp_grades + [float(grade)] * seconds
    dts = ramp_dts + [1.0] * seconds
    time = np.concatenate([[0.0], np.cumsum(dts[1:])])
    dist = np.concatenate([[0.0], np.cumsum(np.asarray(speeds[1:]) * np.asarray(dts[1:]))])
    cycle = fastsim.Cycle.from_dict(
        {
            "init_elev_meters": 0.0,
            "time_seconds": time.tolist(),
            "speed_meters_per_second": [float(x) for x in speeds],
            "dist_meters": dist.tolist(),
            "grade": grades,
            "pwr_max_chrg_watts": [1e9] * len(speeds),
            "temp_amb_air_kelvin": [293.15] * len(speeds),
            "pwr_solar_load_watts": [0.0] * len(speeds),
        }
    )
    sim_drive = fastsim.SimDrive(vehicle, cycle, _sim_params(config, "Error"))
    sim_drive.run()
    return SimFrame(sim_drive.to_dataframe(backend="polars"), len(ramp_speeds))
