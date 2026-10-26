"""Literature anchors: turns "somewhat realistic" into a testable assertion.

Reference points:
* flat ~36 t combination at 25 m/s -> aerodynamic share of the tractive load in
  the 45-55% band (published Class 8 highway energy breakdown);
* AGV-class ~3 t at 2 m/s -> aerodynamic share below 2% (rolling dominates);
* hilly routes are dominated by ``m*g*Δh``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trailer_sim.config import default_config
from trailer_sim.vehicles import sample_vehicle

from conftest import SimFrame, constant_speed_frame


def _shares(frame: SimFrame):
    """Share of the tractive energy split by resistance term (steady phase)."""
    drag = frame.steady("veh.history.energy_drag_joules")
    rolling = frame.steady("veh.history.energy_rr_joules")
    accel = frame.steady("veh.history.energy_accel_joules")
    total = drag + rolling + accel
    assert total > 0
    return {"drag": drag / total, "rolling": rolling / total, "accel": accel / total}


@pytest.fixture()
def heavy_36t():
    cfg = default_config()
    spec = replace(
        cfg.vehicles.classes["long_haul"],
        tractor_mass_kg=(11000.0, 11000.0),
        trailer_mass_kg=(25000.0, 25000.0),
        drag_coef=(0.55, 0.55),
        frontal_area_m2=(10.0, 10.0),
        axle_crr=(0.006, 0.006),
        em_pwr_kw=(450.0, 450.0),
    )
    cfg.vehicles.classes["long_haul"] = spec
    sample = sample_vehicle(np.random.default_rng(1), cfg, force_class="long_haul")
    return cfg, sample


@pytest.fixture()
def agv_3t():
    cfg = default_config()
    spec = replace(
        cfg.vehicles.classes["agv"],
        tractor_mass_kg=(1500.0, 1500.0),
        trailer_mass_kg=(750.0, 750.0),
        n_trailers=[2],
        drag_coef=(0.45, 0.45),
        frontal_area_m2=(3.0, 3.0),
        trailer_drag_coef=0.45,
        trailer_frontal_area_m2=(1.5, 1.5),
        axle_crr=(0.011, 0.011),
    )
    cfg.vehicles.classes["agv"] = spec
    sample = sample_vehicle(np.random.default_rng(2), cfg, force_class="agv")
    return cfg, sample


def test_class8_highway_aero_share(heavy_36t):
    cfg, sample = heavy_36t
    frame = constant_speed_frame(cfg, sample, 25.0, 0.0, seconds=200)
    shares = _shares(frame)
    assert 0.45 <= shares["drag"] <= 0.55, shares


def test_agv_aero_share_is_negligible(agv_3t):
    cfg, sample = agv_3t
    frame = constant_speed_frame(cfg, sample, 2.0, 0.0, seconds=400)
    shares = _shares(frame)
    assert shares["drag"] < 0.02, shares
    assert shares["rolling"] > 0.90, shares


def test_aero_dominance_grows_with_the_square_of_speed(heavy_36t):
    """Long-haul speeds must out-scale the AGV regime by more than an order."""
    cfg, sample = heavy_36t
    slow = constant_speed_frame(cfg, sample, 2.0, 0.0, seconds=400)
    fast = constant_speed_frame(cfg, sample, 30.0, 0.0, seconds=100)
    drag_slow = slow.steady("veh.history.energy_drag_joules") / slow.steady("veh.history.dist_meters")
    drag_fast = fast.steady("veh.history.energy_drag_joules") / fast.steady("veh.history.dist_meters")
    assert drag_fast / max(drag_slow, 1e-12) > 10.0


def test_hilly_route_is_dominated_by_ascent_work(heavy_36t):
    cfg, sample = heavy_36t
    steep = constant_speed_frame(cfg, sample, 20.0, 0.04, seconds=200)
    flat = constant_speed_frame(cfg, sample, 20.0, 0.0, seconds=200)
    ascent = steep.steady("veh.history.energy_ascent_joules")
    flat_total = flat.steady(cfg.sim.energy_column)
    steep_total = steep.steady(cfg.sim.energy_column)
    assert ascent > 0
    # the ascent term alone exceeds the whole flat-route consumption
    assert ascent > flat_total
    assert steep_total > 3.0 * flat_total


def test_ascent_energy_matches_m_g_dh(heavy_36t):
    """``E_ascent`` must equal the textbook ``m*g*Δh`` (within rounding)."""
    cfg, sample = heavy_36t
    seconds, speed, grade = 200, 20.0, 0.04
    frame = constant_speed_frame(cfg, sample, speed, grade, seconds=seconds)
    distance = frame.steady("veh.history.dist_meters")
    delta_h = grade * distance
    expected = sample.total_mass_kg * 9.80665 * delta_h
    ascent = frame.steady("veh.history.energy_ascent_joules")
    assert ascent == pytest.approx(expected, rel=0.01)
