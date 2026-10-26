"""Route synthesis: schema invariants and physical consistency of the features."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trailer_sim.config import default_config
from trailer_sim.routes import synthesize_route


@pytest.fixture()
def spec():
    return default_config().vehicles.classes["mid"]


def test_derived_features_match_the_geometry(spec):
    route = synthesize_route(np.random.default_rng(0), spec)
    n = route.n_segments
    # grade is exactly dH/dd
    assert np.allclose(route.grade, np.diff(route.elevation_m) / route.seg_len_m)
    # elevation is the integral of grade
    assert np.allclose(route.elevation_m, np.concatenate([[0.0], np.cumsum(route.grade) * route.seg_len_m]))
    # target speed is the mean of the boundary speeds (constant-acceleration segment)
    assert np.allclose(route.target_speed_mps, 0.5 * (route.node_speed_mps[:-1] + route.node_speed_mps[1:]))
    # delta_v is the difference between consecutive segment mean speeds
    assert route.delta_v_mps[0] == 0.0
    assert np.allclose(route.delta_v_mps[1:], np.diff(route.target_speed_mps))
    assert route.delta_heading_rad.shape == (n,)
    assert route.heading_rad.shape == (n + 1,)


def test_route_length_and_ranges(spec):
    rng = np.random.default_rng(3)
    for _ in range(20):
        route = synthesize_route(rng, spec)
        assert spec.segments[0] <= route.n_segments <= spec.segments[1]
        assert spec.seg_len_m[0] <= route.seg_len_m <= spec.seg_len_m[1]
        assert np.all(np.abs(route.grade) <= spec.grade_max + 1e-12)
        assert np.all(route.target_speed_mps > 0)
        assert np.all(np.isfinite(route.elevation_m))
        assert np.all(np.isfinite(route.target_speed_mps))


def test_net_elevation_drop_is_bounded(spec):
    rng = np.random.default_rng(11)
    for _ in range(20):
        route = synthesize_route(rng, spec)
        net = abs(route.elevation_m[-1] - route.elevation_m[0])
        assert net <= spec.net_elev_drop_max_m + 1e-9


def test_speed_profile_respects_the_power_envelope():
    """No boundary step may demand more than the machine can deliver/absorb."""
    cfg = default_config()
    from trailer_sim.vehicles import sample_vehicle

    rng = np.random.default_rng(5)
    for index in range(30):
        sample = sample_vehicle(rng, cfg)
        route = synthesize_route(rng, sample.spec, sample=sample)
        mass = sample.total_mass_kg
        for i in range(1, route.n_segments + 1):
            dv = route.node_speed_mps[i] - route.node_speed_mps[i - 1]
            assert abs(dv) <= sample.spec.max_rel_speed_change * sample.spec.speed_mps[1] + 1e-6


@settings(max_examples=25, deadline=None)
@given(st.integers(min_value=0, max_value=2**31 - 1))
def test_route_generation_is_deterministic(seed):
    spec = default_config().vehicles.classes["agv"]
    a = synthesize_route(np.random.default_rng(seed), spec)
    b = synthesize_route(np.random.default_rng(seed), spec)
    assert np.array_equal(a.grade, b.grade)
    assert np.array_equal(a.target_speed_mps, b.target_speed_mps)
    assert np.array_equal(a.delta_heading_rad, b.delta_heading_rad)


def test_stop_and_go_is_a_slowdown_not_a_standstill(spec):
    rng = np.random.default_rng(1)
    route = synthesize_route(rng, spec)
    assert route.target_speed_mps.min() > 0
    assert route.target_speed_mps.min() >= 0.5 * spec.creep_speed_mps
