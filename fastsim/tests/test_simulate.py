"""FASTSim interface: extraction invariants, physics sanity, fidelity levels."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trailer_sim.config import default_config
from trailer_sim.routes import Route, synthesize_route
from trailer_sim.simulate import build_cycle, simulate_route
from trailer_sim.trailers import apply_corrections, correction_terms, effective_params
from trailer_sim.vehicles import build_vehicle, sample_vehicle

from conftest import constant_speed_frame


# --------------------------------------------------------------------------- #
# extraction invariants
# --------------------------------------------------------------------------- #
def test_segment_sum_equals_the_counter_difference_over_the_route(cfg):
    rng = np.random.default_rng(23)
    sample = sample_vehicle(rng, cfg, force_class="mid")
    route = synthesize_route(rng, sample.spec, sample=sample)
    eff = effective_params(sample, cfg.trailers)
    result = simulate_route(route, sample, build_vehicle(sample, eff), cfg, headroom_j=1e7)

    first, last = result.segment_slices[0, 0], result.segment_slices[-1, 1]
    route_total = result.net_counter_j[last] - result.net_counter_j[first - 1]
    assert result.energies_j.sum() == pytest.approx(route_total, rel=1e-9)


def test_segment_indices_are_contiguous(cfg):
    rng = np.random.default_rng(24)
    sample = sample_vehicle(rng, cfg, force_class="agv")
    route = synthesize_route(rng, sample.spec, sample=sample)
    eff = effective_params(sample, cfg.trailers)
    result = simulate_route(route, sample, build_vehicle(sample, eff), cfg, headroom_j=4e5)
    slices = result.segment_slices
    assert slices.shape == (route.n_segments, 2)
    for i in range(route.n_segments - 1):
        assert slices[i, 1] + 1 == slices[i + 1, 0]
    assert result.conditioning_elements == slices[0, 0] - 1


def test_conditioning_is_outside_the_route_window(cfg):
    """The conditioning phase must not contribute to any reported segment."""
    rng = np.random.default_rng(25)
    sample = sample_vehicle(rng, cfg, force_class="mid")
    route = synthesize_route(rng, sample.spec, sample=sample)
    eff = effective_params(sample, cfg.trailers)
    with_cond = simulate_route(route, sample, build_vehicle(sample, eff), cfg, headroom_j=2e7)
    without = simulate_route(route, sample, build_vehicle(sample, eff), cfg, headroom_j=5e6)
    assert with_cond.conditioning_elements > without.conditioning_elements
    # the conditioning changes the starting SOC, which can clip regeneration, so
    # the route total is only approximately invariant
    assert with_cond.energies_j.sum() == pytest.approx(without.energies_j.sum(), rel=0.25)


def test_energy_scales_with_segment_length(cfg):
    """A zero-length segment would cost nothing (linearity in Δd)."""
    rng = np.random.default_rng(26)
    sample = sample_vehicle(rng, cfg, force_class="mid")
    base = synthesize_route(rng, sample.spec, sample=sample)
    eff = effective_params(sample, cfg.trailers)

    def total(seg_len: float) -> float:
        route = Route(
            seg_len_m=seg_len,
            grade=np.full(10, 0.01),
            elevation_m=np.concatenate([[0.0], np.cumsum(np.full(10, 0.01)) * seg_len]),
            heading_rad=np.zeros(11),
            delta_heading_rad=np.zeros(10),
            node_speed_mps=np.full(11, 8.0),
            target_speed_mps=np.full(10, 8.0),
            delta_v_mps=np.zeros(10),
            vehicle_class=sample.vehicle_class,
            n_trailers=sample.n_trailers,
        )
        result = simulate_route(route, sample, build_vehicle(sample, eff), cfg, headroom_j=1e7)
        return float(result.energies_j.sum())

    one = total(50.0)
    two = total(100.0)
    assert one > 0
    assert two == pytest.approx(2.0 * one, rel=0.02)


# --------------------------------------------------------------------------- #
# physics sanity
# --------------------------------------------------------------------------- #
def test_grade_ordering_downhill_flat_uphill(cfg, flat_vehicle):
    config, sample = flat_vehicle
    downhill = constant_speed_frame(config, sample, 25.0, -0.03)
    flat = constant_speed_frame(config, sample, 25.0, 0.0)
    uphill = constant_speed_frame(config, sample, 25.0, 0.03)
    column = config.sim.energy_column
    e_down, e_flat, e_up = (frame.steady(column) for frame in (downhill, flat, uphill))
    assert e_down < e_flat < e_up
    assert e_down < 0.0  # a sustained descent regenerates


def test_energy_increases_with_mass(cfg, flat_vehicle):
    config, sample = flat_vehicle
    column = config.sim.energy_column
    light = constant_speed_frame(config, sample, 25.0, 0.0).steady(column)
    heavy_sample = replace(sample, trailer_mass_kg=sample.trailer_mass_kg * 2.0)
    heavy = constant_speed_frame(config, heavy_sample, 25.0, 0.0).steady(column)
    assert heavy > light


def test_energy_increases_with_rolling_resistance(cfg, flat_vehicle):
    config, sample = flat_vehicle
    column = config.sim.energy_column
    base = constant_speed_frame(config, sample, 25.0, 0.0).steady(column)
    rough = replace(sample, axle_crr=sample.axle_crr * 2.0)
    higher = constant_speed_frame(config, rough, 25.0, 0.0).steady(column)
    assert higher > base


def test_energy_increases_with_speed(cfg, flat_vehicle):
    config, sample = flat_vehicle
    column = config.sim.energy_column
    slow = constant_speed_frame(config, sample, 15.0, 0.0).steady(column)
    fast = constant_speed_frame(config, sample, 30.0, 0.0).steady(column)
    # same duration: higher speed covers more distance and costs more
    assert fast > slow


# --------------------------------------------------------------------------- #
# fidelity levels
# --------------------------------------------------------------------------- #
def test_lumped_never_exceeds_quasi_static_on_turning_routes(cfg):
    """Articulation losses are strictly additive."""
    rng = np.random.default_rng(31)
    sample = sample_vehicle(rng, cfg, force_class="mid")
    route = synthesize_route(rng, sample.spec, sample=sample)
    route.delta_heading_rad = np.full(route.n_segments, 0.3)

    cfg_qs = default_config()
    cfg_lumped = default_config()
    cfg_lumped.trailers.fidelity = "lumped"

    eff = effective_params(sample, cfg.trailers)
    base = simulate_route(route, sample, build_vehicle(sample, eff), cfg, headroom_j=1e7)
    terms_qs = correction_terms(sample, route, eff, cfg_qs.trailers)
    terms_lumped = correction_terms(sample, route, eff, cfg_lumped.trailers)

    total_qs, _ = apply_corrections(base.energies_j, terms_qs, cfg_qs.trailers)
    total_lumped, _ = apply_corrections(base.energies_j, terms_lumped, cfg_lumped.trailers)
    assert terms_lumped.total_j.sum() == 0.0
    assert total_qs.sum() > total_lumped.sum()
    assert np.all(total_qs >= total_lumped - 1e-6)


def test_lumped_effective_parameters_are_position_independent(cfg):
    rng = np.random.default_rng(32)
    sample = sample_vehicle(rng, cfg, force_class="long_haul")
    cfg_lumped = default_config()
    cfg_lumped.trailers.fidelity = "lumped"
    eff = effective_params(sample, cfg_lumped.trailers)
    assert len(set(np.round(eff.crr_by_axle, 12))) == 1


def test_golden_outputs_are_stable(cfg):
    """Regression guard: a fixed seed must reproduce the same per-route energy."""
    rng = np.random.default_rng(99)
    sample = sample_vehicle(rng, cfg, force_class="mid")
    route = synthesize_route(rng, sample.spec, sample=sample)
    eff = effective_params(sample, cfg.trailers)
    result = simulate_route(route, sample, build_vehicle(sample, eff), cfg, headroom_j=1.5e7)
    terms = correction_terms(sample, route, eff, cfg.trailers)
    total, _ = apply_corrections(result.energies_j, terms, cfg.trailers)
    assert result.energies_j.sum() == pytest.approx(1.3031e7, rel=0.02)
    assert total.sum() == pytest.approx(1.3499e7, rel=0.02)


def test_golden_outputs_per_fidelity_level(cfg):
    """Both fidelity levels are pinned for the same route."""
    cfg_lumped = default_config()
    cfg_lumped.trailers.fidelity = "lumped"

    def route_total(config) -> tuple[float, float]:
        rng = np.random.default_rng(99)
        sample = sample_vehicle(rng, config, force_class="mid")
        route = synthesize_route(rng, sample.spec, sample=sample)
        eff = effective_params(sample, config.trailers)
        result = simulate_route(route, sample, build_vehicle(sample, eff), config, headroom_j=1.5e7)
        terms = correction_terms(sample, route, eff, config.trailers)
        total, _ = apply_corrections(result.energies_j, terms, config.trailers)
        return float(result.energies_j.sum()), float(total.sum())

    lumped_base, lumped_total = route_total(cfg_lumped)
    qs_base, qs_total = route_total(default_config())
    # the same FASTSim run underlies both levels: only the corrections differ
    assert lumped_base == pytest.approx(qs_base, rel=1e-9)
    assert lumped_total == pytest.approx(lumped_base, rel=1e-9)
    assert qs_total == pytest.approx(1.3499e7, rel=0.02)
    assert lumped_total == pytest.approx(1.3031e7, rel=0.02)


def test_cycle_layout_matches_the_route(cfg):
    rng = np.random.default_rng(33)
    sample = sample_vehicle(rng, cfg, force_class="agv")
    route = synthesize_route(rng, sample.spec, sample=sample)
    cycle, slices, n_cond = build_cycle(route, sample, cfg, headroom_j=4e5)
    assert cycle.len() == slices[-1, 1] + 1
    assert n_cond == slices[0, 0] - 1
    data = cycle.to_dict()
    assert len(data["speed_meters_per_second"]) == cycle.len()
    assert data["speed_meters_per_second"][0] == 0.0
