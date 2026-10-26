"""Simplified trailer physics: effective parameters and correction terms."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trailer_sim.config import default_config
from trailer_sim.routes import synthesize_route
from trailer_sim.trailers import G, CorrectionTerms, correction_terms, effective_params
from trailer_sim.vehicles import VehicleSample, sample_vehicle


@pytest.fixture()
def setup():
    cfg = default_config()
    rng = np.random.default_rng(17)
    sample = sample_vehicle(rng, cfg, force_class="mid")
    route = synthesize_route(rng, sample.spec, sample=sample)
    return cfg, sample, route


def test_static_vector_is_the_combination_not_the_tractor(setup):
    cfg, sample, _ = setup
    eff = effective_params(sample, cfg.trailers)
    vector = sample.static_vector(eff)
    assert len(vector) == 6  # annex A1.2
    assert vector[0] == pytest.approx(sample.tractor_mass_kg)
    assert vector[1] == sample.n_trailers
    assert vector[2] == pytest.approx(sample.towed_mass_kg)
    assert vector[3] == pytest.approx(eff.axle_crr)
    assert vector[4] == pytest.approx(eff.drag_coef)
    assert vector[5] == pytest.approx(eff.frontal_area_m2)
    # the effective CdA is larger than the bare tractor's
    assert eff.cda_m2 > sample.drag_coef * sample.frontal_area_m2


def test_effective_crr_is_a_load_weighted_mean(setup):
    cfg, sample, _ = setup
    eff = effective_params(sample, cfg.trailers)
    crr = sample.spec.axle_crr_by_position[:3]
    assert min(crr) - 1e-12 <= eff.axle_crr <= max(crr) + 1e-12


def test_effective_cdA_matches_its_factors(setup):
    cfg, sample, _ = setup
    eff = effective_params(sample, cfg.trailers)
    assert eff.cda_m2 == pytest.approx(eff.drag_coef * eff.frontal_area_m2)
    # the combination reference area is the largest cross-section, not the sum
    assert eff.frontal_area_m2 == pytest.approx(
        max(sample.frontal_area_m2, sample.trailer_frontal_area_m2)
    )
    # a trailer adds drag, but less than a second isolated body would
    assert eff.cda_m2 > sample.drag_coef * sample.frontal_area_m2
    assert eff.cda_m2 < sample.drag_coef * sample.frontal_area_m2 + sample.n_trailers * (
        sample.spec.trailer_drag_coef * sample.trailer_frontal_area_m2
    )


def test_lumped_fidelity_has_no_corrections(setup):
    cfg, sample, route = setup
    cfg_lumped = default_config()
    cfg_lumped.trailers.fidelity = "lumped"
    eff = effective_params(sample, cfg_lumped.trailers)
    terms = correction_terms(sample, route, eff, cfg_lumped.trailers)
    assert np.allclose(terms.total_j, 0.0)


def test_straight_segments_have_no_articulation_loss(setup):
    cfg, sample, route = setup
    eff = effective_params(sample, cfg.trailers)
    straight = copy.deepcopy(route)
    straight.delta_heading_rad = np.zeros(route.n_segments)
    terms = correction_terms(sample, straight, eff, cfg.trailers)
    assert np.allclose(terms.articulation_j, 0.0)
    assert np.allclose(terms.yaw_drag_j, 0.0)


def test_articulation_and_yaw_grow_with_heading_change(setup):
    cfg, sample, route = setup
    eff = effective_params(sample, cfg.trailers)
    base = correction_terms(sample, route, eff, cfg.trailers)
    doubled = copy.deepcopy(route)
    doubled.delta_heading_rad = 2.0 * route.delta_heading_rad
    more = correction_terms(sample, doubled, eff, cfg.trailers)
    assert np.all(more.articulation_j >= base.articulation_j - 1e-9)
    assert np.all(more.yaw_drag_j >= base.yaw_drag_j - 1e-9)
    assert more.articulation_j.sum() == pytest.approx(2.0 * base.articulation_j.sum())


def test_load_transfer_is_signed_with_grade(setup):
    cfg, sample, route = setup
    eff = effective_params(sample, cfg.trailers)

    def load_transfer(grade_value: float) -> np.ndarray:
        variant = copy.deepcopy(route)
        variant.grade = np.full(route.n_segments, grade_value)
        return correction_terms(sample, variant, eff, cfg.trailers).load_transfer_j

    e_up = load_transfer(0.05)
    e_down = load_transfer(-0.05)
    e_flat = load_transfer(0.0)
    assert np.allclose(e_flat, 0.0)
    assert np.all(e_up > 0)
    assert np.all(e_down < 0)
    assert np.allclose(e_up, -e_down, rtol=1e-6)


def test_rotational_inertia_follows_the_kinetic_energy_change(setup):
    cfg, sample, route = setup
    eff = effective_params(sample, cfg.trailers)
    terms = correction_terms(sample, route, eff, cfg.trailers)
    m_rot = cfg.trailers.rotational_inertia_fraction * sample.towed_mass_kg
    v_prev = route.target_speed_mps - route.delta_v_mps
    expected = 0.5 * m_rot * (route.target_speed_mps**2 - v_prev**2)
    assert np.allclose(terms.rotational_inertia_j, expected)
    # the inertial term vanishes on a constant-speed route
    steady = copy.deepcopy(route)
    steady.delta_v_mps = np.zeros(route.n_segments)
    terms_steady = correction_terms(sample, steady, eff, cfg.trailers)
    assert np.allclose(terms_steady.rotational_inertia_j, 0.0)


def test_scrub_matches_the_documented_model(setup):
    cfg, sample, route = setup
    eff = effective_params(sample, cfg.trailers)
    terms = correction_terms(sample, route, eff, cfg.trailers)
    expected = (
        cfg.trailers.scrub_friction_coef
        * eff.trailer_axle_load_n
        * sample.spec.hitch_to_axle_m
        * np.abs(route.delta_heading_rad)
    )
    assert np.allclose(terms.articulation_j, expected)


def test_regeneration_caveat_keeps_the_sign_of_positive_segments():
    from trailer_sim.trailers import apply_corrections

    cfg = default_config()
    base = np.array([1000.0, 1000.0, -500.0])
    terms = CorrectionTerms(
        articulation_j=np.array([0.0, 0.0, 0.0]),
        load_transfer_j=np.array([-2000.0, -200.0, -200.0]),
        yaw_drag_j=np.zeros(3),
        rotational_inertia_j=np.zeros(3),
    )
    total, applied = apply_corrections(base, terms, cfg.trailers)
    # segment 0 would flip sign: the negative correction is clamped
    assert total[0] == pytest.approx(0.0)
    assert applied[0] == pytest.approx(-1000.0)
    # segment 1 stays positive, untouched
    assert applied[1] == pytest.approx(-200.0)
    # a segment that is already regeneration may absorb the correction fully
    assert applied[2] == pytest.approx(-200.0)


@settings(max_examples=20, deadline=None)
@given(st.floats(min_value=-0.15, max_value=0.15, allow_nan=False))
def test_load_transfer_never_exceeds_the_rolling_energy(grade):
    cfg = default_config()
    rng = np.random.default_rng(2)
    sample = sample_vehicle(rng, cfg, force_class="long_haul")
    eff = effective_params(sample, cfg.trailers)
    correction = eff.rolling_correction_j(np.array([grade]), 300.0)[0]
    rolling = eff.axle_crr * sample.total_mass_kg * G * 300.0
    assert abs(correction) <= 0.5 * rolling
