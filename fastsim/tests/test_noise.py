"""Noise: statistics, determinism, and the paper's "noise is not an input" rule."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trailer_sim.config import default_config
from trailer_sim.noise import add_output_noise, perturb_sample, sample_rng
from trailer_sim.vehicles import sample_vehicle


@pytest.fixture()
def sample():
    cfg = default_config()
    return cfg, sample_vehicle(np.random.default_rng(4), cfg, force_class="mid")


def test_input_noise_is_zero_mean_multiplicative(sample):
    cfg, base = sample
    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(4000):
        perturbed = perturb_sample(rng, base, cfg)
        ratios.append(perturbed.tractor_mass_kg / base.tractor_mass_kg)
    ratios = np.asarray(ratios)
    assert ratios.mean() == pytest.approx(1.0, abs=0.003)
    assert ratios.std() == pytest.approx(cfg.noise.input.sigma_for("tractor_mass_kg"), rel=0.1)


def test_input_noise_perturbs_every_physical_parameter(sample):
    cfg, base = sample
    perturbed = perturb_sample(np.random.default_rng(1), base, cfg)
    for parameter in cfg.noise.input.parameters:
        assert getattr(perturbed, parameter) != getattr(base, parameter), parameter
    # class/geometry bookkeeping is not a physical input
    assert perturbed.n_trailers == base.n_trailers
    assert perturbed.vehicle_class == base.vehicle_class
    assert perturbed.spec is base.spec


def test_input_noise_can_be_disabled(sample):
    cfg, base = sample
    cfg.noise.input.enabled = False
    try:
        perturbed = perturb_sample(np.random.default_rng(1), base, cfg)
        assert perturbed.tractor_mass_kg == base.tractor_mass_kg
    finally:
        cfg.noise.input.enabled = True


def test_parameters_stay_positive(sample):
    cfg, base = sample
    rng = np.random.default_rng(7)
    for _ in range(500):
        perturbed = perturb_sample(rng, base, cfg)
        assert perturbed.tractor_mass_kg > 0
        assert perturbed.axle_crr > 0
        assert perturbed.frontal_area_m2 > 0
        assert perturbed.trailer_mass_kg >= 0


def test_output_noise_is_zero_mean_multiplicative():
    cfg = default_config()
    clean = np.full(20000, 1000.0)
    noisy = add_output_noise(np.random.default_rng(3), clean, cfg)
    ratio = noisy / clean
    assert ratio.mean() == pytest.approx(1.0, abs=0.002)
    assert ratio.std() == pytest.approx(cfg.noise.output.sigma, rel=0.1)


def test_output_noise_preserves_the_sign_of_the_energy():
    cfg = default_config()
    clean = np.array([-500.0, 0.0, 500.0])
    noisy = add_output_noise(np.random.default_rng(5), clean, cfg)
    assert noisy[0] < 0.0
    assert noisy[1] == 0.0
    assert noisy[2] > 0.0


def test_noise_streams_are_reproducible():
    cfg = default_config()
    base = sample_vehicle(sample_rng(cfg.master_seed, 3, stream=1), cfg)
    a = perturb_sample(sample_rng(cfg.master_seed, 3, stream=2), base, cfg)
    b = perturb_sample(sample_rng(cfg.master_seed, 3, stream=2), base, cfg)
    assert a.tractor_mass_kg == b.tractor_mass_kg
    assert a.axle_crr == b.axle_crr
    # different routes get different draws
    c = perturb_sample(sample_rng(cfg.master_seed, 4, stream=2), base, cfg)
    assert c.tractor_mass_kg != a.tractor_mass_kg


@settings(max_examples=20, deadline=None)
@given(st.floats(min_value=0.001, max_value=0.2))
def test_output_sigma_is_configurable(sigma):
    cfg = default_config()
    cfg.noise.output.sigma = sigma
    clean = np.full(5000, 100.0)
    noisy = add_output_noise(np.random.default_rng(11), clean, cfg)
    assert (noisy / clean).std() == pytest.approx(sigma, rel=0.15)
