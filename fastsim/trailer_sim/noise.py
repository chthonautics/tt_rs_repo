"""Noise: input-parameter uncertainty and output-label noise.

Both levels are seeded from the run's master seed, so a dataset is byte-for-byte
reproducible. The paper's requirement is met by level 1: the noise perturbs the
*physical input parameters* during generation and is **not** a model input --
the dataset's feature columns keep the nominal (as-designed) values while the
simulation ran on the perturbed ones.
"""

from __future__ import annotations

import copy

import numpy as np

from .config import Config
from .vehicles import VehicleSample


def route_rng(master_seed: int, index: int) -> np.random.Generator:
    """Deterministic RNG for route/sample ``index`` of a run."""
    return np.random.default_rng(np.random.SeedSequence(entropy=master_seed, spawn_key=(index,)))


def sample_rng(master_seed: int, index: int, stream: int = 1) -> np.random.Generator:
    """Independent deterministic stream (keeps geometry and noise uncorrelated)."""
    return np.random.default_rng(
        np.random.SeedSequence(entropy=master_seed, spawn_key=(stream, index))
    )


def _mult(rng: np.random.Generator, value: float, sigma: float, *, floor: float = 0.0) -> float:
    return float(max(value * (1.0 + rng.normal(0.0, sigma)), floor))


def perturb_sample(rng: np.random.Generator, sample: VehicleSample, cfg: Config) -> VehicleSample:
    """Return a copy of ``sample`` with multiplicatively perturbed physical inputs.

    Zero-mean multiplicative Gaussian noise simulates the parameter uncertainty of
    the physical inputs (masses, rolling/aerodynamic coefficients, frontal area).
    Geometry/class bookkeeping fields are not perturbed.
    """
    noise = cfg.noise.input
    out = copy.deepcopy(sample)
    out.spec = sample.spec  # class spec is shared, not a physical input
    out.perturbed = True
    if not noise.enabled:
        return out

    def sigma(name: str) -> float:
        return noise.sigma_for(name)

    out.tractor_mass_kg = _mult(rng, sample.tractor_mass_kg, sigma("tractor_mass_kg"), floor=1.0)
    out.trailer_mass_kg = _mult(rng, sample.trailer_mass_kg, sigma("trailer_mass_kg"), floor=0.0)
    out.axle_crr = _mult(rng, sample.axle_crr, sigma("axle_crr"), floor=1e-4)
    out.drag_coef = _mult(rng, sample.drag_coef, sigma("drag_coef"), floor=1e-3)
    out.frontal_area_m2 = _mult(rng, sample.frontal_area_m2, sigma("frontal_area_m2"), floor=0.1)
    out.trailer_frontal_area_m2 = _mult(
        rng, sample.trailer_frontal_area_m2, sigma("frontal_area_m2"), floor=0.05
    )
    return out


def add_output_noise(rng: np.random.Generator, energies_j: np.ndarray, cfg: Config) -> np.ndarray:
    """Multiplicative zero-mean Gaussian noise on the clean per-segment energy.

    The label noise doubles as regularisation for the unmodelled effects listed in
    the dataset manifest (``known_simplifications``).
    """
    out_cfg = cfg.noise.output
    energies = np.asarray(energies_j, dtype=float)
    if not out_cfg.enabled or out_cfg.sigma <= 0:
        return energies.copy()
    return energies * (1.0 + rng.normal(0.0, out_cfg.sigma, size=energies.shape))
