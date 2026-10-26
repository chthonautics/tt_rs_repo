"""Synthetic route synthesis.

A route is ``L`` fixed-length segments (constant ``Δd`` per route). Derived
features (grade, ``Δv``, ``Δθ``) are computed **from the synthesised geometry**,
never sampled independently, so they are always physically consistent. The raw
geometry (per-segment elevation, heading, speed) is returned as well so the
Unity validation phase can re-simulate exact routes.

Two deliberate design decisions:

* elevation is a mean-reverting random walk, so a route is a hill loop and its
  net elevation drop is bounded -- FASTSim's battery cannot absorb unbounded
  descent energy when a BEV starts at ``max_soc``;
* segment speed is strictly positive. A fixed-length segment cannot contain a
  zero-speed dwell, so stop-and-go is represented by a creep speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ClassConfig

#: gravitational acceleration (m/s^2), kept local to avoid a circular import
G = 9.80665
#: constant air density (kg/m^3), matches fastsim's ``f2_const_air_density``
RHO_AIR = 1.225


@dataclass
class Route:
    """One synthesised route (fixed-length segments, physically consistent).

    ``node_speed_mps`` holds the speed at the ``L+1`` segment boundaries; inside a
    segment the vehicle accelerates/decelerates linearly (constant acceleration),
    so the mean speed driven over segment ``i`` is exactly
    ``(node[i] + node[i+1]) / 2`` -- that value is the ``target_speed_mps``
    feature, and FASTSim is given the same trace.
    """

    seg_len_m: float
    grade: np.ndarray  # (L,) dH/dd, dimensionless
    elevation_m: np.ndarray  # (L+1,)
    heading_rad: np.ndarray  # (L+1,) wrapped cumulative heading
    delta_heading_rad: np.ndarray  # (L,) heading change per segment
    node_speed_mps: np.ndarray  # (L+1,) speed at segment boundaries
    target_speed_mps: np.ndarray  # (L,) mean speed driven over the segment
    delta_v_mps: np.ndarray  # (L,) speed difference vs. previous segment
    vehicle_class: str = ""
    n_trailers: int = 0

    @property
    def n_segments(self) -> int:
        return int(self.grade.size)

    @property
    def total_distance_m(self) -> float:
        return self.seg_len_m * self.n_segments

    def to_dict(self) -> dict:
        return {
            "vehicle_class": self.vehicle_class,
            "n_trailers": int(self.n_trailers),
            "n_segments": self.n_segments,
            "seg_len_m": float(self.seg_len_m),
            "elevation_m": [float(x) for x in self.elevation_m],
            "heading_rad": [float(x) for x in self.heading_rad],
            "grade": [float(x) for x in self.grade],
            "node_speed_mps": [float(x) for x in self.node_speed_mps],
            "target_speed_mps": [float(x) for x in self.target_speed_mps],
            "delta_v_mps": [float(x) for x in self.delta_v_mps],
            "delta_heading_rad": [float(x) for x in self.delta_heading_rad],
        }


def synthesize_route(rng: np.random.Generator, spec: ClassConfig, sample=None) -> Route:
    """Synthesise one route for a vehicle class.

    ``sample`` (optional) supplies the sampled machine power and mass so that the
    speed profile stays inside that specific vehicle's power envelope.
    """
    n_seg = int(rng.integers(spec.segments[0], spec.segments[1] + 1))
    seg_len = float(rng.uniform(*spec.seg_len_m))

    raw_grade = _grade_profile(rng, spec, n_seg)
    elevation = _elevation_from_grade(rng, spec, raw_grade, seg_len)
    # The detrend and the net slope inside _elevation_from_grade shift every
    # segment's grade by a constant, so the final grade is the derivative of the
    # elevation, not the raw profile. Clip it to the class limit, then rebuild the
    # elevation from it so the two stay exactly consistent (grade == Δh/Δd).
    grade = np.clip(np.diff(elevation) / seg_len, -spec.grade_max, spec.grade_max)
    elevation = np.concatenate([[0.0], np.cumsum(grade) * seg_len])

    heading, delta_heading = _heading_profile(rng, spec, n_seg)
    node_speed = _speed_profile(rng, spec, n_seg, seg_len, grade, sample)

    speed = 0.5 * (node_speed[:-1] + node_speed[1:])
    delta_v = np.empty(n_seg)
    delta_v[0] = 0.0  # the route window starts already moving
    delta_v[1:] = np.diff(speed)

    return Route(
        seg_len_m=seg_len,
        grade=grade,
        elevation_m=elevation,
        heading_rad=heading,
        delta_heading_rad=delta_heading,
        node_speed_mps=node_speed,
        target_speed_mps=speed,
        delta_v_mps=delta_v,
    )


# --------------------------------------------------------------------------- #
# profiles
# --------------------------------------------------------------------------- #
def _ar1(rng: np.random.Generator, n: int, rho: float, sigma: float) -> np.ndarray:
    """Zero-mean AR(1) noise with unit-free correlation length."""
    eps = rng.normal(0.0, 1.0, size=n)
    out = np.empty(n)
    acc = 0.0
    scale = np.sqrt(1.0 - rho * rho)
    for i in range(n):
        acc = rho * acc + scale * eps[i]
        out[i] = acc
    return out * sigma


def _grade_profile(rng: np.random.Generator, spec: ClassConfig, n_seg: int) -> np.ndarray:
    rho = float(np.exp(-1.0 / max(spec.grade_correlation_segments, 1e-6)))
    grade = _ar1(rng, n_seg, rho, spec.grade_amplitude)
    return np.clip(grade, -spec.grade_max, spec.grade_max)


def _elevation_from_grade(
    rng: np.random.Generator, spec: ClassConfig, raw_grade: np.ndarray, seg_len: float
) -> np.ndarray:
    """Elevation that closes on itself, plus a bounded intentional net drop."""
    n_seg = raw_grade.size
    raw = np.concatenate([[0.0], np.cumsum(raw_grade) * seg_len])
    # remove the residual linear trend so the profile is a closed hill loop
    trend = np.linspace(0.0, raw[-1], raw.size)
    elevation = raw - trend
    # deliberate net elevation change, bounded by the class limit and by the
    # remaining grade headroom (so the recomputed grade stays inside grade_max)
    max_slope = min(
        spec.net_elev_drop_max_m / max(n_seg * seg_len, 1e-9),
        max(spec.grade_max - 2.0 * spec.grade_amplitude, 0.0),
    )
    net_slope = float(rng.uniform(-max_slope, max_slope))
    elevation = elevation + np.linspace(0.0, net_slope * n_seg * seg_len, raw.size)
    return elevation


def _heading_profile(
    rng: np.random.Generator, spec: ClassConfig, n_seg: int
) -> tuple[np.ndarray, np.ndarray]:
    low, high = spec.turn_rad
    turns = np.zeros(n_seg)
    if high > 0:
        mask = rng.random(n_seg) < spec.turn_probability
        magnitude = rng.uniform(low, high, size=n_seg)
        sign = np.where(rng.random(n_seg) < 0.5, -1.0, 1.0)
        turns = np.where(mask, magnitude * sign, 0.0)
    heading = np.zeros(n_seg + 1)
    heading[1:] = np.cumsum(turns)
    wrapped = (heading + np.pi) % (2 * np.pi) - np.pi
    return wrapped, turns


def _speed_profile(
    rng: np.random.Generator,
    spec: ClassConfig,
    n_seg: int,
    seg_len: float,
    grade: np.ndarray,
    sample=None,
) -> np.ndarray:
    """Smooth, drivable boundary-speed profile with stop-and-go dips.

    Returns ``n_seg + 1`` speeds at the segment boundaries. The profile is
    rate-limited with a *power-aware* acceleration cap: fastsim 3.1.0 cannot use
    friction brakes to meet a trace it cannot regenerate, so the synthetic speed
    profile must stay inside the machine's regeneration envelope.
    """
    cruise = float(rng.uniform(*spec.speed_mps))
    creep = min(spec.creep_speed_mps, 0.5 * spec.speed_mps[0])
    rho = float(np.exp(-1.0 / 3.0))
    variation = _ar1(rng, n_seg + 1, rho, 0.12)
    node = cruise * (1.0 + variation)
    node = np.clip(node, creep, spec.speed_mps[1])

    dwell = rng.random(n_seg) < spec.dwell_fraction
    dwell_speed = max(creep, spec.dwell_speed_frac * cruise)
    node[np.nonzero(dwell)[0] + 1] = dwell_speed
    node[0] = float(np.clip(node[0], creep, spec.speed_mps[1]))

    if sample is not None:
        p_em = float(sample.em_pwr_kw) * 1000.0
        mass = float(sample.total_mass_kg)
        crr = float(sample.axle_crr)
        cda = float(sample.drag_coef * sample.frontal_area_m2) + (
            float(sample.n_trailers * spec.wake_factor * spec.trailer_drag_coef * sample.trailer_frontal_area_m2)
        )
    else:  # class-mean fallback
        p_em = 0.5 * (spec.em_pwr_kw[0] + spec.em_pwr_kw[1]) * 1000.0
        n_tr_mean = float(np.mean(spec.n_trailers))
        mass = max(
            0.5 * (spec.tractor_mass_kg[0] + spec.tractor_mass_kg[1])
            + n_tr_mean * 0.5 * (spec.trailer_mass_kg[0] + spec.trailer_mass_kg[1]),
            1.0,
        )
        crr = 0.5 * (spec.axle_crr[0] + spec.axle_crr[1])
        cda = 0.5 * (spec.drag_coef[0] + spec.drag_coef[1]) * 0.5 * (
            spec.frontal_area_m2[0] + spec.frontal_area_m2[1]
        )

    for i in range(1, n_seg + 1):
        # the power peak of a step occurs at its end: evaluate the resistance and
        # grade terms at the higher of the two boundary speeds
        v = max(node[i - 1], node[i], creep)
        dt = 2.0 * seg_len / max(node[i - 1] + node[i], 1e-6)
        p_res = crr * mass * G * v + 0.5 * RHO_AIR * cda * v**3
        p_grade = mass * G * grade[i - 1] * v  # >0 uphill, <0 downhill
        a_acc = (0.75 * p_em - p_res - p_grade) / (mass * v)
        a_dec = (-0.60 * p_em - p_res - p_grade) / (mass * v)
        hi = min(spec.accel_max_mps2, a_acc) * dt
        lo = max(-spec.accel_max_mps2, a_dec) * dt
        if lo > hi:
            lo = hi
        rel = spec.max_rel_speed_change * max(cruise, creep)
        hi = min(hi, rel)
        lo = max(lo, -rel)
        node[i] = float(np.clip(node[i], node[i - 1] + lo, node[i - 1] + hi))
    return np.maximum(node, creep)

