"""Simplified trailer physics -- the paper's "función de física simplificada".

FASTSim 3 has **no native trailer physics**. This module supplies it by

1. computing *effective combination parameters* (total mass, effective axle
   rolling-resistance coefficient, effective aerodynamic drag coefficient,
   effective frontal area) which are injected into the FASTSim ``Vehicle``
   (the effective-parameters design), and
2. computing per-segment *correction terms* for effects that a static parameter
   vector cannot represent (grade-dependent axle-load transfer, articulation
   tyre scrub, yaw drag, trailer wheel rotational inertia).

Fidelity levels: ``lumped`` (point mass, no corrections) and
``quasi_static`` (default: per-segment energy balance). ``dynamic`` is a
stretch goal that is not implemented and is rejected by :mod:`trailer_sim.config`.

All constants that are *assumptions* rather than literature values are documented
as such and exposed in the config so they can be varied in a sensitivity
analysis.
"""

from dataclasses import dataclass, field

import numpy as np

#: gravitational acceleration (m/s^2)
G = 9.80665
#: constant air density used by the correction terms (kg/m^3); matches
#: ``fastsim.SimParams.f2_const_air_density = True``
RHO_AIR = 1.225


@dataclass
class EffectiveParams:
    """Combination-level values injected into the FASTSim vehicle.

    The first six values are exactly the static feature vector of annex A1.2.
    """

    total_mass_kg: float
    axle_crr: float
    drag_coef: float
    frontal_area_m2: float
    #: nominal (unperturbed) per-axle data used to build the correction terms
    crr_by_axle: np.ndarray = field(default_factory=lambda: np.zeros(0))
    axle_loads_n: np.ndarray = field(default_factory=lambda: np.zeros(0))
    load_shift_n_per_sin: float = 0.0
    trailer_axle_load_n: float = 0.0
    cda_m2: float = 0.0

    # ------------------------------------------------------------------ #
    def rolling_correction_j(self, grades: np.ndarray, seg_len_m: float) -> np.ndarray:
        """Rolling-resistance energy that differs from FASTSim's lumped model.

        FASTSim applies a single ``Crr_eff`` to the whole combination mass. The
        real axle loads redistribute with grade (``dN = M*g*sin(a)*h_cg/L``), and
        the steer / drive / trailer tyre groups genuinely have different ``Crr``,
        so the true rolling energy differs. This term is exactly that difference.
        """
        grades = np.asarray(grades, dtype=float)
        cos_a = np.cos(np.arctan(grades))
        total_load = self.axle_loads_n.sum()
        if total_load <= 0 or self.crr_by_axle.size == 0:
            return np.zeros_like(grades)
        front = np.array([1.0] + [0.0] * (self.crr_by_axle.size - 1))
        sin_a = grades / np.sqrt(1.0 + grades**2)
        shift = sin_a * self.load_shift_n_per_sin  # +: load moves rearward (climb)
        loads = (
            cos_a[:, None] * self.axle_loads_n[None, :]
            + shift[:, None] * (1.0 - front)[None, :]
            - shift[:, None] * front[None, :]
        )
        loads = np.maximum(loads, 0.0)
        weighted = (loads * self.crr_by_axle[None, :]).sum(axis=1)
        lumped = self.axle_crr * loads.sum(axis=1)
        return seg_len_m * G * (weighted - lumped)


@dataclass
class CorrectionTerms:
    """Per-segment additive energy terms (Joules, positive = extra consumption)."""

    articulation_j: np.ndarray
    load_transfer_j: np.ndarray
    yaw_drag_j: np.ndarray
    rotational_inertia_j: np.ndarray

    @property
    def total_j(self) -> np.ndarray:
        return (
            self.articulation_j
            + self.load_transfer_j
            + self.yaw_drag_j
            + self.rotational_inertia_j
        )

    @staticmethod
    def zeros(n: int) -> "CorrectionTerms":
        z = np.zeros(n)
        return CorrectionTerms(z.copy(), z.copy(), z.copy(), z.copy())


def _axle_groups(sample) -> tuple[np.ndarray, np.ndarray]:
    """Return (crr per axle, flat-ground load per axle, in N).

    The combination is modelled as three load groups: the tractor steer axle(s),
    the tractor drive axle(s) and the trailer axle groups. Each trailer puts
    ``tongue_weight_frac`` of its weight on the towing vehicle's rear axles.

    ``spec.axle_crr_by_position`` gives the *relative* tyre profile; it is scaled
    so that the load-weighted mean equals the sampled (and, for the simulation,
    noise-perturbed) ``sample.axle_crr`` -- which is the value the paper's static
    vector carries. Without the scaling, redistributing axle loads would not
    change ``Σ Crr·N`` and the load-transfer term would be decorative.
    """
    spec = sample.spec
    profile = np.array(spec.axle_crr_by_position[:3], dtype=float)

    # towed mass distribution: tongue weight onto the tractor rear, rest on the trailer axles
    towed = sample.towed_mass_kg
    tongue = spec.tongue_weight_frac * towed
    trailer_load = towed - tongue
    tractor_load = sample.tractor_mass_kg

    # tractor axle split: 1/3 steer, 2/3 drive is the usual rigid-truck split; the
    # tractor's own weight plus the tongue weight lands on those two groups.
    tractor_total = tractor_load + tongue
    steer_load = tractor_total / 3.0
    drive_load = tractor_total - steer_load

    loads = np.array([steer_load, drive_load, trailer_load], dtype=float) * G
    flat_mean = float(np.average(profile, weights=loads))
    scale = sample.axle_crr / flat_mean if flat_mean > 0 else 1.0
    return profile * scale, loads


def effective_params(sample, trailer_cfg) -> EffectiveParams:
    """Combination static parameters (annex A1.2 vector) for a vehicle sample."""
    spec = sample.spec
    crr_by_axle, axle_loads = _axle_groups(sample)
    crr_eff = float(np.average(crr_by_axle, weights=axle_loads))
    if trailer_cfg.fidelity == "lumped":
        # one point mass, one effective Crr: no axle-position detail survives
        crr_by_axle = np.full(crr_by_axle.size, crr_eff)

    # effective aerodynamic area. The elements run in-line, so the combination's
    # reference area is the largest cross-section, not the sum; each trailer adds
    # only the part of its frontal area that leaves the tractor's wake.
    n_tr = sample.n_trailers
    cda = sample.drag_coef * sample.frontal_area_m2 + (
        n_tr * spec.wake_factor * spec.trailer_drag_coef * sample.trailer_frontal_area_m2
    )
    area = max(sample.frontal_area_m2, sample.trailer_frontal_area_m2)
    cd_eff = cda / area if area > 0 else sample.drag_coef

    # quasi-static grade load transfer: dN = M*g*sin(a) * h_cg / L
    h_cg = (
        sample.tractor_mass_kg * spec.tractor_cg_height_m
        + sample.towed_mass_kg * spec.trailer_cg_height_m
    ) / max(sample.total_mass_kg, 1e-9)
    wheelbase = spec.tractor_wheelbase_m + n_tr * spec.hitch_to_axle_m
    load_shift = sample.total_mass_kg * G * h_cg / max(wheelbase, 1e-9)

    trailer_axle_load = float(axle_loads[2])
    return EffectiveParams(
        total_mass_kg=sample.total_mass_kg,
        axle_crr=crr_eff,
        drag_coef=cd_eff,
        frontal_area_m2=area,
        crr_by_axle=crr_by_axle,
        axle_loads_n=axle_loads,
        load_shift_n_per_sin=load_shift,
        trailer_axle_load_n=trailer_axle_load,
        cda_m2=cda,
    )


def correction_terms(
    sample,
    route,
    eff: EffectiveParams,
    trailer_cfg,
) -> CorrectionTerms:
    """Per-segment additive energy terms on top of FASTSim's own result."""
    n = len(route.grade)
    if trailer_cfg.fidelity == "lumped":
        return CorrectionTerms.zeros(n)

    spec = sample.spec
    grade = np.asarray(route.grade, dtype=float)
    delta_heading = np.asarray(route.delta_heading_rad, dtype=float)
    speed = np.asarray(route.target_speed_mps, dtype=float)
    delta_v = np.asarray(route.delta_v_mps, dtype=float)

    # --- articulation tyre scrub -------------------------------------------
    # A pin hitch dissipates nothing; the real low-speed loss is tyre scrub. The
    # trailer axles run at a slip angle during a turn and scrub laterally over a
    # distance proportional to the heading change and the hitch-to-axle length.
    #   E = f_scrub * N_trailer * L_axle * |dtheta|
    # f_scrub is the tyre/asphalt friction coefficient (same order as FASTSim's
    # own `wheel_fric_coef` default of 0.7); the model is an upper bound because
    # it assumes full sliding over the whole geometric deviation.
    scrub = trailer_cfg.scrub_friction_coef * eff.trailer_axle_load_n * spec.hitch_to_axle_m
    articulation = scrub * np.abs(delta_heading)

    # --- grade-dependent rolling-resistance redistribution ------------------
    load_transfer = eff.rolling_correction_j(grade, route.seg_len_m)

    # --- yaw-augmented drag of the articulated combination ------------------
    # dtheta changes the incidence angle of the air on the combination, the
    # crosswind-analogy argument of the paper; first-order increment on the
    # straight-line effective CdA.
    yaw_drag = (
        0.5
        * RHO_AIR
        * eff.cda_m2
        * trailer_cfg.yaw_drag_per_rad
        * np.abs(delta_heading)
        * speed**2
        * route.seg_len_m
    )

    # --- trailer wheel/axle rotational inertia ------------------------------
    # Translational inertia of the towed mass is already inside FASTSim (design
    # B); only the rotational inertia of the trailer wheels is missing, folded
    # in as an equivalent-mass increment applied to the kinetic-energy change.
    m_rot = trailer_cfg.rotational_inertia_fraction * sample.towed_mass_kg
    v_prev = speed - delta_v
    rotational = 0.5 * m_rot * (speed**2 - v_prev**2)

    return CorrectionTerms(
        articulation_j=articulation,
        load_transfer_j=load_transfer,
        yaw_drag_j=yaw_drag,
        rotational_inertia_j=rotational,
    )


def apply_corrections(
    fastsim_energy_j: np.ndarray,
    corrections: CorrectionTerms,
    trailer_cfg,
) -> tuple[np.ndarray, np.ndarray]:
    """Add the corrections to FASTSim's per-segment energy.

    Returns ``(total_j, applied_correction_j)``. A segment may
    only become negative through FASTSim's own regeneration path, so a correction
    that would flip a positive consumption into regeneration is clamped.
    """
    base = np.asarray(fastsim_energy_j, dtype=float)
    corr = np.asarray(corrections.total_j, dtype=float)
    total = base + corr
    if trailer_cfg.clamp_negative_corrections:
        flipped = (base > 0.0) & (total < 0.0)
        total = np.where(flipped, 0.0, total)
    return total, total - base
