"""Vehicle-class sampling and FASTSim ``Vehicle`` assembly.

One *sample* is one route: a vehicle-class draw (weighted mixture) plus the
physical parameters of the combination. The FASTSim vehicle
is assembled from a **pinned BEV template generated from the resource bundle of
the installed ``fastsim`` package** so that generation is reproducible offline
(the remote vehicle database needs the ``web`` feature and is not pinned). The
template is written on first use and is not committed; see
``templates/README.md``.

The template is a light-duty BEV; the synthetic classes re-parameterise it
(mass, aerodynamics, rolling resistance, electric-machine power, battery
capacity). The efficiency maps are scale-invariant (they are normalised by
``pwr_out_max_watts``), so scaling keeps them valid; this remains an
approximation for the synthetic classes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .config import ClassConfig, Config
from .trailers import EffectiveParams

TEMPLATE_NAME = "bev_tractor_base.yaml"
TEMPLATE_SOURCE = "2022 Tesla Model 3 RWD thrml.yaml"
TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass
class VehicleSample:
    """One sampled vehicle, before and after input-parameter noise."""

    vehicle_class: str
    spec: ClassConfig
    n_trailers: int
    tractor_mass_kg: float
    trailer_mass_kg: float  # per trailer
    axle_crr: float  # nominal combination coefficient (paper A1.2 field 3)
    drag_coef: float
    frontal_area_m2: float
    trailer_frontal_area_m2: float
    em_pwr_kw: float
    battery_kwh: float
    perturbed: bool = False
    #: multiplier of the class' conditioning headroom actually used (0 = none)
    conditioning_used: float = 0.0

    # ------------------------------------------------------------------ #
    @property
    def towed_mass_kg(self) -> float:
        return self.trailer_mass_kg * self.n_trailers

    @property
    def total_mass_kg(self) -> float:
        return self.tractor_mass_kg + self.towed_mass_kg

    def static_vector(self, eff: EffectiveParams) -> list[float]:
        """Annex A1.2 vector: [tractor mass, n trailers, towed mass, Crr, Cd, area]."""
        return [
            float(self.tractor_mass_kg),
            float(self.n_trailers),
            float(self.towed_mass_kg),
            float(eff.axle_crr),
            float(eff.drag_coef),
            float(eff.frontal_area_m2),
        ]

    def to_dict(self) -> dict:
        spec = self.spec
        return {
            "vehicle_class": self.vehicle_class,
            "n_trailers": int(self.n_trailers),
            "tractor_mass_kg": float(self.tractor_mass_kg),
            "trailer_mass_kg": float(self.trailer_mass_kg),
            "towed_mass_kg": float(self.towed_mass_kg),
            "total_mass_kg": float(self.total_mass_kg),
            "axle_crr": float(self.axle_crr),
            "drag_coef": float(self.drag_coef),
            "frontal_area_m2": float(self.frontal_area_m2),
            "trailer_frontal_area_m2": float(self.trailer_frontal_area_m2),
            "em_pwr_kw": float(self.em_pwr_kw),
            "battery_kwh": float(self.battery_kwh),
            "perturbed": bool(self.perturbed),
            "conditioning_used": float(self.conditioning_used),
            "axle_crr_by_position": list(spec.axle_crr_by_position),
            "n_trailer_axles": int(spec.n_trailer_axles),
            "hitch_to_axle_m": float(spec.hitch_to_axle_m),
            "tractor_wheelbase_m": float(spec.tractor_wheelbase_m),
            "tractor_cg_height_m": float(spec.tractor_cg_height_m),
            "trailer_cg_height_m": float(spec.trailer_cg_height_m),
            "trailer_track_m": float(spec.trailer_track_m),
            "tongue_weight_frac": float(spec.tongue_weight_frac),
        }


# --------------------------------------------------------------------------- #
# class sampling
# --------------------------------------------------------------------------- #
def sample_vehicle(rng: np.random.Generator, cfg: Config, force_class: str | None = None) -> VehicleSample:
    """Draw one vehicle from the weighted class mixture."""
    weights = cfg.vehicles.normalised_weights()
    names = list(weights)
    if force_class is not None:
        if force_class not in cfg.vehicles.classes:
            raise KeyError(f"unknown vehicle class '{force_class}'")
        name = force_class
    else:
        name = str(rng.choice(names, p=[weights[n] for n in names]))
    spec = cfg.vehicles.classes[name]

    return VehicleSample(
        vehicle_class=name,
        spec=spec,
        n_trailers=int(rng.choice(spec.n_trailers)),
        tractor_mass_kg=float(rng.uniform(*spec.tractor_mass_kg)),
        trailer_mass_kg=float(rng.uniform(*spec.trailer_mass_kg)),
        axle_crr=float(rng.uniform(*spec.axle_crr)),
        drag_coef=float(rng.uniform(*spec.drag_coef)),
        frontal_area_m2=float(rng.uniform(*spec.frontal_area_m2)),
        trailer_frontal_area_m2=float(rng.uniform(*spec.trailer_frontal_area_m2)),
        em_pwr_kw=float(rng.uniform(*spec.em_pwr_kw)),
        battery_kwh=float(rng.uniform(*spec.battery_kwh)),
    )


# --------------------------------------------------------------------------- #
# FASTSim vehicle assembly
# --------------------------------------------------------------------------- #
def pin_template(dest: Path | None = None) -> Path:
    """Regenerate the BEV template from the installed fastsim resources."""
    import fastsim

    veh = fastsim.Vehicle.from_resource(TEMPLATE_SOURCE)
    dest = dest or (TEMPLATES_DIR / TEMPLATE_NAME)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(veh.to_yaml(), encoding="utf-8")
    return dest


def load_template() -> dict:
    path = TEMPLATES_DIR / TEMPLATE_NAME
    if not path.exists():  # pragma: no cover - generated on first use
        pin_template(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{path}: unreadable vehicle template")
    return data


def build_vehicle(sample: VehicleSample, eff: EffectiveParams):
    """Assemble the FASTSim vehicle for the *effective combination* parameters."""
    import fastsim

    data = copy.deepcopy(load_template())
    chassis = data["chassis"]

    total_mass = eff.total_mass_kg
    data["mass_kilograms"] = total_mass
    chassis["mass_kilograms"] = total_mass
    chassis["drag_coef"] = eff.drag_coef
    chassis["frontal_area_square_meters"] = eff.frontal_area_m2
    chassis["wheel_rr_coef"] = eff.axle_crr
    chassis["wheel_radius_meters"] = sample.spec.wheel_radius_m
    chassis["wheel_inertia_kilogram_square_meters"] = sample.spec.wheel_inertia_kg_m2
    # wheels: tractor axle count plus one axle per trailer axle
    axles = 2 + sample.n_trailers * sample.spec.n_trailer_axles
    chassis["num_wheels"] = int(2 * axles)

    pwr_w = sample.em_pwr_kw * 1000.0
    bev = data["pt_type"]["BEV"]
    bev["em"]["pwr_out_max_watts"] = pwr_w
    # keep the battery from being the power bottleneck; capacity stays realistic
    bev["res"]["pwr_out_max_watts"] = pwr_w * 1.2
    bev["res"]["energy_capacity_joules"] = sample.battery_kwh * 3.6e6
    bev["res"]["min_soc"] = 0.05
    bev["res"]["max_soc"] = 0.98

    return fastsim.Vehicle.from_dict(data)
