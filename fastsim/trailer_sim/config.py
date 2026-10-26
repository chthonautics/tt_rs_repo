"""Configuration dataclasses, YAML loading and validation.

Everything the generator does is driven by one YAML file (see ``configs/``).
Defaults live in this module so that a minimal config (or an empty file) still
produces a valid, documented run.
"""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is structurally invalid."""


Pair = tuple[float, float]


def _pair(value: Any, name: str) -> Pair:
    """Coerce a YAML value into an ordered ``(low, high)`` float pair."""
    if isinstance(value, (int, float)):
        lo = hi = float(value)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
    else:
        raise ConfigError(f"{name} must be a number or a [low, high] list, got {value!r}")
    if lo > hi:
        raise ConfigError(f"{name}: low bound {lo} > high bound {hi}")
    return (lo, hi)


def _build(cls: type, data: Mapping[str, Any] | None, path: str = ""):
    """Instantiate a (possibly nested) dataclass from a plain mapping."""
    data = dict(data or {})
    if not is_dataclass(cls):
        raise TypeError(cls)
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"unknown key(s) {sorted(unknown)} in '{path or cls.__name__}'")
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        if is_dataclass(f.type):
            kwargs[name] = _build(f.type, value, f"{path}{name}.")
        elif f.type == Pair:
            kwargs[name] = _pair(value, f"{path}{name}")
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _deep_merge(base: dict, override: Mapping[str, Any]) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------- #
# noise
# --------------------------------------------------------------------------- #
@dataclass
class InputNoiseConfig:
    """Multiplicative zero-mean Gaussian noise on physical input parameters."""

    enabled: bool = True
    default_sigma: float = 0.03
    #: per-parameter overrides of ``default_sigma`` (fractional standard deviation)
    per_parameter: dict[str, float] = field(default_factory=dict)

    def sigma_for(self, parameter: str) -> float:
        return float(self.per_parameter.get(parameter, self.default_sigma))

    @property
    def parameters(self) -> tuple[str, ...]:
        return ("tractor_mass_kg", "trailer_mass_kg", "axle_crr", "drag_coef", "frontal_area_m2")


@dataclass
class OutputNoiseConfig:
    """Multiplicative zero-mean Gaussian noise on the clean per-segment energy."""

    enabled: bool = True
    sigma: float = 0.015


@dataclass
class NoiseConfig:
    input: InputNoiseConfig = field(default_factory=InputNoiseConfig)
    output: OutputNoiseConfig = field(default_factory=OutputNoiseConfig)


# --------------------------------------------------------------------------- #
# vehicle classes
# --------------------------------------------------------------------------- #
@dataclass
class ClassConfig:
    """Everything that is sampled per vehicle class (one sample = one route)."""

    # mixture
    weight: float = 0.2
    n_trailers: list[int] = field(default_factory=lambda: [1])

    # static physical parameters (sampled ranges) -- paper A1.2 fields
    tractor_mass_kg: Pair = (1000.0, 2000.0)
    trailer_mass_kg: Pair = (300.0, 1000.0)  # per trailer; x n_trailers = towed mass
    axle_crr: Pair = (0.008, 0.015)
    drag_coef: Pair = (0.40, 0.50)
    frontal_area_m2: Pair = (2.0, 4.0)

    # route synthesis (one sample = one route)
    segments: list[int] = field(default_factory=lambda: [20, 60])
    seg_len_m: Pair = (5.0, 15.0)
    speed_mps: Pair = (1.0, 3.0)
    creep_speed_mps: float = 0.35
    grade_max: float = 0.08
    grade_amplitude: float = 0.02
    grade_correlation_segments: float = 4.0
    turn_rad: Pair = (0.1, 0.6)
    turn_probability: float = 0.30
    dwell_fraction: float = 0.10
    dwell_speed_frac: float = 0.20
    accel_max_mps2: float = 1.5
    #: maximum speed change between consecutive segment boundaries, as a fraction
    #: of the cruise speed (drivability constraint on the synthetic profile)
    max_rel_speed_change: float = 0.40
    net_elev_drop_max_m: float = 25.0

    # trailer simplified physics inputs
    n_trailer_axles: int = 2
    hitch_to_axle_m: float = 2.5
    trailer_track_m: float = 1.8
    trailer_cg_height_m: float = 0.9
    trailer_wheelbase_m: float = 3.0
    tongue_weight_frac: float = 0.15
    tractor_cg_height_m: float = 0.9
    tractor_wheelbase_m: float = 3.0
    axle_crr_by_position: list[float] = field(default_factory=lambda: [0.007, 0.008, 0.009])
    trailer_drag_coef: float = 0.55
    trailer_frontal_area_m2: Pair = (2.0, 4.0)
    #: fraction of a trailer's frontal area that acts as *new* drag area when the
    #: trailer runs in the tractor's wake (in-line combination)
    wake_factor: float = 0.15

    # powertrain scaling for the FASTSim BEV (synthetic classes are
    # built by re-parameterising a pinned light-duty BEV template)
    em_pwr_kw: Pair = (4.0, 8.0)
    battery_kwh: Pair = (4.0, 12.0)
    wheel_radius_m: float = 0.28
    wheel_inertia_kg_m2: float = 0.6

    # regen conditioning (workaround for fastsim's BEV "start at max SOC" behaviour)
    conditioning_headroom_j: float = 5.0e5


@dataclass
class VehiclesConfig:
    classes: dict[str, ClassConfig] = field(default_factory=dict)

    def normalised_weights(self) -> dict[str, float]:
        total = sum(c.weight for c in self.classes.values())
        if total <= 0:
            raise ConfigError("vehicle class weights must sum to a positive value")
        return {name: c.weight / total for name, c in self.classes.items()}


# --------------------------------------------------------------------------- #
# trailers / simulation / output
# --------------------------------------------------------------------------- #
@dataclass
class TrailerConfig:
    #: one of ``lumped``, ``quasi_static`` (``dynamic`` is a stretch goal, not implemented)
    fidelity: str = "quasi_static"
    #: tire scrub friction coefficient for the articulation loss
    scrub_friction_coef: float = 0.7
    #: fraction of the rotational inertia of trailer wheels/axles added to towed mass
    rotational_inertia_fraction: float = 0.04
    #: bound of the first-order yaw-drag increment, per radian of heading change
    yaw_drag_per_rad: float = 0.5
    #: clamp: correction terms may not turn a positive FASTSim segment into regen
    clamp_negative_corrections: bool = True


@dataclass
class ConditioningConfig:
    """Battery pre-conditioning: emulates leaving the depot with SOC < 100%.

    fastsim 3.1.0 initialises a BEV to ``max_soc`` and sets the charge-power
    buffer to zero, so the battery can only accept the energy it previously
    discharged. A short climb before the route window creates that
    headroom without touching the route's own per-segment energy.
    """

    enabled: bool = True
    #: fraction of the powertrain limit the conditioning climb may use
    power_frac: float = 0.5
    #: fallback ladder (multiples of each class' ``conditioning_headroom_j``)
    ladder: list[float] = field(default_factory=lambda: [1.0, 2.5, 5.0])
    max_hold_s: float = 1800.0
    min_grade: float = 0.01
    max_grade: float = 0.15


@dataclass
class SimConfig:
    dt_base_s: float = 1.0
    max_substeps_per_segment: int = 60
    trace_miss_opts: str = "Error"
    amb_temp_k: float = 293.15
    energy_column: str = "veh.pt_type.BEV.res.history.energy_out_prop_joules"
    crosscheck_column: str = "veh.pt_type.BEV.em.history.energy_elec_prop_in_joules"
    #: relative tolerance when reconciling the per-segment sum with the counter over the route window
    crosscheck_rtol: float = 1e-6
    #: attempt ``trace_miss_opts='Correct'`` before giving up on a route
    fallback_trace_correct: bool = True
    #: number of times a route is regenerated when the (route, vehicle) pair is
    #: physically infeasible for the powertrain
    max_route_retries: int = 4


@dataclass
class OutputConfig:
    dir: str = "../datasets"
    train_fraction: float = 0.8
    stratify: list[str] = field(default_factory=lambda: ["vehicle_class", "n_trailers"])
    keep_metadata_columns: bool = False


@dataclass
class Config:
    run_name: str = "smoke_v0"
    master_seed: int = 20261018
    n_routes: int = 200
    vehicles: VehiclesConfig = field(default_factory=VehiclesConfig)
    trailers: TrailerConfig = field(default_factory=TrailerConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        data = dict(data or {})
        if "vehicles" in data:
            veh = dict(data["vehicles"])
            classes = veh.pop("classes", None)
            data["vehicles"] = veh
        else:
            classes = None
        cfg = _build(cls, data)
        if classes is not None:
            cfg.vehicles.classes = {
                name: _build(ClassConfig, spec, f"vehicles.classes.{name}.")
                for name, spec in classes.items()
            }
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        if not isinstance(data, Mapping):
            raise ConfigError(f"{path}: top level must be a mapping")
        return cls.from_dict(data)

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        if not self.vehicles.classes:
            raise ConfigError("at least one vehicle class must be configured")
        for name, c in self.vehicles.classes.items():
            if not c.n_trailers:
                raise ConfigError(f"{name}: n_trailers must not be empty")
            if min(c.n_trailers) < 1:
                raise ConfigError(f"{name}: n_trailers must be >= 1")
            if c.creep_speed_mps <= 0:
                raise ConfigError(f"{name}: creep_speed_mps must be > 0")
            if c.speed_mps[0] <= c.creep_speed_mps:
                raise ConfigError(
                    f"{name}: speed_mps low bound must exceed creep_speed_mps"
                )
            if len(c.axle_crr_by_position) < 3:
                raise ConfigError(
                    f"{name}: axle_crr_by_position needs at least [steer, drive, trailer]"
                )
            if len(c.segments) != 2 or c.segments[0] < 1:
                raise ConfigError(f"{name}: segments must be [min, max] with min >= 1")
        if self.trailers.fidelity not in {"lumped", "quasi_static", "dynamic"}:
            raise ConfigError(f"unknown trailer fidelity '{self.trailers.fidelity}'")
        if self.trailers.fidelity == "dynamic":
            raise ConfigError(
                "fidelity 'dynamic' is a stretch goal and is not implemented in v0"
            )
        if not 0.0 < self.output.train_fraction < 1.0:
            raise ConfigError("output.train_fraction must be in (0, 1)")
        unknown_strata = set(self.output.stratify) - {"vehicle_class", "n_trailers"}
        if unknown_strata:
            raise ConfigError(
                f"output.stratify: unknown key(s) {sorted(unknown_strata)}; "
                "valid values are 'vehicle_class' and 'n_trailers'"
            )
        if self.sim.dt_base_s <= 0:
            raise ConfigError("sim.dt_base_s must be > 0")

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def snapshot(self) -> dict[str, Any]:
        """JSON/YAML-safe dict, plus the derived class weights."""
        data = self.to_dict()
        data["vehicles"]["normalised_weights"] = self.vehicles.normalised_weights()
        return data


#: directory holding the shipped vehicle-class presets (one YAML per class)
PRESETS_DIR = Path(__file__).parent / "presets"

#: built-in mixture: which presets are loaded (by default the three classes of
#: the proposal). Add a class = drop ``<name>.yaml`` in ``presets/`` and list
#: its name here (the sampling weight lives in the preset file).
DEFAULT_CLASS_PRESETS: tuple[str, ...] = ("agv", "mid", "long_haul")


class PresetError(ConfigError):
    """A shipped vehicle-class preset is missing, unreadable or malformed."""


def load_class_preset(name: str) -> ClassConfig:
    """Load one vehicle-class preset from ``presets/<name>.yaml``.

    The file only needs the keys it changes; everything else falls back to the
    :class:`ClassConfig` defaults, so a new class can start as a few lines.
    """
    path = PRESETS_DIR / f"{name}.yaml"
    if not path.exists():
        raise PresetError(f"vehicle-class preset not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - malformed file
        raise PresetError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, Mapping):
        raise PresetError(f"{path}: top level must be a mapping")
    return _build(ClassConfig, data, f"presets/{name}.")


def default_config() -> Config:
    """The built-in configuration: global defaults plus the shipped class presets.

    The vehicle-class mixture lives in ``trailer_sim/presets/<class>.yaml`` (one
    file per class, see ``presets/README.md``); those files are editable without
    touching Python. Everything else keeps the dataclass defaults below.
    """
    classes = {name: load_class_preset(name) for name in DEFAULT_CLASS_PRESETS}
    cfg = Config(vehicles=VehiclesConfig(classes=classes))
    cfg.validate()
    return cfg


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> Config:
    """Load a config from YAML, falling back to the built-in three-class mixture."""
    base = default_config().to_dict()
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{path}: top level must be a mapping")
        base = _deep_merge(base, raw)
    if overrides:
        base = _deep_merge(base, overrides)
    return Config.from_dict(base)
