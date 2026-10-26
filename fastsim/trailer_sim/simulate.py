"""FASTSim 3 driver: cycle construction, ``SimDrive`` execution, extraction.

The route is turned into a FASTSim ``Cycle`` (speed/grade vs. time) whose
elements are aligned with route segments. Per-segment energy is obtained by
differencing a **net-energy counter** (a signed cumulative history column), so
no re-integration happens here: the extractor only re-buckets FASTSim's own
energy.

Two FASTSim 3.1.0 behaviours shape this module:

* a BEV is initialised at ``max_soc`` with a zero charge-power buffer, so the
  battery can only accept regeneration it previously discharged. A conditioning
  climb before the route window creates that headroom;
* the speed trace must be achievable from the very first element, hence the
  power-limited spool-up from standstill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .routes import Route
from .trailers import RHO_AIR
from .vehicles import VehicleSample


class SimulationError(RuntimeError):
    """FASTSim could not simulate the route (trace/powertrain solver failure)."""


@dataclass
class SimResult:
    energies_j: np.ndarray  # (L,) per-segment net energy consumption (FASTSim only)
    net_counter_j: np.ndarray  # (N,) running signed integral of power, one row per cycle element
    segment_slices: np.ndarray  # (L, 2) inclusive index range of each segment
    conditioning_elements: int
    conditioning_headroom_j: float
    trace_miss_used: str | None


# --------------------------------------------------------------------------- #
# cycle construction
# --------------------------------------------------------------------------- #
def _spool_up(
    sample: VehicleSample, v_target: float, dt: float
) -> tuple[list[float], list[float], list[float]]:
    """Power-limited acceleration from standstill to ``v_target`` on flat ground."""
    speeds: list[float] = []
    grades: list[float] = []
    dts: list[float] = []
    spec = sample.spec
    p_max = sample.em_pwr_kw * 1000.0
    mass = sample.total_mass_kg
    crr = sample.axle_crr
    cda = sample.drag_coef * sample.frontal_area_m2 + (
        sample.n_trailers * spec.wake_factor * spec.trailer_drag_coef * sample.trailer_frontal_area_m2
    )
    v = 0.0
    a = spec.accel_max_mps2
    while v < v_target - 1e-9:
        if v > 1e-6:
            # iterate on the post-step speed: the power peak of a ramp step occurs
            # at its end, not at its start
            for _ in range(3):
                v_next = min(v + a * dt, v_target)
                p_res = crr * mass * 9.80665 * v_next + 0.5 * RHO_AIR * cda * v_next**3
                a = min(spec.accel_max_mps2, max(0.05, (0.8 * p_max - p_res) / (mass * v_next)))
        v = min(v + a * dt, v_target)
        speeds.append(v)
        grades.append(0.0)
        dts.append(dt)
    return speeds, grades, dts


def _conditioning_hold(
    sample: VehicleSample, v: float, headroom_j: float, cfg: Config
) -> tuple[list[float], list[float], list[float]]:
    """Constant-speed climb that discharges ``headroom_j`` from the battery."""
    cond = cfg.conditioning
    if not cond.enabled or headroom_j <= 0:
        return [], [], []
    spec = sample.spec
    mass = sample.total_mass_kg
    p_max = sample.em_pwr_kw * 1000.0
    cda = sample.drag_coef * sample.frontal_area_m2
    p_res = sample.axle_crr * mass * 9.80665 * v + 0.5 * RHO_AIR * cda * v**3
    p_target = cond.power_frac * p_max
    p_grav = max(p_target - p_res, 0.0)
    grade = float(np.clip(p_grav / max(mass * 9.80665 * v, 1e-9), cond.min_grade, cond.max_grade))
    p_actual = p_grav + p_res
    if p_actual <= 0:
        return [], [], []
    hold_s = min(headroom_j / p_actual, cond.max_hold_s)
    dt = cfg.sim.dt_base_s
    n = max(0, int(np.ceil(hold_s / dt)))
    return [v] * n, [grade] * n, [dt] * n


def build_cycle(route: Route, sample: VehicleSample, cfg: Config, headroom_j: float = 0.0):
    """Build the FASTSim cycle and the route-segment index mapping.

    Returns ``(cycle, slices, conditioning_elements)`` where ``slices[i]`` is the
    inclusive ``(first, last)`` cycle-element index of route segment ``i``.
    """
    import fastsim

    dt_base = float(cfg.sim.dt_base_s)
    max_sub = int(cfg.sim.max_substeps_per_segment)

    # element 0: standstill
    speeds: list[float] = [0.0]
    grades: list[float] = [0.0]
    dts: list[float] = [dt_base]

    v0 = float(route.node_speed_mps[0])
    sp, gr, dt_ = _spool_up(sample, v0, dt_base)
    speeds += sp
    grades += gr
    dts += dt_
    sp, gr, dt_ = _conditioning_hold(sample, v0, headroom_j, cfg)
    speeds += sp
    grades += gr
    dts += dt_
    n_cond = len(speeds) - 1

    slices = np.zeros((route.n_segments, 2), dtype=int)
    for i in range(route.n_segments):
        v_a = float(route.node_speed_mps[i])
        v_b = float(route.node_speed_mps[i + 1])
        v_mean = max(0.5 * (v_a + v_b), 1e-9)
        dt_seg = route.seg_len_m / v_mean
        n_sub = int(np.clip(round(dt_seg / dt_base), 1, max_sub))
        dt_sub = dt_seg / n_sub
        ramp = np.linspace(v_a, v_b, n_sub + 1)[:-1]
        first = len(speeds)
        speeds += [float(v) for v in ramp]
        grades += [float(route.grade[i])] * n_sub
        dts += [dt_sub] * n_sub
        slices[i] = (first, len(speeds) - 1)

    speed = np.asarray(speeds, dtype=float)
    grade = np.asarray(grades, dtype=float)
    dt_arr = np.asarray(dts, dtype=float)
    time = np.concatenate([[0.0], np.cumsum(dt_arr[1:])])
    dist = np.concatenate([[0.0], np.cumsum((speed[1:] * dt_arr[1:]))])

    cycle = fastsim.Cycle.from_dict(
        {
            "init_elev_meters": 0.0,
            "time_seconds": time.tolist(),
            "speed_meters_per_second": speed.tolist(),
            "dist_meters": dist.tolist(),
            "grade": grade.tolist(),
            "pwr_max_chrg_watts": [1e9] * speed.size,
            "temp_amb_air_kelvin": [float(cfg.sim.amb_temp_k)] * speed.size,
            "pwr_solar_load_watts": [0.0] * speed.size,
        }
    )
    return cycle, slices, n_cond


# --------------------------------------------------------------------------- #
# simulation
# --------------------------------------------------------------------------- #
def _sim_params(cfg: Config, trace_miss_opts: str):
    import fastsim

    base = fastsim.SimParams.default().to_dict()
    base["trace_miss_opts"] = trace_miss_opts
    return fastsim.SimParams.from_dict(base, skip_init=True)


def _run(vehicle, cycle, cfg: Config, trace_miss_opts: str):
    import fastsim

    sim_drive = fastsim.SimDrive(vehicle, cycle, _sim_params(cfg, trace_miss_opts))
    sim_drive.run()
    return sim_drive.to_dataframe(backend="polars")


def _guard(exc: BaseException) -> SimulationError:
    """Normalise any fastsim failure (including Rust panics) into SimulationError.

    fastsim raises ``pyo3_runtime.PanicException`` for some solver failures, which
    derives from ``BaseException`` rather than ``Exception``.
    """
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        raise exc
    return SimulationError(f"{type(exc).__name__}: {exc}"[:500])


def simulate_route(
    route: Route,
    sample: VehicleSample,
    vehicle,
    cfg: Config,
    headroom_j: float = 0.0,
) -> SimResult:
    """Run FASTSim on one route and extract per-segment energy."""
    cycle, slices, n_cond = build_cycle(route, sample, cfg, headroom_j=headroom_j)
    trace_miss_used: str | None = None
    try:
        df = _run(vehicle, cycle, cfg, cfg.sim.trace_miss_opts)
    except BaseException as exc:  # noqa: BLE001 - see _guard
        if not cfg.sim.fallback_trace_correct:
            raise _guard(exc) from exc
        trace_miss_used = "Correct"
        try:
            df = _run(vehicle, cycle, cfg, "Correct")
        except BaseException as exc2:  # noqa: BLE001
            raise _guard(exc2) from exc2

    if df.height < len(cycle.to_dict()["time_seconds"]):
        raise SimulationError(f"partial simulation: {df.height} of {len(cycle.to_dict()['time_seconds'])} steps")

    # the history column is a net-energy counter: its increments are signed, so
    # differencing it inside the route window yields the per-segment consumption
    counter = np.asarray(df[cfg.sim.energy_column].to_numpy(), dtype=float)
    energies = np.array(
        [counter[last] - counter[first - 1] for first, last in slices], dtype=float
    )
    _crosscheck(df, counter, slices, energies, cfg)

    return SimResult(
        energies_j=energies,
        net_counter_j=counter,
        segment_slices=slices,
        conditioning_elements=n_cond,
        conditioning_headroom_j=float(headroom_j),
        trace_miss_used=trace_miss_used,
    )


def _crosscheck(df, counter, slices, energies, cfg: Config) -> None:
    """The extractor only re-buckets FASTSim's energy; verify that invariant."""
    first = int(slices[0, 0])
    last = int(slices[-1, 1])
    route_total = counter[last] - counter[first - 1]
    if not np.isclose(energies.sum(), route_total, rtol=cfg.sim.crosscheck_rtol, atol=1e-6):
        raise SimulationError(
            f"per-segment sum {energies.sum()} != counter difference over the route window {route_total}"
        )
    other = cfg.sim.crosscheck_column
    if other in df.columns:
        alt = np.asarray(df[other].to_numpy(), dtype=float)
        alt_total = alt[last] - alt[first - 1]
        if not np.isclose(alt_total, route_total, rtol=1e-3, atol=1e-3):
            raise SimulationError(
                f"energy column cross-check failed: {cfg.sim.energy_column}={route_total} "
                f"vs {other}={alt_total}"
            )
