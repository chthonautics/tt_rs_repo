"""Command line interface: ``python -m trailer_sim <command>``.

Commands
--------
``generate``      build a dataset (the main entry point)
``pin-vehicles``  regenerate the FASTSim BEV template from the installed library
``show-config``   print the effective configuration (defaults + file + overrides)
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml

from . import __version__
from .config import Config, load_config
from .dataset import SampleRecord, split_records, write_run
from .noise import add_output_noise, perturb_sample, sample_rng
from .routes import synthesize_route
from .simulate import SimulationError, simulate_route
from .trailers import apply_corrections, correction_terms, effective_params
from .vehicles import build_vehicle, pin_template, sample_vehicle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trailer-sim", description=__doc__)
    parser.add_argument("--version", action="version", version=f"trailer-sim {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a dataset")
    gen.add_argument("--config", type=Path, default=None, help="YAML configuration file")
    gen.add_argument("--out", type=Path, default=None, help="output directory (default: <out.config.dir>/<run_name>)")
    gen.add_argument("--routes", type=int, default=None, help="override the number of routes")
    gen.add_argument("--seed", type=int, default=None, help="override the master seed")
    gen.add_argument("--run-name", default=None, help="override the run name")
    gen.add_argument("--fidelity", choices=["lumped", "quasi_static"], default=None)
    gen.add_argument("--keep-metadata", action="store_true", help="add metadata columns to the CSVs")
    gen.add_argument("--quiet", action="store_true")
    gen.add_argument("--limit-failures", type=int, default=25, help="maximum tolerated failed routes")

    pin = sub.add_parser("pin-vehicles", help="regenerate the BEV template")
    pin.add_argument("--dest", type=Path, default=None)

    show = sub.add_parser("show-config", help="print the effective configuration")
    show.add_argument("--config", type=Path, default=None)
    return parser


# --------------------------------------------------------------------------- #
def _config_from_args(args) -> Config:
    overrides: dict = {}
    if args.routes is not None:
        overrides["n_routes"] = int(args.routes)
    if args.seed is not None:
        overrides["master_seed"] = int(args.seed)
    if args.run_name is not None:
        overrides["run_name"] = args.run_name
    if getattr(args, "fidelity", None):
        overrides["trailers"] = {"fidelity": args.fidelity}
    if getattr(args, "keep_metadata", False):
        overrides["output"] = {"keep_metadata_columns": True}
    return load_config(args.config, overrides)


def generate(args) -> int:
    cfg = _config_from_args(args)
    out_dir = Path(args.out) if args.out else Path(cfg.output.dir) / cfg.run_name
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()

    log = (lambda *a: None) if args.quiet else (lambda *a: print(*a, file=sys.stderr))
    log(f"[trailer-sim] run '{cfg.run_name}': {cfg.n_routes} routes -> {out_dir}")
    log(f"[trailer-sim] fidelity={cfg.trailers.fidelity} seed={cfg.master_seed}")

    records: list[SampleRecord] = []
    failures: list[dict] = []

    for index in range(cfg.n_routes):
        geo_rng = sample_rng(cfg.master_seed, index, stream=1)
        noise_rng = sample_rng(cfg.master_seed, index, stream=2)
        label_rng = sample_rng(cfg.master_seed, index, stream=3)

        nominal = sample_vehicle(geo_rng, cfg)
        # features come from the *nominal* parameters; the simulation runs on the
        # perturbed ones
        eff_nominal = effective_params(nominal, cfg.trailers)
        static_vector = nominal.static_vector(eff_nominal)

        result = None
        route = None
        sim_sample = None
        corrections = None
        last_error: Exception | None = None
        attempts = 0
        for attempt in range(cfg.sim.max_route_retries + 1):
            attempts = attempt + 1
            route = synthesize_route(geo_rng, nominal.spec, sample=nominal)
            route.vehicle_class = nominal.vehicle_class
            route.n_trailers = nominal.n_trailers

            sim_sample = perturb_sample(noise_rng, nominal, cfg)
            eff_sim = effective_params(sim_sample, cfg.trailers)
            corrections = correction_terms(sim_sample, route, eff_sim, cfg.trailers)

            for multiplier in cfg.conditioning.ladder:
                headroom = multiplier * nominal.spec.conditioning_headroom_j
                # a fresh vehicle per attempt: fastsim carries battery state between runs
                vehicle = build_vehicle(sim_sample, eff_sim)
                try:
                    result = simulate_route(route, sim_sample, vehicle, cfg, headroom_j=headroom)
                    sim_sample.conditioning_used = float(multiplier) if headroom > 0 else 0.0
                    break
                except SimulationError as exc:
                    last_error = exc
            if result is not None:
                break
            log(
                f"[trailer-sim] route {index}: route {attempts} infeasible, regenerating"
            )

        if result is None:
            failures.append(
                {
                    "sample_id": index,
                    "vehicle_class": nominal.vehicle_class,
                    "n_trailers": int(nominal.n_trailers),
                    "n_segments": route.n_segments if route is not None else 0,
                    "attempts": attempts,
                    "error": str(last_error)[:400],
                }
            )
            log(f"[trailer-sim] route {index}: FAILED after {attempts} route attempts")
            continue

        clean, applied = apply_corrections(result.energies_j, corrections, cfg.trailers)
        noisy = add_output_noise(label_rng, clean, cfg)

        records.append(
            SampleRecord(
                sample_id=index,
                sample=sim_sample,
                route=route,
                static_vector=static_vector,
                energies_j=noisy,
                energies_clean_j=clean,
                conditioning_headroom_j=result.conditioning_headroom_j,
                trace_miss_used=result.trace_miss_used,
                correction_split={
                    "articulation_j": float(np.sum(corrections.articulation_j)),
                    "load_transfer_j": float(np.sum(corrections.load_transfer_j)),
                    "yaw_drag_j": float(np.sum(corrections.yaw_drag_j)),
                    "rotational_inertia_j": float(np.sum(corrections.rotational_inertia_j)),
                    "applied_j": float(np.sum(applied)),
                },
            )
        )

        if not args.quiet and (index + 1) % max(1, cfg.n_routes // 10) == 0:
            log(f"[trailer-sim] {index + 1}/{cfg.n_routes} routes ({len(failures)} failed)")

    if not records:
        print("[trailer-sim] no routes could be simulated; nothing written", file=sys.stderr)
        return 1
    if len(failures) > args.limit_failures:
        print(
            f"[trailer-sim] {len(failures)} routes failed (> limit {args.limit_failures}); "
            "inspect the log before using this dataset",
            file=sys.stderr,
        )

    train, val = split_records(records, cfg)
    manifest = write_run(out_dir, cfg, records, train, val, failures=failures)
    log(
        f"[trailer-sim] wrote {out_dir}: {manifest['n_routes_written']} routes, "
        f"{manifest['n_segments']} segments, {manifest['n_routes_failed']} failures"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            return generate(args)
        if args.command == "pin-vehicles":
            dest = pin_template(args.dest)
            print(dest)
            return 0
        if args.command == "show-config":
            cfg = load_config(args.config)
            print(yaml.safe_dump(cfg.snapshot(), sort_keys=False))
            return 0
    except Exception:  # noqa: BLE001 - CLI boundary
        traceback.print_exc()
        return 2
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
