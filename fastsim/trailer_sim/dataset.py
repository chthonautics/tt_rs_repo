"""Dataset assembly and writers.

Output layout for a run ``datasets/<run_name>/``:

- ``train.csv`` / ``val.csv`` -- long format, **one row per segment**, split by
  route (never by segment) and stratified by vehicle class and trailer count;
- ``manifest.json`` -- config snapshot, seeds, versions, counts, ranges, flags;
- ``routes_geometry.jsonl`` -- raw per-route geometry for the Unity phase;
- ``README.md`` -- how to reshape the long CSV into model samples.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from . import SCHEMA_VERSION, __version__
from .config import Config
from .noise import route_rng
from .routes import Route
from .vehicles import VehicleSample

#: fixed column order -- units per annex A1.3/A1.2 of the paper
CSV_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "segment_index",
    "tractor_mass_kg",
    "n_trailers",
    "towed_mass_kg",
    "axle_crr",
    "drag_coef",
    "frontal_area_m2",
    "grade",
    "seg_len_m",
    "target_speed_mps",
    "delta_v_mps",
    "delta_heading_rad",
    "energy_j",
    "energy_j_clean",
)

METADATA_COLUMNS: tuple[str, ...] = (
    "vehicle_class",
    "route_total_energy_j",
    "conditioning_headroom_j",
)

KNOWN_SIMPLIFICATIONS: tuple[str, ...] = (
    "no ambient wind",
    "no accessory/HVAC loads in the target column",
    "no battery temperature or SOC derating beyond FASTSim's defaults",
    "single road surface, constant Crr per axle position",
    "no brake-force distribution across axles",
    "FASTSim has no native trailer physics; trailers enter through effective parameters",
    "battery starts at max SOC, so regeneration is bounded by accumulated discharge",
    "stop-and-go is represented by a creep speed (fixed-length segments cannot dwell)",
    "FASTSim BEV efficiency maps are scaled from a pinned light-duty template",
)


@dataclass
class SampleRecord:
    """One route's worth of samples plus the metadata needed for the outputs."""

    sample_id: int
    sample: VehicleSample
    route: Route
    static_vector: list[float]
    energies_j: np.ndarray
    energies_clean_j: np.ndarray
    conditioning_headroom_j: float
    trace_miss_used: str | None
    correction_split: dict[str, float] = field(default_factory=dict)

    @property
    def route_total_j(self) -> float:
        return float(np.sum(self.energies_j))


def build_dataframe(records: list[SampleRecord], *, metadata: bool = False) -> pl.DataFrame:
    """Long-format table: one row per route segment."""
    columns: dict[str, list] = {name: [] for name in CSV_COLUMNS}
    if metadata:
        columns.update({name: [] for name in METADATA_COLUMNS})

    for rec in records:
        static = rec.static_vector
        n = rec.route.n_segments
        for i in range(n):
            columns["sample_id"].append(rec.sample_id)
            columns["segment_index"].append(i)
            columns["tractor_mass_kg"].append(float(static[0]))
            columns["n_trailers"].append(float(static[1]))
            columns["towed_mass_kg"].append(float(static[2]))
            columns["axle_crr"].append(float(static[3]))
            columns["drag_coef"].append(float(static[4]))
            columns["frontal_area_m2"].append(float(static[5]))
            columns["grade"].append(float(rec.route.grade[i]))
            columns["seg_len_m"].append(float(rec.route.seg_len_m))
            columns["target_speed_mps"].append(float(rec.route.target_speed_mps[i]))
            columns["delta_v_mps"].append(float(rec.route.delta_v_mps[i]))
            columns["delta_heading_rad"].append(float(rec.route.delta_heading_rad[i]))
            columns["energy_j"].append(float(rec.energies_j[i]))
            columns["energy_j_clean"].append(float(rec.energies_clean_j[i]))
            if metadata:
                columns["vehicle_class"].append(rec.sample.vehicle_class)
                columns["route_total_energy_j"].append(rec.route_total_j)
                columns["conditioning_headroom_j"].append(float(rec.conditioning_headroom_j))

    order = list(CSV_COLUMNS) + (list(METADATA_COLUMNS) if metadata else [])
    return pl.DataFrame({name: columns[name] for name in order})


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def split_records(
    records: list[SampleRecord], cfg: Config
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Split **by route**, stratified by vehicle class and trailer count."""
    keys: dict[tuple, list[SampleRecord]] = {}
    for rec in records:
        key = tuple(
            rec.sample.vehicle_class if k == "vehicle_class" else int(rec.sample.n_trailers)
            for k in cfg.output.stratify
        )
        keys.setdefault(key, []).append(rec)

    rng = route_rng(cfg.master_seed, 0xBEEF)
    train: list[SampleRecord] = []
    val: list[SampleRecord] = []
    for key in sorted(keys, key=str):
        group = sorted(keys[key], key=lambda r: r.sample_id)
        order = rng.permutation(len(group))
        n_train = int(round(cfg.output.train_fraction * len(group)))
        n_train = max(1, min(n_train, len(group) - 1)) if len(group) > 1 else len(group)
        for pos, idx in enumerate(order):
            (train if pos < n_train else val).append(group[idx])
    return sorted(train, key=lambda r: r.sample_id), sorted(val, key=lambda r: r.sample_id)


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
def write_geometry(path: Path, records: list[SampleRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            payload = rec.route.to_dict()
            payload["sample_id"] = rec.sample_id
            payload["vehicle"] = rec.sample.to_dict()
            payload["energies_j_clean"] = [float(x) for x in rec.energies_clean_j]
            handle.write(json.dumps(payload) + "\n")


def build_manifest(
    cfg: Config,
    records: list[SampleRecord],
    train: list[SampleRecord],
    val: list[SampleRecord],
    *,
    failures: list[dict],
    dataframe: pl.DataFrame,
) -> dict:
    import fastsim

    classes: dict[str, int] = {}
    trailers: dict[str, int] = {}
    conditioning: dict[str, int] = {}
    for rec in records:
        classes[rec.sample.vehicle_class] = classes.get(rec.sample.vehicle_class, 0) + 1
        key = str(int(rec.sample.n_trailers))
        trailers[key] = trailers.get(key, 0) + 1
        head = f"{rec.conditioning_headroom_j:.3e}"
        conditioning[head] = conditioning.get(head, 0) + 1

    def ranges() -> dict[str, dict[str, float]]:
        out = {}
        for column in CSV_COLUMNS[2:]:
            series = dataframe[column]
            out[column] = {"min": float(series.min()), "max": float(series.max())}
        return out

    snap = cfg.snapshot()
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": __version__,
        "run_name": cfg.run_name,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "master_seed": cfg.master_seed,
        "fastsim_version": fastsim.__version__,
        "python_version": platform.python_version(),
        "n_routes_requested": cfg.n_routes,
        "n_routes_written": len(records),
        "n_routes_failed": len(failures),
        "failures": failures[:50],
        "n_segments": int(dataframe.height),
        "splits": {
            "train_routes": len(train),
            "val_routes": len(val),
            "train_segments": int(dataframe.height - sum(r.route.n_segments for r in val)),
            "val_segments": int(sum(r.route.n_segments for r in val)),
            "stratify": list(cfg.output.stratify),
        },
        "class_histogram": classes,
        "class_weights_configured": cfg.vehicles.normalised_weights(),
        "trailer_count_histogram": trailers,
        "conditioning_headroom_histogram": conditioning,
        "trailer_fidelity": cfg.trailers.fidelity,
        "trailer_corrections": {
            "scrub_friction_coef": cfg.trailers.scrub_friction_coef,
            "rotational_inertia_fraction": cfg.trailers.rotational_inertia_fraction,
            "yaw_drag_per_rad": cfg.trailers.yaw_drag_per_rad,
        },
        "noise": snap["noise"],
        "sim": snap["sim"],
        "output": snap["output"],
        "conditioning": snap["conditioning"],
        "config": snap,
        "value_ranges": ranges(),
        "known_simplifications": list(KNOWN_SIMPLIFICATIONS),
    }


READ_ME = """# {run_name}

Synthetic tractor-trailer energy dataset generated by `trailer_sim` (schema {schema}).

## Files

- `train.csv`, `val.csv` -- long format, **one row per route segment**; split by
  route, stratified by vehicle class and trailer count.
- `manifest.json` -- configuration, seeds, counts, ranges and known simplifications.
- `routes_geometry.jsonl` -- raw per-route geometry (elevation, heading, speed) for
  re-simulation in the Unity validation phase.
- `README.md` -- this file.

## Columns

| # | Column | Unit | Role |
|---|---|---|---|
| 0 | `sample_id` | -- | route identifier (metadata) |
| 1 | `segment_index` | -- | 0..L-1 (metadata) |
| 2 | `tractor_mass_kg` | kg | static 0 |
| 3 | `n_trailers` | -- | static 1 |
| 4 | `towed_mass_kg` | kg | static 2 |
| 5 | `axle_crr` | -- | static 3 |
| 6 | `drag_coef` | -- | static 4 |
| 7 | `frontal_area_m2` | m^2 | static 5 |
| 8 | `grade` | -- | sequence 0 (dH/dd) |
| 9 | `seg_len_m` | m | sequence 1 |
| 10 | `target_speed_mps` | m/s | sequence 2 |
| 11 | `delta_v_mps` | m/s | sequence 3 |
| 12 | `delta_heading_rad` | rad | sequence 4 |
| 13 | `energy_j` | J | target (with label noise) |
| 14 | `energy_j_clean` | J | target without label noise |

`energy_j` is the **net energy consumption** of the segment: positive means energy
drawn from the traction battery, negative means energy returned by regeneration.

## Reshaping into model samples

```python
import polars as pl

frame = pl.read_csv("train.csv").sort(["sample_id", "segment_index"])
static = (
    frame.select(["sample_id", "tractor_mass_kg", "n_trailers", "towed_mass_kg",
                  "axle_crr", "drag_coef", "frontal_area_m2"])
    .unique(subset=["sample_id"], maintain_order=True)
)
sequence = ["grade", "seg_len_m", "target_speed_mps", "delta_v_mps", "delta_heading_rad"]

samples = []
for (sample_id,), group in frame.group_by(["sample_id"], maintain_order=True):
    vector = static.filter(pl.col("sample_id") == sample_id).drop("sample_id").row(0)
    features = group.select(sequence).to_numpy()          # (L, 5)
    target = group["energy_j"].to_numpy()                 # (L,)
    samples.append({{"sample_id": sample_id, "static": vector,
                    "sequence": features, "target": target}})
```

Normalisation (z-score) is deliberately **not** applied here: it is a
training-phase concern, and its statistics must be computed on the training split
only.
"""


def write_run(
    run_dir: Path,
    cfg: Config,
    records: list[SampleRecord],
    train: list[SampleRecord],
    val: list[SampleRecord],
    *,
    failures: list[dict],
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = cfg.output.keep_metadata_columns
    train_df = build_dataframe(train, metadata=metadata)
    val_df = build_dataframe(val, metadata=metadata)
    train_df.write_csv(run_dir / "train.csv")
    val_df.write_csv(run_dir / "val.csv")

    all_df = build_dataframe(records, metadata=False)
    manifest = build_manifest(cfg, records, train, val, failures=failures, dataframe=all_df)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_geometry(run_dir / "routes_geometry.jsonl", records)
    (run_dir / "README.md").write_text(
        READ_ME.format(run_name=cfg.run_name, schema=SCHEMA_VERSION), encoding="utf-8"
    )
    return manifest
