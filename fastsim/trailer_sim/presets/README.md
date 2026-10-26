# Vehicle-class presets

One YAML file per vehicle class. `trailer_sim.config.default_config()` loads the
names listed in `DEFAULT_CLASS_PRESETS` (in `config.py`) from this directory, so
the built-in mixture can be edited without touching Python.

| File | Class | Role |
|---|---|---|
| `agv.yaml` | `agv` | warehouse / yard train: small, slow, 1–3 trailers |
| `mid.yaml` | `mid` | terminal / distribution: the most represented class (60 %) |
| `long_haul.yaml` | `long_haul` | Class 8 / EU 40 t: fast, heavy, single trailer only |

## Format

A preset is a flat mapping of `ClassConfig` fields (`config.py`). Every field is
optional — anything omitted falls back to the `ClassConfig` default, so a new
class can start as a handful of lines.

```yaml
weight: 0.3
n_trailers: [1, 2]
tractor_mass_kg: [8000.0, 12000.0]
speed_mps: [10.0, 20.0]
```

Conventions used by the shipped files:

- **`weight`** — sampling weight of the class. Values do not need to sum to 1;
  they are normalised (`VehiclesConfig.normalised_weights()`).
- **`n_trailers`** — allowed trailer counts, drawn uniformly. Multi-trailer is
  only meaningful for `agv`/`mid`; `long_haul` is single-trailer by decision.
- **`[low, high]` pairs** — sampled uniformly per route (masses, coefficients,
  areas, speeds, powers). Reversed bounds are rejected.
- **`segments`, `seg_len_m`, `speed_mps`** — route geometry per class; `seg_len_m`
  is fixed per route, which is why a segment cannot dwell at zero speed (the
  stop-and-go model uses `dwell_speed_frac` × cruise instead).
- **`em_pwr_kw` / `battery_kwh`** — FASTSim powertrain scaling. `em_pwr_kw` must
  be able to hold the class' speeds on the class' grades: FASTSim errors out
  instead of falling back to friction brakes, so an underpowered class makes many
  routes infeasible (they are regenerated, and counted in the manifest).

## Precedence

```
ClassConfig defaults (config.py)
    < preset file (this directory)
    < user YAML (configs/*.yaml)
    < CLI flags
```

`default_config()` = global dataclass defaults + these presets. `load_config(path)`
deep-merges a user YAML **on top of** that result, so a run config only overrides
what it needs.

## Adding or removing a class

1. Copy the closest existing preset to `<name>.yaml` and adjust it.
2. Add `<name>` to `DEFAULT_CLASS_PRESETS` in `config.py` (removing a class is the
   same edit in reverse).
3. Check the result:

   ```bash
   .venv/bin/python -m trailer_sim show-config | head -40
   .venv/bin/python -m pytest -q tests/test_config.py
   ```

Nothing is cached: the files are read on every `default_config()` call, which
also means the datasets' `manifest.json` snapshots the values actually used for
that run.
