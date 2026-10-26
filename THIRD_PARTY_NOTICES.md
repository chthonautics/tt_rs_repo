# Third-party notices

Everything in this repository is licensed under the MIT License (see
[`LICENSE`](LICENSE)). **No third-party code or data is vendored here** — all
third-party material is installed from its own distribution at install time and
keeps its own license.

## Tools used at build/run time

| Package | License | Role |
|---|---|---|
| `fastsim` | Apache-2.0 (© Alliance for Energy Innovation, LLC) | simulation engine (library dependency) |
| `numpy` | BSD-3-Clause | numerics |
| `polars` | MIT | dataset export |
| `pyyaml` | MIT | config parsing |
| `pandas` | BSD-3-Clause | dev extra only |
| `pytest` | MIT | dev extra only |
| `hypothesis` | MPL-2.0 | dev extra only |

`fastsim` is imported as a library and is not redistributed. Its
`trailer_sim/templates/bev_tractor_base.yaml` template is **generated locally on
first use** from the resource bundle embedded in the installed `fastsim` package
(`python -m trailer_sim pin-vehicles`) and is gitignored, so nothing derived from
FASTSim's Apache-2.0 assets is committed to this repository.

## Generated datasets

Datasets produced by `trailer-sim generate` (`datasets/`) are simulation output,
not part of the source distribution, and are not covered by the MIT license of
this code. Choose a data license (CC-BY-4.0 is the common choice) if you publish
them.
