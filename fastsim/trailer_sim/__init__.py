"""trailer_sim -- tractor-trailer synthetic data generation for the energy model.

Implements the methodology phase "construcción de una simulación para entrenar al
modelo" of the research proposal:

- ``routes``    synthetic route synthesis (geometry, speed profile, derived features)
- ``vehicles``  weighted vehicle-class sampling + FASTSim vehicle assembly
- ``trailers``  simplified trailer physics (``lumped`` / ``quasi_static``)
- ``simulate``  FASTSim 3 driver: cycle building, SimDrive, per-segment extraction
- ``noise``     input-parameter noise and output-label noise (seeded)
- ``dataset``   CSV/manifest/geometry writers and train/val splits
- ``cli``       ``python -m trailer_sim generate --config ... --out ...``

The package is intentionally *not* named ``fastsim`` so it does not shadow the
``fastsim`` library it wraps.
"""

__version__ = "0.1.0"

SCHEMA_VERSION = "1.0"

__all__ = ["__version__", "SCHEMA_VERSION"]
