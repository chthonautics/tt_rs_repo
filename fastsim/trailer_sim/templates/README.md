# FASTSim vehicle templates used by `trailer_sim`
#
# `bev_tractor_base.yaml` is not committed: it is generated on first use (or with
# `python -m trailer_sim pin-vehicles`) from the light-duty BEV resource bundle
# embedded in the installed `fastsim` package, so no network access and no remote
# vehicle database are involved.
#
# The synthetic vehicle classes in `vehicles.py` re-parameterise that template
# (mass, aerodynamics, rolling resistance, machine power, battery size). Deleting
# the generated file is safe — the next simulation recreates it byte-identically
# for a given `fastsim` version.
#
# Regenerate with:  python -m trailer_sim pin-vehicles
