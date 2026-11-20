# eos-ml

This repository is a small synthetic-data demonstration of neutron-star
equation-of-state inference. It samples a three-segment cold polytrope, solves
the TOV structure and quadrupolar tidal equations, then trains an inverse MLP
and a conditional normalizing flow.

## Run

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe main.py
```

The demo creates a small synthetic dataset and plots an inferred mass-radius
curve. The non-interactive test command is:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Model conventions

- Density is in g/cm^3, pressure in dyn/cm^2, energy density in erg/cm^3.
- Reported stellar mass is in solar masses, radius in km, and tidal
  deformability is dimensionless.
- Parameter ranges are read from `data/priors_config.json`; accepted synthetic
  examples must pass monotonicity and causality checks and support a stable
  branch reaching 2 solar masses.

## Scope and limitations

This is a numerical/software prototype, not a research-ready neutron-star
inference package. The polytrope extrapolates to zero density and does not
include a calibrated crust or phase-transition treatment. The priors are
illustrative, and the synthetic observations are not a likelihood for the
included NICER or gravitational-wave data. Scientific use requires validating
the EoS, solver, tidal implementation, and priors against published benchmark
solutions and observational analyses.
