# Automated tests

The test suite focuses on deterministic unit-level behaviour so it runs quickly
without retraining the full production Gradient Boosting pipeline.

Covered areas:

- RUL target creation and capping
- operational risk boundaries
- maintenance recommendations
- HIGH-risk threshold optimisation
- maintenance classification metrics
- financial break-even logic
- cost sensitivity analysis
- latest engine-state selection
- raw-versus-clipped evaluation behaviour
- probability-model evaluation
- model factory behaviour
- engine-safe out-of-fold separation

## Install test dependencies

```powershell
pip install -r requirements-dev.txt
```

## Run all tests

From the repository root:

```powershell
pytest -q
```

## Run with more detail

```powershell
pytest -v
```

These tests deliberately avoid asserting the full NASA model's exact MAE on
every run. The full pipeline validation remains a slower end-to-end check.
