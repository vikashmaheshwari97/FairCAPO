# HEAL-CAPO starter scaffold

HEAL-CAPO: Self-Healing Trust-Aware Prompt Portfolio Optimization.

This scaffold is intentionally framework-light. It is designed to plug into Promptolution/CAPO later while letting us implement and test the new research components first:

- risk-aware multi-objective scoring
- semantic drift guard
- dynamic prompt routing
- verifier-guided repair
- continual Pareto portfolio update

## Suggested setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Minimal run

```bash
python scripts/run_toy_heal_capo.py --config configs/toy.yaml
```

## Main modules

- `heal_capo/core.py`: prompt candidate, evaluation result, portfolio abstractions
- `heal_capo/objectives.py`: performance, cost, risk, drift objective interfaces
- `heal_capo/pareto.py`: Pareto dominance and archive utilities
- `heal_capo/optimizers/risk_aware_mo_capo.py`: initial risk-aware MO-CAPO optimizer skeleton
- `heal_capo/components/router.py`: dynamic routing policy skeleton
- `heal_capo/components/verifier.py`: verifier interface and simple examples
- `heal_capo/components/drift_guard.py`: semantic drift guard interface
- `heal_capo/components/repair.py`: verifier-guided prompt repair skeleton
- `heal_capo/components/failure_memory.py`: continual failure collection and clustering hooks
- `heal_capo/evaluation/metrics.py`: portfolio and deployment metrics
