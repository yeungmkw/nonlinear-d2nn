# Post-RS Evaluation Summary

> These numbers are produced by evaluating archived checkpoints with the current `rs_v1` program.
> They are post-RS program metrics, not post-RS retrained checkpoints.

## Phase-Only Re-Evaluation

| Task | Current program accuracy | Current program contrast |
|---|---:|---:|
| MNIST phase-only | 34.92% | -0.1221 |
| Fashion-MNIST phase-only | 69.32% | 0.1172 |

## Fashion-MNIST Nonlinear Re-Evaluation

| Configuration | Accuracy | Contrast |
|---|---:|---:|
| Phase-only baseline | 69.32% | 0.1172 |
| Incoherent back seed 42 | 43.89% | 0.0234 |
| Incoherent back seed 7 | 48.95% | 0.0814 |
| Incoherent back seed 123 | 51.02% | 0.0901 |

## Grayscale CIFAR-10 Nonlinear Re-Evaluation

| Configuration | seed=42 | seed=7 |
|---|---:|---:|
| Phase-only baseline acc | 26.05% | 24.79% |
| Phase-only baseline contrast | -0.2015 | -0.1945 |
| Incoherent back acc | 23.45% | 22.68% |
| Incoherent back contrast | -0.2550 | -0.2692 |

## RGB CIFAR-10 Nonlinear Re-Evaluation

| Configuration | Mean accuracy (3 seeds) | Mean contrast (3 seeds) |
|---|---:|---:|
| Phase-only baseline | 34.34% | -0.1196 |
| Incoherent back | 18.01% | -0.3495 |
