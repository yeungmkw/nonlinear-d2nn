# Download Audit 2026-04-07

This note records a read-only audit of D2NN-related files under `<downloads-root>`.

No files were deleted during this audit.

## Scope

- Current official fabrication line: `docs/official-artifacts/fmnist5-phaseonly-aligned/`
- Current aligned local export root: `exports/fmnist5-phaseonly-aligned/best_fashion_mnist.fmnist5-phaseonly-aligned/`
- Older pre-nonlinear export still present in the main repo: `exports/best_fashion_mnist/`
- Independent audit: reviewed by a separate agent before any delete recommendations were written

## Main Finding

The loose files in `<downloads-root>\best_fashion_mnist_phase_masks.npy` and `<downloads-root>\best_fashion_mnist_phase_csv\` are not part of the current `fmnist5-phaseonly-aligned` line. They are exact duplicates of the older pre-nonlinear Fashion-MNIST export already stored at `exports/best_fashion_mnist/`, and they are also covered by the release archive `<downloads-root>\d2nn-release-assets\pre-nonlinear-phase-only-v1\pre-nonlinear-phase-only-local-phase-assets.zip`.

## Covered By Existing Repo Or Archive

These items are already represented elsewhere and do not need to stay as loose files in `<downloads-root>`.

- `<downloads-root>\best_fashion_mnist_phase_masks.npy`
  exact SHA-256 match to `exports/best_fashion_mnist/phase_masks.npy`
- `<downloads-root>\best_fashion_mnist_phase_csv\layer_01_phase_rad.csv`
  exact SHA-256 match to `exports/best_fashion_mnist/layers/layer_01_phase_rad.csv`
- `<downloads-root>\best_fashion_mnist_phase_csv\layer_02_phase_rad.csv`
  exact SHA-256 match to `exports/best_fashion_mnist/layers/layer_02_phase_rad.csv`
- `<downloads-root>\best_fashion_mnist_phase_csv\layer_03_phase_rad.csv`
  exact SHA-256 match to `exports/best_fashion_mnist/layers/layer_03_phase_rad.csv`
- `<downloads-root>\best_fashion_mnist_phase_csv\layer_04_phase_rad.csv`
  exact SHA-256 match to `exports/best_fashion_mnist/layers/layer_04_phase_rad.csv`
- `<downloads-root>\best_fashion_mnist_phase_csv\layer_05_phase_rad.csv`
  exact SHA-256 match to `exports/best_fashion_mnist/layers/layer_05_phase_rad.csv`

Conditional duplicate:

- `<downloads-root>\d2nn-release-assets\nonlinear-incoherent-back-fmnist-v1\`
  this extracted directory is redundant if `<downloads-root>\d2nn-release-assets\nonlinear-incoherent-back-fmnist-v1.zip` is kept, because the extracted `*.pth`, `*.json`, and `results-summary.txt` are working copies of the archive contents

## Keep As Release Or Archive Assets

These are packaging or archive files that should be kept unless they are moved into a more deliberate archive location first.

- `<downloads-root>\d2nn-release-assets\nonlinear-incoherent-back-fmnist-v1.zip`
- `<downloads-root>\d2nn-release-assets\nonlinear-incoherent-back-fmnist-v1\release-notes.md`
- `<downloads-root>\d2nn-release-assets\pre-nonlinear-phase-only-v1\pre-nonlinear-phase-only-checkpoints.zip`
- `<downloads-root>\d2nn-release-assets\pre-nonlinear-phase-only-v1\pre-nonlinear-phase-only-fashion-export.zip`
- `<downloads-root>\d2nn-release-assets\pre-nonlinear-phase-only-v1\pre-nonlinear-phase-only-local-phase-assets.zip`
- `<downloads-root>\d2nn-release-assets\pre-nonlinear-phase-only-v1\release-assets-manifest.txt`

## Unresolved: Do Not Delete Yet

These items are related, but the current repo and release archive layout does not give a clean one-to-one replacement yet.

- `<downloads-root>\best_fashion_mnist_phase_bmp\`
  likely old preview bitmaps derived from the pre-nonlinear phase masks, but the BMP files themselves are not tracked in the repo
- `<downloads-root>\D2NN_改进版非线性方案_面向Agent.md`
  topic overlap exists, but no clearly identical tracked copy was confirmed during the audit

## Practical Cleanup Guidance

If you want the lowest-risk cleanup order later:

1. keep all zip archives and release notes
2. remove the loose duplicated old phase files in `best_fashion_mnist_phase_masks.npy` and `best_fashion_mnist_phase_csv\`
3. only remove the extracted `nonlinear-incoherent-back-fmnist-v1\` directory if the `.zip` remains in place
4. leave the BMP previews and standalone nonlinear note alone until you decide whether you still use them

## Naming And Lookup Note

For the current fabrication-target line, search the repository by `fmnist5-phaseonly-aligned` first.

That short name now maps together:

- checkpoint and manifest
- understanding-report figures
- local aligned export package
- repo-tracked official artifact bundle
