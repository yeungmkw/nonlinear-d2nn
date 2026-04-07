# Official Artifacts

This directory is for the small set of artifact bundles that should remain easy to find inside the repository.

## What Goes Here

Only current official artifacts that are expected to be reused directly:

- the active fabrication-target phase package
- the minimum metadata needed to trace how it was generated
- supporting notes that tell someone what the artifact is and where it fits

Historical exports, large STL sets, and ad hoc local runs should stay in local ignored directories or release archives unless they become the current official reference.

## Naming Rule

Use short, readable names that describe the task line without becoming hard to scan.

Current rule:

- dataset first
- model/role second
- special status last

Current official fabrication line:

- `fmnist5-phaseonly-aligned`

This means:

- `fmnist5`: Fashion-MNIST, 5-layer line
- `phaseonly`: phase-only baseline rather than nonlinear
- `aligned`: the physics-aligned line used after the current propagation-path fixes

## Current Official Entry

- `docs/official-artifacts/fmnist5-phaseonly-aligned/`
- `docs/official-artifacts/download-audit-2026-04-07.md`

Use that directory first when someone asks for the currently organized, directly reusable phase package.
