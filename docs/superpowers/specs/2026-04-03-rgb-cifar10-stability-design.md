# RGB CIFAR-10 Stability Confirmation Design

**Goal:** Extend the current RGB CIFAR-10 nonlinear stage from a 2-seed frozen checkpoint to a 5-seed stability confirmation without changing the mechanism, placement, training budget, or optics configuration.

**Scope:** This design only covers the next evidence-building step already implied by the Obsidian reproduction notes: keep the current RGB CIFAR-10 setup fixed and add more seeds so the current `phase-only baseline` versus `incoherent_intensity + back` comparison becomes statistically harder to dismiss as a two-seed coincidence.

## Route Alignment

This design stays aligned with the current Obsidian notes:

- The leading nonlinear line remains `incoherent_intensity + back`.
- The stage objective remains `task-complexity transfer`, not a new mechanism or device-realism phase.
- The immediate question is no longer whether RGB transfer exists at all; that was already shown by `seed=42` and `seed=7`.
- The new question is whether that RGB gain remains stable once the seed count is increased from `2` to `5`.

## Recommended Approach

Run the minimum additional experiment matrix needed to turn the current RGB result from a stage freeze into a stability-confirmed stage:

- keep the existing paired comparison format
- add exactly `3` more seeds
- preserve all non-seed variables
- summarize the five paired runs using per-seed lift, mean, and spread

This keeps the next statement clean:

`incoherent_intensity + back` remains a stable positive line on RGB CIFAR-10 under the current 10-epoch budget.

## Fixed Experiment Surface

The following configuration must remain fixed for the whole stability pass:

- dataset: `cifar10-rgb`
- task: `classification`
- layers: `5`
- size: `200`
- epochs: `10`
- batch size: `64`
- learning rate: `0.01`
- optics: current classifier paper optics
- baseline arm: `phase-only`
- nonlinear arm: `incoherent_intensity + back`
- activation preset: `balanced`

This step explicitly does **not**:

- change `epoch`
- try new placements
- try new nonlinear mechanisms
- retune preset or activation hyperparameters
- alter the RGB embedding path

## Experiment Matrix

Already completed:

- `seed=42`: baseline `44.01%`, nonlinear `46.60%`, lift `+2.59`
- `seed=7`: baseline `44.12%`, nonlinear `47.08%`, lift `+2.96`

This stage adds:

- `3` additional seeds
- for each new seed, one baseline run and one nonlinear run

The result should be a final `5-seed` paired table with:

- baseline accuracy per seed
- nonlinear accuracy per seed
- per-seed lift
- five-seed mean baseline accuracy
- five-seed mean nonlinear accuracy
- mean lift
- observed spread or min/max range

## Naming and Artifact Policy

The current artifact naming pattern must remain explicit and reversible.

For new RGB stability runs:

- baseline names should continue the existing `cifar10_rgb_baseline_10ep_seedX` pattern
- nonlinear names should continue the existing `cifar10_rgb_incoherent_back_10ep_seedX` pattern

Guardrail:

- do not introduce a second naming scheme for the same stage
- do not overwrite existing `seed=42` or `seed=7` artifacts

## Review and Logging Policy

This stability pass ends at a stage node and therefore requires an independent review before any further extension.

Review order:

1. try local Claude Code review if the environment can actually produce usable output
2. if Claude Code still cannot return a usable review, record that fact plainly and do a fallback read-only review

The review must look for:

- manifest and run-name consistency
- accidental baseline/nonlinear mismatch in the paired seed table
- CLI/config drift
- missing or ambiguous artifact naming
- notebook-log drift from the recorded stage interpretation

After review:

- append the result summary to the main reproduction log
- append the stage interpretation to the nonlinear note
- keep writing into the existing progress log sections instead of creating a separate note

## Stage Exit Criteria

This design is complete when:

- RGB CIFAR-10 has a full `5-seed` paired baseline/nonlinear table under the same `10 epoch` setup
- the summary statistics are computed and written down
- an independent review is completed and logged honestly
- both Obsidian notes are updated with the stability conclusion
- the project is ready for the next decision point:
  - continue to longer RGB training budgets
  - or stop and freeze this stronger RGB stability stage

## Risks and Guardrails

### Risk: Mixing stability confirmation with stage extension

Guardrail:

- do not change `epoch` or any nonlinear configuration during this pass

### Risk: Weak paired comparison

Guardrail:

- every new seed must include both baseline and nonlinear arms

### Risk: Narrative drift from Obsidian

Guardrail:

- keep the written conclusion framed as `RGB stability confirmation`, not `new mechanism discovery`

### Risk: Overclaiming review provenance

Guardrail:

- do not claim a Claude review happened unless the tool actually returned usable output
