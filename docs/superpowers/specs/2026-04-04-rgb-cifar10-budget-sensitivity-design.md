# RGB CIFAR-10 Budget Sensitivity Exit Design

**Goal:** Close the current nonlinear-validation phase by testing whether the RGB CIFAR-10 result becomes materially more convincing under a longer training budget, without changing the mechanism, placement, optics, or RGB input path.

**Scope:** This design is intentionally narrower than a new RGB phase. It does not introduce a new mechanism, new placement, or new dataset. It only adds one final budget-sensitivity pass on top of the already completed `10 epoch` RGB evidence so the current nonlinear plan can end with a clear statement instead of an open loop.

## Route Alignment

This design follows the current Obsidian state exactly:

- `Fashion-MNIST` is already closed.
- `grayscale CIFAR-10` is already closed.
- `RGB CIFAR-10, 10 epoch` currently reads as `directional positive / stability inconclusive`.
- The next step in the note is no longer “keep expanding RGB by default,” but “decide whether to stop here or do one final longer-budget check.”

## Recommended Approach

Run one final RGB confirmation pass under a longer budget:

- keep the current `phase-only baseline` arm
- keep the current best nonlinear arm `incoherent_intensity + back`
- keep the current RGB embedding and optics unchanged
- increase only `epoch` from `10` to `20`
- run only the minimum seed set needed to judge direction under higher budget: `42`, `7`, `123`

This is the smallest experiment that can answer the only remaining question:

Is the current RGB result merely budget-limited at `10 epoch`, or is it inherently weaker and more variable than the earlier stages?

## Fixed Experiment Surface

The following stay fixed:

- dataset: `cifar10-rgb`
- task: `classification`
- layers: `5`
- size: `200`
- batch size: `64`
- learning rate: `0.01`
- optics: current classifier paper optics
- baseline arm: `phase-only`
- nonlinear arm: `incoherent_intensity + back`
- activation preset: `balanced`

The only variable changed in this exit pass is:

- epochs: `10 -> 20`

This pass explicitly does **not**:

- add more activation mechanisms
- revisit placement ablation
- revisit RGB encoding design
- add new datasets
- extend beyond the chosen three seeds

## Experiment Matrix

This exit pass runs exactly six jobs:

- baseline, `seed=42`, `20 epoch`
- nonlinear, `seed=42`, `20 epoch`
- baseline, `seed=7`, `20 epoch`
- nonlinear, `seed=7`, `20 epoch`
- baseline, `seed=123`, `20 epoch`
- nonlinear, `seed=123`, `20 epoch`

The `10 epoch` RGB evidence remains the reference stage.

The `20 epoch` pass is interpreted as a budget-sensitivity check, not a new freeze by default.

## Naming and Artifact Policy

Use explicit, non-overlapping names:

- baseline: `cifar10_rgb_baseline_20ep_seedX`
- nonlinear: `cifar10_rgb_incoherent_back_20ep_seedX`

Guardrails:

- do not overwrite any existing `10ep` RGB artifacts
- do not reuse the `nonlinear-incoherent-back-cifar10-rgb-v1` release tag

## Stage Exit Interpretation

This pass exists to let the current nonlinear plan stop cleanly.

After the three paired `20 epoch` runs:

- if the RGB gap becomes clearly stronger and more consistent than the `10 epoch` pass, the current nonlinear line can be declared closed through RGB
- if the RGB gap stays weak or variable, the current nonlinear line is still considered complete, but the final statement must say that RGB remains budget-sensitive or not fully stabilized

In both cases, this phase ends here.

## Review and Logging Policy

This is a stage node and requires the same review rule as earlier milestones:

1. try local Claude Code review if it can actually produce usable output
2. if not, record the failure reason plainly and do fallback read-only review

The review must look for:

- `20ep` naming consistency
- baseline/nonlinear pairing integrity across the three seeds
- accidental config drift from the fixed experiment surface
- whether the written conclusion matches the observed data

After review:

- append the result to the main reproduction log
- append the stage interpretation to the nonlinear note
- stop at the next decision boundary instead of rolling directly into repo cleanup or new experiments

## Success Criteria

This design is complete when:

- the three `20 epoch` RGB seed pairs are finished
- the `20 epoch` summary is written down
- an independent review is honestly recorded
- both Obsidian notes are updated
- the nonlinear-validation phase has a final written exit statement

## Risks and Guardrails

### Risk: Scope expands again

Guardrail:

- this pass changes only `epoch`, nothing else

### Risk: A stronger budget check gets mistaken for a new RGB phase

Guardrail:

- write this explicitly as an `exit pass` or `budget-sensitivity pass`

### Risk: Negative or mixed results trigger endless continuation

Guardrail:

- this phase ends after the three `20 epoch` seed pairs regardless of outcome

### Risk: Overclaiming external review

Guardrail:

- do not claim Claude review unless the command returns usable output
