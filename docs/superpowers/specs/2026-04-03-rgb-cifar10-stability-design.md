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
- add exactly `3` more seeds: `seed=123`, `seed=0`, `seed=2025`
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

| seed | baseline test | nonlinear test | lift | baseline best-val | nonlinear best-val |
|------|--------------|---------------|------|-------------------|--------------------|
| 42   | 44.01%       | 46.60%        | +2.59 | 44.20%           | 46.78%             |
| 7    | 44.12%       | 47.08%        | +2.96 | —                | —                  |

> Note: `seed=7` val accuracy was not recorded at freeze time; leave as `—` rather than backfill.

This stage adds:

- seeds `123`, `0`, `2025` (in that order)
- for each new seed, one baseline run and one nonlinear run
- the reference naming base for comparison is the `nonlinear-incoherent-back-cifar10-rgb-v1` release

The result should be a final `5-seed` paired table with:

- baseline test accuracy per seed
- nonlinear test accuracy per seed
- per-seed lift
- five-seed mean baseline accuracy
- five-seed mean nonlinear accuracy
- mean lift
- observed lift spread (max − min across seeds)

Example CLI for each new seed pair (substitute `X` and `SEEDVAL`):

```bash
# baseline arm
uv run python train.py --task classification --dataset cifar10-rgb \
  --layers 5 --size 200 --epochs 10 --batch-size 64 --lr 0.01 \
  --seed SEEDVAL --run-name cifar10_rgb_baseline_10ep_seedX

# nonlinear arm
uv run python train.py --task classification --dataset cifar10-rgb \
  --layers 5 --size 200 --epochs 10 --batch-size 64 --lr 0.01 \
  --seed SEEDVAL --activation-type incoherent_intensity \
  --activation-placement back --activation-preset balanced \
  --run-name cifar10_rgb_incoherent_back_10ep_seedX
```

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

- manifest and run-name consistency (compare against the `nonlinear-incoherent-back-cifar10-rgb-v1` release naming as the reference baseline)
- accidental baseline/nonlinear mismatch in the paired seed table
- CLI/config drift from the Fixed Experiment Surface above
- missing or ambiguous artifact naming, especially `seed` and `10ep` encoding in filenames
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

### Minimum Stability Condition

Stability is **confirmed** if all of the following hold:

- at least `4` out of `5` seeds show lift `> +1.5 pt`
- no single seed shows lift `< 0` (i.e., nonlinear never falls below baseline)
- five-seed mean lift `> +2.0 pt`

If the data falls short of this bar, record it plainly as "stability inconclusive" rather than forcing a positive conclusion.

### Next-Step Decision Criteria

After the 5-seed table is complete, decide as follows:

- **Continue to longer RGB training budgets** if: mean lift `> +2.5 pt` AND spread (max − min lift) `< 1.5 pt`. This indicates a sufficiently stable advantage worth investing more compute.
- **Freeze this stage and stop** if: mean lift is between `+1.5 pt` and `+2.5 pt`, or spread `≥ 1.5 pt`. The result is real but not compelling enough to justify further budget increases without a new idea.
- **Flag for re-examination** if: minimum stability condition above is not met. Do not extend before understanding why a seed failed.

This decision is made by human judgment after reviewing the table, not by the automated pipeline.

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

### Risk: Lift spread too large to support stability claim

Guardrail:

- if spread (max − min lift across 5 seeds) `≥ 2.0 pt`, do not write "stable"; write "directional but variable" and note which seed drove the outlier before continuing

### Risk: Overclaiming review provenance

Guardrail:

- do not claim a Claude review happened unless the tool actually returned usable output
