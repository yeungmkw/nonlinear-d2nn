# RGB CIFAR-10 Minimal Entry Design

**Goal:** Extend the current nonlinear reproduction line from grayscale CIFAR-10 to a first RGB CIFAR-10 classification entrypoint without changing the nonlinear mechanism, training loop, or stage-comparison protocol.

**Scope:** This design only covers the minimum code and experiment surface needed to run `phase-only baseline` versus `incoherent_intensity + back` on RGB CIFAR-10 under the same core budget used for the grayscale stage: `5 layers / 200x200 / 10 epochs / seed=42`.

## Route Alignment

This design stays aligned with the Obsidian reproduction notes:

- The main nonlinear line remains `incoherent_intensity + back`.
- The current stage objective is still “transfer to a more complex dataset,” not “redesign the optics stack.”
- RGB CIFAR-10 is treated as the next complexity increase after grayscale CIFAR-10, not as a new nonlinear-mechanism phase.

## Recommended Approach

Implement a minimal RGB input path for classification while keeping the existing D2NN classifier and experiment plumbing intact.

The first RGB encoding should be simple and stable:

- Add an RGB classification dataset entry alongside `cifar10_gray`.
- Preserve RGB tensors as 3-channel inputs; do not collapse them to grayscale.
- Embed RGB inputs into a single optical field by assigning `R`, `G`, and `B` amplitude maps to three fixed horizontal subregions inside the existing centered input window.

This keeps the experiment interpretable:

- The model change is limited to input embedding.
- Baseline and nonlinear remain directly comparable.
- Any performance movement can still be attributed mainly to dataset complexity and nonlinear transfer, not to a new training regime.

## Architecture Changes

### Dataset Layer

Add a new classification dataset config:

- Key: `cifar10_rgb`
- Aliases accepted by the CLI/config parser: `cifar10-rgb` and `cifar10_rgb`
- Display name: `CIFAR-10 (RGB)`
- Checkpoint stem: `best_cifar10_rgb.pth`
- Output directory: `figures/cifar10_rgb`

The classification transform builder should support two modes:

- `grayscale=True`: current behavior
- `grayscale=False` with explicit RGB retention: use `ToTensor()` only so the input remains shape `(3, H, W)`

### Field Embedding Layer

Keep the current grayscale embedding helper for single-channel inputs.

Add a second helper for RGB inputs that:

- accepts `(B, 3, H, W)`
- resizes each channel to the same square subpatch size
- places three non-overlapping subpatches inside the same centered input window
- returns a `(B, size, size)` complex field with real-valued amplitude initialization

The classifier should route input embedding by channel count:

- 1 channel -> existing grayscale embedding path
- 3 channels -> new RGB embedding path
- anything else -> explicit error

This keeps the model interface unchanged while making RGB entrypoint support local and testable.

## Experiment Policy

The first RGB stage should use the same comparison style as the grayscale stage:

- baseline: phase-only
- nonlinear: `incoherent_intensity + back`
- fixed optics: current classifier paper optics
- fixed budget: `10 epochs`
- fixed seed: `42`

No RGB-specific activation sweep, no new placement ablation, and no new optics tuning in this step.

## Testing

This step needs only minimum regression coverage:

- dataset alias/config resolution for `cifar10_rgb`
- RGB transform preserves 3 channels
- RGB embedding returns the correct field shape/dtype
- the three channel patches land in disjoint regions with nonzero values when given distinct channel inputs
- a classifier built on the existing architecture can accept a `(B, 3, H, W)` batch and produce logits

## Risks and Guardrails

### Risk: Route drift from the Obsidian plan

Guardrail:

- keep the stage goal as “RGB dataset entry + first transfer comparison”
- do not redesign nonlinear activation or placement in this step

### Risk: Input encoding change muddies comparison

Guardrail:

- keep the RGB encoding intentionally simple
- change only the classifier input embedding, not training logic

### Risk: Overly large code footprint

Guardrail:

- restrict changes to `tasks.py`, `d2nn.py`, `artifacts.py` if needed, `train.py` help text, and `tests/test_d2nn_core.py`

## Success Criteria

This design is complete when:

- RGB CIFAR-10 can be selected from the classification CLI
- the classifier can ingest RGB tensors without grayscale conversion
- the first RGB baseline/nonlinear pair can be run with the existing experiment skeleton
- the resulting stage remains narratively aligned with the Obsidian reproduction notes
