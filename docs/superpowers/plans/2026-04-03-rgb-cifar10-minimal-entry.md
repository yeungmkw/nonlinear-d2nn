# RGB CIFAR-10 Minimal Entry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal RGB CIFAR-10 classification entrypoint that keeps the current nonlinear route intact and enables the first `phase-only baseline` versus `incoherent_intensity + back` comparison.

**Architecture:** Extend the classification dataset registry with an RGB CIFAR-10 option and teach the classifier input embedding path to map 3-channel tensors into one optical field via three fixed subregions. Keep the classifier, nonlinear mechanism plumbing, and experiment naming flow unchanged so the first RGB comparison remains directly comparable to the grayscale stage.

**Tech Stack:** Python 3.13, PyTorch, torchvision, unittest

---

### Task 1: Add failing tests for RGB dataset config and embedding

**Files:**
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\tests\test_d2nn_core.py`
- Test: `C:\Users\Jiangqianxian\source\repos\d2nn\tests\test_d2nn_core.py`

- [ ] **Step 1: Write the failing tests**

```python
    def test_get_classification_dataset_config_supports_cifar10_rgb_alias(self):
        cfg = get_classification_dataset_config("cifar10-rgb")
        self.assertEqual(cfg["display_name"], "CIFAR-10 (RGB)")
        self.assertEqual(cfg["checkpoint_name"], "best_cifar10_rgb.pth")

    def test_build_classification_transform_preserves_cifar10_rgb_channels(self):
        transform = build_classification_transform(get_classification_dataset_config("cifar10_rgb"))
        image = Image.new("RGB", (32, 32), color=(255, 128, 0))
        tensor = transform(image)
        self.assertEqual(tensor.shape, (3, 32, 32))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_classifier_rgb_embed_input_places_channels_into_disjoint_regions(self):
        model = D2NN(**CLASSIFIER_PAPER_OPTICS.with_overrides(size=30, num_layers=2).classifier_model_kwargs())
        x = torch.zeros(1, 3, 12, 12)
        x[:, 0].fill_(1.0)
        x[:, 1].fill_(2.0)
        x[:, 2].fill_(3.0)

        field = model._embed_input(x)

        self.assertEqual(field.shape, (1, 30, 30))
        self.assertEqual(field.dtype, torch.cfloat)
        amplitude = field.abs()[0]
        nonzero_cols = torch.where(amplitude.sum(dim=0) > 0)[0]
        self.assertGreaterEqual(len(nonzero_cols), 3)

    def test_classifier_accepts_rgb_batches(self):
        model = D2NN(**CLASSIFIER_PAPER_OPTICS.with_overrides(size=24, num_layers=2).classifier_model_kwargs())
        x = torch.rand(2, 3, 32, 32)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 10))
        self.assertEqual(logits.dtype, torch.float32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_d2nn_core.D2NNCoreTests.test_get_classification_dataset_config_supports_cifar10_rgb_alias tests.test_d2nn_core.D2NNCoreTests.test_build_classification_transform_preserves_cifar10_rgb_channels tests.test_d2nn_core.D2NNCoreTests.test_classifier_rgb_embed_input_places_channels_into_disjoint_regions tests.test_d2nn_core.D2NNCoreTests.test_classifier_accepts_rgb_batches -v`

Expected: FAIL because `cifar10_rgb` is unsupported and the classifier currently only embeds single-channel input.

### Task 2: Implement the minimal RGB dataset and embedding path

**Files:**
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\tasks.py`
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\d2nn.py`
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\train.py`

- [ ] **Step 1: Add the RGB CIFAR-10 dataset config**

```python
    "cifar10_rgb": {
        "dataset_cls": datasets.CIFAR10,
        "display_name": "CIFAR-10 (RGB)",
        "checkpoint_name": "best_cifar10_rgb.pth",
        "paper_target": None,
        "default_output_dir": "figures/cifar10_rgb",
        "grayscale": False,
        "preserve_rgb": True,
    },
```

- [ ] **Step 2: Keep RGB tensors in the classification transform**

```python
def build_classification_transform(dataset_cfg):
    transform_steps = []
    if dataset_cfg.get("grayscale"):
        transform_steps.append(transforms.Grayscale(num_output_channels=1))
    transform_steps.append(transforms.ToTensor())
    return transforms.Compose(transform_steps)
```

Expected implementation note: `cifar10_rgb` should use the no-grayscale path and remain `(3, H, W)`.

- [ ] **Step 3: Add an RGB embedding helper in `d2nn.py`**

```python
def embed_rgb_amplitude_image(x, size, target_size=None):
    batch, channels, _, _ = x.shape
    if channels != 3:
        raise ValueError("RGB embedding expects exactly 3 channels")

    field = torch.zeros(batch, size, size, dtype=torch.cfloat, device=x.device)
    target_size = target_size or size // 3
    patch_size = max(target_size // 3, 1)
    offset_y = (size - patch_size) // 2
    total_width = patch_size * 3
    offset_x = (size - total_width) // 2

    resized = F.interpolate(x, size=(patch_size, patch_size), mode="bilinear", align_corners=False)
    for channel_idx in range(3):
        start_x = offset_x + channel_idx * patch_size
        field[:, offset_y : offset_y + patch_size, start_x : start_x + patch_size] = resized[:, channel_idx].to(torch.cfloat)
    return field
```

- [ ] **Step 4: Route classifier embedding by channel count**

```python
    def _embed_input(self, x):
        channels = x.shape[1]
        if channels == 1:
            return embed_amplitude_image(x, self.size, target_size=self.size // 3)
        if channels == 3:
            return embed_rgb_amplitude_image(x, self.size, target_size=self.size // 3)
        raise ValueError(f"Unsupported classification input channels: {channels}")
```

- [ ] **Step 5: Update the CLI dataset help text**

```python
        help="classification: mnist/fashion-mnist/cifar10-gray/cifar10-rgb; imaging: stl10/imagefolder",
```

- [ ] **Step 6: Run the focused tests to verify they pass**

Run: `uv run python -m unittest tests.test_d2nn_core.D2NNCoreTests.test_get_classification_dataset_config_supports_cifar10_rgb_alias tests.test_d2nn_core.D2NNCoreTests.test_build_classification_transform_preserves_cifar10_rgb_channels tests.test_d2nn_core.D2NNCoreTests.test_classifier_rgb_embed_input_places_channels_into_disjoint_regions tests.test_d2nn_core.D2NNCoreTests.test_classifier_accepts_rgb_batches -v`

Expected: PASS

### Task 3: Run broader regression and document the first RGB experiment commands

**Files:**
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\README.md` (only if the dataset help text needs a matching mention)
- Test: `C:\Users\Jiangqianxian\source\repos\d2nn\tests\test_d2nn_core.py`

- [ ] **Step 1: Run the full unit test suite**

Run: `uv run python -m unittest discover -s tests -v`

Expected: PASS with all tests green.

- [ ] **Step 2: Print the first RGB baseline command**

Run: `uv run python train.py --task classification --dataset cifar10-rgb --epochs 10 --size 200 --layers 5 --batch-size 64 --lr 0.01 --seed 42 --experiment-stage cifar10-rgb-baseline --run-name cifar10_rgb_baseline_10ep`

Expected: command starts training without input-shape errors.

- [ ] **Step 3: Print the first RGB nonlinear command**

Run: `uv run python train.py --task classification --dataset cifar10-rgb --epochs 10 --size 200 --layers 5 --batch-size 64 --lr 0.01 --seed 42 --experiment-stage cifar10-rgb-nonlinear --run-name cifar10_rgb_incoherent_back_10ep --activation-type incoherent_intensity --activation-placement back --activation-preset balanced`

Expected: command starts training without input-shape errors.

- [ ] **Step 4: Commit the implementation**

```bash
git add d2nn.py tasks.py train.py tests/test_d2nn_core.py docs/superpowers/specs/2026-04-03-rgb-cifar10-minimal-entry-design.md docs/superpowers/plans/2026-04-03-rgb-cifar10-minimal-entry.md
git commit -m "feat(classification): add rgb cifar10 entrypoint"
```
