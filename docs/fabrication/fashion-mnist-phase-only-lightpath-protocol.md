# Fashion-MNIST Phase-Only Lightpath Protocol

## Purpose

This protocol maps the regenerated physics-aligned baseline to the first optical calibration round. Success is judged by repeatable agreement between the normalized full output-plane intensity map and the simulation reference, with detector-mask hotspots used as a secondary sanity check rather than as the only readout.

## Simulation Inputs

- Task: classification
- Dataset: Fashion-MNIST
- Layer count: 5
- Layer resolution: 200 x 200
- Wavelength: 0.75 mm
- Layer spacing: 30 mm
- Pixel size: 0.4 mm

## Current Simulation References

- Regenerated baseline figures: `figures/fashion_mnist/baseline_5layer_physics_aligned/`
- Baseline artifacts: `phase_masks.png`, `output_energy.png`, `confusion_matrix.png`, `sample_output_patterns.png`, `quantization_sensitivity.png`
- Nonlinear comparison figures: `figures/fashion_mnist/incoherent_back_20ep/`
- Nonlinear reference: `checkpoints/best_fashion_mnist.incoherent_back_20ep.pth` with `incoherent_intensity` activation at position `[5]`
- Use the regenerated baseline figures as the primary optical reference for first-round calibration. Use the nonlinear figures only as a later comparison branch.

## Optical Mapping

- Input loading method: use the classification preprocessing path from `tasks.py` and `d2nn.py` exactly. For Fashion-MNIST that means `transforms.ToTensor()` only, with no extra dataset normalization, followed by `D2NN._embed_input`, which calls `embed_amplitude_image` to bilinearly resize each `1 x 28 x 28` sample into a centered `66 x 66` amplitude patch (`size // 3`) inside the `200 x 200` complex field and leaves the rest of the field at zero amplitude.
- Phase plate ordering: keep the phase plates in ascending layer order, from layer 01 through layer 05, and preserve the same physical sequence used in simulation.
- Plate orientation convention: the repository currently preserves raw array indexing during export, but it does not yet encode a physical fiducial or front/back-face convention. For the first lab round, treat exported row `0` / column `0` as the indexing reference, keep that reference attached to each fabricated plate, and preserve the same indexing across `phase_masks.npy`, per-layer CSVs, and STL files. If the fabrication flow cannot preserve that labeling, orientation must be treated as an unresolved calibration variable rather than assumed correct.
- Output-plane measurement location: measure one final `30 mm` free-space propagation step beyond layer 05, because the classifier model applies an output transfer function `H_out` with the same `layer_distance = 30 mm` after the fifth diffractive layer.
- Detector/comparison region: compare the full output-plane intensity map after normalization, then check the detector masks on that same map as a task-linked sanity check. Do not crop to an undocumented readout window.
- Normalization rule: for first-round full-plane comparison, normalize each simulated or measured output intensity map by its full-plane summed energy, `I_norm(x, y) = I(x, y) / (sum_{x,y} I(x, y) + 1e-8)`, before pixelwise comparison and detector-hotspot checks. This keeps the comparison consistent with the classification-side energy normalization used in `tasks.py` (`d2nn_mse_loss` normalizes detector readout by summed energy).

## Current Dry-Run Package

- Export root: `exports/best_fashion_mnist.baseline_5layer_physics_aligned/best_fashion_mnist.baseline_5layer_physics_aligned/`
- Generated files: `phase_masks.npy`, `height_map.npy`, `height_map_manufacturable.npy`, `thickness_map.npy`, `height_map_quantized.npy`, `report.md`, `metadata.json`, `layers/`, `stl/`
- Quantization levels: `256`
- Base thickness: `500.0 um`
- Refractive index: `1.7227`
- Ambient index: `1.0`
- Relief limit enabled: `false`
- Max relief: `none`
- Raw height max: `1037.775 um`
- Current exported relief max in `height_map_manufacturable.npy`: `1037.775 um` (identical to the raw height map in this dry-run because no relief limit was applied)
- Thickness max: `1537.775 um`
- Clipped pixels: `0 of 200000`
- Clipped fraction: `0.00%`
- Pixel size: `400.000 um`
- This dry-run confirms that the export chain is traceable and currently unclipped, but it does not yet establish manufacturability against a real process limit. The final fabrication export still depends on lab-confirmed parameters, especially `max_relief`.

## Failure Triage

If the fabricated output does not match the simulation trends, debug in this order:

1. plate ordering and orientation
2. layer spacing
3. output-plane distance
4. input loading method
5. measurement normalization
6. fabrication fidelity

## Notes

This protocol is now backed by a regenerated `fabrication_baseline` checkpoint, saved understanding-report figures, and a complete export dry-run. The next fabrication handoff should preserve this optical mapping while replacing the dry-run material/process assumptions with lab-confirmed values.
