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

## Optical Mapping

- Input loading method: load the same Fashion-MNIST amplitude encoding used by the regenerated simulation checkpoint, without changing the preprocessing path between simulation and lightpath calibration.
- Phase plate ordering: keep the phase plates in ascending layer order, from layer 01 through layer 05, and preserve the same physical sequence used in simulation.
- Plate orientation convention: treat every exported phase map as an image viewed from the input side. Mount each plate without mirroring, rotation, or flipping; the first array row stays at the physical top edge, the first array column stays at the physical left edge, and the edge marked as downstream points toward the next propagation step.
- Output-plane measurement location: measure at the same output-plane distance used in the simulation model, directly after the final propagation step.
- Detector/comparison region: compare the full output-plane intensity map after normalization, then check the detector masks on that same map as a task-linked sanity check. Do not crop to an undocumented readout window.
- Normalization rule: normalize each measured output pattern with the same per-pattern rule used in simulation before comparing the full map and the detector-mask hotspots.

## Failure Triage

If the fabricated output does not match the simulation trends, debug in this order:

1. plate ordering and orientation
2. layer spacing
3. output-plane distance
4. input loading method
5. measurement normalization
6. fabrication fidelity
