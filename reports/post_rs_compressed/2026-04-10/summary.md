# Post-RS Compressed Proxy Summary

Configuration: `Fashion-MNIST`, `size=100`, `layers=5`, `epochs=5`, `seed=42`, `rs-backend=fft`, `batch-size=64`

## Results

| Run | Activation | Best val acc | Best val contrast | Best epoch | Test acc | Test contrast |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fashion_mnist_phase_only_5ep_size100_seed42_post_rs_fft_proxy` | `none` | 82.68 | 0.4946 | 4 | 82.08 | 0.4925 |
| `fashion_mnist_incoherent_back_5ep_size100_seed42_post_rs_fft_proxy` | `incoherent_intensity@back` | 82.92 | 0.4997 | 4 | 82.22 | 0.4964 |

## Comparison

- Test accuracy delta (`incoherent_back - phase_only`): `+0.14`
- Test contrast delta (`incoherent_back - phase_only`): `+0.0039`
- Both runs exceeded the paper target accuracy `81.13%` under the proxy configuration.

## Artifacts

- Phase-only manifest: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints_proxy\best_fashion_mnist.fashion_mnist_phase_only_5ep_size100_seed42_post_rs_fft_proxy.json`
- Incoherent-back manifest: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints_proxy\best_fashion_mnist.fashion_mnist_incoherent_back_5ep_size100_seed42_post_rs_fft_proxy.json`
- Phase-only figures: `C:\Users\Jiangqianxian\source\repos\d2nn\figures\post_rs_compressed\2026-04-10\fashion_mnist_phase_only_5ep_size100_seed42_post_rs_fft_proxy`
- Incoherent-back figures: `C:\Users\Jiangqianxian\source\repos\d2nn\figures\post_rs_compressed\2026-04-10\fashion_mnist_incoherent_back_5ep_size100_seed42_post_rs_fft_proxy`
