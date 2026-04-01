# D2NN

简洁版 D2NN 复现仓库，对应 Lin et al., Science 2018。

当前覆盖三条主线：
- MNIST 分类
- Fashion-MNIST 分类
- 成像透镜

仓库中的主要程序分工如下：
- `train.py`: 统一训练入口，负责启动分类或成像任务。
- `visualize.py`: 统一可视化入口，负责生成相位掩膜、输出能量和重建结果图。
- `export_phase_plate.py`: 相位板导出入口，负责把 checkpoint 转成 phase mask、height map、CSV 和 STL。
- `d2nn.py`: 光学传播和 D2NN 模型定义，是核心前向实现。
- `tasks.py`: 任务层封装，负责数据集、训练循环、评估和可视化调度。
- `artifacts.py`: checkpoint、manifest、制造导出和共享工具函数。
- `tests/`: 最小测试集，负责 smoke check 和核心工具回归验证。

## 当前状态

这套仓库现在更适合作为 `主文数值复现 + 相位板导出前处理` 的交付包，而不是完整实验系统。

当前已经完成：
- 主文分类主线的数值复现
- 成像链路的功能性复现
- 从 checkpoint 到 phase mask / height map / CSV / STL 的导出

当前还没有完成：
- 与论文原始 ImageNet 成像展示的严格对齐
- 真实光路实验验证
- fabrication-aware 训练与误差鲁棒性扫描

## 结构与程序作用

- `d2nn.py`: 核心光学模型
- `tasks.py`: 分类与成像任务逻辑
- `artifacts.py`: optics / checkpoint / manifest / 可视化 / 制造导出共用工具
- `train.py`: 统一训练入口
- `visualize.py`: 统一可视化入口
- `export_phase_plate.py`: 相位板数值导出
- `tests/`: 最小测试集

## 推荐复现路径

如果只想快速确认当前仓库能否复现主线，推荐按下面顺序跑：

1. 分类基线：

```bash
uv run python visualize.py --task classification --dataset fashion-mnist --checkpoint checkpoints/best_fashion_mnist.pth --no-show
```

预期产物：
- `figures/fashion_mnist/phase_masks.png`
- `figures/fashion_mnist/output_energy.png`
- `figures/fashion_mnist/confusion_matrix.png`

2. 成像基线：

```bash
uv run python visualize.py --task imaging --dataset stl10 --checkpoint checkpoints/best_imager_stl10.pth --no-show
```

预期产物：
- `figures/imager/phase_masks.png`
- `figures/imager/sample_reconstructions.png`

3. 相位板导出基线：

```bash
uv run python export_phase_plate.py --task classification --checkpoint checkpoints/best_fashion_mnist.pth --export-stl
```

预期产物：
- `exports/best_fashion_mnist/metadata.json`
- `exports/best_fashion_mnist/report.md`
- `exports/best_fashion_mnist/layers/`
- `exports/best_fashion_mnist/stl/`

## 推荐 checkpoint

- 论文主线 `5 层 / 200x200` 的分类基线，优先使用 `checkpoints/best_fashion_mnist.pth`
- 如果要看 MNIST 的 5 层主线备份，优先使用 `checkpoints/best_mnist.pre_retrain_20260327.pth`
- 如果要看成像链路，使用 `checkpoints/best_imager_stl10.pth`
- 不要把 `checkpoints/best_mnist.pth` 当作当前 paper-faithful 主线，它现在实际是 `3 层 / 100x100`

## 论文对照结果

| 任务 | 论文记录 | 当前仓库结果 | 对齐情况 |
|------|------|------|------|
| MNIST 分类 | `97.61%` | `97.63% (9763/10000)` | 数值结果基本对齐，但当前默认 `best_mnist.pth` 已不是 5 层主线 checkpoint |
| Fashion-MNIST 分类 | `81.13%` | `87.49%` | 功能与结构对齐，当前训练结果高于论文记录 |
| 成像透镜 | 主文展示 ImageNet 自然图像成像 | `STL10`, `Test MSE = 0.0259` | 完成功能链路复验，但数据源不是论文原始展示 |

## 常用命令

```bash
uv run python train.py --task classification --dataset mnist --epochs 20 --size 200 --layers 5
uv run python train.py --task classification --dataset fashion-mnist --epochs 20 --size 200 --layers 5
uv run python train.py --task imaging --dataset stl10 --epochs 10 --size 200 --layers 5 --image-size 64 --batch-size 4

uv run python visualize.py --task classification --dataset mnist --checkpoint checkpoints/best_mnist.pth --no-show
uv run python visualize.py --task classification --dataset fashion-mnist --checkpoint checkpoints/best_fashion_mnist.pth --no-show
uv run python visualize.py --task imaging --dataset stl10 --checkpoint checkpoints/best_imager_stl10.pth --no-show

uv run python export_phase_plate.py --task classification --checkpoint checkpoints/best_fashion_mnist.pth
uv run python export_phase_plate.py --task imaging --checkpoint checkpoints/best_imager_stl10.pth
```

## 导出产物

`export_phase_plate.py` 会在 `exports/<checkpoint_stem>/` 下生成：
- `phase_masks.npy`
- `height_map.npy`
- `height_map_manufacturable.npy`
- `thickness_map.npy`
- `height_map_quantized.npy`
- `metadata.json`
- `report.md`
- `layers/`
- `stl/`（传 `--export-stl` 时）

其中 `layers/` 下会按层导出：
- `layer_XX_phase_rad.csv`
- `layer_XX_height_um.csv`
- `layer_XX_thickness_um.csv`
- `layer_XX_height_quantized.csv`

这一步是从数值复现走向真实相位板的公共桥接层。

## 验证

```bash
uv run python -m unittest discover -s tests -v
```

最小 smoke check 已覆盖：
- `train.py --help`
- `visualize.py --help`
- `export_phase_plate.py --help`

## 当前注意事项

- 当前顶层主文件已经收敛到 `d2nn.py`、`tasks.py`、`artifacts.py`、`train.py`、`visualize.py`、`export_phase_plate.py`
- `checkpoints/best_mnist.pth` 当前实际是 `3 层 / 100x100`
- 如果要沿主文 `5 层 / 200x200` 基线继续推进，更稳的是：
  - `checkpoints/best_mnist.pre_retrain_20260327.pth`
  - `checkpoints/best_fashion_mnist.pth`
  - `checkpoints/best_imager_stl10.pth`
- `checkpoints/best_fashion_mnist.pth` 的完整导出与 `STL` 已验证通过，当前更适合作为第一版样片基线
- 以当前 `wavelength = 750 um`、`refractive index = 1.7227`、`ambient index = 1.0` 计算，完整 `2pi` 相位包裹对应的最大浮雕高度约为 `1037.8 um`
- `--max-relief-um 300` 会让 `best_fashion_mnist.pth` 每层约 `50%` 的像素被截平，这个值不适合作为当前默认工艺上限
