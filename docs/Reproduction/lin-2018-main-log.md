---
created: 2026-03-06
type: project
status: active
category: 学业
tags:
  - project
  - deep-learning
  - optics
  - D2NN
paper: Lin et al. 2018 D2NN
doi: 10.1126/science.aat8084
---

# 论文复现 · Lin et al. 2018

> 论文: Lin et al., *All-optical machine learning using diffractive deep neural networks*, Science 361, 1004–1008 (2018)
> 论文文件: 主文与 supplementary PDF 仍保留在 Zotero 外部文献库，本仓库不镜像论文原文
> 仓库入口: [`README.md`](../../README.md)
> 运行环境: Python 3.13 + PyTorch 2.10 + CUDA 12.6 + RTX 4060 Laptop

> [!summary]
> 主文主线现在已经闭环：MNIST 分类、Fashion-MNIST 分类、成像透镜三条线都已有当前工作树下可复跑的代码、checkpoint 和可视化产物。当前需要单独记住的不是“主文还没做完”，而是“成像透镜的当前复验使用 STL10 自然图像完成链路验证，而不是论文原始的 ImageNet 图像示例”。

> [!note]
> 本文件已经从 Obsidian 迁入仓库，正文以仓库内可核对的代码、checkpoint 和实验记录为准。论文原文 PDF 与外部文献索引仍保留在 Zotero / Obsidian，不再通过 wiki link 从这里跳转。

## 主要程序入口

| 程序 | 入口文件 | 用途 | 什么时候看 |
|------|------|------|------|
| 核心模型 | [`d2nn.py`](../../d2nn.py) | 定义 `DiffractiveLayer`、`D2NN`、`D2NNImager` | 想看网络结构、物理传播和输入编码时 |
| 任务逻辑 | [`tasks.py`](../../tasks.py) | 统一承载分类与成像的训练、评估、可视化逻辑 | 想看各任务具体怎么组织时 |
| 共用工具 | [`artifacts.py`](../../artifacts.py) | 集中管理 optics 预设、checkpoint / manifest、可视化辅助与制造导出工具 | 想看共享参数和导出工具时 |
| 统一训练入口 | [`train.py`](../../train.py) | 通过 `--task classification|imaging` 负责分类与成像训练 | 想看当前正式入口怎么训练时 |
| 统一可视化入口 | [`visualize.py`](../../visualize.py) | 通过 `--task classification|imaging` 导出分类图与成像重建图 | 想看当前正式入口怎么出图时 |
| 相位板导出 | [`export_phase_plate.py`](../../export_phase_plate.py) | 从 checkpoint 导出 phase mask、height map、CSV、report 与 STL | 想看真实相位板前处理时 |

## 论文范围边界

| 范围                 | 对应内容                           | 备注                                    |
| ------------------ | ------------------------------ | ------------------------------------- |
| 主文主线               | MNIST 分类、Fashion-MNIST 分类、成像透镜 | 主文摘要、Fig. 1 到 Fig. 4 直接覆盖             |
| Supplementary 细化   | 训练细节、物理参数、损失函数、制造映射、实验附图       | Materials and Methods 与 Fig. S1 到 S16 |
| Supplementary 扩展实验 | 迁移学习、层数和神经元数消融、相位与复值调制对比       | 主要见 Fig. S1、S2、S5、S16                 |

> [!note]
> 后续规划时，主文复现先以前四个模块为边界；模块 5 到模块 6 统一放到 Supplementary 扩展处理。

## 当前判断

| 项目 | 结论 | 依据 |
|------|------|------|
| 当前源码是否可运行 | ✅ | 当前六文件结构下训练、可视化、相位板导出入口均可正常执行 |
| MNIST 主线是否闭环 | ✅ | 当前工作树已完成重训、测试和可视化导出 |
| Fashion-MNIST 主线是否闭环 | ✅ | 当前工作树已完成训练、测试和可视化导出 |
| 成像透镜主线是否闭环 | ✅ | 当前工作树已通过 `train.py --task imaging`、`visualize.py --task imaging`、正式 checkpoint 和重建图导出完成闭环 |
| 是否与论文原始数据完全一致 | ⚠ | 成像透镜当前复验使用 STL10 自然图像，论文主文展示的是 ImageNet 图像成像 |

## 主文结果对照

| 主文任务 | 论文原始记录 | 当前复验结果 | 备注 |
|------|------|------|------|
| MNIST 分类 | `97.61%` | `97.63% (9763/10000)` | 当前代码重训 20 epoch，耗时约 `350.5s` |
| Fashion-MNIST 分类 | `81.13%` | `87.49%` | 当前代码重训 20 epoch，结构仍为 5 层相位调制 |
| 成像透镜 | 主文展示太赫兹自然图像成像功能 | `STL10`, 10 epoch, `Test MSE = 0.0259` | 当前复验完成了功能链路与结果导出，但数据源不是论文原始 ImageNet |

## 模块总览

### 模块 1：基础框架 ✅

| 内容 | 状态 | 论文归属 | 文件 | 说明 |
|------|------|------|------|------|
| 衍射层 `DiffractiveLayer` | ✅ | 主文依赖 + Supplementary 细节 | [`d2nn.py`](../../d2nn.py) | 已通过实例化、checkpoint 加载和前向传播验证 |
| D2NN 完整网络 | ✅ | 主文依赖 + Supplementary 细节 | [`d2nn.py`](../../d2nn.py) | 已支撑 MNIST 与 Fashion-MNIST 全流程 |
| 输入编码：振幅编码 | ✅ | 主文主线 | [`d2nn.py`](../../d2nn.py) | 分类任务当前沿用主文振幅编码 |
| GPU 加速环境 | ✅ | 工程实现 | [`pyproject.toml`](../../pyproject.toml) | 已在 CUDA 下完成训练与可视化 |

### 模块 2：MNIST 手写数字分类 ✅

| 内容 | 状态 | 论文归属 | 文件 | 说明 |
|------|------|------|------|------|
| 训练流程 | ✅ | 主文主线 | [`train.py`](../../train.py) | 已完整训练 20 epochs |
| 训练产物 | ✅ | 主文主线 | [`checkpoints/best_mnist.pth`](../../checkpoints/best_mnist.pth) | 当前 checkpoint 已由本轮重训刷新 |
| 历史 checkpoint 备份 | ✅ | 工程记录 | [`checkpoints/best_mnist.pre_retrain_20260327.pth`](../../checkpoints/best_mnist.pre_retrain_20260327.pth) | 保留了重训前版本 |
| 可视化脚本 | ✅ | 主文主线 | [`visualize.py`](../../visualize.py) | 已用新 checkpoint 重新跑通 |
| 可视化结果 | ✅ | 主文主线 | [`figures/`](../../figures/) | 已与新 checkpoint 对齐重导 |

### 模块 3：Fashion-MNIST 分类 ✅

| 内容               | 状态  | 论文归属             | 说明                                                                                                                                                                                                                                                                             |
| ---------------- | --- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 输入编码             | ✅   | 主文主线             | 依据 Supplementary Methods，当前沿用与 digit classifier 相同的振幅编码                                                                                                                                                                                                                        |
| 5 层相位调制训练        | ✅   | 主文主线             | 已完成 20 epoch 训练，测试精度 `87.49%`                                                                                                                                                                                                                                                  |
| 训练产物             | ✅   | 主文主线             | 当前最佳模型为 [`checkpoints/best_fashion_mnist.pth`](../../checkpoints/best_fashion_mnist.pth)，并保留 [`checkpoints/best_fashion_mnist.pre_full_20260327.pth`](../../checkpoints/best_fashion_mnist.pre_full_20260327.pth) 备份 |
| 可视化与论文 Fig. 4 对比 | ✅   | 主文主线             | 已生成 [`figures/fashion_mnist/`](../../figures/fashion_mnist/) 下三张图                                                                                                                                                                     |
| 5 层复值调制训练        | ⬜   | Supplementary 扩展 | 更高结果与复值调制主要在补充材料展开                                                                                                                                                                                                                                                             |

### 模块 4：太赫兹成像透镜 ✅

| 内容 | 状态 | 论文归属 | 说明 |
|------|------|------|------|
| 成像网络训练 | ✅ | 主文主线 + Supplementary 细节 | 已实现 [`D2NNImager`](../../d2nn.py)，并通过 [`train.py`](../../train.py) 的 `--task imaging` 入口在 `STL10` 上完成 10 epoch 正式训练，`Test MSE = 0.0259` |
| 输出图像可视化 | ✅ | 主文主线 + Supplementary 细化 | 已通过 [`visualize.py`](../../visualize.py) 的 `--task imaging` 入口生成 [`figures/imager/sample_reconstructions.png`](../../figures/imager/sample_reconstructions.png) |
| 成像 checkpoint | ✅ | 工程记录 | 当前最佳模型为 [`checkpoints/best_imager_stl10.pth`](../../checkpoints/best_imager_stl10.pth)，并保留 [`checkpoints/best_imager_stl10.pre_full_20260327.pth`](../../checkpoints/best_imager_stl10.pre_full_20260327.pth) 备份 |
| 与论文数据的一致性 | ⚠ | 主文主线 | 当前复验使用 `STL10` 自然图像完成链路验证；若要严格对齐论文，还需补一版 ImageNet 图像训练与展示 |
| 点扩散函数 PSF 评估 | ⬜ | Supplementary 细化 | 主要对应 Fig. S7、S8，不阻塞当前主文闭环 |

### 模块 5：迁移学习实验 ⬜

| 内容 | 状态 | 论文归属 | 说明 |
|------|------|------|------|
| 冻结 5 层 + 新增 2 层训练 | ⬜ | Supplementary 扩展 | 对应 Fig. S2，主文只顺带引用 |
| Lego-like 拼接行为验证 | ⬜ | Supplementary 扩展 | 属于补充材料里的额外能力展示 |

### 模块 6：消融实验 ⬜

| 内容 | 状态 | 论文归属 | 说明 |
|------|------|------|------|
| 层数消融 | ⬜ | Supplementary 扩展 | 主要见 Fig. S1、S5 |
| 神经元数量消融 | ⬜ | Supplementary 扩展 | 主要见 Fig. S1 |
| 相位 vs 复值调制对比 | ⬜ | Supplementary 扩展 | 主要见 Fig. S1、S16 |

> [!done]
> 当前最适合的任务边界是：模块 1 到 4 视为主文复现主线，模块 5 到 6 统一作为 Supplementary 扩展。

## 技术参数

| 参数 | 论文值 | 当前项目设置 | 来源 |
|------|------|------|------|
| 工作波长 | 0.75 mm (0.4 THz) | 0.75 mm | 主文 + Supplementary |
| 层面积 | 8 cm × 8 cm | 分类器 `200×200 × 0.4 mm = 8 cm` | 主文 |
| 层间距 | 分类 `30 mm`，成像 `4 mm` | 分类 `30 mm`，成像 `4 mm` | Supplementary Methods |
| 像素间距 | 分类 `400 um`，成像 `300 um` | 分类 `0.4 mm`，成像 `0.3 mm` | Supplementary Methods |
| 层数 | 5 | 5 | 主文 |
| 可训练参数 | `200000` | `200000` | 工程换算 |
| 传播方法 | Angular spectrum implementation | 角谱法 | Supplementary Methods |

## 文件结构

```text
source/repos/d2nn/
├── pyproject.toml
├── uv.lock
├── d2nn.py
├── tasks.py
├── artifacts.py
├── train.py
├── visualize.py
├── export_phase_plate.py
├── checkpoints/
│   ├── best_mnist.pth
│   ├── best_mnist.pre_retrain_20260327.pth
│   ├── best_fashion_mnist.pth
│   ├── best_fashion_mnist.pre_full_20260327.pth
│   ├── best_imager_stl10.pth
│   └── best_imager_stl10.pre_full_20260327.pth
├── exports/
├── figures/
│   ├── fashion_mnist/
│   └── imager/
├── tests/
└── data/
    ├── MNIST/
    ├── FashionMNIST/
    └── stl10_binary/
```

## 运行方式

```bash
cd ~/source/repos/d2nn
uv run python train.py --task classification --dataset mnist --epochs 20 --size 200 --layers 5
uv run python train.py --task classification --dataset fashion-mnist --epochs 20 --size 200 --layers 5
uv run python train.py --task imaging --dataset stl10 --epochs 10 --size 200 --layers 5 --image-size 64 --batch-size 4
uv run python visualize.py --task classification --dataset mnist --checkpoint checkpoints/best_mnist.pth --no-show
uv run python visualize.py --task classification --dataset fashion-mnist --checkpoint checkpoints/best_fashion_mnist.pth --no-show
uv run python visualize.py --task imaging --dataset stl10 --checkpoint checkpoints/best_imager_stl10.pth --no-show
uv run python export_phase_plate.py --task classification --checkpoint checkpoints/best_fashion_mnist.pth --export-stl
```

## 当前计划

### P0：主文复现的最小闭环

- [x] 修复 [`d2nn.py`](../../d2nn.py) 中的实例化错误
- [x] 确认 `D2NN()` 可以成功创建
- [x] 用现有 checkpoint 跑通 [`visualize.py`](../../visualize.py)
- [x] 重新训练一轮 MNIST
- [x] 记录重新验证后的测试精度和耗时

### P1：主文主线补全

- [x] 完成 Fashion-MNIST 的主文版结果对齐
- [x] 完成成像透镜正式训练与结果导出
- [x] 把主文结果与当前复验结果分开记录

> [!note]
> 如果后续追求与论文原始展示完全一致，再单独补一版基于 ImageNet 图像的成像实验；这不再阻塞当前主文主线的闭环判断。

### P2：再进入 Supplementary 扩展

- [ ] 迁移学习实验
- [ ] 层数与神经元数量消融
- [ ] 相位与复值调制对比
- [ ] 其他 Supplementary 附图对齐

## 进展日志

- 2026-03-06：启动项目，确认论文来源
- 2026-03-08：生成 `visualize.py` 和 `figures` 下的可视化结果
- 2026-03-09：生成 `checkpoints/best_mnist.pth`
- 2026-03-27：核对主文与 Supplementary 的任务边界，确认模块 1 到 4 更适合作为主文复现主线，模块 5 到 6 归入补充扩展
- 2026-03-27：修复 `d2nn.py` 中的 `sizec` typo，恢复模型实例化、checkpoint 加载与前向传播
- 2026-03-27：用现有 checkpoint 重新跑通 `visualize.py`，并复验旧模型达到 `97.58% (9758/10000)`
- 2026-03-27：重新训练 20 个 epoch，得到新的 `best_mnist.pth`，测试精度 `97.63% (9763/10000)`，总耗时约 `350.5s`，并完成 `figures` 重导出
- 2026-03-27：为 `train.py` 和 `visualize.py` 增加 Fashion-MNIST 支持，完成 1 epoch 冒烟训练，得到 `best_fashion_mnist.pth`，测试精度 `82.69%`，并导出 `figures/fashion_mnist`
- 2026-03-27：完成 Fashion-MNIST 20 epoch 完整训练，得到新的 `best_fashion_mnist.pth`，测试精度 `87.49%`，并保留冒烟阶段备份 checkpoint
- 2026-03-27：实现 `D2NNImager`、`train_imager.py` 与 `visualize_imager.py`，完成成像透镜 1 epoch 冒烟训练，得到 `best_imager_stl10.pth`，`Test MSE = 0.0299`
- 2026-03-27：完成成像透镜 10 epoch 正式训练，刷新 `best_imager_stl10.pth` 为 `Test MSE = 0.0259`，并导出 `figures/imager/sample_reconstructions.png`
- 2026-03-28：补充共享配置、checkpoint / manifest 工具、可视化辅助与相位板导出公共层，形成统一的公共底座；新增最小单元测试，验证三条主线 checkpoint 的相位板导出链路。
- 2026-03-28：将顶层入口收敛为 train.py 与 visualize.py，并把分类与成像逻辑整理为更清楚的统一结构，补充仓库说明 README。
- 2026-03-28：增强相位板导出，新增 manufacturable height、thickness、quantized height、逐层 CSV 与 report，形成第一版制造前数值包。
- 2026-03-28：完成 best_fashion_mnist.pth 的完整相位板导出与逐层 STL 验证；当前材料参数下完整 2pi 相位包裹对应最大浮雕高度约 1037.8 um，而 max-relief-um = 300 会让每层约 49.9% 到 53.0% 的像素被截平，因此不能作为当前默认工艺上限。
- 2026-03-28：将仓库顶层 Python 文件收敛到 6 个主文件：d2nn.py、tasks.py、artifacts.py、train.py、visualize.py、export_phase_plate.py；删除旧的 imaging wrapper 与分散 helper 文件。
- 2026-03-29：将 README 收成正式复现交付说明，补充当前状态、推荐复现路径、推荐 checkpoint、论文对照结果，并把 train.py / visualize.py / export_phase_plate.py 的最小 smoke check 固化进测试。
- 2026-04-02：为后续非线性层实验补了训练产物命名隔离能力。`train.py` 新增 `--run-name` 参数，`tasks.py` 与 `artifacts.py` 增加基于实验名生成 checkpoint / manifest 变体文件名的逻辑，使 `baseline_5layer`、后续不同非线性机制和不同插入位置的实验不再覆盖同一个 `best_*.pth`。同时补充了对应的单元测试，覆盖默认命名、带实验后缀命名以及 Windows 非法文件名字符清洗。
- 2026-04-02：完成非线性层开工前的最后一轮工程收口。`train.py` 正式加入 `--seed` 与 `--experiment-stage`，分类与成像训练的 manifest 现在统一记录实验阶段、随机种子和光学配置；`visualize.py` 新增 `--seed`，并在成像 `imagefolder` 场景下优先从 checkpoint 相邻 manifest 继承训练时的种子，避免训练与可视化测试划分不一致；仓库内新增 `docs/baselines/fashion-mnist-phase-only-5layer-baseline.md`，把当前 `5 层 phase-only + Fashion-MNIST` 对照组正式冻结为后续非线性机制与插入位置消融的基线记录。
- 2026-04-02：通过 Claude Code 的 ACP 风格只读 review 对非线性层前的收尾代码做了独立审查。结论是当前不存在阻塞进入 nonlinear layer implementation 的问题；主要剩余建议是后续再顺手收敛 `D2NN._embed_input` 与 `embed_amplitude_image` 的重复实现，并清理 `resolve_imaging_optics` 里对 `resolve_optics` 的冗余默认值注入。与此同时，已将 phase-only 阶段的关键大文件产物整理到 GitHub Release `pre-nonlinear-phase-only-v1`，并把阶段产物统一走 Release、源码仓库只保留文档/manifest/小体积可编辑参考文件的规则固化到仓库说明与长期记忆。
- 2026-04-02：完成 Claude review 建议中的最后两项结构收尾：`D2NN._embed_input` 现已直接复用 `embed_amplitude_image`，分类与成像输入嵌入共享同一套基底实现；`resolve_imaging_optics` 中对 `resolve_optics` 的冗余默认值注入已删除，默认光学参数的回退统一收口到底层 `OpticalConfig.with_overrides` 逻辑。至此，进入 nonlinear layer implementation 前的代码结构清理工作全部完成。
- 2026-04-02：吸收 `implementation_plan.md` 中对 nonlinear 工程 `Step 0 / Step 1` 的拆分方式，并已在代码中完成对应落地：新增 `safe_abs()`、`FieldActivationBase`、`IdentityActivation` 以及按层插入 activation 的 placement skeleton，当前默认关闭；`train.py` / `tasks.py` / `artifacts.py` 已能记录 `activation_type`、`activation_positions` 和占位超参数到 manifest；同时补齐 baseline checkpoint 兼容、`identity` 前向等价、shape / dtype 稳定性与一次 backward smoke test。当前完整测试结果更新为 `Ran 33 tests ... OK`，可以正式进入第一种真实非线性机制的实现。
- 2026-04-03：完成第一种真实非线性机制 `CoherentAmplitudeActivation` 的最小可用实现。当前实现采用强度相关的 sigmoid gain，对复场只做相干振幅门控、不改相位，可通过 `activation_type="coherent_amplitude"` 与 `activation_positions` 接入现有骨架；训练入口已开放 `threshold / temperature / gain_min / gain_max`。同时补齐了对应的物理边界测试与一次 backward smoke test，完整测试结果更新为 `Ran 37 tests ... OK`。这意味着后续可以直接进入下一步：围绕该机制做超参数收敛和位置消融，而不必再回头补第一版接口。
- 2026-04-03：继续沿第一种真实机制推进，补齐了 `CoherentAmplitudeActivation` 的实验诊断支撑。当前每个 activation 位置都会记录 `mean_intensity / mean_gain / min_gain / max_gain`，模型侧可通过 `activation_diagnostics()` 汇总，训练日志会按轮输出 activation stats，manifest 也会保存该摘要。对应测试已补齐并通过，完整测试结果更新为 `Ran 39 tests ... OK`。这一步完成后，后续做阈值、温度和插入位置实验时，不再只能盯最终精度，而可以直接观察每层 gate 行为。
- 2026-04-03：正式确认代码分支边界：`phase-only-baseline` 继续保留为无线性层阶段的完整可编辑主线，`pre-nonlinear-phase-only-v1` 作为其冻结快照 tag；`main` 作为当前 nonlinear 工程推进主线继续向前。后续每完成一个重要非线性阶段，都按“主线提交 + tag/release 冻结”的方式管理，而不回写旧的 phase-only 分支。
- 2026-04-03：补完了 nonlinear 实验的自动 `run-name` 生成逻辑。当前当 `activation_type != none` 且未显式传入 `--run-name` 时，会根据 `experiment_stage`、`activation_type`、`activation_positions`、`activation_hparams` 和 `seed` 自动生成稳定且可回溯的实验名；同时保持 phase-only 默认不改名、显式 `--run-name` 原样优先。对应单元测试与全量回归已更新为 `Ran 42 tests ... OK`。
- 2026-04-03：为 `CoherentAmplitudeActivation` 的后续收敛实验补了第一层超参数 preset 支撑。`train.py` 现已接受 `--activation-preset {conservative|balanced|aggressive}`，当前仅对 `coherent_amplitude` 生效；`tasks.py` 会在未显式给出阈值、温度和 gain 区间时，用 preset 填入推荐默认值，并继续保持显式参数优先。对应单元测试与全量回归已更新为 `Ran 44 tests ... OK`。
- 2026-04-03：继续为后续位置消融补实验入口。`train.py` 新增 `--activation-placement {front|mid|back|all}`，`tasks.py` 现可在未显式给出 `--activation-positions` 时，按当前层数把 `front / mid / back / all` 映射为稳定的 1-based 插入位置；若同时给出显式位置，则仍以显式位置优先。这样后续做同机制、不同插入位置的消融时，不必反复手填层号。对应单元测试与全量回归已更新为 `Ran 46 tests ... OK`。
- 2026-04-03：继续把非线性实验支撑从“可配置”推进到“可执行”。`tasks.py` 新增了固定实验网格生成与命令格式化入口，当前先覆盖 `coherent_amplitude_positions` 和 `coherent_amplitude_presets` 两类 sweep；`train.py` 新增 `--print-experiment-grid`，可以直接打印对应的一组训练命令而不启动训练。这样后续做同一机制的位置消融和 preset 收敛时，可以先从统一模板展开，避免手工拼命令导致实验矩阵不一致。对应单元测试、全量回归以及 `train.py --print-experiment-grid coherent_amplitude_positions` 的实命令输出验证均已通过，当前测试结果为 `Ran 48 tests ... OK`。
- 2026-04-03：完成第二种非线性机制 `CoherentPhaseActivation` 的第一版最小实现。当前实现采用强度相关的相位偏移 `phase_shift = gamma * I`，只改变复场相位、不改变模长，并已接入现有 activation skeleton 与训练入口 `--activation-type coherent_phase`。同时补齐了对应的物理边界测试与一次 backward smoke test，并让 activation diagnostics 开始记录 `mean_phase_shift` 等相位统计量，为后续机制消融做好准备；本轮全量回归继续通过。
- 2026-04-03：继续把 `CoherentPhaseActivation` 从“已实现”推进到“可实验”。当前 `tasks.py` 已补入 `COHERENT_PHASE_PRESETS`，`resolve_activation_config()` 现在会对 `coherent_phase` 正确应用 `conservative / balanced / aggressive` 三档 `gamma` 默认值；同时新增了 `coherent_phase_presets` 与 `coherent_activation_mechanisms` 两类 printable experiment grid，`train.py --print-experiment-grid coherent_activation_mechanisms` 已可直接打印 amplitude-vs-phase 的同位置机制消融命令。对应单元测试与全量回归均已通过，当前测试结果更新为 `Ran 56 tests ... OK`。
- 2026-04-03：完成第三种机制 `IncoherentIntensityActivation` 的第一版最小实现，并继续补齐到“可实验”层级。当前实现采用 `relu(responsivity * I - threshold)` 生成新的输出振幅，并在 `emission_phase_mode='zero'` 下显式丢弃输入相位、重建零相位输出；训练入口已开放 `--activation-type incoherent_intensity`。与此同时，`tasks.py` 已补入 `INCOHERENT_INTENSITY_PRESETS`、`incoherent_intensity_presets` grid 与覆盖三类机制的 `activation_mechanisms` grid，`train.py --print-experiment-grid activation_mechanisms` 现在可直接打印 amplitude / phase / incoherent 三种机制的同位置消融命令。当前全量回归结果更新为 `Ran 64 tests ... OK`。
- 2026-04-03：继续把机制消融从“可打印命令”推进到“可顺序执行”。`tasks.py` 新增 `execute_experiment_grid()`，会把 experiment grid 逐条展开成独立的参数集并交给训练回调；`train.py` 新增 `--run-experiment-grid`，现在可以直接顺序运行预定义的机制或位置实验网格，而不再需要手工一条条复制命令。对应 parser、grid runner 与全量回归测试均已通过，当前测试结果更新为 `Ran 66 tests ... OK`。
- 2026-04-03：完成第一轮真实机制消融冒烟实验，配置为 `Fashion-MNIST + 5 layers + mid placement + balanced preset + 1 epoch`，通过 `train.py --run-experiment-grid activation_mechanisms` 顺序跑通三类机制。当前测试集结果分别为：`coherent_amplitude = 81.04%`、`coherent_phase = 81.08%`、`incoherent_intensity = 80.29%`；对应验证集结果分别为 `82.22% / 81.78% / 80.88%`。这一轮的意义主要是验证三类机制都能稳定训练、命名与 manifest 链路闭环、activation diagnostics 能正常输出；暂时还不把这组 1 epoch 结果当作最终机制优劣结论。
- 2026-04-03：将 mechanism ablation 从 1 epoch 冒烟放大到 5 epoch 正式对比，配置仍保持 `Fashion-MNIST + 5 layers + mid placement + balanced preset`。当前测试集结果为：`coherent_amplitude = 83.96%`、`coherent_phase = 83.22%`、`incoherent_intensity = 85.28%`；对应最佳验证集结果为 `84.60% / 83.88% / 85.56%`。这说明在当前实现与当前数据集下，`incoherent_intensity` 已经从“可训练”进一步表现成机制消融里的领先候选，因此下一步转入位置消融是合理的。
- 2026-04-03：围绕当前领先机制 `incoherent_intensity` 完成了第一轮位置消融，配置为 `Fashion-MNIST + 5 layers + balanced preset + 3 epochs`。当前测试集结果为：`front = 82.92%`、`mid = 84.19%`、`back = 84.98%`、`all = 65.06%`；对应最佳验证集结果为 `83.46% / 84.90% / 85.24% / 65.48%`。这说明后部插入明显优于前部与中部，而“每层都加”会导致层间输出振幅快速爆炸，当前 diagnostics 已直接暴露出后层幅度级联失稳，因此后续位置线优先围绕 `back` 继续放大训练，而 `all` 暂不作为主力候选。
- 2026-04-03：将当前最优配置 `incoherent_intensity + back` 从 5 epoch 继续放大到 10 / 20 epoch，并显式保存为 `incoherent_back_10ep` 与 `incoherent_back_20ep`，避免覆盖前面的消融产物。结果显示：`10 epoch` 时 test `86.67%`、best val `87.42%`；`20 epoch` 时 test `87.61%`，已经首次超过当前 phase-only Fashion-MNIST 主线 `87.49%`。同时后层 diagnostics 继续保持稳定收敛，`L5` 输出振幅均值逐步下降到约 `0.004`，没有再出现 “all placement” 那种级联爆炸现象。
- 2026-04-03：完成 `incoherent_intensity + back + 20 epoch` 的多 seed 稳定性验证，新增 `seed=7` 与 `seed=123` 两条复现实验。当前三组 test accuracy 分别为 `87.61% (seed=42)`、`87.36% (seed=7)`、`87.28% (seed=123)`，均值约 `87.42%`，波动约 `±0.14%`；对应最佳验证集结果分别为 `87.84% / 88.26% / 88.38%`。这说明当前领先结果不是单次偶然，而是在不同随机划分与训练顺序下都比较稳定地维持在 phase-only 主线附近或略高的位置。
- 2026-04-03：为“更复杂数据集迁移”先补齐了 `grayscale CIFAR-10` 的代码入口，而没有直接切到 RGB 版本。当前 `tasks.py` 已新增 `cifar10_gray` 分类配置、单通道灰度 transform 与按数据集总量自适应的 train/val split 逻辑，`train.py` 与 `visualize.py` 的 dataset 帮助信息也同步补入 `cifar10-gray`。对应新增测试覆盖了配置解析、灰度 transform 输出为单通道以及 50k/60k 训练集场景下的 split 长度，完整回归结果更新为 `Ran 69 tests ... OK`。这一步的目的不是直接给出 CIFAR 结论，而是把下一阶段“baseline vs nonlinear on grayscale CIFAR-10”的代码通路先打通。
- 2026-04-03：完成 `grayscale CIFAR-10` 上的第一轮正式迁移对照，先在 `seed=42` 下跑通 `5 epoch` 的 phase-only baseline 与 `incoherent_intensity + back`。当前 baseline test accuracy 为 `33.17%`，对应 nonlinear 配置达到 `37.73%`，领先 `4.56` 个点；同时 `L5` 的 activation diagnostics 从 `A≈0.010` 继续收敛到 `A≈0.004`，没有出现幅度级联爆炸。
- 2026-04-03：继续补 `grayscale CIFAR-10` 的第二个随机种子验证。在 `seed=7` 下，`5 epoch` phase-only baseline 的 test accuracy 为 `33.43%`，对应的 `incoherent_intensity + back` 为 `37.29%`，领先 `3.86` 个点。到目前为止，这条 nonlinear 配置已经不只是在 Fashion-MNIST 上有效，而是在更复杂的 grayscale CIFAR-10 上也表现出稳定收益。
- 2026-04-03：将 `grayscale CIFAR-10` 的 `seed=42` 对照从 `5 epoch` 继续放大到 `10 epoch`。当前 phase-only baseline 的 test accuracy 提升到 `35.09%`，而对应的 `incoherent_intensity + back` 提升到 `39.48%`，仍领先 `4.39` 个点；最佳验证集从 `35.40%` 提升到 `40.00%`，同时 `L5` 的 activation diagnostics 继续平稳收敛到 `A≈0.002`、`I≈0.023`，说明这条非线性线在更长训练预算下仍保持稳定优势。
- 2026-04-03：完成通过 Agent 对非线性层代码的前置独立审查，确认机制逻辑匹配、诊断管线完整、70 项单测全部跑通。顺延修复 4 项周边非阻塞问题：去除 `IncoherentIntensityActivation` 内 `emission_phase_mode="zero"` 时的零相位冗余指数运算；补充明确的 `--activation-preset` 帮助信息；修正 `execute_experiment_grid` 构造规格时外层与内层 `activation_positions` 的优先覆盖问题；补充对 `front / back / all` 映射机制的单元测试保障。目前系统稳定性完全收敛，下一步可正式开始 CIFAR-10/10-epoch 基线确认或消融实验。
- 2026-04-03：补完 `grayscale CIFAR-10` 的 `10 epoch, seed=7` 对照。`phase-only baseline` test acc `35.10%`，`incoherent_intensity + back` test acc `39.88%`，领先 `4.78` 个点；对应 best val 分别为 `34.78%` 和 `39.84%`。这说明在 `10 epoch` 预算下，非线性收益不只在 `seed=42` 下成立。
- 2026-04-03：完成 RGB CIFAR-10 引入前的全量代码重构与清理。新增 `D2NNBase` 基类，统一收敛 `D2NN` 与 `D2NNImager` 内部重复的层初始化、传播逻辑与非线性诊断流；重构 `_embed_input` 并修正 `patch_size` 校验策略，确保 3 通道 RGB 输入能够无损注入目标视场中心。目前所有前置重构均不影响既有 phase-only 与 nonlinear 机制，对应的 77 项全量单元测试已重新通过。代码整体复用率和健壮性大幅提升，正式为 RGB 实验和版本终态做好准备。
- 2026-04-03：按阶段节点规则尝试使用本机 Claude Code CLI 做独立 review，但当前环境未登录，未产生可用输出，因此不记作外部审查；随后改做 fallback 只读 review。结论是当前没有阻塞是否冻结新阶段的硬问题，但需注意 `grayscale CIFAR-10` 非线性产物的命名仍不完全统一，旧的 `seed=42` 10 epoch artifact 文件名没有显式带 `10ep`，后续若做 tag/release 需要在说明中明确这条线的命名边界。
- 2026-04-03：已补完 `RGB CIFAR-10` 的最小分类入口，并保持现有 nonlinear 主线不变：数据集入口新增 `cifar10_rgb`，分类输入嵌入现已支持把 `R / G / B` 三个通道映射到同一输入窗口中的三个固定子区域，训练与可视化 CLI 也已同步接受 `cifar10-rgb`。
- 2026-04-03：完成 `RGB CIFAR-10, 10 epoch, seed=42` 的第一轮正式对照。`phase-only baseline` test acc `44.01%`，`incoherent_intensity + back` test acc `46.60%`，领先 `2.59` 个点；对应 best val 分别为 `44.20%` 和 `46.78%`。这说明非线性收益在从 grayscale CIFAR-10 继续迁移到 RGB CIFAR-10 时尚未消失，只是幅度较灰度阶段有所收窄。
- 2026-04-03：按阶段节点规则再次尝试使用本机 Claude Code CLI 做独立 review，但当前环境仍未登录，未产生可用输出，因此继续采用 fallback 只读 review。该轮 review 未发现阻塞当前代码节点提交的硬问题，只顺手收掉了一处 `visualize.py` dataset help 未同步 `cifar10-rgb` 的 CLI 一致性问题；修正后全量回归更新为 `Ran 75 tests ... OK`。
- 2026-04-03：根据独立审查意见，完成 RGB CIFAR-10 相关分支的两项小型非阻塞问题修复：为极小 target_size 的 RGB patch 嵌入增加异常拦截保护；在 `D2NNImager` 中同步补齐对三通道输入的兼容分发逻辑，并新增对应单测防回归。当前回归全量通过 (`Ran 77 tests ... OK`)，工程基线干净，执行提交。
- 2026-04-03：补完 `RGB CIFAR-10, 10 epoch, seed=7` 对照。`phase-only baseline` test acc `44.12%`，`incoherent_intensity + back` test acc `47.08%`，领先 `2.96` 个点；与 `seed=42` 的 `44.01% -> 46.60%` 一起看，当前 RGB 阶段两条 10 epoch 线的平均提升约为 `+2.78` 个点。
- 2026-04-03：在当前代码节点 `b985442` 上按阶段节点规则再次尝试使用本机 Claude Code CLI 做独立 review，但当前环境仍未登录，未产生可用输出，因此继续采用 fallback 只读 review。结论仍是没有阻塞 RGB 阶段冻结的硬问题。
- 2026-04-03：已将当前 `RGB CIFAR-10 + incoherent_intensity + back + 10 epoch + seed 42/7` 的阶段产物整理并发布到 GitHub Release `nonlinear-incoherent-back-cifar10-rgb-v1`，对应 tag 同名。至此，非线性收益从 Fashion-MNIST -> grayscale CIFAR-10 -> RGB CIFAR-10 的迁移链条都已有阶段性冻结节点。
- 2026-04-03：补完 `RGB CIFAR-10, 10 epoch` 稳定性确认所需的新增三组 seed 对照。`seed=123` 下 `phase-only baseline` test acc `43.66%`，`incoherent_intensity + back` test acc `44.86%`，领先 `1.20` 个点；`seed=0` 下对应为 `44.47% -> 45.89%`，领先 `1.42` 个点；`seed=2025` 下对应为 `44.07% -> 45.45%`，领先 `1.38` 个点。
- 2026-04-03：完成 `RGB CIFAR-10 + incoherent_intensity + back + 10 epoch` 的五 seed 汇总。当前 `phase-only baseline` 五 seed 平均 test acc 为 `44.07%`，对应 nonlinear 五 seed 平均为 `45.98%`，平均提升 `+1.91` 个点，lift 区间为 `+1.20 ~ +2.96`，spread 为 `1.76`。五个 seed 全部保持正收益，但按当前预设门槛（至少 `4/5` seed `> +1.5 pt` 且平均 lift `> +2.0 pt`）仍不足以记为“stable”，因此本轮结论记为 `stability inconclusive`，而不是强行上调成稳定结论。
- 2026-04-03：按阶段节点规则再次尝试做独立 review。本机 `claude` 路径仍存在，但当前 ACP 环境缺少其依赖的 `node`，因此未产生可用外部审查输出，不计作 Claude review；随后执行 fallback 只读 review，确认五组 RGB manifest 的 `run_name / seed / experiment_stage / activation` 配对关系一致，未发现阻塞当前阶段判断的命名、CLI 或产物管理硬问题。当前需要回头判断的不是 pipeline 是否有 bug，而是这条 RGB 收益线是否已经达到继续追加更长训练预算的证据强度。
- 2026-04-04：按“计划出口前最后一轮预算敏感性确认”的定义，补完 `RGB CIFAR-10, 20 epoch` 的三组对照。`seed=42` 下 `phase-only baseline` test acc `45.10%`，`incoherent_intensity + back` test acc `47.72%`，领先 `2.62` 个点；`seed=7` 下对应为 `45.38% -> 47.96%`，领先 `2.58` 个点；`seed=123` 下对应为 `45.33% -> 47.75%`，领先 `2.42` 个点。
- 2026-04-04：完成 `RGB CIFAR-10 + incoherent_intensity + back + 20 epoch` 的三 seed 汇总。当前 baseline mean test acc 为 `45.27%`，对应 nonlinear mean 为 `47.81%`，mean lift `+2.54` 个点，lift spread 仅 `0.20`。与此前 `10 epoch` 下的 `mean lift = +1.91`、`spread = 1.76` 相比，当前更长训练预算明显提高了 RGB 阶段的证据强度与一致性。
- 2026-04-04：按阶段节点规则再次尝试独立 review。本机 `claude` 命令路径仍存在，但当前 ACP 环境依然缺少其依赖的 `node`，因此未产生可用外部审查输出，不计作 Claude review；随后执行 fallback 只读 review，确认六个 `20ep` manifest 的 `run_name / seed / experiment_stage / activation` 配对与命名一致，未发现阻塞当前阶段结论的硬问题。基于这一轮结果，当前非线性验证主线可以在 RGB 节点收口：`incoherent_intensity + back` 在更长预算下继续保持稳定正收益，因此“从原始任务迁移到更复杂 RGB CIFAR-10”这一段已完成闭环。
- 2026-04-05：修复当前 D2NN 数值主干中的三处明确问题，并已推送到 `main`（commit `94bb34b`）。`d2nn.py` 现已把层间传播与输出传播统一改为带 zero-padding 的 ASM 传播路径，以避免原先未补零 FFT 带来的周期边界混叠；同时把强度计算从 `safe_abs(u) ** 2` 改为显式 `real^2 + imag^2`，去掉不必要的 `sqrt -> square` 往返与 `eps` 对物理强度的污染；`D2NN._build_detectors()` 也修正为严格尊重偶数 `detector_size`，不再默认扩成奇数边长。为避免旧 checkpoint 因 `H / H_out / detector_masks` buffer 形状变化而失配，当前 `load_state_dict()` 路径会忽略这些派生 buffer 并按现配置重建。对应新增 4 条回归测试覆盖 padded transfer grid、偶数 detector 尺寸、无偏强度 helper 与旧 buffer 兼容加载；完整验证结果为 `uv run python -m pytest`，`81 passed`。

## 非线性层扩展文献

> 当前最值得单独拉出来的一条扩展线不是复值调制，而是 activation-style nonlinearity：在衍射层之间插入强度相关的非线性模块，再比较插入位置与任务复杂度对收益的影响。

- `UU9LC2D7` A surface-normal photodetector as nonlinear activation function in diffractive optical neural networks
  作用：把“非线性层”具体化成 photodetector-based activation，适合作为 activation baseline。
- `4QED5A8M` Multilayer nonlinear diffraction neural networks with programmable and fast ReLU activation function
  作用：最完整的 multilayer nonlinear D2NN 参考，覆盖 interleaved linear/nonlinear architecture、可调 ReLU 和更复杂任务。
- `Q568L2D4` Scalable multilayer diffractive neural network with all-optical nonlinear activation
  作用：支持“复杂任务和更深网络下，非线性收益更值得看”这条判断。
- `P22VML5G` Second-harmonic generation for enhancing the performance of diffractive neural networks
  作用：最贴当前痛点，优先拿来参考非线性层插入位置消融。
- `R74Y3P9E` Nonlinear encoding in diffractive information processing using linear optical materials
  作用：保留作边界对照，避免把 nonlinear encoding 的收益误判为层间 activation 的收益。

> 当前阅读顺序：`P22VML5G -> UU9LC2D7 -> 4QED5A8M -> Q568L2D4 -> R74Y3P9E`。

## 非线性层专项入口

> [!important]
> 非线性层相关的阶段性编程方案、实验边界和后续任务拆解，统一转到 [`nonlinear-layer-plan.md`](./nonlinear-layer-plan.md)。主文复现总文档这里只保留总入口，不再把细节继续堆在这里。

- 2026-04-06：完成 `Fashion-MNIST phase-only 5-layer physics-aligned` fabrication baseline 的当前阶段收口。已按 `fabrication_baseline` / `seed=42` 重训生成 `checkpoints/best_fashion_mnist.baseline_5layer_physics_aligned.pth` 与相邻 manifest，当前 `best val accuracy = 85.34%`、`test accuracy = 84.53%`。同时已生成 understanding-report 图集（`figures/fashion_mnist/baseline_5layer_physics_aligned/`）并保留 `incoherent_back_20ep` 作为 nonlinear 对照图集；量化敏感性显示 `16-128` 级基本保持不变，`8` 级开始出现可见下降。fabrication dry-run 已跑通，导出包位于 `exports/best_fashion_mnist.baseline_5layer_physics_aligned/best_fashion_mnist.baseline_5layer_physics_aligned/`，当前 dry-run 下 `raw height max = 1037.775 um`、`exported relief max = 1037.775 um`、`thickness max = 1537.775 um`、`clipped pixels = 0`，但 `max_relief` 仍未冻结，因此这次只记录为 traceable export，不把 manufacturable 结论写死。该阶段节点已完成独立 review：本机 Claude Code review 可用且已返回 APPROVED，未发现阻塞当前 fabrication baseline 收口的硬问题；剩余风险已明确集中在实验室侧的 relief limit / orientation labeling / 真实光路参数确认。
- 2026-04-07：补完进入实验室前的统一 handoff note，文件位于 `docs/fabrication/fashion-mnist-phase-only-lab-handoff.md`。当前已把“待确认实验室参数清单”“final export 命令模板”“加工前检查项”和“第一轮上光路 SOP”收口到同一页里；这样在实验室参数尚未到手时，主线动作就不再是继续改模型，而是明确缺什么参数、参数一到手后如何直接导出最终制造包并完成加工前判定。
- 2026-04-07：完成 mnist5-phaseonly-aligned 的 final export preset + validator wrapper 工程化，新增 export_fmnist5_phaseonly_aligned_final.py，内部冻结官方 checkpoint / optics 预设，只暴露实验室参数，并在导出后追加 alidation_summary.json。本轮 stage review 已实际通过本机 Claude Code CLI 执行；返回的外部 review 中有若干与当前事实不符的误报，复核后仅吸收了 checkpoint 路径校验这一条并补上回归测试。当前 wrapper 已在本地 dry-run 跑通：当工作树缺少官方 checkpoint 时，会先由 docs/official-artifacts/fmnist5-phaseonly-aligned/phase_masks.npy 受控反建本地 checkpoint，再调用底层 export_phase_plate.py 完成导出。以 efractive_index = 1.7227、mbient_index = 1.0、ase_thickness_um = 500、max_relief_um = 900、quantization_levels = 256 的 dry-run 为例，alidation_summary.json 给出的结论是 STOP，因为 aw height max = 1037.775 um 而 relief limit 只有 900 um，导致 clipped_pixels = 63982、clipped_fraction = 31.99%；这说明当前主线已经从『如何手工拼 final export 命令』收口到『等实验室确认真实 relief/material 参数后，用 wrapper 直接出最终包并根据 summary 做 PASS/WARN/STOP 判定』。
- 2026-04-08：完成导师要求的 RS-only 主干重构。当前分类/成像主前向已从 `d2nn.py` 中移除 `torch.fft` 主路径，改为直空间 Rayleigh-Sommerfeld 传播；结构上明确拆成 `RayleighSommerfeldPropagation`、`DiffractiveLayer`、`DetectorLayer` 和 `DiffractiveNetwork`。分类训练改为 composite loss：`alpha * MSE + beta * CrossEntropy + gamma * phase regularization`，默认权重为 `alpha=1.0, beta=0.1, gamma=0.01`；checkpoint 选择规则改为 `val accuracy -> val contrast -> later epoch`；manifest 新增 `model_version = rs_v1`、`loss_config` 和逐 epoch 的 `history`；`visualize.py` 对应补齐了 accuracy/contrast history 图输出。按阶段节点规则尝试过外部 subagent review，但当前平台额度限制导致未返回可用输出，因此不计作外部审查；随后执行了 fallback 本地独立 review，并据此修正了一处契约偏差：`forward()` 不再错误依赖 `forward_with_metrics()`，重新保持为轻量推理路径。当前验证结果：`uv run python -m pytest` 通过，`116 passed`；剩余主要风险不是正确性而是计算代价，即默认 `200x200`、`5 层` 的 RS 直积分训练会明显慢于此前 ASM/FFT 版本。
- 2026-04-08：完成 `train.py` 精简瘦身阶段。当前把 argparse 构造移入 `train_options.py`，composite loss 移入 `train_losses.py`，history/checkpoint/manifest helper 移入 `train_metrics.py`；`train.py` 保留训练入口、classification epoch、loader/model 构建和训练流程编排，并去掉此前为导师 prompt 对照添加的说明注释与无信息增量的薄包装函数。本轮阶段 review 已实际通过本机 Claude Code CLI 对当前 diff 和新增文件执行只读审查，返回 `No findings`；本地验证为 `uv run python -m pytest` 通过，`123 passed`。
