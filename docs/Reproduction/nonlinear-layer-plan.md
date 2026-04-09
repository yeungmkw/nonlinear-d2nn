---
created: 2026-03-31
type: project
status: active
category: 学业
tags:
  - project
  - D2NN
  - optics
  - nonlinear
paper: Lin et al. 2018 D2NN
source_note: docs/Reproduction/lin-2018-main-log.md
---


# 论文复现 · Lin et al. 2018 · 非线性层方案

> 关联主线：[`lin-2018-main-log.md`](./lin-2018-main-log.md)
> 文献入口：外部文献索引仍保留在 Zotero / Obsidian，本仓库只保留实现与实验决策记录
> 仓库入口：[`README.md`](../../README.md)

> [!summary]
> 当前最合适的编程主线不是一上来做真实器件级非线性，而是先把非线性层框架改成可插拔、可扩展的多机制接口，然后按“工程准备 -> 机制消融 -> 位置消融 -> 复杂数据集迁移”的顺序推进。第一阶段主打相干振幅型非线性，第二阶段再扩展到相干相位型和非相干光电型。

## 与 implementation_plan.md 并轨

Gemini 新给出的 `implementation_plan.md` 里，真正值得吸收的是把开工前工作重新压缩成两个非常明确的阶段：

1. `Step 0` 只做工程收口和回归保护，不提前引入真实非线性公式。
2. `Step 1` 只做默认关闭的可插拔骨架，先证明“接得进去而且不破坏旧主线”。

这两步与本笔记原来的路线一致，因此不单独分叉新方案，而是直接并入本方案的阶段 0/1。

### 当前并轨后的状态

- [x] Step 0：统一传播接口、冻结 phase-only 基线、扩展 run-name / seed / experiment-stage / manifest 元数据
- [x] Step 0：补齐“默认无非线性时行为不变”的回归测试，包括 baseline checkpoint 加载、默认关闭前向等价、manifest 可序列化
- [x] Step 1：新增 `safe_abs()`、`FieldActivationBase`、`IdentityActivation`
- [x] Step 1：在 `D2NN` / `D2NNImager` 中接入按层插入 activation 的 placement skeleton，默认关闭
- [x] Step 1：在训练入口加入 `--activation-type`、`--activation-positions` 及占位超参数接口，当前仅开放 `none / identity`
- [x] Step 1：补齐显式插入 `identity` 时的 shape / dtype / forward 等价性与一次 backward smoke test

> [!done]
> 到这里为止，非线性工程的“开工前准备”和“默认关闭骨架”都已经完成。下一步不再继续扩展接口，而是直接进入第一种真实机制：`CoherentAmplitudeActivation`。

### 当前阶段更新

- [x] `CoherentAmplitudeActivation` 已完成第一版最小实现
- [x] 当前形式采用强度相关 sigmoid gain，对应 `gain_min ~ gain_max` 区间内的相干振幅门控
- [x] 已接入现有 activation skeleton，可通过 `activation_type="coherent_amplitude"` 和 `activation_positions` 启用
- [x] 训练入口已开放 `coherent_amplitude`，并支持 `threshold / temperature / gain_min / gain_max`
- [x] 已补齐物理边界测试：相位保持不变、输出模长不放大、低高强度 gain 有序、一次 backward smoke test
- [x] 已补齐实验诊断接口：每个 activation 位置现在会记录 `mean_intensity / mean_gain / min_gain / max_gain`

> [!note]
> 当前这一版仍然是阶段 2 的“最小可用机制版”，还没有加入动态阈值初始化、按层可学习 hparams 或位置消融实验。

### 当前阶段再更新

为了支撑后续对 `CoherentAmplitudeActivation` 的超参数收敛和位置消融，当前模型与训练侧已经能导出每层 activation 的即时统计：

- `D2NN.activation_diagnostics()` / `D2NNImager.activation_diagnostics()`
- 训练过程中的每轮 `activation stats` 摘要输出
- manifest 中落盘的 `activation_diagnostics`

这一步的意义不是增加新机制，而是让第一种机制从“可运行”进入“可调试、可比较”的状态。后续如果某组参数训练不收敛，可以直接先看是 gain 过低、过高，还是层间强度分布出了问题。

## 一句话结论

这条线是可行的，但还不适合不加准备就直接开工。真正开始实现前，必须先补齐传播接口收口、checkpoint/manifest 扩展、实验命名规范和基线回归测试；否则中途最容易在旧 checkpoint 失配、可视化与训练逻辑不一致、实验结果互相覆盖这三件事上返工。

## 目标与边界

### 当前目标

在不破坏现有主文复现闭环的前提下，为当前仓库增加一条独立的 nonlinear activation 实验线，回答四个问题：

1. 非线性层是否在当前 D2NN 基线上带来稳定收益
2. 三类不同物理机制里，哪一类最值得继续做
3. 非线性层加在什么位置更有效
4. 这种收益在更复杂数据集上是否仍然存在

### 当前不做的事

- 不把 `nonlinear encoding` 当作主线实现
- 不在第一阶段绑定某一种真实器件工艺
- 不同时改层数、调制方式、检测方案和数据集，避免变量缠在一起
- 不把 Supplementary 里的复值调制和非线性层混成一个实验
- 不在第一版就引入全尺寸 `H x W` 空间异质性可学习参数图

> [!note]
> 这条线的第一阶段是“算法与结构验证”，不是“器件实现收口”。

## 文献结论如何转成代码路线

### 主线文献

主线文献当前仍从外部文献索引进入。当前实际起支撑作用的几篇分别承担的是：

- photodetector-based activation：说明“非线性层”应被理解为层间 activation module，而不是输入端编码技巧
- programmable fast ReLU：说明 interleaved linear/nonlinear architecture 是一条完整系统路线，而且复杂任务上能见到收益
- scalable all-optical nonlinear activation：支持“任务更复杂时，非线性更值得看”
- second-harmonic generation：最直接支持“非线性层插入位置消融”
- nonlinear encoding：作为边界对照，提醒后续不要把 encoding 改进误判成 activation 收益

### 从文献中提炼出的三类算子

当前方案不再只围绕一个抽象 `IntensityGateActivation` 展开，而是把第一阶段框架设计成兼容三种不同物理表征：

1. 相干振幅型非线性
2. 相干相位型非线性
3. 非相干光电型非线性

这三类的差别不只是公式不同，而是它们对复场历史信息保留的程度不同。后续实验应该先比较“哪种机制有效”，再比较“哪种位置有效”。

## 开工前准备

> [!important]
> 下面四项在真实写非线性算子前必须先完成，否则大概率中途返工。

### 1. 收口传播接口

当前分类训练、评估和可视化并不完全共用同一条传播路径。非线性层一旦接入，如果只有模型 `forward()` 变了，而某些可视化还在手写旧传播逻辑，训练结果和图就会互相打架。

开工前应先收口为一个统一的场传播入口，例如：

- `propagate_field()`
- `forward_field()`
- 或者等价的内部统一接口

要求：

- 分类训练调用它
- 分类可视化调用它
- 后续 activation 前后场强分析也调用它

### 2. 扩展 checkpoint / manifest 结构

当前 checkpoint 与 manifest 只足够描述 phase-only baseline，不足以区分 nonlinear 实验。

开工前应统一增加这些信息：

- `activation_type`
- `activation_positions`
- `activation_hparams`
- `experiment_stage`
- `seed`

否则后面很容易出现“训练跑完了，但不知道这个 checkpoint 到底是哪种机制和哪个位置”的情况。

### 3. 先定义实验命名规范

当前分类 checkpoint 文件名仍是按数据集固定命名，不适合 nonlinear 实验线。必须先定命名规则，至少包含：

- 数据集
- activation 类型
- 插入位置
- 阶段
- 随机种子

推荐思路：

`fashion_mnist__act-coherent_amp__pos-mid__stage-mechanism__seed-42.pth`

### 4. 先补基线回归测试

当前测试能证明主线入口能跑，但还不足以保护这次重构。开工前应先补下面这些测试位：

- `activation=None` 时与旧模型前向等价
- 插入 activation 后 shape / dtype 不变
- 默认参数下 checkpoint 加载与可视化不崩
- manifest 能完整记录新的 nonlinear 配置

## 推荐的阶段性方案

### 阶段 0：冻结当前主线基线

保留现有 5 层 phase-only D2NN 作为对照组，不改当前主训练入口和推荐 checkpoint。后续所有 nonlinear 实验都相对这条基线做增量比较。

当前对照指标固定为：

- `MNIST` / `Fashion-MNIST` 分类精度
- 收敛速度
- 不同随机种子下稳定性
- 训练时间与显存开销

### 阶段 1：先把框架改成“允许插非线性”，但默认关闭

目标不是先追结果，而是先把架构接口理顺。

建议新增的核心结构：

- `FieldActivationBase`
  - 输入：复场 `u`
  - 输出：激活后的复场 `u_out`
  - 额外能力：记录当前层平均强度、平均 gain、平均相位偏移等诊断量
- `IdentityActivation`
  - 默认占位，用于兼容现有基线
- `CoherentAmplitudeActivation`
  - 相干振幅饱和型，保留原始相位
- `CoherentPhaseActivation`
  - 相干相位调制型，只改变相位不改变振幅
- `IncoherentIntensityActivation`
  - 光电探测 / O-E-O 型，丢弃输入相位并重建输出场
- `NonlinearPlacementConfig`
  - 定义非线性插在哪些 layer 之后
- `NonlinearD2NN` 或在现有 `D2NN` 内增加可选 activation 链
  - 保证 `activation=None` 时退化回当前实现

> [!important]
> 第一阶段虽然要把三种 activation 的接口都设计出来，但不要求三种都同时深做。先让框架支持它们，再按实验顺序逐个启用。

### 阶段 2：三类核心算子的实现优先级

#### 2.1 CoherentAmplitudeActivation

这是第一阶段最推荐先实现的主力算子。

目标形式：

```python
I = safe_abs(u) ** 2
gain = g_min + (g_max - g_min) * torch.sigmoid((I - threshold) / temperature)
u_out = gain * u
```

它对应的是相干振幅非线性，应该满足：

- 输出相位与输入相位一致
- 输出模长不大于输入模长
- 更容易与当前复场传播实现兼容

#### 2.2 CoherentPhaseActivation

这是第二优先级的机制扩展。

目标形式：

```python
I = safe_abs(u) ** 2
phase_shift = gamma * I
u_out = u * torch.exp(1j * phase_shift)
```

它对应的是 Kerr-like / self-phase modulation 路线，应满足：

- 输出模长与输入模长一致
- 相位发生强度相关偏移
- 物理解释比抽象 gain 更贴近全光相位非线性

#### 2.3 IncoherentIntensityActivation

这是最值得保留、但不应最先实现的机制。

目标形式：

```python
I = safe_abs(u) ** 2
new_amplitude = activation_fn(responsivity * I - threshold)
u_out = new_amplitude * torch.exp(1j * emission_phase)
```

它的关键特征是：

- 输入场原始相位信息被丢弃
- 输出相位由新的发射相位控制
- 更接近 photodetector / O-E-O activation 路线

> [!note]
> 这一类虽然物理含义清晰，但与当前纯相干传播框架的差异最大，因此不建议放在第一版最先验证。

### 阶段 3：数值稳定性与初始化

这一部分应从一开始就写进方案，而不是出了问题再补。

建议新增两个基础约束：

1. `safe_abs(u)`

```python
safe_abs = torch.sqrt(u.real ** 2 + u.imag ** 2 + 1e-8)
```

避免在复数零点附近因为 `abs` 求导引入不稳定行为。

2. 动态阈值初始化

不要把 `threshold` 写死成固定常数。更稳的路线是：

- 第一版用相对输入强度均值的比例初始化
- 或者维护 running mean / running scale 作为 buffer
- 再在其上学习一个小范围偏移

> [!caution]
> `H x W` 级别的全空间阈值图、`gamma` 图、`emission_phase` 图虽然有研究价值，但第一版不建议直接做成全自由度参数。先用标量、按层参数或低分辨率参数图更稳。

### 阶段 4：实验顺序

这条线最合理的顺序是“先机制，后位置”。

#### 阶段 4.1：算子基建与单元测试

先写带物理边界断言的单元测试：

- `CoherentAmplitudeActivation`：输出模长必须不超过输入模长，相位不变
- `CoherentPhaseActivation`：输出模长保持不变，相位改变
- `IncoherentIntensityActivation`：输出相位不依赖输入相位
- `activation=None` 时与旧模型前向一致
- 插入 activation 后张量 shape / dtype 不变

#### 阶段 4.2：单层机制消融

固定非线性层插在一个统一中部位置，例如第 2-3 层之间，只比较三类机制：

- `CoherentAmplitudeActivation`
- `CoherentPhaseActivation`
- `IncoherentIntensityActivation`

数据集先固定 `Fashion-MNIST`。

这一阶段回答的是：

“到底是改振幅、改相位，还是丢掉原相位后重建，更值得继续做？”

#### 阶段 4.3：位置消融

从阶段 4.2 里挑表现最好的那一种算子，再比较位置：

1. 输入后立即加一层非线性
2. 第 1-2 层之间
3. 第 3 层附近
4. 倒数第二层后
5. 每层之间都加

这一步只允许改“位置”一个变量，其余固定：

- 层数固定 5 层
- 传播参数固定当前默认值
- 数据集固定 `Fashion-MNIST`
- 非线性机制固定为阶段 4.2 中最优者

#### 阶段 4.4：复杂数据集迁移

位置结果确定后，再升级数据集，不要反过来。

建议顺序：

1. `Fashion-MNIST`
2. `grayscale CIFAR-10`
3. `CIFAR-10` 多通道版本或更复杂成像任务

这样可以把“非线性本身有没有用”和“输入分布变难了”拆开看。

## 代码改动建议

### [`d2nn.py`](../../d2nn.py)

需要承载的改动：

- 新增 `safe_abs` 或等价安全模长函数
- 新增 `FieldActivationBase`
- 新增 `IdentityActivation`
- 新增 `CoherentAmplitudeActivation`
- 新增 `CoherentPhaseActivation`
- 新增 `IncoherentIntensityActivation`
- 在 `D2NN` 前向传播里增加 activation hook
- 支持按 layer index 插入 activation

### [`tasks.py`](../../tasks.py)

需要承载的改动：

- 把 activation 配置纳入任务构建
- 统一记录 activation 类型、位置、超参数
- 输出更清楚的实验标识，避免 checkpoint 混淆
- 记录机制消融和位置消融所属阶段

### [`train.py`](../../train.py)

需要承载的改动：

- 增加 `--activation-type`
- 增加 `--activation-positions`
- 增加 `--activation-threshold`
- 增加 `--activation-temperature`
- 增加 `--activation-gamma`
- 增加 `--activation-responsivity`
- 增加 `--activation-emission-phase-mode`
- 保证默认参数下仍等价于当前 baseline

### [`visualize.py`](../../visualize.py)

需要承载的改动：

- 在图上标出 activation 位置
- 可选导出 activation 前后场强分布
- 对相干相位型机制，可额外导出相位偏移图
- 帮助判断非线性到底在做什么，而不只是看最终精度

### `tests/`

至少补四类测试：

- `activation=None` 时与旧模型前向一致
- 三类 activation 的物理边界断言成立
- 单个 activation 插入后张量 shape / dtype 不变
- 配置不同位置时，模型能正常构建与跑通一个 batch

## 开工过程中最可能出现的意外

### 1. 旧 checkpoint 无法恢复或恢复后语义不清

一旦模型结构里出现 activation 配置，旧 checkpoint 的恢复逻辑就会不够描述完整实验。即使能加载，也可能不知道它到底属于哪一种机制和哪一个位置。

### 2. 可视化和训练结论不一致

如果传播逻辑没有彻底收口，训练时走的是“带 activation 的新路径”，可视化时走的是“手写旧路径”，最后图和数值会互相矛盾。

### 3. 相干振幅型非线性把能量压得过低

如果阈值、温度或 gain 上限设得不合适，层间能量会迅速衰减，训练表现会变成“数值正常但没有有效信号”。

### 4. 非相干光电型机制比预期更重

这一类机制的物理含义清楚，但它对当前纯相干复场框架的改动最大，很容易把第一阶段复杂度抬高。

### 5. 实验矩阵膨胀过快

如果同时展开三类机制、多个位置、多个数据集和多个随机种子，实验数量会迅速失控。必须强制执行“机制消融完成前不开位置消融，位置消融完成前不开复杂数据集”的顺序。

## 阶段出口标准

### 阶段 0 完成标准

- 统一传播接口已经收口
- checkpoint / manifest / 命名规则已扩展
- 基线回归测试可保护默认行为

### 阶段 1 完成标准

- 默认关闭 activation 时，训练和可视化与当前基线一致
- 三类 activation 至少都能完成单 batch 前向与反向
- 物理边界单元测试全部通过

### 阶段 2 完成标准

- 在 `Fashion-MNIST` 上完成一轮机制消融
- 得到一个明确更值得继续做的主力机制

### 阶段 3 完成标准

- 在 `Fashion-MNIST` 上完成一轮位置消融
- 得到一个明确的最优位置或至少排除明显无效位置

### 阶段 4 完成标准

- 最优机制与最优位置在更复杂数据集上复现出可解释趋势
- 能写出“非线性是否值得继续做”的阶段结论

## 当前建议的执行顺序

1. 先做工程准备：传播接口、命名、manifest、回归测试
2. 先做框架改造，保证 activation 是可插拔的
3. 先实现 `CoherentAmplitudeActivation`
4. 再补 `CoherentPhaseActivation` 与 `IncoherentIntensityActivation`
5. 先做机制消融，再做位置消融
6. 再迁移到更复杂数据集
7. 最后才考虑 SHG 或更物理的器件绑定

- [x] CoherentPhaseActivation 已完成第一版最小实现，当前形式采用 phase_shift = gamma * I，保模长、改相位，并补齐 mean_phase_shift 诊断与对应回归测试
- [x] CoherentPhaseActivation 已补齐第一层实验支撑：COHERENT_PHASE_PRESETS、coherent_phase_presets grid、coherent_activation_mechanisms grid 均已接入，机制消融现在可以直接从 printable 命令模板起跑
- [x] IncoherentIntensityActivation 已完成第一版最小实现，当前形式采用 relu(responsivity * I - threshold) 生成新振幅，并在 emission_phase_mode='zero' 下显式丢弃输入相位
- [x] IncoherentIntensityActivation 已补齐第一层实验支撑：INCOHERENT_INTENSITY_PRESETS、incoherent_intensity_presets grid，以及覆盖三类机制的 activation_mechanisms grid 均已接入
- [x] experiment grid runner 已接入：tasks.execute_experiment_grid() 与 train.py --run-experiment-grid 现已可顺序执行机制消融/位置消融预定义网格
- [x] 第一轮真实 mechanism ablation 冒烟已完成：Fashion-MNIST、5 layers、mid placement、balanced preset、1 epoch 下，三类机制均可稳定训练；当前 test acc 为 coherent_amplitude 81.04%、coherent_phase 81.08%、incoherent_intensity 80.29%
- [x] 第一轮 5 epoch mechanism ablation 已完成：当前 test acc 为 coherent_amplitude 83.96%、coherent_phase 83.22%、incoherent_intensity 85.28%，因此下一步位置消融优先围绕 incoherent_intensity 展开
- [x] 第一轮 3 epoch placement ablation 已完成：在 incoherent_intensity + balanced preset 下，front / mid / back / all 的 test acc 分别为 82.92% / 84.19% / 84.98% / 65.06%，当前 back 明显最优，而 all 因后层幅度级联爆炸暂时排除
- [x] 当前最优配置 incoherent_intensity + back 已放大到更正式训练：5 epoch test 86.32%、10 epoch test 86.67%、20 epoch test 87.61%，当前已首次超过 phase-only Fashion-MNIST 主线 87.49%
- [x] 当前最优配置 incoherent_intensity + back + 20 epoch 已完成多 seed 稳定性验证：seed 42 / 7 / 123 下 test acc 分别为 87.61% / 87.36% / 87.28%，均值约 87.42%，当前已可视为稳定领先或至少稳定持平于 phase-only 主线
- [x] “更复杂数据集迁移”的代码入口已先行补齐到 grayscale CIFAR-10：当前 classification 数据集配置已支持 `cifar10_gray`，并统一走单通道灰度 transform 与按数据集总量自适应的 train/val split；这意味着下一步可以直接开始做 grayscale CIFAR-10 上的 phase-only baseline 与 `incoherent_intensity + back` 对照，而不必先插手 RGB 多通道改造
- [x] grayscale CIFAR-10 的第一轮正式对照已完成：`5 epoch, seed=42` 下，phase-only baseline test acc 为 `33.17%`，`incoherent_intensity + back` 提升到 `37.73%`，领先 `4.56` 个点
- [x] grayscale CIFAR-10 的第二个随机种子验证已完成：`5 epoch, seed=7` 下，phase-only baseline test acc 为 `33.43%`，`incoherent_intensity + back` 为 `37.29%`，领先 `3.86` 个点；当前可初步判断该 nonlinear 配置对更复杂数据集的收益并非单次偶然
- [x] grayscale CIFAR-10 的 `10 epoch, seed=42` 放大对照已完成：phase-only baseline test acc 为 `35.09%`，`incoherent_intensity + back` 提升到 `39.48%`，仍领先 `4.39` 个点；最佳验证集达到 `40.00%`，当前可判断这条优势在更长训练预算下没有回落

- [x] grayscale CIFAR-10 的 `10 epoch, seed=7` 对照已补完：phase-only baseline test acc `35.10%`，`incoherent_intensity + back` test acc `39.88%`，领先 `4.78` 个点；至此 `10 epoch` 下 `seed=42 / 7` 两个随机种子都保持约 `+4.4 ~ +4.8` 个点收益。
- [x] 2026-04-03 阶段节点 review 已完成 fallback 记录：本机 Claude Code CLI 当前未登录，未拿到可用外部审查输出，因此不计作 Claude review；后续改用只读 review pass，结论是当前无阻塞冻结的硬问题，但 release 前需要在说明中明确 `seed=42` 的 `10 epoch` nonlinear artifact 文件名未显式编码 `10ep`，避免与更早的 grayscale CIFAR nonlinear 产物混淆。

- [x] 已补完 `RGB CIFAR-10` 的最小分类入口：当前 `cifar10_rgb` 已接入 classification 数据集配置，输入嵌入支持把 `R / G / B` 三个通道放入同一输入窗口的三个固定子区域，且 `train.py` / `visualize.py` 的 dataset help 已同步对齐，不再出现训练入口与可视化入口对数据集支持边界不一致的问题。
- [x] `RGB CIFAR-10, 10 epoch, seed=42` 的第一轮正式对照已完成：phase-only baseline test acc `44.01%`，`incoherent_intensity + back` test acc `46.60%`，领先 `2.59` 个点；对应 best val 分别为 `44.20%` 和 `46.78%`。这说明最优线在进一步进入 RGB CIFAR-10 后仍然保留正收益，但当前收益幅度低于 grayscale CIFAR-10 阶段。
- [x] 2026-04-03 该节点 review 已完成 fallback 记录：本机 Claude Code CLI 仍未登录，未拿到可用外部审查输出，因此继续不计作 Claude review；随后进行只读 review pass，未发现阻塞本轮代码节点提交的硬问题，仅发现并修复了一处 `visualize.py` dataset help 未同步 `cifar10-rgb` 的 CLI 一致性问题。修正后全量回归结果更新为 `Ran 75 tests ... OK`。

- [x] `RGB CIFAR-10, 10 epoch, seed=7` 的第二条对照已完成：phase-only baseline test acc `44.12%`，`incoherent_intensity + back` test acc `47.08%`，领先 `2.96` 个点。结合 `seed=42` 的 `44.01% -> 46.60%`，当前 RGB 阶段在两个随机种子下都维持了稳定正收益，双 seed 平均提升约 `+2.78` 个点。
- [x] 2026-04-03 RGB 阶段冻结前的 review 继续记录为 fallback：本机 Claude Code CLI 仍未登录，未拿到可用外部审查输出，因此继续不计作 Claude review；随后进行只读 review pass，未发现阻塞当前 RGB 阶段冻结的硬问题。
- [x] 当前 RGB 阶段已完成冻结：GitHub Release / tag `nonlinear-incoherent-back-cifar10-rgb-v1` 已发布，包含 `10 epoch, seed=42 / 7` 的 baseline 与 nonlinear 核心产物和结果摘要。至此，当前“最优线是否能继续迁移到 RGB CIFAR-10”已经有一版可回看的阶段快照。

- [x] 已完成 `RGB CIFAR-10 + 10 epoch` 的五 seed 稳定性确认补充：新增 `seed=123 / 0 / 2025` 三组对照后，五个 seed 的 baseline / nonlinear test acc 分别为 `44.01 -> 46.60`、`44.12 -> 47.08`、`43.66 -> 44.86`、`44.47 -> 45.89`、`44.07 -> 45.45`。
- [x] 当前 RGB 五 seed 汇总为：baseline mean `44.07%`，nonlinear mean `45.98%`，mean lift `+1.91 pt`，lift spread `1.76 pt`。虽然 `5/5` seed 仍全部为正收益，但只有 `2/5` seed 超过 `+1.5 pt`，且 mean lift 未超过当前设定的 `+2.0 pt` 稳定性门槛，因此这一轮应记为 `stability inconclusive`，而不是继续写成“stable”。
- [x] 2026-04-03 本轮阶段 review 继续记录为 fallback：本机 `claude` 命令路径存在，但当前 ACP 环境缺少其依赖的 `node`，因此未拿到可用外部审查输出，不计作 Claude review；随后执行只读 review，未发现 RGB 五 seed artifact 的 `run_name / seed / stage / activation` 配对错误，也未发现阻塞当前阶段结论的命名或 CLI 漂移问题。

- [x] 已完成 `RGB CIFAR-10 + 20 epoch` 的计划出口前预算敏感性确认：三组 baseline / nonlinear 对照分别为 `45.10 -> 47.72 (seed=42)`、`45.38 -> 47.96 (seed=7)`、`45.33 -> 47.75 (seed=123)`。
- [x] 当前 `20 epoch` RGB 三 seed 汇总为：baseline mean `45.27%`，nonlinear mean `47.81%`，mean lift `+2.54 pt`，lift spread `0.20 pt`。这相较于 `10 epoch` 的 `mean lift = +1.91 pt`、`spread = 1.76 pt` 明显更强也更稳，说明此前 RGB 阶段的“不够稳”主要更像预算不足，而不是机制迁移本身失效。
- [x] 2026-04-04 本轮阶段 review 继续记录为 fallback：本机 `claude` 命令路径存在，但当前 ACP 环境缺少其依赖的 `node`，因此未拿到可用外部审查输出，不计作 Claude review；随后执行只读 review，未发现 `20ep` RGB artifact 的命名、seed 配对、stage 或 activation 配置错误。

> [!done]
> 当前 nonlinear validation phase 可以在这里正式收口：`incoherent_intensity + back` 已在 Fashion-MNIST、grayscale CIFAR-10 与 RGB CIFAR-10 上完成由低复杂度到高复杂度的数据集迁移验证；其中 RGB 阶段在 `20 epoch` 预算下也表现出稳定正收益，因此当前主线的剩余工作不再是继续追这条非线性结论本身，而是进入后续仓库整理、README 重写、public 发布与下一轮科研问题定义。

## 下一阶段计划：从非线性验证转入相位冻结与实物验证

> [!important]
> 2026-04-05 导师反馈的重点不是“继续把非线性精度往上追”，而是借这条线把网络部分真正理解透，然后把训练得到的相位稳定地保存下来，整理成可加工的 phase plate 产物，后续放到真实光路里做测量闭环。

### 当前阶段判断

- 当前 `incoherent_intensity` 这条最优线本质上仍是“相对光强驱动的层间非线性处理”，它已经足够支撑数值层面的研究结论，但还不能直接等同于“只加工若干相位板后即可原样在光路中复现”的器件方案。
- 如果下一步的真实动作是“先把训练相位加工出来，再放进现有光路测试”，那么第一优先加工目标应是 `phase-only` 基线，而不是当前的 `incoherent_intensity + back` checkpoint。
- 当前非线性专项的价值没有消失，而是发生角色切换：从“继续扩实验矩阵”切到“为后续器件实现和实测定义边界、提供对照、提前暴露潜在失配点”。

### 本阶段总目标

把“非线性验证阶段的结论”收口成一套能直接服务后续加工与实测的执行包，明确三件事：

1. 先加工哪一个 checkpoint。
2. 相位如何导出成制造参数。
3. 实物测量时仿真链路与光路链路如何一一对应。

### 具体执行顺序

### 阶段0：网络机制理解（新增）

> [!important]
> 这是导师强调的"练的时候理解网络"环节，必须在进入加工前完成。

**目标**：通过可视化和对照分析，真正理解当前网络学到了什么，为后续加工和测量建立直觉。

**具体内容**：
1. **相位分布可视化** — 把5层相位板的分布画出来，观察是否有规律（边缘/中心结构、频率选择性等）
2. **场传播可视化** — 选几个典型样本，观察光场在5层之间的演变过程
3. **Baseline vs Nonlinear 对照** — 对比 `phase-only` 和 `incoherent_intensity + back` 的相位分布差异，理解非线性机制改变了什么
4. **量化敏感性分析** — 模拟不同量化级数（8级/16级/连续）对精度的影响，为加工精度提供依据

**交付物**：
- 一页"网络行为解读"摘要图（5层相位分布 + 典型样本场传播）
- 关键发现笔记（记录相位分布规律、量化敏感度结论）

> [!note]
> 阶段0的目标是服务于首发 `phase-only` checkpoint 的选择与加工，不把这里重新扩成新一轮 nonlinear 机制研究。


#### A. 先冻结第一版实物基线

- 第一版实物加工优先冻结 `phase-only` 分类基线，不以当前 nonlinear 最优线作为首发加工目标。
- 当前优先候选按稳妥性排序为：
  1. `Fashion-MNIST phase-only`，便于与当前非线性专项直接对照。
  2. `MNIST phase-only`，作为更简单、更容易先打通光路闭环的兜底版本。
- 当前判断下，不建议直接把 `best_fashion_mnist.incoherent_back_20ep*.pth` 作为第一版加工源，除非后续同时明确非线性器件链路要如何在实验里实现。

#### B. 把“推荐 checkpoint”固定成单一入口

- 在笔记和仓库说明里明确写死“第一版加工基线”的 checkpoint 名称，避免后续在 `baseline_5layer`、`best_fashion_mnist.pth`、`incoherent_back_*` 等多个产物之间反复漂移。
- 推荐优先检查并固定这两个候选：
  - `checkpoints/best_fashion_mnist.pth`
  - `checkpoints/best_mnist.pth`
- 只允许保留一个“首发加工主目标”，另一个作为备选，不再并行推进多个加工版本。

#### C. 先形成实验对照单

- 在进入加工前，必须先写清楚“仿真里的每一个量在实验里对应什么”。
- 至少要显式对齐：
  - 5 层相位板的真实层间距
  - 入射波前形式
  - 输入图样加载方式
  - 输出探测面位置
  - 分类 detector 区域或成像采样区域
  - 归一化方式与实验测量值之间的换算
- 这一步的目标是避免后面出现“相位板加工没问题，但实验搭法与仿真读取方式根本不是一回事”的伪失效。

#### D. 正式导出制造包，而不是只停留在 checkpoint

- 对选定 checkpoint 重新执行一遍完整导出，确保产物来源、参数和日期可追溯。
- 导出包至少包含：
  - `phase_masks.npy`
  - `height_map.npy`
  - `height_map_manufacturable.npy`
  - `thickness_map.npy`
  - 每层 `phase / height / thickness / quantized` CSV
  - `report.md`
  - 如工艺方需要，则补 `stl/`
- 导出动作必须连同下面这些参数一起冻结记录：
  - `wavelength`
  - `layer_distance`
  - `pixel_size`
  - `refractive_index`
  - `ambient_index`
  - `base_thickness_um`
  - `max_relief_um`
  - `quantization_levels`

#### E. 做一次“加工前可制造性检查”

- 当前要回答的不是“网络还能不能再训练”，而是“这个相位板现在能不能被工艺真实做出来”。
- 加工前至少检查四项：
  1. 每层最大 relief 是否超出当前工艺上限。
  2. 限高后是否出现大面积截平，导致相位信息严重失真。
  3. 量化级数是否与可用加工精度匹配。
  4. 层尺寸、像素尺寸、总口径是否与现有光路机械尺寸兼容。
- 如果 `max_relief_um` 一旦收紧就导致大面积 clipping，那么此时优先回到“材料 / 工艺参数重选”，而不是立刻重训网络。

#### F. 给非线性专项留一个“影子对照出口”

- 虽然第一版不直接加工 nonlinear checkpoint，但当前最优线不应被丢掉。
- 建议把 `incoherent_intensity + back` 保留为“数值对照参考包”：
  - 保留 checkpoint
  - 保留 manifest
  - 如有必要，也导出一份 phase package，但明确标记为“非首发加工目标”
- 这样后续如果实验条件允许增加器件级非线性，能够直接回到当前最佳数值结论，而不需要重跑整条搜索线。

### 当前建议的里程碑拆分

#### 里程碑 0：网络机制理解（新增）

完成标准：

- 5层相位分布可视化已完成，能描述出主要规律（如边缘/中心结构、频率选择性等）。
- 典型样本的场传播可视化已完成，能解释光场在层间的演变。
- `phase-only` 与 `incoherent_intensity + back` 的相位分布对照已完成，能说出非线性机制改变了什么。
- 量化敏感性分析已完成，明确8级/16级/连续量化对精度的影响。


#### 里程碑 1（原A+B）：选定首发加工 checkpoint

完成标准：

- 明确只选 `Fashion-MNIST phase-only` 或 `MNIST phase-only` 其中之一作为首发加工目标。

- 在笔记里写明为什么选它，而不是 `incoherent_intensity + back`。
#### 里程碑 2（原E）：形成实验对照单

完成标准：

- 仿真参数与光路搭建参数形成一页式对照。
- 已明确输入、层间距、输出面和读出方式的实验映射。


#### 里程碑 3（原C+D）：导出并检查第一版制造包

完成标准：

- 选定 checkpoint 的完整 phase package 已导出。
- `report / metadata / layer CSV / STL` 已齐全。
- 已确认没有明显超工艺上限的硬性问题，或已明确知道问题卡在哪个制造参数上。

#### 里程碑 4（原F）：准备进入真实加工与首轮测量

完成标准：

- 相位板制造文件已冻结，不再随意变更 checkpoint。
- 已经有一份“加工后上光路该怎么测”的简短 SOP。
- 已明确首轮实验是“验证 phase-only 基线能否闭环”，不是“直接验证非线性器件收益”。

### 本阶段暂不继续做的事

- 暂不继续扩大 nonlinear 机制搜索或位置搜索。
- 暂不因为追求更高 test accuracy 再追加新的 CIFAR seed 或更长 epoch。
- 暂不把“非线性数值最优线”直接包装成第一版实物加工目标。
- 暂不把“器件级非线性如何真实实现”与“先把相位板加工出来”这两件事混成一个阶段。

> [!note]
> 这一步的核心不是否定非线性专项，而是把它从“主动作”降为“研究对照与后续器件实现储备”。真正排在最前面的动作，已经变成 `冻结一个可真实落地的 phase-only 模型 -> 导出制造包 -> 对齐实验光路 -> 做首轮实测`。

### 执行时的决策规则

- 如果目标是“尽快拿到第一块可上光路的相位板”，优先 `MNIST phase-only`。
- 如果目标是“尽量贴近当前专项主线并保留与 nonlinear 对照的直接可比性”，优先 `Fashion-MNIST phase-only`。
- 如果后续实验条件无法同步实现层间非线性器件，就不把 `incoherent_intensity + back` 作为首发加工目标。
- 如果导出的 height map 在当前材料参数下明显不可制造，就优先调整制造参数，不优先回去改训练目标。

### 当前最合理的下一动作

1. 在 `phase-only Fashion-MNIST` 与 `phase-only MNIST` 之间定一个首发加工目标。
2. 单独补一页“仿真参数 -> 光路参数”的实验对照笔记。
3. 对选定 checkpoint 重新导出一版完整制造包。
4. 记录导出时采用的材料与工艺参数，并完成一次加工前检查。

> [!todo]
> 当前笔记对应的下一条执行指令不再是“继续跑 nonlinear 实验”，而是“选定首发加工基线并完成 phase package 导出与加工前检查”。
