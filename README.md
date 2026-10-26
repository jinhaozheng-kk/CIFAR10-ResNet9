# CIFAR-10 图像分类：ResNet-9 快速训练实验

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.5.1 cu121">
  <img src="https://img.shields.io/badge/CIFAR--10-40k%2F10k%2F10k-blue" alt="CIFAR-10">
  <img src="https://img.shields.io/badge/ResNet--9-lightweight-blueviolet" alt="ResNet9">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/best_acc-89.35%25-brightgreen" alt="best accuracy">
</p>

ELEC5304 Project 2 **Group 45** 的课程项目归档与完整重跑记录：用自建 **ResNet-9** 在 CIFAR-10 上完成图像分类，在**单卡 T4、1 分钟训练预算**的约束下同时满足两个指标：

1. 完整训练后测试准确率 **> 80%**；
2. **1 分钟内**测试准确率 **≥ 70%**。

仓库保留了原始 Jupyter Notebook（含保存的训练输出与截图）、历史训练日志、找回并整理后的结果文件，以及一次在 RTX 4060 Laptop 上的完整复现实验。

**小组作者**：Jinhao Zheng、Yilin Wu、Huiwen Zhang、Yiqian Guo、Chengnan Zhu。

> 这是小组课程项目，开源副本移除了学号与过时的交互控件元数据，**未将小组成果改写为个人独立完成**。

---

## 目录

- [1. 项目目标与方法](#1-项目目标与方法)
- [2. 数据与预处理](#2-数据与预处理)
- [3. 模型结构](#3-模型结构)
- [4. 训练策略](#4-训练策略)
- [5. 原始课程实验记录](#5-原始课程实验记录)
- [6. 完整重跑（RTX 4060）](#6-完整重跑rtx-4060)
- [7. 如何运行](#7-如何运行)
- [8. 仓库结构](#8-仓库结构)
- [9. 结果解读与学术诚信说明](#9-结果解读与学术诚信说明)
- [10. 已知限制](#10-已知限制)
- [11. 数据来源](#11-数据来源)

---

## 1. 项目目标与方法

课程作业要求在 CIFAR-10 上：

1. 下载数据集，并按 80% / 20% 划分训练与测试；
2. 设计一个适合该任务的神经网络；
3. 训练并评估性能；
4. 进一步优化结构与训练策略，使模型**在 T4 GPU 上 1 分钟内达到 ≥ 70% 测试准确率**，同时完整训练后 **> 80%**。

小组最初的方案是"两个模型分别对应两个指标"（一个轻量 CNN 负责 1 分钟指标，一个更深的网络负责最终精度）。经过多轮实验发现：**同一个 ResNet-9 就能同时完成两件事**——它在 1 分钟内已经能超过 70%，完整 15 轮后又能超过 89%。因此最终删掉了基线 CNN，统一使用 ResNet-9。

设计思路（摘自 Notebook 的讨论部分）：

- **ResNet-9 轻量**：参数量小，前向/反向计算便宜，同样时间内能完成更多参数更新；
- **残差连接**：缓解退化与梯度消失，让浅层信息直达深层，加速前几轮收敛；
- **AdamW**：对 CIFAR-10 这种小图像数据集更不容易过拟合；
- **OneCycleLR**：前期大学习率帮助快速跳出差的局部最优，满足"1 分钟"约束；
- **AMP（自动混合精度）**：利用 T4 的 Tensor Cores 加速训练，几乎不损精度。

## 2. 数据与预处理

| 项目 | 说明 |
| --- | --- |
| 数据来源 | Hugging Face `uoft-cs/cifar10`（`datasets.load_dataset`） |
| 原始训练集 | 50,000 张 32×32 RGB |
| 划分方式 | `train_test_split(test_size=0.2, random_state=5304, shuffle=True, stratify=y_train0)` → **40,000 训练 / 10,000 验证** |
| 测试集 | 官方 10,000 张，单独保留 |
| 验证集用途 | **创建了但训练循环没有使用**（原始 notebook 的行为，保留未改） |

**训练增强**：

```python
transforms.RandomCrop(32, padding=4)
transforms.RandomHorizontalFlip()
transforms.ToTensor()
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
```

**测试预处理**：只做 `ToTensor()` + 同样的 `Normalize()`，不做随机增强。

均值和标准差用的是 CIFAR-10 的经典统计量（沿用既有研究/课程给定值），不是在本数据集上重新统计的。

数据以 NumPy 数组形式一次性载入，再用自定义 `CifarSub(Dataset)` 包装成 `DataLoader`：

- `batch_size=512`
- 训练 `shuffle=True`，测试 `shuffle=False`
- `num_workers=2`、`pin_memory=True`（Windows 上建议改为 `num_workers=0`，见第 7 节）

## 3. 模型结构

`ResNet-9` 的实现（`ELEC5304_Project_2.ipynb` 第 12 个代码单元）：

```text
输入 3×32×32
└─ prep      : Conv(3→32, 3×3, BN, ReLU)
└─ layer1    : Conv(32→64, 3×3, BN, ReLU, MaxPool2)
└─ res1      : Conv(64→64)×2 后的残差相加        ← 残差块 1
└─ layer2    : Conv(64→128, 3×3, BN, ReLU, MaxPool2)
└─ layer3    : Conv(128→256, 3×3, BN, ReLU, MaxPool2)
└─ res3      : Conv(256→256)×2 后的残差相加      ← 残差块 2
└─ classifier: AdaptiveMaxPool2d(1) → Flatten → Linear(256, 10)
```

```python
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)
```

- 所有卷积都**不使用 bias**（后面接 BatchNorm），下采样用 2×2 MaxPool；
- 两个残差块分别位于 64 通道与 256 通道阶段；
- 分类头用 `AdaptiveMaxPool2d(1)` 把 256×4×4 压成 256×1×1，再线性映射到 10 类，参数量远小于全连接堆叠。

> 注：原始报告的文字说明里曾把起始卷积描述为"全连接压缩"，**代码实际是卷积**，以代码为准。

## 4. 训练策略

| 配置 | 值 |
| --- | --- |
| Epochs | 15 |
| Loss | `CrossEntropyLoss` |
| 优化器 | `AdamW(lr=0.003, weight_decay=5e-3)` |
| 学习率调度 | `OneCycleLR(max_lr=0.006, epochs=15, steps_per_epoch=len(train_loader), pct_start=0.04)` |
| Batch size | 512 |
| 混合精度 | CUDA 上开启 `torch.autocast` + `GradScaler`（AMP） |
| 测试时机 | **从第 3 个 epoch 开始**，每轮结束后在测试集上评估 |
| 计时方式 | 从训练循环开始计时；每轮打印累计耗时与测试准确率 |
| 60 秒指标 | 记录累计耗时 ≤ 60s 时达到的最佳测试准确率（`best_acc60`） |

一个细节：`scheduler.step()` 放在 `GradScaler` 更新之后，并且只在 scale 未下降时执行（`new_scale >= old_scale`），避免 AMP 梯度缩放调整时误触发学习率变化。

评估从第 3 轮开始的原因：1 分钟指标要求模型在约 3–4 轮内达标，前两轮测试纯属浪费时间。代价是**测试集被反复使用**，第 9 节会专门讨论这一点。

## 5. 原始课程实验记录

以下数字来自找回的 notebook 中**已经保存的输出**，不是本次恢复操作新测得的：

| 指标 | 原 notebook 记录 |
| --- | ---: |
| 60 秒内最佳测试准确率 | **70.29%**（第 4 轮，累计 58.7 秒） |
| 15 轮最佳测试准确率 | **89.35%**（第 15 轮） |
| 第 15 轮累计耗时 | 235.9 秒 |

逐轮日志（`results/historical/training_log.csv`）：

| Epoch | 累计耗时 (s) | 测试准确率 (%) |
| ---: | ---: | ---: |
| 1 | 15.1 | — |
| 2 | 27.9 | — |
| 3 | 43.9 | 65.01 |
| 4 | 58.7 | 70.29 |
| 5 | 73.7 | 75.76 |
| 6 | 95.5 | 80.17 |
| 7 | 110.7 | 79.39 |
| 8 | 126.0 | 84.44 |
| 9 | 141.4 | 84.69 |
| 10 | 156.8 | 86.33 |
| 11 | 171.8 | 87.89 |
| 12 | 187.6 | 88.71 |
| 13 | 205.0 | 88.89 |
| 14 | 221.0 | 89.20 |
| 15 | 235.9 | 89.35 |

原课程以 T4 GPU 为目标；这些数字不构成对任意硬件的速度保证。代码没有固定所有随机性，重跑不保证得到相同准确率。

![原 notebook 保存的测试准确率曲线](results/historical/accuracy_curve.png)

历史 CSV 与两张原实验截图存放在 `results/historical/`（图片从原 notebook 中提取）。

## 6. 完整重跑（RTX 4060）

2026-10-18，使用找回的原代码在 **NVIDIA GeForce RTX 4060 Laptop GPU** 上完成 15 轮真实训练与测试：

| 指标 | 本次实测 |
| --- | ---: |
| 最佳测试准确率 | **88.93%** |
| 60 秒内最佳测试准确率 | **78.42%**（第 6 轮，累计 59.6 秒） |
| 15 轮累计耗时 | **153.2 秒** |
| 训练 / 未使用的验证 / 测试样本数 | 40,000 / 10,000 / 10,000 |

**环境**（`results/reproduction/summary.json`）：

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.14 |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| datasets | 3.6.0 |
| numpy | 1.26.4 |
| scikit-learn | 1.7.1 |
| matplotlib | 3.9.4 |
| Pillow | 11.0.0 |
| GPU | RTX 4060 Laptop |

**逐轮日志**（`results/reproduction/training_log.csv`）：

| Epoch | 累计耗时 (s) | 测试准确率 (%) |
| ---: | ---: | ---: |
| 1 | 9.6 | — |
| 2 | 18.8 | — |
| 3 | 29.2 | 65.83 |
| 4 | 39.3 | 62.02 |
| 5 | 49.4 | 75.40 |
| 6 | 59.6 | 78.42 |
| 7 | 70.0 | 81.34 |
| 8 | 80.3 | 81.48 |
| 9 | 90.5 | 83.12 |
| 10 | 100.8 | 86.27 |
| 11 | 111.2 | 87.13 |
| 12 | 121.7 | 88.09 |
| 13 | 132.2 | 88.76 |
| 14 | 142.7 | 88.89 |
| 15 | 153.2 | 88.93 |

![RTX 4060 完整重跑的测试准确率曲线](results/reproduction/accuracy_curve.png)

**本次重跑做的调整**（`summary.json` 的 `adaptations`）：

- DataLoader `num_workers` 从 2 改为 **0**，适配 Windows 下 notebook 内定义 `Dataset` 的多进程导入问题；
- 正式训练前跑了一次 **eval 模式前向检查**；
- 没有固定训练随机种子。模型、增强、优化器、学习率策略、轮数均沿用原 notebook。

因此这是"同一方法 + 同一流程"的重跑，不是逐位可复现实验；耗时只统计训练循环与第 3 轮起的测试，不含下载数据、构建数组、构造模型等准备时间。

**验证内容**：notebook 格式与全部 **11 个代码单元的语法检查**、GPU 前向/反向传播数值检查，以及完整 15 轮真实 CIFAR-10 训练。仅出现无界面绘图的 `plt.show()` 提示，曲线已实际保存。新训练权重保存在本地 `results/reproduction/resnet9_rerun_final.pth`，**未纳入 Git**（`.gitignore` 规则 `*.pth`）。

## 7. 如何运行

### 环境

- Python 3.10 / 3.11（重跑使用 3.10.14）
- 建议使用 NVIDIA GPU；CPU 也能跑但会慢很多
- PyTorch 版本请按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择与本机 CUDA 匹配的组合（仓库 requirements 记录的是 `torch==2.5.1` / `torchvision==0.20.1`）

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
jupyter lab
```

打开 `ELEC5304_Project_2.ipynb`，选择对应内核，从头执行全部单元。

### Google Colab

把 notebook 上传到 Colab，选择 GPU 运行时，先安装 `datasets`（其余依赖 Colab 已自带或按需安装）：

```python
!pip install datasets
```

### 注意事项

- 首次运行会从 Hugging Face 下载 CIFAR-10，需要网络和磁盘空间；
- **Windows 本地运行建议把两个 DataLoader 的 `num_workers=2` 改为 `num_workers=0`**，避免 notebook 中定义的 `Dataset` 类在多进程下导入失败；
- 原 notebook 的 `autocast`/`GradScaler` 用 CUDA，完整复现建议使用 NVIDIA GPU；
- 数据加载是把 5 万张图逐张 `np.array(img)` 后整体载入内存，内存占用较大（约 1 GB 级）；
- 仓库不包含训练数据与权重，`data/`、`*.pth`、`*.pt`、`*.ckpt` 均被 `.gitignore` 忽略。

## 8. 仓库结构

```text
CIFAR10-ResNet9/
├─ ELEC5304_Project_2.ipynb          # 原始课程 notebook（含保存输出与内嵌截图）
├─ requirements.txt                  # 依赖版本
├─ results/
│  ├─ historical/
│  │  ├─ accuracy_curve.png          # 原 notebook 的准确率曲线
│  │  ├─ training_log.csv            # 原 notebook 的逐轮日志
│  │  ├─ experiment_screenshot_1.png # 原实验截图
│  │  └─ experiment_screenshot_2.png
│  └─ reproduction/
│     ├─ accuracy_curve.png          # RTX 4060 重跑曲线
│     ├─ training_log.csv            # RTX 4060 逐轮日志
│     └─ summary.json                # 环境、数据规模、调整与结果摘要
├─ README.md
└─ .gitignore
```

## 9. 结果解读与学术诚信说明

- **测试集被反复使用**：原作业从第 3 轮起每轮都在测试集上评估，因此 89.35% / 88.93% 是"课程实验记录"，**不等于独立盲测或严格的最终泛化评估**。做新的模型比较时，应使用验证集选择超参数与模型，只在最终模型确定后评估测试集。
- **验证集未被使用**：notebook 建了 10,000 张验证集，但训练循环没有用它——这是原始代码行为，仓库如实保留。
- **无原权重**：仓库未找回原始训练 checkpoint，保留的曲线与日志不能代替可加载的模型文件。
- **非可复现实验**：未固定全部随机种子，且不同 GPU 的耗时差异明显（T4 vs 4060），重跑结果会有合理波动。
- **小组作品**：本项目为 5 人小组课程作业，重新整理、重跑与文档工作不代表独立完成原作业。

## 10. 已知限制

- 依赖 `datasets` 在线下载数据，离线环境需预先缓存；
- 测试集参与调参（评估泄漏风险），指标偏乐观；
- 未使用验证集，也没有早停 / 模型选择流程；
- 训练随机性未完全固定，复现只保证流程一致；
- 仓库不含权重文件，无法直接加载模型推理；
- Windows 多进程 DataLoader 需要手动改 `num_workers`；
- 没有单元测试或 CI；
- 仓库未附带 LICENSE 文件（课程作业，默认保留所有权利）。

## 11. 数据来源

- [CIFAR-10 官方数据集](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Hugging Face 镜像 `uoft-cs/cifar10`](https://huggingface.co/datasets/uoft-cs/cifar10)
- 课程：ELEC5304 Project 2（Group 45）
