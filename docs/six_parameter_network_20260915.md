# 六参数深度学习网络 SixParamNet（2026-09-15）

用 Field II 带噪数据的六参数标签（`data/labels_fieldii.jsonl`，1848 行，11 个体模）训练一个网络，同时给出深度、频率、聚焦、增益、TGC、动态范围的调整建议，并在仿真网格上做闭环优化评估。本文只描述代码，尚未训练，没有结果。

## 1. 文件

| 文件 | 作用 |
|---|---|
| `tools_build_fieldii_training_cache.py` | HDF5 分片 → `data/fieldii_dl_cache/cache.npz`（需要 h5py 与 bmode_opt） |
| `bmode_dl/constants.py` | 从 bmode_opt 复制的常数（增益/TGC 每级 dB、灰阶支点、动态范围窗宽、起点抽样范围） |
| `bmode_dl/render.py` | GPU 后端渲染，与 `hisense_backend_sim.render` 逐像素一致 |
| `bmode_dl/labels.py` | 标签编码；后端起点重抽与标签重算 |
| `bmode_dl/dataset.py` | 载入、按体模分折、组批、输入特征 |
| `bmode_dl/model.py` | 网络 |
| `bmode_dl/losses.py` | 掩膜多任务损失 |
| `bmode_dl/metrics.py` | 指标、只看设置的查表基线 |
| `bmode_dl/closed_loop.py` | 闭环优化仿真 |
| `bmode_dl/checkpoint.py` | 检查点存取 |
| `tools_train_six_param.py` | 训练 |
| `tools_evaluate_six_param.py` | 评估（单步 + 基线 + 闭环） |
| `tests/verify_six_param_model.py` | 不变量检查与 CPU 冒烟测试 |

训练与评估只依赖 numpy 与 torch，服务器上不需要 HDF5、h5py 和 bmode_opt。

## 2. 数据

缓存每个分片存：深度方向 1851 点按强度分块平均到 512 行、横向 128 线的 dB 图（float32）；逐行底噪（`fieldii_noise.noise_floor_db`）；深度轴首末点。1848 片约 380 MB，整个放进 GPU。

**后端起点重抽**是主要的数据增强：后端最优值只取决于图像，与起点无关，所以每个批次按 `labels.draw_start` 的同一分布重抽起点，重算增益/TGC 的修正量与方向，再按新起点渲染输入灰阶。前端标签绑定分片的设置，不能重抽。另做左右翻转（所有标签不变）。

**分折**：按体模（`group_id`）留出，囊肿与均匀体模分别轮流分到各折。标签里出现 `split=val` 时自动改按 split 字段。

## 3. 网络

输入：

| 输入 | 形状 | 内容 |
|---|---|---|
| 图像 | 3×512×128 | 当前增益/TGC/动态范围渲染的灰阶；绝对 dB（标准化）；物理深度 |
| 深度剖面 | 5×512 | 逐行 dB 的 10/50/90 分位；逐行灰阶中位数；逐行 dB 中位数减底噪 |
| 当前参数 | 17 | 深度、频率、聚焦、增益、8 个 TGC 滑块、动态范围、曝光参考、成像模式、顶部/底部底噪 |

衰减、电子噪声、声速是仿真真值，不作输入，只作辅助监督目标。底噪作输入，因为它对应实机上已知的接收机常数；可用 `--no-noise-floor` 关掉做对照。

结构：二维残差编码器（每级用当前参数做 FiLM 调制）+ 一维剖面编码器 → 深度方向 32 个位置的序列，前面加一个参数 token → 2 层 Transformer → 共享 MLP → 各轴输出头。TGC 头把深度序列池化成 8 段，每段结合该段当前滑块值预测该段修正量。约 5 M 参数。

输出与损失：

| 参数 | 输出 | 损失 |
|---|---|---|
| 增益 | 修正量 dB；三类方向 | 死区内不罚的 Huber；加权交叉熵 |
| TGC | 8 段修正量 dB；近/中/远三组方向 | Huber；曲率不超过教师曲线的惩罚；加权交叉熵 |
| 深度 | 6 档最优；三类方向 | 交叉熵 + 期望档距离；加权交叉熵 |
| 频率 | 4 档最优；三类方向 | 可接受集合似然（borderline 降权）；加权交叉熵 |
| 聚焦 | 8 档最优（按当前深度可选档掩膜）；三类方向 | 交叉熵 + 期望档距离；加权交叉熵 |
| 动态范围 | 修正量；三类方向 | 按 `dr_determined` 掩膜，现为 0 |

方向类别权重取频数倒数的平方根；训练集里没有出现的类别（深度"太深"）权重为 0。

## 4. 评估

- **单步**：增益/TGC 平均绝对误差（dB）、落入死区比例、方向 macro-F1；前端最优档准确率、±1 档准确率、频率可接受集合命中率。方向既报方向头，也报"由数值输出推出"的方向（闭环实际执行的是后者）。
- **只看设置的查表基线**：同一 深度/频率/聚焦 下训练体模的众数最优档、中位最优增益与滑块。网络必须胜过它，才能说明看了图像。
- **闭环**：从验证体模的每个分片及其起点出发。前端建议与当前不同就换到对应设置的分片（后端保持同一绝对曝光），否则执行后端修正；直到不再调整或达到步数上限。报告收敛比例、打转比例、步数分布，以及停下时该分片标签是否三轴都"正确"、增益与 TGC 离教师最优还有多少 dB。

## 5. 服务器上运行

环境：Python 3.10、PyTorch 2.6.0 + cu118、numpy。在仓库根目录执行：

```bash
# 1. 缓存（二选一）：本地 cubdl 环境生成后把 data/fieldii_dl_cache/ 拷到服务器；或在服务器装 h5py 后生成
python tools_build_fieldii_training_cache.py --workers 8

# 2. 检查（可选，CPU 上约 1 分钟；需要 bmode_opt 与 data/labels_fieldii.jsonl）
python tests/verify_six_param_model.py

# 3. 4 折交叉验证（默认即第 7 节的改动后配置）
python tools_train_six_param.py --fold all --amp --name fieldii_v2

# 4. 评估（每折默认用 best_frontend.pt + best_backend.pt 组合；--select best 只用 best.pt）
python tools_evaluate_six_param.py --run runs/fieldii_v2 --amp

# 5. 对照实验
python tools_train_six_param.py --fold all --amp --input-mode no_image --name fieldii_no_image
python tools_train_six_param.py --fold all --amp --no-noise-floor --name fieldii_no_floor

# 6. 全部体模训练（做实机微调的起点）
# --frontend-epoch 取第 3 步汇总里报告的前端最佳轮次中位数；后端用 last.pt
python tools_train_six_param.py --fold none --epochs 150 --frontend-epoch 30 --amp --name fieldii_v2_all
```

服务器上要拷贝：`bmode_dl/`、`tools_train_six_param.py`、`tools_evaluate_six_param.py`、`data/labels_fieldii.jsonl`、`data/fieldii_dl_cache/`。第 1、2 步另需 `bmode_opt/` 与 `data/field_ii/full_noise/hdf5/`。

## 6. 已知限制

1. 11 个体模全部是 train，只能留体模交叉验证，每折验证 2–3 个体模，方差会大。仿真全部完成后重跑标签与缓存即可，代码不用改。
2. Field II 的深度标签从不说"太深"（`docs/fieldii_labels_20260915.md` §5.1），网络学不到减小深度。
3. 动态范围头没有监督，输出无意义，直到定出判据。
4. 增益"正确"只占约 3%；闭环以数值修正量加固定死区作停止判据。
5. 闭环里的停止死区用固定 0.5 级（实机没有逐帧死区），评判用各分片标签自己的死区。

## 7. 修改记录：fieldii_v1 训练日志分析之后（2026-09-15）

`runs/fieldii_v1` 的 4 折结果与只看设置的查表基线相比：TGC、频率明显更好；增益反而更差（平均绝对误差 0.39 对 0.22 dB）；深度与查表持平。前端训练损失降到 0.001 量级、验证在第 10–40 轮见顶，后端到第 150 轮仍在变好。据此改了三处，默认值即新配置：

| 改动 | 做法 | 恢复旧行为 |
|---|---|---|
| 增益 / TGC 改为预测最优值 | 网络输出最优增益（dB，相对 reference_db）与 8 段最优 TGC（dB），按训练集均值方差标准化，末层置零（起点即训练集均值）；修正量 = 最优 − 当前，在网络外精确相减。损失、指标、闭环不变 | `--backend-output delta` |
| 前后端分开选检查点 + 前端正则 | 每次验证按综合、后端（增益方向 F1、滑块方向 F1）、前端（深度、频率、聚焦准确率）三个分数分别保存 `best.pt` / `best_backend.pt` / `best_frontend.pt`；评估默认用 `bmode_dl.checkpoint.CombinedModel` 组合两者。前端头丢弃率 0.3、标签平滑 0.1（只摊到可选档）。`--fold none` 时用 `--frontend-epoch` 另存 `frontend.pt` | `--frontend-dropout 0.1 --frontend-label-smoothing 0`；评估 `--select best` |
| metrics.csv 保留全部指标 | 每轮按所有轮次列的并集重写；列表型指标展开为 `_0.._n` | — |

旧检查点（没有 `backend_output` 字段）仍按 delta 方式载入，`tests/verify_six_param_model.py` 检查 `runs/fieldii_v1/fold_0/best.pt` 严格载入。组合模型的交叉验证数字与 `best.pt` 一样是在验证体模上选轮次，偏乐观。

## 8. 修改记录：fieldii_v2 训练日志分析之后（2026-09-16）

`runs/fieldii_v2` 的单步指标全面好于 v1，也高于只看设置的查表基线（增益 0.39→0.24 dB，聚焦准确率 0.78→0.86 且折间标准差 0.14→0.02）。问题都在闭环：25% 的轨迹在两个设置之间打转、一次后端修正也做不了，最终增益误差 5–15 dB，把平均值从停下轨迹的 0.22 dB 拉到 2.55 dB；停下的轨迹里三轴全对只有 16%，但 76% 在 1 档以内，深度与聚焦平均偏浅 0.4 档。改了两处：

| 改动 | 做法 | 恢复旧行为 |
|---|---|---|
| 闭环加滞回与打转冻结，记录逐步路径 | 只有新档概率比当前档高出 `--frontend-margin`（默认 0.1）才换档；回到走过的设置时记为打转并冻结前端，之后只做后端修正。轨迹记录增加 `path`（逐步走到哪、做了什么）、`axis_changes`、`frozen`、`frontend_steps_to_optimum`；汇总增加停下轨迹单独的增益/TGC 误差、各轴平均偏档、打转时是哪一轴在反复改 | `--frontend-margin 0 --no-freeze-on-revisit` |
| 近最优帧指标，并用它选前端检查点 | 只统计当前设置与最优相差不超过 1 档的帧（`*_near`），闭环停点全在这一带；`best_frontend.pt` 默认按 `frontend_score_near` 选。同时报告期望档决策（`*_expected`）与各轴平均偏档，用来判断偏浅是不是 argmax 造成的 | `--frontend-select frontend_score`；评估里两套指标都会报告 |

已有的 `runs/fieldii_v2` 检查点不用重训就能拿到新的闭环与近最优帧结果：直接重跑 `tools_evaluate_six_param.py`。前端检查点选择的改动要重训才生效。
