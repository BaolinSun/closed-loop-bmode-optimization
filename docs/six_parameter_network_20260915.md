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

## 9. 修改记录：fieldii_v3 训练日志分析之后（2026-09-16）

`runs/fieldii_v3`（15 个体模、2520 行）的闭环问题基本解决：停下 70%→92%，最终增益误差 2.55→0.28 dB、TGC 0.686→0.218 dB。只看停下的轨迹两次几乎一样（增益 0.264→0.246 dB），说明改善来自不再打转，而不是后端预测变准。剩下两件事各改一处：

| 改动 | 做法 | 恢复旧行为 |
|---|---|---|
| 滞回只拦回头 | fieldii_v3 对每一次换档都要概率余量，等于整体倾向不动，深度仍平均停早 0.35 档。默认改为 `--margin-mode revisit`：只有建议回到这条轨迹走过的设置时才要余量，没走过的新设置照常换 | `--margin-mode always` |
| 每折打印物理量覆盖范围 | 训练报告按折打印衰减、电子噪声、声速的训练/验证范围，验证集超出训练范围时标 EXTRAPOLATES，并写进 summary.json | — |

第 2 折的验证集同时拿走了衰减最低（0.358）与最高（0.739）的体模，四折里只有它的衰减估计比猜训练集均值更差（+10%，另外三折为 −28%、−31%、−47%），频率命中 0.62、深度 0.47，都比其余三折低 0.2 以上。这件事此前要手工查标签才看得出来，现在直接写在训练日志里。是否让极端体模固定留在训练集，属于交叉验证口径的改变，未做。

期望档决策（`*_expected`）三次都不如 argmax，且 argmax 在近最优帧上的平均偏差已降到 −0.06 档，这条不再跟进。

## 10. 闭环四组对照的结论（2026-09-17）

用 fieldii_v3 的同一批检查点跑了四种闭环设置（单步指标逐位相同，差异只来自循环逻辑）：

| 设置 | 停下 | 最终增益误差 | 最终 TGC 误差 | 三轴全对 | 深度平均停早 |
|---|---|---|---|---|---|
| 余量 0，不冻结（fieldii_v2 行为） | 0.799 | 2.121 dB | 0.615 dB | 0.211 | +0.345 档 |
| 余量 0.10，每次换档（fieldii_v3） | 0.924 | 0.277 dB | 0.218 dB | 0.216 | +0.349 档 |
| 余量 0.10，只拦回头 | 0.924 | 0.273 dB | 0.214 dB | 0.216 | +0.343 档 |
| **余量 0 + 冻结（新默认）** | **0.925** | **0.271 dB** | **0.214 dB** | **0.219** | +0.345 档 |

- **起作用的只有打转冻结。** 被冻结的 376 条轨迹（14.9%）在不冻结时只有 7.4% 能停下、平均增益误差 11.75 dB、平均只做 0.38 次后端修正；冻结后 87% 停下、误差 0.33 dB。其余轨迹在两种设置下 98.8% 逐条相同。
- **滞回贡献为零。** 余量 0 与 0.10 有 99.05% 的轨迹逐条相同，24 条不同里 13 条更好、11 条更差；汇总指标要么持平要么略好。`--frontend-margin` 默认因此改为 0，它与 `--margin-mode` 一并保留供实验。第 9 节那处"滞回只拦回头"的改动经此判定为无效改动，保留只是因为它与关闭滞回等价。
- **深度停早与闭环无关**：四组都是 +0.343 到 +0.349 档。它是停点的选择效应——轨迹只停在网络说"不用动"的地方——要靠前端模型本身和体模数量解决。

闭环这条线到此收尾。剩余差距（聚焦停点正确率 0.46、三轴全对 0.22、1 档内 0.84）都在模型侧。

## 11. 实机（海信）微调（2026-09-18）

`labels_console.jsonl`（422 帧、18 个场次、谐波 269 / 基波 153）与 Field II 标签的差异，决定了原有代码不能直接微调：

| 差异 | 不处理的后果 | 做法 |
|---|---|---|
| 数据是 BC0（870 深度点 × 256 线），不是 Field II 的 HDF5 包络 | 缓存脚本读不了 | 新增 `tools_build_console_training_cache.py`：BC0 / 本组 counts_per_db + 本组深度响应（即生成后端标签时求解的那幅 dB），降到 512 × 128 |
| 标签行没有 `reference_db`、底噪 | `encode_rows` 直接报错 | 缓存写入本组 pivot_db 与底噪（先经 `floors_in_counts`，与生成标签同一步），载入时覆盖标签行 |
| 实机 dB 刻度与 Field II 差一个任意常数（中位约 29 对 −41 dB） | 写死的标量换算把参考推到 2.3–3.2、底噪 8.6–11.3，预训练只见过约 −1 到 1.4 | `--scalar-norm data`：增益按训练集标准化，参考与底噪用图像 dB 的均值方差标准化；旧检查点仍按 `fixed` |
| 档位不同：深度 7 档（25.1–75.4 mm）、频率 9 档（谐波 4.4–5.7 与基波 5–11.4 的并集）、聚焦 6 档 | 前端输出层维度对不上 | `--init-checkpoint` 部分装入：只有前端三个头的最后一层重新初始化，其余 147 个张量装入；最优增益 / TGC 的标准化按实机重新估计 |
| 全部来自同一个体模，独立单位是探头摆放；E8、E9 把同一次摆放的基波、谐波存成 `_GEN` / `_THI` 两个目录 | 按目录分组会让同一次摆放同时进训练与验证 | `--group-by placement`：按 family_id 分组并合并只差模式后缀的目录，18 个摆放单位 |
| 没有完整的 深度 × 频率 × 聚焦 网格 | 闭环仿真无意义 | 评估脚本在实机数据上自动跳过闭环 |

冻结范围 `--freeze encoders`（图像与剖面编码器，3.23 M 冻结、1.73 M 可训练）；`backbone` 只训输出头，`none` 全部训练。

本地冒烟（CPU、第 0 折、从 `runs/fieldii_v3/fold_0/best_backend.pt` 起、20 轮）：链路完整，增益误差从第 1 轮的 7.1 dB 降到 1.34 dB。与同一折的查表基线相比各有胜负——增益平均误差 1.34 对 3.38 dB、深度 0.768 对 0.696 更好，但增益落入死区 0.24 对 0.53、聚焦 0.653 对 0.806 更差。单折 5 个摆放、114 帧，只能说明能学，不是结论。**实机全部来自同一个体模，查表基线相当于"记住这个体模"，是很强的对照**，交叉验证必须与它比。

服务器上的命令见本节下方；建议同时跑"从零训练"对照，才能判断 Field II 预训练有没有用。

```bash
# 1. 用全部 15 个 Field II 体模重新预训练（标量改为按数据标准化，两个域刻度一致）
python tools_train_six_param.py --fold none --epochs 150 --scalar-norm data --amp --name fieldii_v3_all
# 2. 实机微调，4 折按摆放交叉验证
python tools_train_six_param.py --cache data/console_dl_cache --labels data/labels_console.jsonl \
    --init-checkpoint runs/fieldii_v3_all/all_data/last.pt --freeze encoders --group-by placement \
    --scalar-norm data --fold all --epochs 40 --lr 1e-4 --warmup-epochs 2 --eval-every 1 --batch 16 \
    --amp --name console_ft_v1
# 3. 评估（含查表基线；闭环自动跳过）
python tools_evaluate_six_param.py --run runs/console_ft_v1 --cache data/console_dl_cache \
    --labels data/labels_console.jsonl --amp
# 4. 对照：不用预训练
python tools_train_six_param.py --cache data/console_dl_cache --labels data/labels_console.jsonl \
    --freeze none --group-by placement --scalar-norm data --fold all --epochs 40 --lr 3e-4 \
    --warmup-epochs 2 --eval-every 1 --batch 16 --amp --name console_scratch
```

## 12. 增益正反馈的发现与防护（2026-09-21）

### 12.1 现象

`fieldii_v4`（新数据：32 个体模，训练 24 / 验证 6 / 测试 2）用 `--scalar-norm data` 训练，单步指标很好，闭环却发散：验证集最终增益误差 11.9 dB、测试集 36.4 dB，最坏的轨迹后端修正逐步变成 +368、−472、+671、−997、+1530 级。固定一幅图扫描当前增益，预测最优增益对当前增益的斜率中位数 **1.45**，40 帧里全部大于 1——正反馈，每做一次修正目标跑得更远。改用 `--scalar-norm fixed` 重训后斜率 **0.002**，验证集停下 0.89、最终增益误差 0.47 dB，测试集停下 0.95、0.22 dB，单步指标与 data 模型几乎相同。实机微调模型（data）斜率 −0.15 / −0.05，不受影响。

### 12.2 `--scalar-norm` 的三种取值

只影响 17 个输入标量中的 4 个，图像、剖面与其余 13 个标量完全相同：

| 标量 | `fixed` | `data` | `levels`（新） |
|---|---|---|---|
| 当前增益 | 增益 / 10 | （增益 − 训练均值）/ 训练标准差 | 增益 / 10 |
| 曝光参考 | （参考 + 45）/ 30 | （参考 − 图像 dB 均值）/ 图像 dB 标准差 | 同 data |
| 顶部 / 底部底噪 | （底噪 + 90）/ 10 | （底噪 − 图像 dB 均值）/ 图像 dB 标准差 | 同 data |

`fixed` 的常数按 Field II 的 dB 刻度写死，实机的参考与底噪会落到 2.3–11.5（预训练只见过约 −1 到 1.4）；`data` 为此而加。但 Field II 训练集增益标准差只有 3.8 dB，标准化后当前增益成了很强的输入，而训练起点是在最优值附近抽的，网络学会了"最优 ≈ 当前 + 修正"的捷径。`levels` 把两件事分开：增益照 `fixed`（捷径的来源），参考与底噪按 `data`（跨域刻度差的来源）。**Field II 预训练与实机微调都推荐 `levels`。**

### 12.3 防护（默认开启）

| 改动 | 做法 | 关闭 |
|---|---|---|
| 不变性约束 | 每个训练批次对同一批图像另抽一个后端起点再前向一次，惩罚两次预测的最优增益 / TGC 之差（Huber，dB）。最优值是图像的属性，本就与当前设置无关；这一项从根上堵住捷径。训练时间约增加一半 | `--invariance-weight 0` |
| 斜率指标与选模 | 每次验证计算预测最优对当前增益的斜率（`val_gain_feedback_slope`、`val_gain_feedback_frac_gt1`），中位数 ≥ 0.5 时日志标 UNSTABLE，这样的轮次不能成为 `best.pt` / `best_backend.pt`（除非还没有任何稳定轮次）。评估报告也打印斜率，≥ 1 时提示闭环会发散 | — |
| 闭环限幅 | 单步增益修正不超过 40 级、相对起点累计不超过 120 级（`--max-gain-step-clicks`、`--max-gain-total-clicks`，0 = 不限），被限幅的轨迹比例记为 `gain_clamped`。只是最后一道保险 | 设为 0 |
| 运行目录防覆盖 | 运行目录已有训练报告时拒绝启动，除非加 `--overwrite`（此时旧报告被删除，不再首尾相接） | `--overwrite` |

评估输出文件名里加了限幅设置（如 `m000_rev_frz_c40-120`），不会覆盖此前未限幅的结果。

不变性约束在完整训练上的效果还没有验证：只在假模型上确认了它对"跟随当前增益"的输出给出正损失。下次训练的日志里每次验证都会打印斜率，可以直接看到。
