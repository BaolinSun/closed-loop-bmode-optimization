# 海信人体颈动脉五参数联合训练（2026-11-15）

训练深度、频率、聚焦、增益、TGC；动态范围不训练。沿用现有 SixParamNet 主干，
不混入可穿戴数据。需要在服务器重新训练，本次本地只做标签和合成梯度检查。

## 本次变化

- 训练命令明确恢复 depth_cls / depth_ord / depth_dir、frequency_cls / frequency_dir。
  这些项原本就有实现，之前的人体运行通过 loss-weight=0 将其关闭；旧检查点不会自动变成五参数模型。
- 新选项 `--frontend-set-targets`：深度使用标签已有的 `depth_acceptable_mm`，集合内任一档均可；
  有序距离改成到最近可接受档的距离。没有集合的旧标签退回单一最优档，未定出的标签仍排除。
- 集合监督下，深度和频率方向遵循同一集合：当前档可接受就监督保持，否则监督通向集合的方向。
  频率方向与档位损失使用相同的 frequency_loss_weight / borderline 降权。
- 增加 depth_hit、各轴 balanced_hit 与每个主标签档位的样本数、集合命中率。
  聚焦没有集合，balanced_hit 就是各主标签档位等权的准确率。
- 新选模指标 `frontend_score_balanced` 是深度、频率、聚焦三个 balanced_hit 的均值。
  它对主标签档位等权，可以暴露少数档表现；并未通过重采样改变训练分布。
  仍保留原先的严格 top1、near、方向 F1 和查表基线，便于与旧实验对照。
- 默认训练行为保持兼容：不开 `--frontend-set-targets` 时仍用旧损失。

集合可以包含多个方向，但当前档已可接受时方向监督只选“保持”。推理仍按档位概率 argmax
决定最优档，因此可能建议从一个可接受档转到另一个可接受档；方向头是辅助监督。
旧方向准确率 / 混淆矩阵仍按标签的单一主答案统计，可能把集合内的另一种正确决策算错，
应结合集合命中率看，不能单独作为新损失的成败依据。

## 服务器文件

在服务器仓库根目录执行命令，环境需有 numpy、PyTorch（支持 torch.amp.GradScaler 的版本，
现有记录使用 PyTorch 2.6.0 + CUDA 11.8）。训练和评估不需要原始 BC0、h5py 或 bmode_opt。

必须同步最新的：

- `bmode_dl/` 全部 Python 文件。
- `tools_train_six_param.py`、`tools_evaluate_six_param.py`。
- `tests/verify_five_param_training.py`（检查用，不执行训练）。
- `data/labels_console_invivo_carotid_target.jsonl`（17,545 行，约 49 MB，gitignored，需要单独传）。
- `data/console_dl_cache_invivo_carotid/cache.npz` 与 `index.json`。

缓存无需重建。当前本机缓存位于
`E:/数据/closed_loop_bmode_optimization/console_dl_cache_invivo_carotid/`，
cache.npz 为 4,636,794,826 字节。复制到服务器上面的 data 目录即可；也可以将命令的 --cache
改为服务器上已有缓存的绝对路径。缓存 index 中保留旧本机路径是生成来源信息，不要求服务器有 E 盘。

也可使用本次生成的 `data/server_packages/invivo_five_v1.zip`，已包含上述代码、标签与本文，
解压到服务器仓库根目录；大缓存不在压缩包内，需要另传。压缩包可重新生成：

```powershell
& C:/Users/sunbaolin/miniconda3/envs/cubdl/python.exe tools_package_invivo_training.py
```

## 服务器执行顺序（Linux / Bash）

先激活服务器上有 PyTorch 的环境，然后在仓库根目录运行。

```bash
python tests/verify_five_param_training.py
```

这个检查不会启动训练、不执行 optimizer.step、不保存权重；本地已通过。

首轮从头联合训练，batch / amp / scalar_norm / focus_head 沿用 invivo_bf_v2，
先保留 softmax 聚焦头，不叠加远壁辅助头，便于判断加入两个任务后的变化。
深度、频率标签不均衡仍然存在，这一轮不声称已经解决标签质量问题。

```bash
python tools_train_six_param.py \
  --cache data/console_dl_cache_invivo_carotid \
  --labels data/labels_console_invivo_carotid_target.jsonl \
  --name invivo_five_v1 \
  --device cuda --group-by subject --fold 0 \
  --epochs 40 --batch 16 --lr 3e-4 --warmup-epochs 5 \
  --eval-every 2 --amp --scalar-norm levels \
  --recorded-start-prob 0.3 --focus-head softmax \
  --frontend-set-targets --frontend-select frontend_score_balanced \
  --loss-weight depth_cls=1 --loss-weight depth_ord=0.5 --loss-weight depth_dir=0.5 \
  --loss-weight frequency_cls=1 --loss-weight frequency_dir=0.5 \
  --loss-weight dr_reg=0 --loss-weight dr_dir=0
```

开头应打印 `active parameter tasks: depth, frequency, focus, gain, tgc` 和
`front-end set targets: True`。原有按人划分保持不变：训练 132 人 / 12,232 帧，
验证 28 人 / 2,631 帧，测试 29 人 / 2,682 帧。`--fold 0` 在这种显式划分下不是重新交叉验证。
不要用旧命令中将 depth / frequency 置零的参数，也不要先用 --fold none 合并验证集。
本命令不使用 --init-checkpoint，不需要拷贝旧权重，不覆盖旧实验。

训练结束先看验证集，同时给出查表基线和后端闭环：

```bash
python tools_evaluate_six_param.py \
  --run runs/invivo_five_v1 --select combined \
  --cache data/console_dl_cache_invivo_carotid \
  --labels data/labels_console_invivo_carotid_target.jsonl \
  --device cuda --groups val --closed-loop backend \
  --stop-deadband-levels 2 --amp --tag val_bloop_db2
```

根据验证集确定配置后，做最终测试：

```bash
python tools_evaluate_six_param.py \
  --run runs/invivo_five_v1 --select combined \
  --cache data/console_dl_cache_invivo_carotid \
  --labels data/labels_console_invivo_carotid_target.jsonl \
  --device cuda --groups test --closed-loop backend \
  --stop-deadband-levels 2 --amp --tag test_bloop_db2
```

如显存不足，将训练 batch 改为 8，并在实验记录中说明。评估当前固定 batch_size=64，
显存要求可能高于训练单批；需要时再单独调整评估批量，不要因此改输入分辨率或标签。

## 查看结果

- `runs/invivo_five_v1/train_report.txt` 与 `fold_0/metrics.csv`：五个任务的训练损失、验证曲线、选中轮次。
- `fold_0/best_frontend.pt`：按 balanced 分数选出的前端模型；`best_backend.pt`：后端模型。
- `evaluation_summary_combined_val_bloop_db2.txt`：验证集汇总。
- `evaluation_summary_combined_test_bloop_db2.txt`：独立测试集汇总。
- 折目录里的 evaluation_report / evaluation JSON：分档样本数、集合命中率、查表基线、后端闭环。

重点比较深度与频率的分档结果及 balanced_hit，防止多数档高分掩盖少数档失败；同时检查聚焦、
增益、TGC 是否因多任务联合训练退步。`gain_feedback_slope` 应远小于 1。
目前人体数据没有完整的深度×频率×聚焦组合网格，前端只能评单步；backend 闭环的收敛率只代表后端。
远壁仍为模拟阅片产生的暂定标签，本次没有替换成医生真值。
