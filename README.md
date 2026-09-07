# closed_loop_bmode_optimization

六个 B 模成像参数的闭环优化：**Depth、Frequency、Focus(TX)、Gain、TGC、动态范围**。
目标机型为海信超声主机 + L15-4 线阵探头，颈动脉预设。

## 这个项目为什么能做成

海信主机导出的 `Algo_BC0.bin` 取自**整个后端之前**：改 TGC 时它的变化在帧噪声以内，
而显示图像可以动 20 dB 以上。所以 BC0 是一份**与参数无关的组织观测**，
后端的每一个旋钮都可以离线仿真，不必回到主机上试。

参数因此分成两类，这个划分决定了采集量：

| 类别 | 参数 | 为什么 | 采集代价 |
|---|---|---|---|
| **前端** | Depth、Frequency、Focus | 改变发射与波束合成，**BC0 会变** | 必须逐点实采 |
| **后端** | Gain、TGC、动态范围 | 在 BC0 之后，**BC0 不变** | 从 BC0 合成，只需标定 |

## 目录

```
bmode_opt/                     前向模型、指标、标定与数据读取
  hisense_loader.py            解析主机导出目录（.pdt 参数 + BC0 + 截图）
  hisense_backend_sim.py       后端前向模型：BC0 -> dB -> 灰阶；含全部标定函数
  hisense_metrics.py           体模指标：深度带电平、斑点 SNR、囊肿 CNR、点目标宽度
  hisense_display_response.py  由多曝光对恢复灰阶响应曲线
  hisense_tgc_optimizer.py     TGC 的解析求逆控制律
  hisense_analyze.py           成套分析出图
  fieldii_loader.py            读 Field II 仿真 shard，接入同一条渲染链路

docs/
  phantom_acquisition_protocol_v4.md   主协议：采集设计、七个判定性实验的结论、分析规则
  hisense_acquisition_protocol.md      前置协议：TGC 执行器标定
  *.pdf                                两篇作为方法参照的论文

data/
  hisense_medical/             主机实采，179 帧，见协议 §12
  field_ii/full/               Field II 仿真，4560 个 shard，见协议 §9b
```

## 运行环境

模块本身只依赖 numpy / matplotlib / Pillow。
**读 Field II 数据需要 h5py**，用 conda 环境 `cubdl`：

```bash
C:/Users/sunbaolin/miniconda3/envs/cubdl/python.exe bmode_opt/fieldii_loader.py
```

## 常用入口

```bash
cd bmode_opt

# 列出一个场次的全部采集帧与关键参数
python hisense_loader.py --data-dir ../data/hisense_medical/20260819

# 从一次 TGC 扫描重新拟合后端模型常量，并逐帧比对仿真与截图
python hisense_backend_sim.py --data-dir ../data/hisense_medical/20260819

# 对某一帧求 TGC 建议值
python hisense_tgc_optimizer.py --data-dir ../data/hisense_medical/20260819 \
    --capture ../data/hisense_medical/20260819/S0-a

# 成套分析出图
python hisense_analyze.py --data-dir ../data/hisense_medical/20260819 --out-dir ../output
```

## 当前进度

前端采集与七个判定性实验（E0 / E0b / E1 / E2 / E3 / E4 / E5）**全部完成**，
后端前向模型已按 E5 的结果改造并通过灰阶域保真度检验（1.86 灰阶）。

剩余工作全部在代码侧，见协议 §14：

1. **阶段 2** —— 用 Field II 的真值标定目标函数（进行中的下一步）
2. 阶段 3 —— 逐帧求 θ_optimal
3. 阶段 4 —— Field II 预训练 → 实机微调

> 协议 `docs/phantom_acquisition_protocol_v4.md` 是这个项目的**主文档**，
> 记录了每一个结论的实测依据，也记录了推翻过的说法和错误链。
> 改动模型或采集方案之前先读它。
