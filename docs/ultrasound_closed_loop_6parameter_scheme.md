# 基于多维图像质量感知的六参数分层闭环超声自动优化方案

## 1. 方案目标

本方案面向超声成像系统的自动参数优化，构建一套 **AI 驱动、分层控制、闭环验证、可回滚** 的六参数调节框架。

闭环参数分为两类：

### 1.1 前端采集参数

1. **Depth**
2. **Transmit Frequency**
3. **Transmit Focus**

### 1.2 后端数字处理参数

4. **Digital Gain**
5. **Digital TGC**
6. **Dynamic Range**

定义参数向量：

\[
\Theta=
\left[
D,\ f,\ z_F,\ G_d,\ \mathbf T,\ DR
\right]
\]

其中：

- \(D\)：成像深度；
- \(f\)：发射中心频率；
- \(z_F\)：发射聚焦深度；
- \(G_d\)：数字全局增益；
- \(\mathbf T=[t_1,\ldots,t_K]\)：数字 TGC 控制点；
- \(DR\)：B-mode 对数压缩 / 显示动态范围。

总体闭环思想：

\[
\boxed{
\text{Quality perception}
\rightarrow
\text{Cause diagnosis}
\rightarrow
\text{Hierarchical control}
\rightarrow
\text{Acquisition / Reconstruction}
\rightarrow
\text{Quality verification}
}
\]

即：

> 图像质量感知 → 图像质量劣化原因诊断 → 分层参数决策 → 前端重新采集 / 后端重新重建 → 图像质量复核 → 接受、回滚或继续局部搜索。

---

## 2. 研究基础与方法来源

本方案综合以下代表性工作的关键思想：

- **El-Zehiry et al., 2013**  
  通过专家图像质量评分学习 \(Q(I)\)，并使用低维 manifold 降低 Depth、Frequency、Focus 等采集参数的搜索空间。

- **Annangi et al., 2020**  
  将图像质量拆分为 Global IQ、Structure IQ、Depth IQ，并根据综合质量选择 Frequency 与 Depth。

- **TGC-Net, 2025**  
  使用 U-Net 从未补偿图像中预测空间增益分布，为 AI 自动 TGC 提供参考。

- **Hwang et al., 2026**  
  将图像质量问题进一步分类为 artifact、incorrect gain、incorrect positioning，为 cause-aware quality diagnosis 提供依据。

- **OATS, 2026**  
  使用 differentiable beamforming 与梯度优化学习 depth-dependent apodization，说明 TGC、Focus、F-number 等参数存在物理耦合。

---

## 3. 总体系统架构

推荐采用：

\[
\boxed{
\text{Slow outer loop}
+
\text{Fast inner loop}
}
\]

其中：

- 慢环控制：Depth、Frequency、TX Focus；
- 快环控制：Digital Gain、Digital TGC、Dynamic Range。

```text
                 Current parameter state
       Θt=[Depth, Freq, TX Focus, Gain, TGC, DR]
                            │
                            ▼
                ┌─────────────────────┐
                │ Ultrasound system   │
                └─────────┬───────────┘
                          │
                     RF acquisition
                          │
                          ▼
                   DAS beamforming
                          │
             Digital Gain / Digital TGC
                          │
                    Log compression
                          │
                    Dynamic Range
                          │
                          ▼
                       B-mode
                          │
                          ▼
             ┌────────────────────────┐
             │ AI Quality Assessment  │
             └────────────┬───────────┘
                          │
        ┌─────────────────┼───────────────────┐
        ▼                 ▼                   ▼
  View/Artifact      Anatomy/Structure    Image quality
      Gate               analysis            vector
        │                                     │
        │          ┌─────────┬────────┬────────┼────────┐
        │          ▼         ▼        ▼        ▼        ▼
        │        Depth     Freq     Focus    Gain     TGC/DR
        │          │         │        │        │        │
        │          └─────────┴────────┴────────┴────────┘
        │                           │
        ▼                           ▼
 reject/hold                Cause-aware controller
                                    │
                    ┌───────────────┴──────────────┐
                    │                              │
               Slow outer loop                Fast inner loop
             D → f → TX Focus             Gain → TGC → DR
                    │                              │
                reacquire                      same RF
                    │                              │
                    └──────────────┬───────────────┘
                                   ▼
                             Verify ΔQ
                         ┌─────────┴────────┐
                      improve             worsen
                         │                   │
                       accept          rollback
                         │                   │
                         └──────────↺────────┘
```

---

## 4. 为什么不能只使用一个整体图像质量分数

六个参数对图像的影响高度耦合。

例如远场偏暗可能来自：

- Frequency 过高；
- Digital TGC 远场增益不足；
- Depth 过大；
- TX Focus 不合理；
- Global Gain 偏低；
- acoustic shadow；
- Dynamic Range 过小。

因此只输出：

\[
Q=6.5/10
\]

并不能告诉控制器：

> 下一步到底应该改哪个参数。

推荐输出一个多维质量向量：

\[
\boxed{
\mathbf q=
[
Q_{\rm view},
Q_{\rm anatomy},
Q_{\rm depth},
Q_{\rm penetration},
Q_{\rm focus},
Q_{\rm brightness},
Q_{\rm axial},
Q_{\rm contrast},
Q_{\rm speckle},
Q_{\rm artifact},
Q_{\rm expert}
]
}
\]

各分量定义：

| 质量分量 | 含义 | 主要关联参数 |
|---|---|---|
| \(Q_{view}\) | 标准切面 / 目标器官有效性 | Gate |
| \(Q_{anatomy}\) | anatomy 完整性与结构可见性 | Depth / Frequency / Focus |
| \(Q_{depth}\) | 成像深度是否合理 | Depth |
| \(Q_{penetration}\) | 远场穿透与深部结构可见性 | Frequency |
| \(Q_{focus}\) | ROI 聚焦质量 | TX Focus |
| \(Q_{brightness}\) | 全局亮度 | Digital Gain |
| \(Q_{axial}\) | 深度方向亮度均匀性 | Digital TGC |
| \(Q_{contrast}\) | 局部/组织对比度 | Frequency / Focus / DR |
| \(Q_{speckle}\) | 散斑纹理合理性 | TGC / DR |
| \(Q_{artifact}\) | 阴影、运动、截断等 | Gate |
| \(Q_{expert}\) | 专家整体主观质量 | Overall objective |

---

## 5. Validity Gate：先判断当前图像是否允许自动调参

闭环开始前首先判断：

- 是否为目标切面；
- 目标器官是否存在；
- anatomy 是否严重截断；
- 是否有大面积 acoustic shadow；
- 是否有严重 motion artifact；
- 探头是否明显偏离目标；
- 图像是否已经失去可诊断性。

若：

\[
Q_{\rm view}<T_{\rm view}
\]

或：

\[
Q_{\rm artifact}<T_{\rm artifact}
\]

则系统进入：

\[
\boxed{
\text{HOLD}
}
\]

即：

- 暂停六参数自动调整；
- 保持当前参数；
- 输出“重新定位探头 / 保持稳定”等提示；
- 等待图像重新进入有效状态。

这样可以避免：

\[
\text{shadow造成图像暗}
\Rightarrow
\text{误判为Gain低}
\Rightarrow
\text{Gain/TGC持续上调}
\]

---

## 6. Anatomy / Landmark Quality

该模块主要服务：

- Depth；
- Frequency；
- Focus。

推荐采用：

- anatomy segmentation；
- landmark detection；
- landmark visibility confidence。

若有 \(N_L\) 个关键解剖 landmark：

\[
Q_{\rm landmark}
=
\frac{1}{N_L}
\sum_{i=1}^{N_L}
P(l_i\ \text{visible}|I)
\]

同时：

\[
Q_{\rm anatomy}
=
w_1Q_{\rm completeness}
+w_2Q_{\rm visibility}
+w_3Q_{\rm delineation}
\]

其中：

- \(Q_{completeness}\)：结构完整性；
- \(Q_{visibility}\)：关键结构清晰度；
- \(Q_{delineation}\)：边界可分辨程度。

---

## 7. Depth 闭环优化

### 7.1 优化目标

Depth 的目标不是越深越好，而是：

\[
\boxed{
\text{完整覆盖目标 anatomy 的最小合理成像深度}
}
\]

若 anatomy 最深位置为：

\[
z_{\max}
\]

则初始深度：

\[
D_0=z_{\max}+M
\]

其中 \(M\) 为 far-field margin。

### 7.2 Depth 质量函数

\[
Q_{\rm depth}
=
w_cQ_{\rm completeness}
-w_eP_{\rm empty}
-w_tP_{\rm truncation}
\]

其中：

- \(Q_{completeness}\)：ROI 是否完整；
- \(P_{empty}\)：无用远场区域比例；
- \(P_{truncation}\)：远场 anatomy 截断惩罚。

### 7.3 闭环流程

1. segmentation 获得目标 anatomy；
2. 求 \(z_{max}\)；
3. 计算 \(D_0=z_{max}+M\)；
4. 映射到设备合法 Depth 档位；
5. 重新采集 RF；
6. 重新评价 \(Q_{depth}\)；
7. 改善则接受，否则 rollback。

---

## 8. Frequency 闭环优化

### 8.1 物理目标

Frequency 是：

\[
\text{resolution}
\leftrightarrow
\text{penetration}
\]

的折中。

随着：

\[
f\uparrow
\]

通常：

\[
resolution\uparrow
\]

同时：

\[
attenuation\uparrow,
\quad
penetration\downarrow
\]

因此推荐：

\[
\boxed{
\text{在关键结构保持可见时选择尽可能高的发射频率}
}
\]

### 8.2 Frequency 质量函数

\[
Q_f
=
w_LQ_{\rm landmark}
+w_RQ_{\rm resolution}
+w_PQ_{\rm penetration}
\]

可写为约束优化：

\[
\boxed{
f^*=\max f
}
\]

subject to

\[
Q_{\rm landmark}>T_L
\]

\[
Q_{\rm penetration}>T_P
\]

### 8.3 调节策略

1. 从中间频率开始；
2. 若远场清楚且 landmark 保持完整：
   - 尝试更高频率；
3. 若深部结构丢失：
   - 回退一个频率档位；
4. 选择满足结构可见性的最高频率。

---

## 9. TX Focus 闭环优化

### 9.1 初始 Focus

根据 anatomy segmentation：

\[
z_{\rm ROI}
\]

设：

\[
z_F^{(0)}=z_{\rm ROI}
\]

### 9.2 Focus 质量函数

\[
Q_F
=
w_sQ_{\rm sharpness}
+w_lQ_{\rm landmark}
+w_cQ_{\rm local\ contrast}
\]

### 9.3 局部搜索

仅在：

\[
z_F
\in
[
z_{\rm ROI}-\Delta z,
z_{\rm ROI}+\Delta z
]
\]

进行搜索。

### 9.4 Focus 与 TGC 顺序

由于 Focus 会影响焦点附近回波强度：

\[
\boxed{
\text{Focus 必须先调，TGC 后调}
}
\]

避免 TGC 把 Focus 错误造成的亮度变化“补掉”。

---

## 10. Digital Gain 闭环优化

Digital Gain 负责：

\[
\boxed{
\text{global brightness}
}
\]

不负责深度方向均衡。

### 10.1 Tissue Mask

建立有效组织区域：

\[
M_T
\]

排除：

- background；
- fluid；
- cyst；
- acoustic shadow；
- 极强 reflector。

### 10.2 Robust Brightness

\[
B_{\rm med}
=
{\rm median}_{(x,z)\in M_T}
I(x,z)
\]

定义：

\[
Q_G
=
\exp
\left[
-\frac{(B_{\rm med}-B_0)^2}
{2\sigma_B^2}
\right]
\]

其中 \(B_0\) 建议由专家优选图像统计学习得到。

### 10.3 更新

模型输出：

\[
\Delta G_d
\]

更新：

\[
G_{d,t+1}
=
G_{d,t}
+
\lambda_G\Delta G_d
\]

其中：

\[
0<\lambda_G\le1
\]

防止过冲。

---

## 11. Digital TGC 闭环优化

### 11.1 参数化

Digital TGC 不建议输出上百个 depth samples。

定义：

\[
\boxed{
\mathbf T=
[t_1,t_2,\ldots,t_K]
}
\]

建议：

\[
K=6\sim8
\]

再通过 spline：

\[
T(z)=Spline(\mathbf T)
\]

数字增益：

\[
x_T(x,z)
=
x(x,z)
10^{T(z)/20}
\]

### 11.2 深度分区

把有效组织区域分为：

\[
M_1,M_2,\ldots,M_K
\]

计算：

\[
b_k=
{\rm median}
\{
I(x,z):(x,z)\in M_k
\}
\]

得到：

\[
e_k=b_k^*-b_k
\]

形成：

\[
\mathbf e_T=[e_1,\ldots,e_K]
\]

模型 / 控制器输出：

\[
\Delta\mathbf T
\]

---

## 12. Digital Gain 与 Digital TGC 显式解耦

计算深度误差平均：

\[
\bar e
=
\frac1K
\sum_{k=1}^{K}e_k
\]

残差：

\[
\tilde e_k=e_k-\bar e
\]

然后：

\[
\boxed{
\bar e
\rightarrow
Digital\ Gain
}
\]

\[
\boxed{
\tilde{\mathbf e}
\rightarrow
Digital\ TGC
}
\]

例如：

\[
[-5,-5,-5,-5,-5,-5]
\]

说明整幅图偏暗，应主要提高 Gain。

而：

\[
[0,0,-1,-2,-4,-6]
\]

说明远场逐渐变暗，应主要增加中远场 TGC。

---

## 13. TGC 物理与稳定性约束

### 13.1 一阶平滑

\[
L_{\rm smooth}
=
\frac1{K-1}
\sum_{k=1}^{K-1}
(T_{k+1}-T_k)^2
\]

### 13.2 二阶曲率

\[
L_{\rm curvature}
=
\frac1{K-2}
\sum_{k=2}^{K-1}
(T_{k+1}-2T_k+T_{k-1})^2
\]

### 13.3 最大 slope

\[
\left|
\frac{\Delta T}{\Delta z}
\right|
<S_{\max}
\]

### 13.4 最大单次更新量

\[
|\Delta T_k|<\Delta T_{\max}
\]

建议单次控制步长控制在约：

\[
1\sim3\ {\rm dB}
\]

---

## 14. Dynamic Range 闭环优化

这里的 Dynamic Range 是：

\[
\boxed{
\text{B-mode log-compression / display dynamic range}
}
\]

不是 ADC 动态范围。

若 envelope：

\[
E(x,z)
\]

则：

\[
L(x,z)
=
20\log_{10}
\frac{E(x,z)}{E_{\max}}
\]

显示映射：

\[
B(x,z)
=
{\rm clip}
\left[
\frac{L(x,z)+DR}{DR},
0,1
\right]
\]

### 14.1 DR 影响

较小 DR：

- 视觉对比更强；
- 灰度层次减少；
- 暗部更易 clipping。

较大 DR：

- 灰度层次增加；
- 视觉 contrast 降低。

### 14.2 DR 质量函数

\[
Q_{DR}
=
w_cQ_{\rm local\ contrast}
+
w_lQ_{\rm landmark}
+
w_sQ_{\rm speckle}
-
w_{clip}P_{\rm clipping}
\]

其中：

\[
P_{\rm clipping}
=
P(B<\epsilon)
+
P(B>1-\epsilon)
\]

### 14.3 为什么 DR 最后调

DR 会改变视觉 contrast，因此建议六参数顺序固定为：

\[
\boxed{
Depth
\rightarrow
Frequency
\rightarrow
TX\ Focus
\rightarrow
Digital\ Gain
\rightarrow
Digital\ TGC
\rightarrow
Dynamic\ Range
}
\]

这样减少参数之间的相互混淆。

---

## 15. 推荐 AI 模型架构

推荐：

\[
\boxed{
\text{Shared Encoder + Multi-task Heads}
}
\]

### 15.1 主干网络

优先推荐：

- **ConvNeXt-Tiny**

轻量部署可选：

- **EfficientNet-B0**
- **EfficientNet-B3**

### 15.2 输入

推荐：

\[
4\sim8
\]

帧短 cine，而不是单帧。

### 15.3 Temporal aggregation

可采用：

- temporal average pooling；
- attention pooling；
- TSM；
- 小型 GRU。

### 15.4 总体结构

```text
               4–8 B-mode frames
                       │
                       ▼
               Shared CNN encoder
                       │
                  Temporal pooling
                       │
        ┌──────────────┼────────────────────────────┐
        │              │             │              │
        ▼              ▼             ▼              ▼
 Anatomy decoder   Global IQ     Validity head   Parameter heads
 U-Net style       ordinal       artifact/view   D/f/F/G/TGC/DR
```

---

## 16. 六参数 Head 输出设计

| 参数 | 推荐输出 | 任务形式 |
|---|---|---|
| Depth | \(\Delta D\) + shallow/correct/deep | ordinal + regression |
| Frequency | lower/correct/higher | ordinal classification |
| TX Focus | \(\Delta z_F\) | Huber regression |
| Digital Gain | \(\Delta G_d\) | Huber regression |
| Digital TGC | \(K\)-point \(\Delta\mathbf T\) | vector regression |
| Dynamic Range | \(\Delta DR\) | Huber regression |

推荐预测：

\[
\boxed{
\Delta\Theta
}
\]

而不是：

\[
\Theta^*
\]

原因：

- 更适合闭环；
- 更容易跨设备；
- 更容易加入步长约束；
- 更方便 rollback。

---

## 17. 专家标签设计

推荐 overall quality：

\[
Q_{\rm expert}
\in
\{1,2,3,4,5\}
\]

同时增加 cause-specific 参数方向标签：

### Depth

\[
\{shallow,correct,deep\}
\]

### Frequency

\[
\{low,correct,high\}
\]

### Focus

\[
\{shallow,correct,deep\}
\]

### Gain

\[
\{dark,correct,bright\}
\]

### TGC

按：

- near；
- mid；
- far；

分别标注。

### Dynamic Range

\[
\{too\ narrow,correct,too\ wide\}
\]

---

## 18. Pairwise Preference 标注

对同一个患者、同一个稳定切面，给专家展示：

\[
I(\Theta_a),\ I(\Theta_b)
\]

让专家选择：

\[
a>b,
\quad
a=b,
\quad
a<b
\]

尤其适合：

- Frequency；
- Focus；
- Dynamic Range。

对于闭环系统，排序能力往往比绝对评分精度更重要：

\[
\boxed{
Q_{new}>Q_{old}?
}
\]

才是控制器最关心的问题。

---

## 19. 总训练损失函数

推荐：

\[
\boxed{
\mathcal L_{\rm total}
=
\sum_i\lambda_i\mathcal L_i
}
\]

展开为：

\[
\begin{aligned}
\mathcal L_{\rm total}
={}&
\lambda_vL_{\rm valid}
+\lambda_aL_{\rm anatomy}
+\lambda_qL_{\rm quality}
\\
&+\lambda_DL_D
+\lambda_fL_f
+\lambda_FL_F
\\
&+\lambda_GL_G
+\lambda_TL_T
+\lambda_RL_{DR}
\\
&+\lambda_{\rm rank}L_{\rm rank}
+\lambda_{\rm temp}L_{\rm temporal}
\end{aligned}
\]

---

## 20. Validity / Artifact Loss

多标签：

\[
L_{\rm valid}
=
BCE(
\hat{\mathbf y}_{valid},
\mathbf y_{valid}
)
\]

若类别不平衡，使用：

\[
\boxed{
Focal\ Loss
}
\]

可包含标签：

\[
[
valid\ view,
shadow,
motion,
wrong\ plane,
truncated\ ROI
]
\]

---

## 21. Anatomy Loss

### Segmentation

\[
L_{\rm anatomy}
=
\lambda_{Dice}L_{Dice}
+
\lambda_{CE}L_{CE}
\]

### Landmark heatmap

\[
L_{\rm landmark}
=
MSE(
H_{\rm pred},
H_{\rm GT}
)
\]

对 Depth 和 Focus，推荐优先使用 segmentation / heatmap，因为可以获得更准确的：

\[
z_{min},z_{max},z_{ROI}
\]

---

## 22. Expert Quality Loss

专家评分为有序标签：

\[
1<2<3<4<5
\]

因此推荐：

\[
\boxed{
Ordinal\ Regression
}
\]

例如：

- CORAL；
- CORN。

即：

\[
L_{\rm quality}=L_{\rm ordinal}
\]

不推荐普通五分类 CE 作为唯一方案。

---

## 23. Pairwise Ranking Loss

若：

\[
I_a>I_b
\]

定义：

\[
s_{ab}=+1
\]

则：

\[
L_{\rm rank}
=
\log
\left[
1+
\exp
\left(
-s_{ab}
(\hat Q_a-\hat Q_b)
\right)
\right]
\]

用于训练模型正确学习相对质量顺序。

---

## 24. 参数回归 Loss

连续参数建议统一使用：

\[
\boxed{
Huber\ Loss
}
\]

例如：

\[
L_G
=
Huber(
\Delta\hat G,
\Delta G^{GT}
)
\]

\[
L_F
=
Huber(
\Delta\hat z_F,
\Delta z_F^{GT}
)
\]

\[
L_{DR}
=
Huber(
\Delta\widehat{DR},
\Delta DR^{GT}
)
\]

---

## 25. Digital TGC 专用 Loss

\[
L_T
=
L_{\rm value}
+
\lambda_sL_{\rm smooth}
+
\lambda_cL_{\rm curvature}
+
\lambda_0L_{\rm zeroMean}
\]

### 25.1 控制点误差

\[
L_{\rm value}
=
\frac1K
\sum_{k=1}^{K}
Huber(
\Delta\hat T_k,
\Delta T_k^{GT}
)
\]

### 25.2 一阶平滑

\[
L_{\rm smooth}
=
\frac1{K-1}
\sum_{k=1}^{K-1}
(\hat T_{k+1}-\hat T_k)^2
\]

### 25.3 二阶曲率

\[
L_{\rm curvature}
=
\frac1{K-2}
\sum_{k=2}^{K-1}
(\hat T_{k+1}-2\hat T_k+\hat T_{k-1})^2
\]

### 25.4 与 Gain 解耦

\[
L_{\rm zeroMean}
=
\left(
\frac1K
\sum_k
\Delta\hat T_k
\right)^2
\]

用于迫使 TGC 主要负责 depth-relative correction，而不是整体亮度。

---

## 26. Temporal Stability Loss

对于稳定 probe / anatomy：

\[
L_{\rm temporal}
=
\|
\hat{\mathbf q}_t
-
\hat{\mathbf q}_{t-1}
\|_1
\]

控制目标：

\[
J
=
Q_{\rm image}
-
\lambda_\theta
\|\Delta\Theta\|^2
\]

意味着只有在图像质量收益足够明显时才改参数。

---

## 27. 后端三参数 Fast Inner Loop

后端参数：

\[
\Theta_B=
[
G_d,\mathbf T,DR
]
\]

它们不需要重新发射。

定义重建：

\[
I
=
\mathcal R
(
RF;
G_d,\mathbf T,DR
)
\]

理论上可进行：

\[
\Theta_B^*
=
\arg\max_{\Theta_B}
Q_\phi
[
\mathcal R(RF;\Theta_B)
]
\]

推荐采用：

\[
\boxed{
\text{Prediction + Local refinement}
}
\]

而不是纯 gradient search 或纯 CNN。

---

## 28. 后端优化流程

### Step 1：模型预测

\[
[
\Delta G,
\Delta\mathbf T,
\Delta DR
]
=
F_\phi(I)
\]

### Step 2：安全投影

\[
\Theta_B'
=
Proj_\Omega
(
\Theta_B+\Delta\Theta_B
)
\]

### Step 3：使用同一 RF 重建

得到：

\[
I'
\]

### Step 4：重新计算质量

\[
Q(I')
\]

### Step 5：接受 / 回滚

若：

\[
Q(I')>Q(I)+\epsilon
\]

接受。

否则 rollback，并尝试：

\[
\Delta\Theta/2
\]

---

## 29. 前端三参数 Slow Outer Loop

定义：

\[
\Theta_F=
[
D,f,z_F
]
\]

由于改变这些参数需要：

\[
\boxed{
\text{重新采集 RF}
}
\]

因此要强调 sample efficiency。

推荐：

\[
\boxed{
\text{Hierarchical initialization}
+
\text{Constrained Bayesian Optimization}
}
\]

---

## 30. 前端初始化

顺序：

\[
\boxed{
Depth
\rightarrow
Frequency
\rightarrow
Focus
}
\]

### Depth

\[
D_0=z_{max}+M
\]

### Frequency

选择满足：

\[
Q_{landmark}>T_L
\]

\[
Q_{penetration}>T_P
\]

的最高频率。

### Focus

\[
z_F^{(0)}=z_{ROI}
\]

---

## 31. Bayesian Optimization

定义：

\[
J(\Theta_F)
=
Q_{clinical}
[
I(\Theta_F)
]
\]

下一参数：

\[
\Theta_{F,t+1}
=
\arg\max_{\Theta_F}
Acquisition(\Theta_F)
\]

推荐：

- Gaussian Process；
- TPE；
- Expected Improvement；
- UCB。

搜索限制：

\[
D\in D_0\pm\Delta D
\]

\[
f\in\mathcal F_{allowed}
\]

\[
z_F
\in
z_{ROI}\pm\Delta z
\]

目标：

\[
\boxed{
3\sim6
\text{ 次额外 acquisition 完成局部优化}
}
\]

---

## 32. Manifold Prior

从专家优选参数中收集：

\[
[D,f,z_F]
\]

采用：

- PCA；
- Diffusion Map；
- VAE。

得到：

\[
z_{latent}\in\mathbb R^{1\sim2}
\]

让 BO 只在：

\[
\boxed{
\text{good-parameter feasible region}
}
\]

中搜索，以减少组合爆炸。

---

## 33. 六参数最终调节顺序

推荐严格按照：

\[
\boxed{
Depth
\rightarrow
Frequency
\rightarrow
TX\ Focus
\rightarrow
Digital\ Gain
\rightarrow
Digital\ TGC
\rightarrow
Dynamic\ Range
}
\]

原因：

1. Depth 决定 FOV 和后续 depth coordinate；
2. Frequency 依赖所需 penetration；
3. Focus 依赖 ROI depth；
4. Gain 先解决 global brightness；
5. TGC 再解决 depth-dependent residual；
6. DR 最后调整 visual contrast。

---

## 34. 总质量函数

推荐：

\[
\begin{aligned}
Q_{\rm total}
={}&
w_EQ_{\rm expert}
+w_AQ_{\rm anatomy}
+w_LQ_{\rm landmark}
\\
&+w_PQ_{\rm penetration}
+w_FQ_{\rm focus}
\\
&+w_BQ_{\rm brightness}
+w_TQ_{\rm axial}
\\
&+w_CQ_{\rm contrast}
+w_SQ_{\rm speckle}
\\
&-\lambda_AP_{\rm artifact}
-\lambda_UP_{\rm unstable}
\end{aligned}
\]

推荐先使用 heuristic 权重作为 baseline，再通过专家 pairwise preference 学习权重。

---

## 35. 闭环稳定性策略

### 35.1 Confidence Gate

\[
P_{AI}<T_c
\Rightarrow
\text{不自动修改参数}
\]

### 35.2 Deadband

\[
|\Delta Q|<\epsilon
\Rightarrow
\text{保持当前参数}
\]

### 35.3 Maximum Step

每次只允许有限参数变化。

### 35.4 Hysteresis

连续：

\[
N=3\sim5
\]

帧都预测需要修改，才执行动作。

### 35.5 Rollback

若：

\[
Q_{new}<Q_{old}-\epsilon
\]

恢复上一参数状态。

### 35.6 Cooldown

修改 Depth / Frequency / Focus 后等待若干帧稳定，再做下一次评价。

---

## 36. 双时间尺度策略

### Slow Loop

控制：

\[
Depth,Frequency,Focus
\]

触发：

- probe 明显移动；
- ROI depth 变化；
- anatomy 变化；
- image quality 持续下降；
- 用户重新定位。

### Fast Loop

控制：

\[
Digital\ Gain,
Digital\ TGC,
Dynamic\ Range
\]

其中：

- Gain / TGC 可频繁更新；
- DR 仅在 Gain/TGC 稳定后更新。

---

## 37. 数据集构建

### 37.1 前端数据

由专业 sonographer 获得稳定切面后，在专家 preset 周围采集：

\[
D_i,f_j,z_{F,k}
\]

推荐：

- local sweep；
- orthogonal design；
- Latin hypercube。

保存：

\[
RF
+
B-mode
+
Parameter\ Vector
\]

专家标注：

- overall quality；
- Depth direction；
- Frequency direction；
- Focus direction；
- pairwise preference。

### 37.2 后端数据

一份 RF 可以离线生成大量：

\[
G_d,\mathbf T,DR
\]

组合。

例如 Gain：

\[
G_d
\in
\{-12,-9,-6,-3,0,+3,+6,+9,+12\}
\]

Digital TGC：

\[
\mathbf T=[T_1,\ldots,T_6]
\]

Dynamic Range：

\[
DR\in\{DR_1,\ldots,DR_n\}
\]

因此：

\[
\boxed{
1\ RF
\rightarrow
\text{几十到上百个后端图像变体}
}
\]

可自动获得：

\[
\Delta G^{GT},
\Delta\mathbf T^{GT},
\Delta DR^{GT}
\]

---

## 38. 三阶段训练策略

### Stage A：Representation / Anatomy

训练：

- organ/view classification；
- anatomy segmentation；
- landmarks；
- artifact detection。

目标：

\[
\boxed{
\text{先让模型知道“图里是什么”}
}
\]

### Stage B：Quality + Parameter Heads

增加：

- expert IQ；
- Depth；
- Frequency；
- Focus；
- Gain；
- TGC；
- DR。

### Stage C：Closed-loop Fine-tuning

利用同一 patient/cine 中：

\[
I_t,I_{t+1}
\]

的参数变化与专家 preference 做 ranking fine-tuning。

最终在：

- phantom；
- volunteer；
- clinical research setting；

中逐步验证。

---

## 39. 实验验证指标

### 39.1 参数选择准确性

\[
Acc_D
\]

\[
Acc_f
\]

\[
MAE_{focus}
\]

\[
MAE_G
\]

\[
MAE_{TGC}
\]

\[
MAE_{DR}
\]

### 39.2 专家图像质量评价

由 blinded experts 对：

\[
Before/After
\]

进行：

- 1–5 ordinal rating；
- pairwise preference。

建议报告：

- ICC；
- Cohen/Fleiss kappa；
- expert agreement。

### 39.3 工程指标

包括：

- FWHM；
- CNR；
- gCNR；
- SNR；
- ROI contrast；
- background speckle statistics；
- clipping ratio；
- landmark confidence。

### 39.4 闭环控制指标

必须加入：

\[
N_{iterations}
\]

\[
T_{convergence}
\]

\[
\Delta Q
\]

\[
N_{rollback}
\]

\[
Parameter\ oscillation\ rate
\]

\[
Failure-to-safe\ rate
\]

以及：

\[
\boxed{
\text{mean number of acquisitions required for convergence}
}
\]

---

## 40. Ablation Study

建议至少比较：

1. Manual/default preset；
2. Global IQ only；
3. Multi-task IQ；
4. Multi-task IQ + hierarchical controller；
5. Full closed-loop；
6. Full closed-loop + BO；
7. Full closed-loop + backend differentiable refinement。

同时逐个移除：

- artifact gate；
- pairwise ranking；
- TGC smoothness；
- rollback；
- temporal hysteresis；

评估性能下降。

---

## 41. 推荐研发版本

### V1：后端闭环

先完成：

\[
\boxed{
Digital\ Gain
+
Digital\ TGC
+
Dynamic\ Range
}
\]

优势：

- 不改 TX；
- 不重新采集；
- 数据扩展方便；
- 快速验证闭环可行性。

### V2：加入前端参数

加入：

\[
Depth,
Frequency,
TX\ Focus
\]

采用：

\[
Depth
\rightarrow
Frequency
\rightarrow
Focus
\]

层级初始化。

### V3：完整六参数系统

最终采用：

\[
\boxed{
Hierarchical\ initialization
+
Bayesian\ optimization
+
Fast\ backend\ optimization
}
\]

---

## 42. 最终算法伪代码

```text
Input:
    current parameters Θ0
    ultrasound RF/cine
    trained quality network Φ

1. Acquire current cine
2. Evaluate validity:
       q = Φ(I)

3. if invalid view / severe artifact:
       HOLD parameters
       return

========== Slow acquisition loop ==========

4. Estimate anatomy ROI

5. Initialize Depth from ROI extent
6. Reacquire
7. Evaluate and accept / rollback

8. Select highest Frequency satisfying:
       landmark visibility > TL
       penetration > TP

9. Reacquire
10. Evaluate and accept / rollback

11. Initialize TX Focus at ROI depth
12. Perform local Focus search
13. Reacquire
14. Evaluate and accept / rollback

15. Optional:
       constrained Bayesian Optimization
       over [Depth, Frequency, Focus]

========== Fast reconstruction loop ==========

16. Fix RF data

17. Predict:
       ΔGain
       ΔTGC[1:K]
       ΔDR

18. Update Digital Gain
19. Reconstruct
20. Evaluate
21. Accept / rollback

22. Update Digital TGC
23. Enforce:
       smoothness
       slope
       step limit
       zero-mean constraint
24. Reconstruct
25. Evaluate
26. Accept / rollback

27. Update Dynamic Range
28. Reconstruct
29. Evaluate
30. Accept / rollback

========== Final verification ==========

31. Compute Qtotal

32. if Qtotal > Qold + ε:
       accept Θnew
    else:
       rollback Θold

33. Enter monitoring state

34. Restart slow loop only if:
       anatomy changes
       probe moves
       quality drops persistently
```

---

## 43. 数学形式总结

图像生成：

\[
I_t
=
\mathcal U
\left(
RF;
D_t,
f_t,
z_{F,t},
G_{d,t},
\mathbf T_t,
DR_t
\right)
\]

多维质量感知：

\[
\mathbf q_t
=
\Phi(I_t)
\]

总体质量：

\[
Q_t
=
\Psi(\mathbf q_t)
\]

控制策略：

\[
\Delta\Theta_t
=
\pi(
\mathbf q_t,
\Theta_t
)
\]

安全更新：

\[
\boxed{
\Theta_{t+1}
=
Proj_{\Omega}
[
\Theta_t+\Delta\Theta_t
]
}
\]

最终目标：

\[
\boxed{
\Theta^*
=
\arg\max_{\Theta\in\Omega}
\left[
Q_{clinical}(I(\Theta))
-
\lambda_\Delta
\|\Delta\Theta\|^2
\right]
}
\]

其中 \(\Omega\) 包含：

- Depth 合法范围；
- Frequency 合法档位；
- Focus 合法范围；
- Digital Gain 范围；
- TGC slope；
- TGC smoothness；
- DR 范围；
- 最大单次参数变化；
- 参数组合物理可行区域。

---

## 44. 推荐的最终技术路线

\[
\boxed{
\begin{aligned}
&\text{B-mode / cine}
\\
&\downarrow
\\
&\text{Anatomy-aware + cause-aware multi-task IQA}
\\
&\downarrow
\\
&\text{Validity gate}
\\
&\downarrow
\\
&Depth
\rightarrow Frequency
\rightarrow TX\ Focus
\\
&\downarrow
\\
&\text{Reacquire RF}
\\
&\downarrow
\\
&Digital\ Gain
\rightarrow Digital\ TGC
\rightarrow Dynamic\ Range
\\
&\downarrow
\\
&\text{Clinical quality verification}
\\
&\downarrow
\\
&\text{Accept / rollback / local refinement}
\end{aligned}
}
\]

---

## 45. 推荐的方法名称

### 中文

**基于解剖感知与质量诊断的超声六参数分层闭环自动优化框架**

### 英文

**An Anatomy-Aware and Cause-Aware Hierarchical Closed-Loop Framework for Automatic Ultrasound Parameter Optimization**

更简洁的工程命名：

**AI-Driven Hierarchical Closed-Loop Ultrasound Auto-Knobology**

---

## 46. 核心创新点

本方案的核心创新不是“用 CNN 直接预测六个参数”，而是：

\[
\boxed{
\textbf{
Cause-aware
+
Anatomy-aware
+
Physics-constrained
+
Hierarchical
+
Closed-loop
}
}
\]

具体体现在：

1. 将整体 IQ 拆为多个可解释质量分量；
2. 先判断图像无效原因，再决定是否允许自动调参；
3. 将前端和后端参数放在不同时间尺度；
4. 前端采用 sample-efficient 局部搜索 / Bayesian Optimization；
5. 后端基于同一 RF 快速重建和验证；
6. Digital Gain 与 Digital TGC 显式解耦；
7. Dynamic Range 放在最后调节；
8. 所有参数更新加入约束、deadband、hysteresis、rollback；
9. 最终以临床结构可见性、专家偏好和物理合理性共同决定是否接受新参数。

---

## 47. 推荐实施优先级

### 第一阶段

\[
\boxed{
Digital\ Gain
+
Digital\ TGC
+
Dynamic\ Range
}
\]

### 第二阶段

\[
\boxed{
Depth
}
\]

### 第三阶段

\[
\boxed{
Frequency
+
TX\ Focus
}
\]

### 最终阶段

\[
\boxed{
6\text{-parameter hierarchical closed loop}
}
\]

这种分阶段实施方式便于逐步验证：

- 图像质量模型；
- 参数独立贡献；
- 参数耦合；
- 闭环稳定性；
- 实时性；
- 专家一致性；
- 临床可接受性。
