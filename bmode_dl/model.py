# -*- coding: utf-8 -*-
"""SixParamNet：图像 + 深度剖面 + 当前参数，输出六参数的建议。

    image (B,3,512,128) ─ 二维残差编码器（每级 FiLM 受当前参数调制）─ 横向平均 ─┐
    profile (B,5,512) ── 一维残差编码器 ────────────────────────────────────────┤ 深度序列 (B,32,d)
                                                                               │
    scalars (B,17) ──── MLP 参数嵌入 ── 作为一个 token 前置 ──── Transformer（沿深度注意力）
                                                                               │
                                          全局池化 + 参数嵌入 ── 共享 MLP ── 各轴输出头
    TGC 头额外把深度序列池化成 8 段，每段特征 + 该段当前滑块 + 段编号 -> 该段 delta。

为什么调制要用当前参数：同一幅原始 dB 图可以对应任意当前增益/TGC，"偏暗还是偏亮"只有和当前
设置放在一起才有定义（docs/six_parameter_model_recommendations_20260914.md §6.3）。

输出（全部是 logits 或 dB 数值，解码见 decode_predictions）：
    gain_delta_db (B,)        gain_dir (B,3)           gain_optimal_db (B,)   仅 optimum 输出方式
    tgc_delta_db (B,8)        slider_dir (B,3,3)       tgc_optimal_db (B,8)   near/mid/far 各 3 类
    depth_logits (B,nd)       depth_dir (B,3)
    frequency_logits (B,nf)   frequency_dir (B,3)
    focus_logits (B,nz)       focus_dir (B,3)
    dr_delta_ui (B,)          dr_dir (B,3)             目前无监督
    aux (B,2)                 衰减、电子噪声（标准化后）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import constants as K
from .dataset import IMAGE_CHANNELS, NUM_SCALARS, PROFILE_CHANNELS, INPUT_MODES

# 后端输出方式：
#   optimum  网络预测最优增益 / 最优 TGC 曲线（dB，按训练集均值方差标准化），修正量 = 最优 - 当前，
#            减法在网络外精确完成。fieldii_v1 的日志显示网络自己做减法残留约 0.3 dB，输给只看设置的查表。
#   delta    网络直接预测修正量（fieldii_v1 及更早的检查点）。
BACKEND_OUTPUTS = ("optimum", "delta")


def _groups(channels):
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class FiLM(nn.Module):
    """x * (1 + gamma) + beta，gamma/beta 由参数嵌入给出，初始化为 0（恒等）。"""

    def __init__(self, embed_dim, channels):
        super().__init__()
        self.proj = nn.Linear(embed_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, embed):
        gamma, beta = self.proj(embed).chunk(2, dim=1)
        shape = (x.shape[0], x.shape[1]) + (1,) * (x.dim() - 2)
        return x * (1.0 + gamma.view(shape)) + beta.view(shape)


class ResBlock2d(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1), dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(cout), cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(cout), cout)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (nn.Identity() if cin == cout and tuple(stride) == (1, 1)
                     else nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                                        nn.GroupNorm(_groups(cout), cout)))

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)), inplace=True)
        y = self.drop(y)
        y = self.norm2(self.conv2(y))
        return F.relu(y + self.skip(x), inplace=True)


class ResBlock1d(nn.Module):
    def __init__(self, cin, cout, stride=1, kernel=5, dropout=0.0):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(cin, cout, kernel, stride=stride, padding=pad, bias=False)
        self.norm1 = nn.GroupNorm(_groups(cout), cout)
        self.conv2 = nn.Conv1d(cout, cout, kernel, padding=pad, bias=False)
        self.norm2 = nn.GroupNorm(_groups(cout), cout)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip = (nn.Identity() if cin == cout and stride == 1
                     else nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                        nn.GroupNorm(_groups(cout), cout)))

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)), inplace=True)
        y = self.drop(y)
        y = self.norm2(self.conv2(y))
        return F.relu(y + self.skip(x), inplace=True)


class ImageEncoder(nn.Module):
    """(B,3,512,128) -> (B,C,32,16)。深度方向先降，横向后降。"""

    def __init__(self, embed_dim, widths=(32, 64, 128, 256), dropout=0.05):
        super().__init__()
        w0, w1, w2, w3 = widths
        self.stem = nn.Sequential(nn.Conv2d(IMAGE_CHANNELS, w0, (7, 3), stride=(2, 1), padding=(3, 1), bias=False),
                                  nn.GroupNorm(_groups(w0), w0), nn.ReLU(inplace=True))
        self.stages = nn.ModuleList([
            nn.Sequential(ResBlock2d(w0, w0, (2, 1), dropout)),                            # 128 x 128
            nn.Sequential(ResBlock2d(w0, w1, (2, 2), dropout)),                            # 64 x 64
            nn.Sequential(ResBlock2d(w1, w2, (1, 2), dropout), ResBlock2d(w2, w2, (1, 1), dropout)),  # 64 x 32
            nn.Sequential(ResBlock2d(w2, w3, (2, 2), dropout), ResBlock2d(w3, w3, (1, 1), dropout)),  # 32 x 16
        ])
        self.films = nn.ModuleList([FiLM(embed_dim, c) for c in (w0, w1, w2, w3)])
        self.out_channels = w3

    def forward(self, image, embed):
        x = self.stem(image)
        for stage, film in zip(self.stages, self.films):
            x = film(stage(x), embed)
        return x


class ProfileEncoder(nn.Module):
    """(B,5,512) -> (B,C,32)。"""

    def __init__(self, embed_dim, width=128, dropout=0.05):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(PROFILE_CHANNELS, 64, 7, padding=3, bias=False),
                                  nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([ResBlock1d(64, 64, 2, dropout=dropout),     # 64 (after pool 128)
                                     ResBlock1d(64, width, 2, dropout=dropout)])  # 32
        self.films = nn.ModuleList([FiLM(embed_dim, 64), FiLM(embed_dim, width)])
        self.out_channels = width

    def forward(self, profile, embed, length):
        x = F.adaptive_avg_pool1d(profile, length * 4)
        x = self.stem(x)
        for block, film in zip(self.blocks, self.films):
            x = film(block(x), embed)
        return x


class MLP(nn.Module):
    def __init__(self, cin, hidden, cout, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cin, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, cout))

    def forward(self, x):
        return self.net(x)


class SixParamNet(nn.Module):
    def __init__(self, ladders, input_mode="full", d_model=256, seq_len=32, transformer_layers=2,
                 use_transformer=True, dropout=0.1, backend_output="optimum", backend_norm=None,
                 frontend_dropout=None):
        super().__init__()
        if input_mode not in INPUT_MODES:
            raise ValueError("input_mode must be one of %s" % (INPUT_MODES,))
        if backend_output not in BACKEND_OUTPUTS:
            raise ValueError("backend_output must be one of %s" % (BACKEND_OUTPUTS,))
        self.input_mode = input_mode
        self.backend_output = backend_output
        self.seq_len = int(seq_len)
        self.num_depth = len(ladders["depth_mm"])
        self.num_frequency = len(ladders["frequency_mhz"])
        self.num_focus = len(ladders["focus_mm"])
        embed_dim = d_model

        self.param_encoder = nn.Sequential(nn.Linear(NUM_SCALARS, embed_dim), nn.GELU(),
                                           nn.Linear(embed_dim, embed_dim), nn.GELU())
        seq_channels = 0
        if input_mode == "full":
            self.image_encoder = ImageEncoder(embed_dim)
            seq_channels += self.image_encoder.out_channels
        if input_mode in ("full", "no_image"):
            self.profile_encoder = ProfileEncoder(embed_dim)
            seq_channels += self.profile_encoder.out_channels
        self.has_sequence = seq_channels > 0

        if self.has_sequence:
            self.seq_proj = nn.Linear(seq_channels, d_model)
            self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.use_transformer = bool(use_transformer)
            if self.use_transformer:
                layer = nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=2 * d_model,
                                                   dropout=dropout, activation="gelu", batch_first=True,
                                                   norm_first=True)
                self.sequence_model = nn.TransformerEncoder(layer, transformer_layers, enable_nested_tensor=False)
            else:
                self.sequence_model = nn.Sequential(ResBlock1d(d_model, d_model, 1, 3, dropout),
                                                    ResBlock1d(d_model, d_model, 1, 3, dropout))
            self.seq_norm = nn.LayerNorm(d_model)

        trunk_in = embed_dim + (d_model if self.has_sequence else 0)
        self.trunk = nn.Sequential(nn.Linear(trunk_in, d_model), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(d_model, d_model), nn.GELU())

        hidden = d_model // 2
        front_dropout = dropout if frontend_dropout is None else float(frontend_dropout)
        self.gain_head = MLP(d_model, hidden, 1 + 3, dropout)
        self.tgc_band_head = MLP(d_model + (d_model if self.has_sequence else 0) + 1 + K.NUM_TGC_BANDS,
                                 hidden, 1, dropout)
        self.slider_dir_head = MLP(d_model, hidden, 3 * len(K.SLIDER_GROUPS), dropout)
        self.depth_head = MLP(d_model, hidden, self.num_depth + 3, front_dropout)
        self.frequency_head = MLP(d_model, hidden, self.num_frequency + 3, front_dropout)
        self.focus_head = MLP(d_model, hidden, self.num_focus + 3, front_dropout)
        self.dr_head = MLP(d_model, hidden, 1 + 3, dropout)
        self.aux_head = MLP(d_model, hidden, 2, dropout)
        self.register_buffer("band_onehot", torch.eye(K.NUM_TGC_BANDS), persistent=False)

        if backend_output == "optimum":
            # 最优值 = 训练集均值 + 标准差 * 网络输出；末层置零，起点就是"按训练集均值给建议"
            norm = backend_norm or {}
            self.register_buffer("gain_opt_mean", torch.tensor(float(norm.get("opt_gain_mean", 0.0))))
            self.register_buffer("gain_opt_std", torch.tensor(float(norm.get("opt_gain_std", 1.0))))
            self.register_buffer("tgc_opt_mean", torch.tensor(
                norm.get("opt_tgc_db_mean", [0.0] * K.NUM_TGC_BANDS), dtype=torch.float32))
            self.register_buffer("tgc_opt_std", torch.tensor(
                norm.get("opt_tgc_db_std", [1.0] * K.NUM_TGC_BANDS), dtype=torch.float32))
            for head in (self.gain_head, self.tgc_band_head):
                nn.init.zeros_(head.net[-1].weight)
                nn.init.zeros_(head.net[-1].bias)

    def forward(self, inputs):
        scalars = inputs["scalars"].float()
        embed = self.param_encoder(scalars)
        b = scalars.shape[0]

        band_features = None
        if self.has_sequence:
            parts = []
            if self.input_mode == "full":
                x = self.image_encoder(inputs["image"], embed)              # (B,C,32,16)
                x = x.mean(dim=3)                                           # (B,C,32)
                parts.append(F.adaptive_avg_pool1d(x, self.seq_len))
            parts.append(self.profile_encoder(inputs["profile"], embed, self.seq_len))
            seq = torch.cat(parts, dim=1).transpose(1, 2)                   # (B,32,C)
            seq = self.seq_proj(seq)
            tokens = torch.cat([embed[:, None, :], seq], dim=1) + self.pos_embed
            if self.use_transformer:
                tokens = self.sequence_model(tokens)
            else:
                tokens = self.sequence_model(tokens.transpose(1, 2)).transpose(1, 2)
            tokens = self.seq_norm(tokens)
            depth_seq = tokens[:, 1:, :]                                    # (B,32,d)
            band_features = F.adaptive_avg_pool1d(depth_seq.transpose(1, 2), K.NUM_TGC_BANDS).transpose(1, 2)
            # 参数 token（已与深度序列交互）+ 深度序列平均
            global_in = torch.cat([tokens[:, 0, :], depth_seq.mean(dim=1)], dim=1)
        else:
            global_in = embed
        h = self.trunk(global_in)

        out = {}
        gain = self.gain_head(h)
        out["gain_dir"] = gain[:, 1:]
        if self.backend_output == "optimum":
            # 当前增益由输入标量还原（gain_db / 10），在网络外做减法
            out["gain_optimal_db"] = self.gain_opt_mean + self.gain_opt_std * gain[:, 0].float()
            out["gain_delta_db"] = out["gain_optimal_db"] - scalars[:, 3] * 10.0
        else:
            out["gain_delta_db"] = gain[:, 0]

        current_tgc = scalars[:, 4:4 + K.NUM_TGC_BANDS]                     # (level-127)/127
        band_in = [h[:, None, :].expand(b, K.NUM_TGC_BANDS, h.shape[1]), current_tgc[:, :, None],
                   self.band_onehot[None, :, :].expand(b, K.NUM_TGC_BANDS, K.NUM_TGC_BANDS)]
        if band_features is not None:
            band_in.insert(1, band_features)
        tgc_raw = self.tgc_band_head(torch.cat(band_in, dim=2)).squeeze(2)
        if self.backend_output == "optimum":
            mode = scalars[:, 14]
            slope = torch.where(mode > 0.5, torch.full_like(mode, K.TGC_DB_PER_LEVEL[K.MODE_HARMONIC]),
                                torch.full_like(mode, K.TGC_DB_PER_LEVEL[K.MODE_FUNDAMENTAL]))
            current_db = current_tgc * float(K.TGC_CENTER_LEVEL) * slope[:, None]   # (level-127)*dB/级
            out["tgc_optimal_db"] = self.tgc_opt_mean + self.tgc_opt_std * tgc_raw.float()
            out["tgc_delta_db"] = out["tgc_optimal_db"] - current_db
        else:
            out["tgc_delta_db"] = tgc_raw
        out["slider_dir"] = self.slider_dir_head(h).view(b, len(K.SLIDER_GROUPS), 3)

        depth = self.depth_head(h)
        out["depth_logits"], out["depth_dir"] = depth[:, :self.num_depth], depth[:, self.num_depth:]
        freq = self.frequency_head(h)
        out["frequency_logits"], out["frequency_dir"] = freq[:, :self.num_frequency], freq[:, self.num_frequency:]
        focus = self.focus_head(h)
        out["focus_logits"], out["focus_dir"] = focus[:, :self.num_focus], focus[:, self.num_focus:]
        dr = self.dr_head(h)
        out["dr_delta_ui"], out["dr_dir"] = dr[:, 0], dr[:, 1:]
        out["aux"] = self.aux_head(h)
        return out


def masked_logits(logits, valid):
    """不可选的档位置为很小的数（float32 下计算）。"""
    return logits.float().masked_fill(valid <= 0, -1e4)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
