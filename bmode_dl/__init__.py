# -*- coding: utf-8 -*-
"""六参数（深度、频率、聚焦、增益、TGC、动态范围）深度学习模型。

只依赖 numpy 与 torch，服务器上训练不需要 HDF5、h5py 或 bmode_opt。输入数据是
tools_build_fieldii_training_cache.py 写出的缓存，标签是 data/labels_fieldii.jsonl。

    constants    从 bmode_opt 复制的实机/仿真常数（tests/verify_six_param_model.py 断言一致）
    render       GPU 上的后端渲染：dB + TGC 曲线 + 增益 -> 显示灰阶
    labels       jsonl 行编码成张量；后端起点重抽与标签重算
    dataset      缓存与标签载入、按体模分折、组批与输入特征
    model        SixParamNet
    losses       掩膜多任务损失
    metrics      评估指标
    closed_loop  在缓存的设置网格上做闭环优化仿真
"""
