"""话语级机器人颈部运动生成（第一版基线）。

模块：
- dataset.py        数据加载（JSONL + 音频 + RPY 轨迹 + 过滤）
- audio_features.py 音频读取 / 16k mono / 80 维 log-Mel
- text_features.py  词表 / tokenization / 词到 30Hz 帧对齐
- model.py          共享编码器 + Speaker/Listener 双头 Transformer
- losses.py         掩码 Huber 损失（pose / velocity / acceleration）
- metrics.py        验证指标（MAE/RMSE/分角色/首帧误差）
- train.py          训练 / 验证 / dry-run / checkpoint
- infer.py          推理（相对 RPY + 可选机器人旋转矩阵）
"""

__version__ = "0.1.0"
