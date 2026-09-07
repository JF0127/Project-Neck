# Algorithm

`algorithm/` 是 Project-Neck 独立的深度学习研究与训练项目：

```text
Canonical Dataset → DataLoader/Preprocessing → Model → Loss → Train/Eval → Model Package
```

当前目录正在从零重建，本次仓库重构只建立骨架，尚未实现 Dataset Loader、模型、训练、
评估或推理。后续实现必须以 Dataset V1 的真实 timestamps、neck valid mask、grouped split
和 GT 语义为准。

Algorithm 的产物是可独立部署的 checkpoint/model package，而不是机器人控制命令。Audio、
ASR、Dialogue、TTS、Motor Socket、IK、EtherCAT 和真机实验都属于 `runtime/`。

目录：

```text
algorithm/
├── configs/   # 后续实验配置
├── data/      # 后续训练数据接口
├── models/    # 后续模型
└── tests/     # 后续纯软件测试
```

不要从 `runtime/motion_model/` 复制旧 V3 结构继续开发新 Algorithm。
