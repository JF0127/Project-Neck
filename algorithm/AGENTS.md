# Algorithm 开发规则

`algorithm/` 是纯深度学习训练与研究项目，当前正在从零重建。

- 数据入口只能是 `dataset/` 发布的 canonical manifests/artifacts。
- 产物是 checkpoint、完整配置、词表/预处理 metadata 和评估结果。
- 不包含 Audio 设备、WebSocket server、ASR/TTS 服务、Dialogue、Motor、IK 或 EtherCAT。
- 不 import `runtime/`；Runtime 未来通过独立 model package 加载训练产物。
- 不以 `runtime/motion_model/` 中冻结的旧 V3 inference 代码作为新架构基础。
- 不把旧 CVAE、MultiCandidate、speaker/listener dual-head 或 previous context 写成当前设计。
- Dataset V1 原始 artifact 不可修改；使用真实 neck timestamps，保留 invalid mask，不假设 30 fps。
- 只使用 `split_v1`，不得随机 fragment split；所有拟合统计仅从 Train 估计。
- 新实现应从数据语义和时间对齐测试开始，再实现 baseline；避免 registry、插件系统和过度抽象。

当前阶段禁止新增训练、模型或 Dataset Loader 实现；本目录只保留干净骨架。
