# YAW：代码与数据附件

这是论文 **“YAW: Predictive Reward Composition via Task-Advantage Alignment
for Wheel-Legged Locomotion”** 的官方实现与实验数据。YAW（You Always Walk）
将 22 个轮腿运动奖励动态组合为 5 个有界且均值为 1 的奖励组权重，并使用固定
任务优势验证虚拟 PPO 更新。

作者：Lexing Yang、Houbao Ji、Shaolong Shen、Hongji Huang、Xiangxiao Chen、
Yufeng Ding。

English: [README.md](./README.md)

本仓库包含：

- 双轮、六动作的 Isaac Lab 任务；
- YAW 及论文中的五种对比方法；
- 训练、模型选择、独立评估和绘图入口；
- CPU 回归测试与机器可读实验配置；
- 论文使用的训练曲线和独立评估结果。

开发机器路径、训练日志和检查点未包含在发布包中。复现与许可证合规所必需的
第三方署名均予以保留。

## 环境

论文实验使用 Ubuntu 22.04、Isaac Sim 5.1、Isaac Lab、Python 3.11、
PyTorch 2.7 和 `rsl-rl-lib==3.1.2`。请在可用的 Isaac Lab 环境中安装：

```bash
python -m pip install -e source/yaw_bot
```

机器人 USD、URDF 和网格文件位于 `assets/robots/yaw_bot`，无需重新转换。
完整硬件和依赖版本见 `configs/environment.json`，实验协议细节见
`PROTOCOL_NOTES.md`。

## 快速验证

以下 CPU 测试不会启动 Isaac Sim：

```bash
python -B -m unittest discover -s tests -p "test_*.py"
python scripts/validate_release.py
```

## 复现实验

只打印训练命令：

```bash
python scripts/reproduce.py train --dry-run
```

训练六种方法、选择检查点、独立评估并生成论文图表：

```bash
python scripts/reproduce.py train
python scripts/reproduce.py select
python scripts/reproduce.py evaluate
python scripts/reproduce.py plot
```

仅运行一个小规模 YAW 冒烟测试：

```bash
python scripts/reproduce.py train --method yaw --num-envs 2 --iterations 2
```

论文公式与实现的逐项对应关系见 `PAPER_CODE_MAP.md`；完整协议见
`configs/paper_experiments.json`；22 个原子奖励定义与尺度见
`configs/reward_terms.json`。

## 许可证

项目代码采用 BSD 3-Clause License。详见 `LICENSE` 和 `NOTICE.md`；源自上游
项目的文件保留原有版权和 SPDX 标记。
