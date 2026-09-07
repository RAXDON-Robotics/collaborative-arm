# RAXDON Collaborative Arm (Y1)

## 目录结构

```
collaborative-arm/
├── sdk/                       # 机械臂底层 SDK 与驱动
│   ├── y1_cpp/                #   Y1 C++ SDK + ROS1/ROS2 driver
│   └── y1_python/             #   Y1 Python SDK + ROS1/ROS2 driver（含 y1_ros）
├── tools/
│   └── data_collection/       # Y1 机械臂数据采集工具
└── research/                  # 学习算法研究与示教复现
    ├── aloha_act/             #   ALOHA ACT 训练/推理（RAXDON 适配）
    └── raxdon_lerobot/        #   LeRobot ACT / Diffusion / VLA 评估与真机部署
```

## 模块说明

- **sdk/y1_cpp**：`liby1_sdk` 动态库 + `y1_controller`（ROS）等 C++ 驱动；提供 CAN 通信、状态读取、示教与轨迹复现接口。
- **sdk/y1_python**：Python 封装 SDK 及 `y1_ros`（ROS1/ROS2 包，含 `y1_description` URDF/模型），附带 `raxdon_y1_can0` 等 CAN 启动脚本与设备规则。
- **tools/data_collection**：机械臂遥操作数据采集（含 conda 环境定义），产出用于 ACT/策略训练的 `v30/*` 数据集。
- **research/aloha_act**：基于 ALOHA/ACT 的动作学习复现。
- **research/raxdon_lerobot**：基于 LeRobot 框架的策略训练与真机部署（ACT / Diffusion / VLA 微调）。**LeRobot 为第三方依赖**（官方仓库固定 commit `0217e1e3`），已不随仓库附带，按该目录 README 自行拉取。

## 依赖关系

```
tools/data_collection ──采集──▶ v30/* 数据集 ──训练──▶ research/raxdon_lerobot（+ research/aloha_act）
                                                              │
sdk/y1_cpp · sdk/y1_python ◀──────真机执行/复现────────────────┘
```

## 快速上手

各模块独立构建/使用，完整步骤见各子目录 README：

- C++ SDK（含 ROS driver）：`sdk/y1_cpp/README.md`
- Python SDK / y1_ros：`sdk/y1_python/README.md`
- 数据采集：`tools/data_collection/README.md`
- 算法训练与真机评估：`research/raxdon_lerobot/README.md`、`research/aloha_act/README.md`


