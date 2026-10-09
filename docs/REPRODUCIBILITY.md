# 复现状态与外部依赖

## 已验证

1. 包内 Python 源码全部通过 AST 语法解析（不等于运行验证）。
2. 最终结果图可从包内 4 个 CSV / JSON 输入生成。
3. 原电路后处理可从保留的 meter 记录重新得到 732 / 732 nominal 事件通过，以及约 57.10 / 61.31 fJ 的最大 detector subtotal。
4. 能耗投影采用论文的 1.929 pJ 基准及 0.63 系数，生成两档 gross / net 和剩余能量；该系数为参考设计标定的一阶估计。

## GPU 训练：尚未做全新环境重跑

需要 Linux NVIDIA GPU、PyTorch、NumPy、CUDA toolkit / nvcc 与 C++ 编译器。WRN 历史 manifest 记录 PyTorch 2.8.0+cu128、CUDA 12.8；不能将当前 Mac 的绘图依赖当作 GPU 训练环境。

旧脚本使用 `/root/autodl-tmp/adc_wrn28_10_v1`、`adc_wrn28_10_extend_v1`、`adc_deit_cloud_v1`、`adc_wrn_final_nogate_v1` 等绝对路径，以及相邻 `bf16_fp_cim_reproduction/data/cifar-100-python/`。需要按原布局恢复依赖或另做路径参数化；本包未自动写入 `/root`，未启动训练。

需准备：
- CIFAR-100 原数据及已归档的数据划分；不要另抽测试集后称为相同结果。
- 各模型 clean / W4A4 起始权重和校准数据。
- ResNet 续训分支需原始前 5 轮的 last / best 权重、AdamW 状态和 history。
- WRN BN-only 对照需与训练分支匹配的校准状态。
- `docs/WEIGHTS_LOCAL_INVENTORY.json` 是已找到的本地权重位置和大小清单，不是所有依赖齐全的保证。权重未收入代码 ZIP。

原脚本部分输出目录使用 `exist_ok=False` / exclusive marker，且会进行昂贵训练。不要把运行 `run.py` 当作简单的安装检查。源码中的 parent manifest 哈希检查应保留。

## 电路重新仿真

需合法可用的 Spectre / Verilog-A 及 55 nm PDK。网表保留当时的 PDK include 和实验目录绝对路径；先改成自己的路径。包内不含 foundry models、PDK 库、OA 数据库或许可证。`prepare_netlists.py` 可在独立目录生成仅替换 include 路径的副本。

`spectre.out` 为原始日志的 meter 行摘录，包含原日志已有的成功标记，用于后处理复核。它不含完整诊断或 PSF 波形，不能代替重新仿真。观测数据是有限采样事件，不应解释为全范围误差保证。

## 架构分析重新运行

获取 `docs/EXPERIMENT_MAP.md` 指定的 CiMLoop commit，按上游文档安装 Timeloop / Accelergy 及其环境。包内只保留本项目适配脚本、配置、记录结果和上游许可证，没有复制整份第三方仓库。

数值模拟还需要历史 operand trace / baseline checkpoint / 数据等大文件。`run_isaac_cost.py` 所需 `isaac_adc_pre_trace.npz` 未打包。完整架构重跑尚未在本包中验证。

## 发布前剩余项

- 整理并单独提供必要权重/trace/逐样本预测的下载包和校验值。
- 统一训练及仿真路径，完成新 GPU 环境的 smoke check。
- 补齐 ResNet20 sparsity 专用对照的代码与结果来源。
- 明确作者代码 LICENSE 并核对第三方来源；不要因文件位于本包就认为全部可任意再授权。

本包不包含 SSH 密码、GitLab 登录配置、传输脚本或 PDK 文件。自动扫描未发现常见凭据模式，但不将自动扫描当作公开发布审核的替代品。
