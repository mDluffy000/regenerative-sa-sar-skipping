# Regenerative-SA Similarity Sensing and SAR Skipping

Regenerative-SA Similarity Sensing with Hardware-Aware Training for Aggressive SAR Skipping in Analog Compute-in-Memory

本仓库保存论文的实验实现、结果记录与复现材料，不包含论文 LaTeX 工程。当前为私有研究归档，完整训练环境的可移植化尚在整理。

## 从这里开始

| 内容 | 入口 | 当前状态 |
|---|---|---|
| 最终四联结果图 | `plots/plot_results.py`，`results/final/` | 已用包内数据生成 PDF / PNG |
| 能耗估算 | `scripts/energy_budget.py` | 可用 Python 标准库重算 |
| SA 电路与 testbench | `circuits/nominal_detector/` | Spectre / Verilog-A 原始源码；重仿真需 PDK |
| 电路结果后处理 | `circuits/nominal_detector/final_analyze.py` | 已从提取的原始 meter 记录重算 732 个事件 |
| WRN / ResNet20 / DeiT 训练 | `snapshot/`，见 `docs/EXPERIMENT_MAP.md` | 原始源码及协议归档；尚未统一服务器路径 |
| ISAAC / CiMLoop 架构分析 | `snapshot/timeloop-accelergy-exercises/workspace/adc_architecture_baselines_v1/` | 自定义适配代码；上游地址/commit 见说明 |

## 立即可以运行

在本目录下使用已有 Python 环境（结果图和电路后处理需要 matplotlib）：

```sh
python plots/plot_results.py
python scripts/energy_budget.py
python circuits/nominal_detector/final_analyze.py
python scripts/verify_package.py
```

结果图与能耗表输出到 `generated/`。电路后处理按原脚本在其目录输出 CSV / PNG。
`requirements-plot.txt` 为绘图依赖；归档中的 `paper_plots/requirements-lock.txt` 是原绘图环境记录。

## 复现范围

这是一份经过筛选的代码交接包，**不是已经在全新服务器上验证的一键训练发行版**。训练实现、CUDA 内核、扰动表、数据划分和最终结果协议已整理；大体积模型权重、CIFAR-100 数据、第三方 PDK、Spectre 及其许可证不包含在包内。训练脚本中的旧绝对路径被保留，避免改动历史实验。

查看 `docs/REPRODUCIBILITY.md` 了解需配置的路径、依赖和尚未闭合的复现项。所有论文曲线应以 `results/final/` 为准，不能把早期诊断结果替换进去。

## 文件来源与使用权

`MANIFEST.json` 记录来源路径、原始 SHA-256 和包内 SHA-256。`SHA256SUMS.json` 校验本包交付文件。源目录未被修改。

仅对最终绘图入口做路径适配；Spectre 日志只保留原始测量行及原日志中的成功标记，不是新运行日志。部分源码保留历史本地路径，作为来源信息，不应据此访问原服务器。

未替作者和第三方添加新的开源授权。正式公开发布前需确定作者代码的许可证，并核对上游模型的授权与引用。CiMLoop 的上游 LICENSE 已保留。
