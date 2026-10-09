# 论文实验与代码对应

下列路径均相对于本包，历史原文件位于 `snapshot/`。

## WRN-28-10

- 基础模型 / 量化 / 初始训练：`snapshot/timeloop-accelergy-exercises/workspace/adc_wrn28_10_v1/`；`adc_deit_cloud_v1/model.py` 提供共用 QuantOp，`data.py` 提供数据读取。
- 最终电路误差前向实现：`snapshot/cloud_migration/remote_results/adc_wrn_final_nogate_v1/` 的 `hardware.py`、`events_cisa.cu`、`oracle.py`、`held_20.bin`、`held_40.bin`。
- 20 mV / S4：该目录 `run.py`，最终 `results.json` 的 `aware_20_skip4`。
- 20 mV / S5、S6；40 mV / S4、S5：`snapshot/cloud_migration/wrn_bn_repair_v1/repair.py`；结果在 `remote_results/adc_wrn_bn_repair_v1/`，使用 `combined_*_best`。
- 40 mV / S6：`snapshot/cloud_migration/wrn40_skip6_paired.py`；结果在 `remote_results/adc_wrn40_skip6_paired_v1/`，使用 `combined_40_skip6_best`。不要替换为更早的 S6 结果。
- 40 mV 未误差训练对照使用各分支 `combined_40_skip{k}_BN`，与对应 trained descendant 配对；不是未经 BN 校准的直接测试。
- 稀疏性分解实验：`remote_results/adc_wrn_sparsity_cause_v1/`，含 ordinary / prefix_only / voltage_only / combined。该诊断的零门控口径与主结果图不同，不能混作主图节省量。

## ResNet20 / DeiT

- 共用实现：`snapshot/timeloop-accelergy-exercises/workspace/adc_crossmodel_cisa_nogate_v1/`。
- 核心文件：`run.py`、`models.py`、`resnet_quant.py`、`deit_model.py`、`deit_data.py`、`hardware.py`、`events_cisa.cu`、`oracle.py`。`input/` 包含模型定义、数据划分和量化校准元数据。
- DeiT 最终结果：`snapshot/cloud_migration/remote_results/adc_crossmodel_complete_20260913/deit/results.json`，`aware_20/40_skip4/5/6`；20 mV 对照为 `direct_20_skip{k}`。同级 continuation 源码保留。
- ResNet20 S5：`snapshot/cloud_migration/crossmodel_cisa_nogate_v1_status/diagnostics/skip5_extend20.py` + `skip5_extend_loop.py`；结果元数据在 `results/resnet_skip5_delivery/`。
- ResNet20 其他四点：同目录 `remaining_matrix20.py`；结果在 `snapshot/cloud_migration/remote_results/remaining_matrix20_v1/`。
- ResNet20 20 mV 的 BN-only 对照分别来源于 `adc_crossmodel_complete_20260913/bn_refresh_finetune`、`sa20_skip5_bn_v1`、`sa20_skip6_bn_v1`。
- ResNet20 既有点采用前 5 轮加后续 15 轮，不是重新从头跑 20 轮 cosine；新增 40 mV / S4 为 fresh 20。保留协议说明，勿声称所有点的训练日程完全一致。
- ResNet20“只学 encoding error 对稀疏性的影响”专用驱动及完整配对结果本次未定位，不能用 WRN 的稀疏性脚本冒充已闭合的 ResNet 复现。此项列为待补。

## 测试与适用范围

这些训练结果是历史 CIFAR-100 1000 张测试子集上的结果；主硬件微调通常使用 2000 张训练和 500 张验证样本，精确设置以对应 protocol 为准。硬件误差注入到三个选定映射层，不能写成所有层均进行了完整模拟。主结果禁用零门控。teacher 为 clean W4A4；student 执行硬件误差前向，训练损失/STE 以源码为准。

## 电路、架构和图表

- 电路：`circuits/nominal_detector/`，双阈值 SA 核心、决策/存储、偏置前端、held / ideal / detector-removed testbench、Verilog-A meters。
- 电路结果：`final_events.csv`、`final_summary.csv`、`held_disturbance_vs_removed.csv`；nominal TT schematic，不是 PEX / silicon / 全 PVT。
- ISAAC 适配：`snapshot/timeloop-accelergy-exercises/workspace/adc_architecture_baselines_v1/`。上游 CiMLoop commit `481c044e96034e125c1a442e92c73b5d05c1f506`，https://github.com/mit-emze/cimloop 。数值 ADC 行为与公共成本模型的职责不同；完整假设见原 `provenance.json` / `isaac_acceptance.json`。
- 曲线提取历史：`snapshot/paper_plots/plot_wrn.py`、`plot_wrn_pair.py`、`plot_crossmodel.py`。这些历史脚本需要未打包的逐样本 npz，不是本包的直接运行入口。
- 最终绘图：`plots/plot_results.py`，只改路径，不改数据、曲线或算法。
