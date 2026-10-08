# Causal-SeisField：截面前输入的最小补充训练实验

## 做什么

在新 `input_acceleration` 上重新训练 Cross-Attention Base，再冻结该新 Base、训练 `A4_under_only`（论文中的 CA-URC）。新流程内部比较 `Prefix_CAURC - Prefix_Base`。旧结果仅作同一测试队列上的单独参照。

本包不会重新进行原始波形预处理，不会覆盖旧 HDF5，不会训练 mean-pooling Base / Graph / IDW / PLUM，不会声称已经验证实时预警。归档到时、原标签及数据筛选的回溯性边界仍保留。

## 文件

- `train_snapshot_experiment.py`：统一运行入口，阶段 audit / base / head / evaluate / all。
- `snapshot_experiment_data.py`：严格数据适配器、事件预检与固定台站抽样。
- `snapshot_input_reader.py`：此前提供的原前缀读取模块，逐字保留。
- `source45_baselines.py`：本轮上传的45号脚本，逐字保留，仅作为库导入。
- `source63_risk.py`：上传的63号脚本，逐字保留，仅作为库导入。
- `test_snapshot_training.py`：合成数据测试，不是论文数据。
- `checks.json`：来源SHA256及本地检查结果。

**不要单独运行 `source45_baselines.py` 或 `source63_risk.py`。它们的原 main 仍是旧实验逻辑。使用统一入口。也不要用本包覆盖原项目45/63脚本。**

## 放置与运行目录

将整个 `Causal_SeisField_prefix_training` 文件夹放在项目根目录下，例如：

```text
F:\数据集\世界模型\causal_seisfield_data_scripts\causal_seisfield_data_scripts\
    Causal_SeisField_prefix_training\
        train_snapshot_experiment.py
        snapshot_experiment_data.py
        ...
    data\scedc\processed_full_v4\events\
    data\scedc\snapshot_available_t0_5s_k5\events\
    runs\cadrg_gate_ablation_A3_A5\locked_test_ablation_predictions.csv
```

在项目根目录的 **Windows CMD** 中执行（不是在代码包子目录中执行）：

```bat
D:\python3\python.exe Causal_SeisField_prefix_training\train_snapshot_experiment.py --stage audit
```

默认事件清单是 `data/scedc/model_manifests/scenario_t0_5s_k5.csv`。
若实际使用另一份清单，例如 `model_manifests/scenario_t0_5s_k5_linux.csv`，显式传入：

```bat
D:\python3\python.exe Causal_SeisField_prefix_training\train_snapshot_experiment.py ^
  --stage audit ^
  --manifest model_manifests/scenario_t0_5s_k5_linux.csv
```

所有路径相对于当前工作目录；模型源码相对于代码包目录定位，不需要改名已有文件。

## 第一步：只做预检

默认就是 audit。即使不写 `--stage`，也不会开始训练。

预检：
1. 核对主清单及 train / validation / test 原归属。
2. 检查每个事件前缀文件的 ready / write_complete / all_eligible_inputs_valid。
3. 对原候选池中**全部合资格输入台站**读取新输入，检查有限值、台站顺序、到时、坐标和时间索引。
4. 验证旧 HDF5 的SHA256与预处理时的源文件相同；记录前缀文件内容指纹。
5. 核对侧文件中已有的未来扰动审计记录，没有重新执行原始波形扰动。报告覆盖范围，不宣称审计了全部台站或模型预测。
6. 复现固定测试抽样，并按 event_id / repeat / target_slot / target_station_index 与旧锁定CSV逐键核对；核对真实标签及已有尾部标记，不用旧预测表现调参。

输出在 `runs/snapshot_available_validation/audit/`：

```text
event_input_checks.csv
incomplete_events.csv
failed_station_details.csv
exclusions_REVIEW_REQUIRED.csv
cohort_counts_by_split.csv
cohort_manifest.csv
cohort_audit.json
checked_test_keys_and_truth.csv       # 仅队列通过后产生
checked_test_station_draws.csv        # 仅队列通过后产生
locked_draw_audit.json                # 仅队列通过后产生
```

当前用户报告17个事件未ready。本程序不会直接将1603个事件认定为最终实验队列，也不会因为某事件尚有5个有效站就重抽输入。默认发现未完成事件就停止，训练尚未启动。

先查看 `failed_station_details.csv` 和此前预处理目录的 `station_prefix_qc.csv`、`event_prefix_qc.csv`。修复后，用原预处理脚本在相同设置下重新生成对应侧文件，再重跑audit。

### 明确审定共同子集时

只有确认原因并决定采用共同子集后，才新建一个CSV：

```csv
event_id,reason
<实际事件编号>,<核实后的排除原因>
```

再加 `--excluded-events-csv reviewed_exclusions.csv`。这是显式排除整个事件，不是删除或替换其中的台站。模板 `exclusions_REVIEW_REQUIRED.csv` 的 reason 留空，未经填写不能直接运行。不可按模型预测好坏选择排除事件。

`--expected-events 1620` 核对的是排除之前的原清单数量，不要因为显式排除了17个事件便改成1603。各集合最终数量由实际清单统计，不预设按比例减少。

所有阶段必须使用相同的清单、排除文件和训练参数。旧结果会在相同保留测试事件与目标上重算；旧模型不因此变成在缩小训练集上重新训练的模型，故新旧绝对差异不完全隔离预处理的因果影响。

## 第二步：运行最小训练实验

仅在预检通过、队列已确认后运行：

```bat
D:\python3\python.exe Causal_SeisField_prefix_training\train_snapshot_experiment.py --stage all
```

如使用其他清单或已审定排除文件，在每一条命令中保留相同参数。

也可以分开执行：

```bat
D:\python3\python.exe Causal_SeisField_prefix_training\train_snapshot_experiment.py --stage base
D:\python3\python.exe Causal_SeisField_prefix_training\train_snapshot_experiment.py --stage head
D:\python3\python.exe Causal_SeisField_prefix_training\train_snapshot_experiment.py --stage evaluate
```

每个阶段重新做只读预检。`all`完成一套实验；已完成的Base和head只有协议ID相同才会复用。未完成或不兼容的权重不静默覆盖/恢复，用新输出目录保留旧记录。

### 默认训练设置

取自这次提供的45/63脚本默认值，不代表已经核验历史最终checkpoint的全部训练设置：

- Cross-Attention，hidden_dim=128，heads=4；不是45号中名为 `base` 的 mean-pooling 模型。
- Base最多50轮，Smooth-L1，AdamW，学习率0.001、权重衰减0.0001，验证3次组合；原来的学习率调度、early stopping和检查点选择函数不改。
- 校正头40轮，risk_hidden=64，最大修正1.5 log10；原风险/回归/方向/不必要修正损失及权重不改。
- 校正头3次组合监控、20次验证组合联合选择epoch与gamma，原验证约束0.005 / 0.003 / 0.05不改。
- 新Base残差生成新的严重低估标签，不能复制旧标签；新的epoch/gamma由验证集确定，不强制3和5。
- 模型初始化/数据种子默认20260713。头初始化默认seed+2000，匹配原完整 A3,A4,A5 列表中A4的位置；若历史运行只选择A4等情况，需核对原protocol并显式传 `--head-seed`。
- `num_workers=0` 固定，防止persistent workers拿不到更新后的epoch和patched seed协议。不要在本最小版本中改成多进程后声称抽样仍相同。
- 只能读取前缀加速度作为输入，不输出旧观测PGV字段，不运行需要输入速度的传播基线；未来旧速度只用于PGV标签。

### 至关重要的种子差异

上传45号脚本单独运行的stable_seed是 `SHA256(seed:text)`。
63号后续实验将它改成 `SHA256(text:seed)`，且：
- train / validation 去掉 `strong:` 前缀；
- test 用 `locked:test:...`。

新的Dataset使用63号的完成版协议，并以旧锁定CSV核对测试目标。不靠启动一次临时猴子补丁来维持Dataset行为。
如果旧锁定CSV不匹配，程序停止，不会关闭检查、换种子重试至“碰巧通过”，也不会修改原预测。

### 验证未找到可行校正模型时

程序写出验证扫描并停止，不按测试表现放宽约束。
若选择的gamma在搜索上边界，也保留原63号的保护：停止测试性能评价。应先分析验证扫描，而不是看测试结果后修改搜索范围。

## 分组bootstrap

原45/63脚本中没有指定构造时空组的列名；本包不猜测哪个字段才是原分组。
`--group-column` 需填写实际分组字段名，而不是把 `split_grouped` 这个Train/Val/Test标签当成组ID。

例如，**只有确认真实时空组列确实叫 group_id 时**：

```bat
D:\python3\python.exe Causal_SeisField_prefix_training\train_snapshot_experiment.py ^
  --stage all ^
  --group-column group_id
```

不传该参数可以训练并计算配对点估计，但CI留空、明确注明未计算，绝不冒充按组bootstrap。字段名在audit输出中列出。原清单没有组ID时，需要按event_id关联实际分组表，不能自行从震级、日期或台站信息猜组。

相同权重下补充分组统计可使用 `--stage evaluate --group-column 实际列名 --analysis-subdir evaluation_grouped`，沿用完全相同的训练配置和队列。此时会重新读数据和运行固定模型，不再训练或选模型。

有组ID时用10,000次配对 group -> event重采样。空尾部事件仍为缺失、不是0；重采样总体是全部保留测试事件，再对每个指标略去缺失子集。事件均衡点估计与原方法一致。

## 输出

```text
runs/snapshot_available_validation/
    audit/...
    experiment_protocol.json
    base/cross_attention/best_model.pt
    base/cross_attention/history.csv
    base/base_complete.json
    head/training_prevalence.json
    head/A4_under_only/epoch_checkpoints/...
    head/A4_under_only/epoch_gamma_validation_scan.csv
    head/A4_under_only/selected_head.pt
    head/A4_under_only/selected_epoch_gamma.json
    head/head_result.json
    evaluation/locked_prefix_test_predictions.csv
    evaluation/paired_metrics.csv
    evaluation/paired_changes.csv
    evaluation/event_level_paired_changes.csv
    evaluation/evaluation_audit.json
```

原CSV中的 `A2_cross_attention_base` / `A4_under_only` 列若存在，按键重排并分别标为 `FullRecord_Base` / `FullRecord_CAURC`。不会把旧dual-risk文件的一般 `final_log10_*` 列冒充CA-URC；需要明确的 `--old-base-prefix` / `--old-caurc-prefix`。
新Base指标始终来自新权重、新输入上的输出；不从旧CSV抽取。

`paired_metrics.csv` 中的factor2/under05是0–1比例；`paired_changes.csv` 中对应差值乘100才是百分点。MAE和Bias为log10单位。
主要结果看新条件内部 `Prefix_CAURC - Prefix_Base`。旧模型在同一子集上的重汇总只是背景参照，不自动构成单因素预处理因果估计。

## 检查与限制

本包保持上传45/63的模型、两阶段loss及训练/选择函数。来源文件逐字复制，其SHA256见checks.json。
已用合成HDF5、合成旧锁定CSV在CPU上跑通audit -> 1轮Base -> 1轮CA-URC -> 验证选择 -> 测试配对 -> 指标与bootstrap导出，并测试缺失事件阻断、禁止旧权重混用和文件变更检测。
合成冒烟测试为了快速覆盖调用链，使用小网络、少轮数和明确放宽的测试约束；不是论文训练参数，更不代表科研效果。
未取得你的真实HDF5、StationXML、QC CSV、阈值文件、分组清单或权重，未执行真实科研训练；当前17个失败事件的具体原因仍未知。
