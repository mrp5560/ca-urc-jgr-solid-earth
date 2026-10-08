# Causal-SeisField：已完成前缀实验的地震物理条件诊断

本代码是新的、只读的事后诊断，不改45/63训练代码，不加载权重、不推理、不训练，不修改预测或标签，不重新抽测试台站。需要 Python、numpy、pandas、h5py（原训练环境通常已具备）。不需要GPU、ObsPy或原始miniSEED。

## 1. 放置与运行

把 `diagnose_physical_conditions.py` 复制到项目根目录，即 `train_snapshot_experiment.py` 所在目录：

```text
F:\数据集\世界模型\causal_seisfield_data_scripts\causal_seisfield_data_scripts
```

先审核配对、到时字段覆盖和全局指标复现：

```powershell
D:\python3\python.exe .\diagnose_physical_conditions.py --stage audit
```

预检通过后完成分层统计：

```powershell
D:\python3\python.exe .\diagnose_physical_conditions.py --stage analyze
```

两条命令使用相同默认路径：

```text
--run-dir runs/snapshot_available_validation_common1603
--evaluation-subdir evaluation_grouped
--h5-root data/scedc/processed_full_v4/events
--out-dir runs/physical_diagnostics_prefix
```

可显式指定上述参数。不要把输出目录放进已完成的训练目录，也不要修改旧协议文件。两个阶段参数应一致；改变方案后用新的输出目录，不覆盖旧诊断。

## 2. 实际读取哪些文件

从已完成实验读取：

- experiment_protocol.json：T0、K、Q、重复次数、种子、原尾部阈值、保留事件及文件来源。
- evaluation_grouped/locked_prefix_test_predictions.csv：逐目标真值、Base和CA-URC预测、风险分数及实际校正量。
- evaluation_grouped/paired_metrics.csv：用来检查全局指标是否复现。
- evaluation_grouped/evaluation_audit.json：协议ID、测试数量、选定gamma和分组字段。
- audit/cohort_manifest.csv：保留训练/验证/测试事件、sequence_group和原HDF5路径。
- audit/checked_test_station_draws.csv：本次实验实际保存的输入/目标台站索引。
- audit/checked_test_keys_and_truth.csv：本次实验锁定目标键和真值。

从每个原 `processed_full_v4/events/<event_id>.h5` 只读取：

- station_coords（纬度、经度、海拔；不改变顺序）
- station_id
- p_offset_sec、s_offset_sec（相对于同一最早P参考时刻）
- event_id、first_p_time等属性；只查询acceleration的维度，不读取波形数组。

缺失文件或错配时停止。缺失/无效到时不删除目标行，标为Unknown。不从旧波形补数据，不重新进行拾取，不以理论到时补齐缺失S。

默认不重新计算整个原HDF5的SHA256；记录读取元数据的指纹，并核对与实验协议的事件/划分/目标键匹配。需要重新核实原HDF5整文件来源时，在两个阶段均加 `--verify-source-hash`，程序会匹配原协议保存的哈希，代价为额外磁盘读取。

本次已有CSV仅为汇总；真正的分层还需要本地逐目标预测和上述元数据。脚本可在本地运行，但提供脚本不等于已经得到物理分层结果。

## 3. 分层定义：在检查分层误差之前固定

### A. 目标P到时状态（主分析）

设 `p_lead_sec = p_offset_sec - T0`，T0来自已完成实验协议（当前为5秒）。

- Pre-P：有效P拾取且 `p_offset_sec > T0`。
- Post-P：有效P拾取且 `p_offset_sec <= T0`。
- Unknown/invalid P：缺失或明显早于最早P参考（小于-0.001秒）。

Pre-P指目标站本地P尚未到达，不是地震尚未发生，也不意味着输入台站没有波形信息。

### B. 目标P/S到时状态（覆盖允许时的补充分析）

三种可解释状态统一要求P和S均有限且 `S>P`，P也满足上述有效条件：

- Pre-P：`T0<P`。
- P-to-S：`P<=T0<S`。
- Post-S：`S<=T0`。
- 缺P、缺S或时序不合法：Unknown/invalid PS。

字段存在不代表每个目标都有有效拾取。不要把缺S理解成S尚未到达。P-only与P/S-complete分析总体不同，不直接横向比较其均值。

### C. 目标到本次五个输入站的最近距离（主分析）

使用原45号代码的局部平面坐标换算（x为经向差×111.32×cos参考纬度，y为纬向差×110.57）。参考点是该次选定输入站坐标的均值；不是震中。

`d_min = min(distance(target, selected_input_i))`。

同一目标在不同repeat中可能有不同最近距离。因此保留event/repeat/target键，不把所有台站的全网最近邻当作输入距离。

距离Near/Middle/Far边界由保留TRAIN事件元数据计算：每事件3次固定抽样、每次与原协议相同K/Q、采用63号text:seed抽样顺序。不读取训练预测、波形或幅值；边界是全部这些训练距离的1/3和2/3分位数（每事件的重复和目标数量相同，相当于同权事件）。

边界保存于 `distance_bins_train_only.json`。测试集不重新计算分位数；幅值尾部阈值也仍用原协议值，不在每个物理分层内部重新定义top 10%。

### D. 输入台站S波记录覆盖（补充观测条件）

统计本次K个输入站中，多少站有有效S拾取且S严格早于T0，即输入前缀含正长度的post-S时段。对应0、1-2、3+站分组。

为了不把缺失S拾取当成未到达，只有全部K站具备有效P/S对时才分入这些类别；否则为Unknown/partial S，同时保留已知S数、已确认到达数和未知数。

本项是利用归档拾取的回溯性状态诊断，不把S拾取作为模型输入，也不是自动识别实时候选触发。

## 4. 统计方法

主要方法固定为 `Prefix_CAURC - Prefix_Base`，不把旧FullRecord模型混进主分层比较。

所有指标先筛选物理状态及幅值总体，再依次平均：目标→repeat→事件。空条件保持缺失，不填零。分别输出overall、non_tail和high_motion_tail。

- mae：log10绝对误差。
- bias：预测减观测的有符号残差；更负表示条件平均低估更严重。
- under05：残差<=-0.5的事件均衡比例。
- factor2：绝对残差<=log10(2)的事件均衡比例。
- delta=CA-URC-Base；MAE/under05负值有利，factor2正值有利。

条件均值变化的95%区间沿用完成版实验的分组→事件配对bootstrap分布，默认10,000次，抽样总体仍为全部保留219个测试事件（具体数量读取文件），不存在某条件的事件在该条件中为NaN。不是先平均22组的均值。

默认少于10个事件或少于5个贡献分组时，仅报告描述统计，不给CI；这是本诊断建议的保守显示门槛，不是普遍统计定理，也不是达到门槛后即可宣称充分可靠。边界规则、样本量及空重采样均被记录。各CI是探索性的逐项区间，没有多重比较或训练种子不确定性的保证。

`physical_common_event_contrasts.csv` 另外检验共享事件上的直接条件差异：

- Pre-P minus Post-P；
- P-to-S minus Post-S；
- Far minus Near。

每个对比仅纳入同时对两个条件有贡献的事件，比较Base under05/Bias和校正MAE变化，且使用同一组bootstrap抽样。用于避免把“一组显著、另一组不显著”误当作组间差异显著。这仍不能消除目标幅值、传播路径、场地及几何的全部混杂。

## 5. 输出及单位

```text
runs/physical_diagnostics_prefix/
  diagnostic_configuration.json
  distance_bins_train_only.json
  training_geometry_source.csv
  metadata_fingerprints.csv
  physical_target_diagnostics.csv
  physical_arrival_coverage.csv
  physical_stratum_counts.csv
  overall_metric_reconciliation.csv
  physical_diagnostics_audit.json
  physical_stratified_metrics.csv             # analyze阶段
  physical_event_metrics.csv                  # analyze阶段
  physical_common_event_contrasts.csv          # analyze阶段
  physical_primary_summary.csv                # analyze阶段
```

- 主结果看 `physical_primary_summary.csv` 和 `physical_common_event_contrasts.csv`。
- 到时是否足够看 `physical_arrival_coverage.csv`。
- 各组台站行数/唯一事件-台站对/事件/分组数看 `physical_stratum_counts.csv`。同一事件可进入多个条件，不能简单相加当成独立样本总数。
- `physical_stratified_metrics.csv` 的under05和factor2是0-1比例；delta和CI乘100才是百分点。
- `physical_primary_summary.csv` 额外提供 `*_display` 列并注明单位：under05的绝对值是百分数，变化/CI是百分点；MAE不乘100。
- `inference_status=descriptive_only_sparse`时不要根据点估计给出强排序。
- 新的各条件均值不能简单平均回原总体指标，因为各条件的事件/重复权重不同。程序单独先检查完整总体与旧汇总表一致。

## 6. 如何回答两类问题

“什么条件容易低估”：看Base under05是否较高、Base Bias是否更负；MAE单独升高不能区分低估和高估。

“什么条件校正更有效”：看对应条件的MAE配对变化、相对MAE降幅、under05变化，同时检查非尾部MAE和整体Bias，不能只看实际校正量大不大。风险分数不是校准概率；风险分数与修正量的关联包含模型公式本身，不能据此证明物理机制。

“远比近改善更强”：需要看直接条件差异而不是两个独立的显著性判断。较困难条件中绝对降幅更大不代表最终误差更低；同时给出Base与CA-URC绝对指标。

允许的表述：误差/校正收益与归档相位状态、输入几何覆盖存在条件关联。
不允许凭这些结果直接声称：波传播因果机制已经识别、缺乏S波就是失败的唯一原因、物理可预测性极限被证明、实时预警已得到验证。

## 7. 测试与限制

附带 `test_physical_diagnostics.py` 可在脚本目录运行：

```powershell
D:\python3\python.exe -m unittest test_physical_diagnostics -v
```

合成测试核对时序边界、缺失S处理、台站索引不变、坐标计算、事件均衡、空组不填零、分组重采样、全局指标复现、元数据/协议保护及audit→analyze全流程。没有使用用户真实HDF5或43,800行逐目标预测；没有绘制或编造论文分层结果。

本包只生成统计和作图源表，不改变已有论文图片。完成统计后再根据有效组数选择图形布局，避免先设计结论式图再追求某个分层结果。
