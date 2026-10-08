# Causal-SeisField：预测截面可用预处理补强实验

## 1. 这个补丁做什么

基于你上传的 `07_preprocess_ground_motion_full_safe(2).py`，新增一个独立实验脚本，不替换、不覆盖旧脚本和 `processed_full_v4`。

原始生产链：完整 miniSEED → 原始记录QC → merge/interpolate → 去均值/去趋势/taper → 仪器响应去除 → 旋转 → 零相位滤波 → 共同时间轴 → 完整标签档案。

新增输入链：miniSEED → 对每条原始trace严格保留 `t < first_P + T0` → QC → 原来的merge/interpolate/去趋势/taper/去响应/旋转/滤波/重采样 → `[first_P-2 s, first_P+5 s)` 的350点加速度输入。

新脚本逐字保留了原代码中的7个函数：`parse_utc_timestamp`、`validate_raw_stream`、`prepare_raw`、`remove_response_stream`、`rotate_to_zne`、`choose_components`、`exact_array`。因此没有在同一次比较中顺手更换水位、滤波频带、滤波阶数、插值方法或QC阈值。

只生成加速度输入，因为已提供的方法用加速度作为网络波形输入。速度和未来峰值仍从旧HDF5取得，不重复生成或改写标签。

## 2. 为什么输出一个独立HDF5而不是缩短旧HDF5

旧文件包含未来速度/加速度、台站顺序、P/S偏移等，是锁定标签及抽样索引的来源。如果直接把它改成截至5秒的短记录，下游 `snapshot:` 的剩余峰值计算就会变空，或者标签/台站索引被悄悄改变。

新输出（称为sidecar，即配套输入文件）逐站与旧文件对齐。台站数和索引完全保留，不删除失败或未到P波台站。

- `input_acceleration`: `(N_original_stations, 3, 350)`，单位m/s²，尚未作sign-log变换。
- `input_valid`: 对应台站的前缀输入是否成功生成。
- `snapshot_eligible`: 沿用旧 `p_offset_sec` 的主实验资格判断。
- `station_index`: 原来的 `0..N-1`。
- `station_id`, `station_coords`, `p_offset_sec`, `s_offset_sec`（原文件有时）：原样保留。
- 原始文件SHA256、时间轴、QC、边界审计报告：保存以便核对。
- 未达到截面P波资格的台站保留为NaN占位行，不是零值训练输入；它们仍可从旧档案作为预测目标。

新文件故意不命名为旧的 `acceleration`/`velocity` 数据集，以防旧训练器无声地把它当作完整标签档案。

## 3. 文件

- `08_preprocess_snapshot_available_inputs.py`：独立、完整的新增预处理程序。
- `snapshot_input_reader.py`：训练和预测时读取新输入的严格校验函数。
- `test_snapshot_preprocessing.py`：时间索引、HDF5配对、源文件不变性和拒绝错误输入的测试。
- `validation_report.json`：本次在工作环境中实际执行的检查结果及未完成项目。

不需要下载新的地震数据；但本地必须保留原始miniSEED和StationXML。只有处理后的HDF5无法重做这个实验。

## 4. 环境和试运行

使用能够运行原07脚本的同一Python环境，避免同时引入依赖版本变化。脚本需要numpy、pandas、h5py、ObsPy及其SciPy依赖。

将两个生产用`.py`文件放在原项目工作目录。相对路径从当前工作目录解释，与原脚本一致。

### 先做3个事件（仅调试，不是完整性能验证）

```bat
D:\python3\python.exe 08_preprocess_snapshot_available_inputs.py ^
  --events data/scedc/model_manifests/scenario_t0_5s_k5.csv ^
  --legacy-h5-root data/scedc/processed_full_v4/events ^
  --raw-root data/scedc/raw_m3_2010_2025 ^
  --out-root data/scedc/snapshot_available_t0_5s_k5 ^
  --limit-events 3 ^
  --audit-stations-per-event 2 ^
  --debug
```

如需只在训练/验证集做调试，使用 `--splits train,validation`（前提是清单的 `split_grouped` 确实使用这些值）。若实际叫 `val`，使用实际标签。程序对不存在的标签报错，不猜测划分。

原空间脚本另一个清单路径为 `model_manifests/scenario_t0_5s_k5_linux.csv`。若你实际使用的是该清单，把 `--events` 改成那个文件即可；不要创建新的训练/测试划分。

### 完整主实验

```bat
D:\python3\python.exe 08_preprocess_snapshot_available_inputs.py ^
  --events data/scedc/model_manifests/scenario_t0_5s_k5.csv ^
  --legacy-h5-root data/scedc/processed_full_v4/events ^
  --raw-root data/scedc/raw_m3_2010_2025 ^
  --out-root data/scedc/snapshot_available_t0_5s_k5 ^
  --expected-events 1620 ^
  --audit-stations-per-event 2
```

Linux用 `python3` 替换解释器，使用反斜杠续行，或者将命令写为一行。`1620`是已报告主实验总事件数；不应把它用于其他实验清单。

首次小样本和全量可用同一输出目录。复用之前会核对配置、脚本版本、旧HDF5及原始miniSEED/StationXML哈希。配置改变或某事件之前不完整时需核查原因后加 `--overwrite`。覆盖仅发生在新目录，不会改写旧HDF5。

## 5. 审计的含义

默认每个事件对原站序中前2个成功处理的合资格台站做审计，分别把截面后的原始样本：

1. 乘2；
2. 替换为固定随机种子的噪声（仅检查信息依赖，不是地震模拟）。

截面前原始样本和元数据不变。新流程分别重算，检查：

- 裁剪后原始前缀的指纹一致；
- 生成的加速度输入逐元素完全一致；
- sign-log变换后的输入最大差值。

同时运行原完整处理链，报告未来变化对其输入的影响；若原链因QC或去响应失败，明确记录失败，不把它记为零差异。

未来没有实际被改变的记录记为`inconclusive_no_changed_future`，不是通过。审计只覆盖指定台站数，不自动等于每个台站都做了扰动实验。将 `--audit-stations-per-event` 设为很大的整数可覆盖每个事件全部合资格台站，但计算开销会增加。

该脚本不加载神经网络，所以不包含预测输出不变性或模型性能验证。它准备输入和预处理证据；随后仍需重新训练/测试。

## 6. 输出与失败处理

```text
data/scedc/snapshot_available_t0_5s_k5/
  events/<event_id>.h5
  event_prefix_qc.csv
  station_prefix_qc.csv
  prefix_invariance_audit.csv
  paired_manifest.csv
  run_configuration.json
  run_summary.json
```

`paired_manifest.csv`保留原始列（包括原h5_path和split），只新增输入文件路径和QC列。不会自动过滤不完整事件，也不会把h5_path改成输入配套文件。

如任何原本合资格台站处理失败，则该事件不是`ready_for_paired_use`。这是为了阻止训练器静默缩小输入池，重新抽出不同的五台站。失败事件仍可写出诊断用sidecar；它不能直接进入配对训练。

失败值为NaN，不回退旧完整记录输入。程序结束状态码2表示主队列未全部准备好；应查看QC。解决原始文件/响应/边界问题后重跑；若最终需要共同有效子集，须明确报告并对新旧方法共同应用，不应无说明地改变样本量。

## 7. 训练器必须真正接入新输入

仅运行08脚本还没有完成补强实验；只修改 `h5_path` 也不对。保持旧HDF5用于所有真值和元数据，在原台站抽样得到 `input_indices` 后，将原来的波形读取与sign-log变换替换为：

```python
from pathlib import Path
import h5py
from snapshot_input_reader import load_snapshot_input

prefix_root = Path("data/scedc/snapshot_available_t0_5s_k5/events")

# h5_path仍为原processed_full_v4的HDF5；input_indices沿用原抽样。
with h5py.File(h5_path, "r") as h5:
    input_waveforms = load_snapshot_input(
        original_h5=h5,
        sidecar_path=prefix_root / f"{event_id}.h5",
        input_indices=input_indices,
        t0_sec=5.0,
        input_pre_sec=2.0,
        apply_log_transform=True,
    )
    # 已包含sign(a)*log1p(abs(a)/1e-3)，不要重复变换。
    # 原来的完整波形仍只用于标签/其他既有评价计算：
    acceleration_for_labels = h5["acceleration"][:]
    velocity_for_labels = h5["velocity"][:]
```

这只是接入位置示例，不是完整训练器。训练、验证、测试、CA-URC风险标签生成、推理缓存都必须使用同一输入策略。不能只修改绘图脚本或只修改测试数据读取。

新Base应重新训练；新CA-URC的严重低估标签从新Base生成；模型选择只用验证集。旧的标签、尾部阈值、事件划分、固定测试台站组合继续保留。与旧输入有关的缓存必须另命名或重建。

尚未提供两阶段训练器的完整实现，因此本补丁没有声称已经把训练/评估数据加载器全面接入，更没有执行模型训练。

## 8. 解释边界与方法中可用的表述

保留原来的零相位滤波，是为了最小化除了信息边界之外的参数变化。零相位、FFT去响应等运算仅在已获得的前缀内部进行，因此满足的是固定截面可用性，不是逐采样点的严格因果流式运算。

默认保留原miniSEED中所有截面前的历史样本，并记录实际缓冲长度；不人为补造不存在的历史。原始记录开头和预测截面附近的边界效应仍需在训练/验证数据上检查，特别是0.1Hz频带与较短前缀的组合。不要根据测试性能反复调缓冲或滤波参数。

归档P到时、最早台网到时和旧档案的离线QC仍保留；旧标签也保持离线定义。实验通过只能支持“输入无截面后波形依赖条件下的回溯性结果”，不能声称完整实时预警已被验证。

可用英文表述（仅在实际运行后据结果填写）：

> As a sensitivity experiment, all raw input traces were truncated strictly before the prediction snapshot prior to quality checks, merging, detrending, response removal, rotation, filtering, and resampling. The original target labels and station indices were retained. This prefix-only batch processing evaluates snapshot-level information availability; it does not constitute a sample-wise causal filtering pipeline or validation of real-time triggering.

## 9. 本次检查范围

本地通过了语法检查、CLI帮助、原7函数逐字一致性检查，以及20项单元测试。1项依赖ObsPy的实际Trace截断测试被跳过：本工作环境未安装ObsPy，安装尝试未成功。没有原始miniSEED、StationXML、旧HDF5和训练权重，因此未运行完整物理预处理或真实性能验证。

不要把单元测试的通过数、审计模板或合成测试记录当作论文实验结果。
