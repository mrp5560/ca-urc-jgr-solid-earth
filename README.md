# Causal-SeisField and CA-URC

Research code, trained-model artifacts, numerical source data, and selected processed waveform data prepared for submission to *Journal of Geophysical Research: Solid Earth*.

**Release scope:** this is a results-recalculation and sample-inference package. It provides archived predictions/results, selected checkpoints, four spatial-case HDF5 files, an 81-event full-record convenience sample, and 1,620 snapshot sidecars. It does **not** contain the complete full-record training/label dataset.

Repository: <https://github.com/mrp5560/ca-urc-jgr-solid-earth>

This repository preserves the study's numbered scripts and archived outputs. It does not imply manuscript acceptance.

## Model and benchmark

CA-URC means **Cross-Attention Underprediction-Risk Correction**. It freezes a target-conditioned Cross-Attention Base and applies a nonnegative correction:

```text
prediction = frozen_base + underprediction_risk ** gamma * correction
```

The final model is `A4_under_only` in `63_train_and_evaluate_cadrg_gate_ablations_A3_A5.py`. Validation selected epoch 3 and gamma 5 for the primary grouped experiment. Earlier `cadrg` scripts also implement dual-risk models and ablations; their filename alone does not identify CA-URC. Newly trained and shifted-domain experiments require their own validation selection.

The primary setting uses a 5 s snapshot, 5 input stations, and 10 non-input target stations. There are 1,620 eligible events: 1,189 training, 207 validation, and 224 test events. Twenty draws per test event yield 44,800 target predictions. The complete processed archive contains 3,239 events. Targets are remaining horizontal PGA/PGV peaks after the snapshot, in log10 units of m/s² and m/s. Metrics average targets within draws, draws within events, and then events; repeated rows are not independent earthquakes.

**Interpretation limit:** primary waveforms were preprocessed offline before snapshot extraction. Full-record detrending and zero-phase filtering can make pre-snapshot processed samples depend on later raw samples. Archived picks define retrospective arrival states. This is a retrospective snapshot benchmark, not a validated end-to-end real-time processing or warning system. A separate prefix-input experiment truncates raw inputs before preprocessing while retaining retrospective picks, archived labels, and cohort-selection limits.

## Installation

Use Python 3.11 in a fresh environment. Source parsing was checked with Python 3.11.5. Dependency ranges are installation requirements, not a recovered exact lockfile for every historical run. GPU training needs an appropriate PyTorch/CUDA installation.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pip install -r requirements-figures.txt
```

Activate `.venv` using your operating system's activation command before installation or execution. Figure extras are optional for numerical analysis. Run all examples from the repository root.

## Contents and data

| Location | Contents |
| --- | --- |
| Numbered root scripts | Acquisition, preprocessing, split construction, models, and evaluation |
| `snapshot_*.py`, `train_snapshot_experiment.py`, `source45_baselines.py`, `source63_risk.py` | Prefix-input experiment and original model implementation |
| `diagnose_physical_conditions.py` | Post-hoc physical-condition diagnostics |
| `data/scedc/model_manifests/` | Eligibility and fixed event/split manifests |
| `runs/` | Selected checkpoints, locked predictions, selection records, and numerical outputs |
| `画图/` | Figure scripts and supporting tables, including historical layout variants |
| `Causal_SeisField_*`, `SeisField_physical_diagnostics/` | Original source bundles and run/check notes |

CSV HDF5 paths are normalized to repository-relative paths. Numerical values, event identities, and locked draws are preserved. Archived JSON protocols may retain original machine paths and hashes as provenance. The prefix protocol ID uses source-code hashes, input-content hashes, cohort order, and scientific settings rather than directory names. Moving unchanged files does not by itself invalidate that protocol. Preserve exact source/data bytes and use new output directories for fresh experiments.

Four complete processed HDF5 examples are included:

| Event ID | Case |
| --- | --- |
| `39464360` | Representative preservation |
| `39111991` | Successful tail correction |
| `40865216` | Residual failure |
| `40865192` | Supplementary non-tail overcorrection |

These examples do not replace the full training dataset.

### Published waveform samples and snapshot sidecars

The [v1.0.0 release](https://github.com/mrp5560/ca-urc-jgr-solid-earth/releases/tag/v1.0.0) distributes two independent ZIP assets:

| Release asset | Content and scope |
| --- | --- |
| `ca-urc-full-record-sample-81-events.zip` | 81 full-record event HDF5 files, selected as the first event IDs in sorted order for convenient packaging. This is neither a random nor a representative training sample. |
| `ca-urc-snapshot-1620-events.zip` | All 1,620 snapshot-input sidecars from the supplementary scenario: 1,603 ready events and 17 incomplete/excluded events retained for audit. These short input files do not contain the complete full-record target/label archive. |

The four spatial-case files listed above are included in the repository separately. The study's complete full-record archive contains 3,239 events, approximately **16.97 GiB**; that full archive is **not uploaded in this release**. The full archive remains part of the original local study data. SCEDC acquisition and preprocessing scripts are supplied for independent reconstruction from source observations.

Verify each downloaded asset's SHA256 and extract each ZIP independently into the repository root; do not concatenate them. Internal paths restore:

```text
data/scedc/processed_full_v4/events/<event_id>.h5
data/scedc/snapshot_available_t0_5s_k5/events/<event_id>.h5
```

Use the download commands below to obtain all *published* assets. `--dataset all` means these two assets, not the full 3,239-event archive. Archived-table analyses below run without downloading the full waveform archive. Full retraining, complete waveform-based lead-time regeneration, and full prefix preflight require additional full-record label files.

The complete raw miniSEED and StationXML archive is also not duplicated here. Source observations are available from SCEDC and contributing networks, and scripts `00`–`11` retain the acquisition/preprocessing route. For a newly reconstructed full-label archive, regenerate matching prefix sidecars from raw inputs as needed: the published sidecars verify the exact original source-HDF5 SHA256. Do not disable that integrity check to pair a different label file with an archived sidecar.

## Recompute primary numerical analyses

These commands read archived locked predictions, perform no training/model selection, and write to fresh directories. Bootstrap analyses use 10,000 replicates by default.

```bash
python 69_sequence_aware_paired_bootstrap_caurc.py --manifest data/scedc/model_manifests/scenario_t0_5s_k5_linux.csv --out-dir reproduced/caurc_sequence_aware_bootstrap
python 70_caurc_warning_lead_time_analysis.py --out-dir reproduced/caurc_warning_lead_time
python 78_recompute_final_propagation_metrics_caurc_protocol.py --manifest data/scedc/model_manifests/scenario_t0_5s_k5_linux.csv --out-dir reproduced/propagation_metrics
```

Inputs:

- `runs/cadrg_gate_ablation_A3_A5/locked_test_ablation_predictions.csv`
- `runs/final_strong_baselines_reuse_locked/final_strong_baseline_reused_locked_predictions.csv`
- `runs/phase2_warning_lead_time/unique_event_station_lead_times.csv` for lead-time analysis
- `data/scedc/model_manifests/scenario_t0_5s_k5_linux.csv` for event groups/splits

Script `69` reuses the bootstrap implementation in `62_sequence_aware_paired_bootstrap_cadrg.py`. Script `70` uses stored physical lead times; script `64` can regenerate them from full waveforms with explicit manifest/HDF5 paths. Script `78` applies the final metric/pairing protocol to propagation baselines. Preserve every pairing and truth check.

## Training and full reproduction map

This is a code/dependency map for the original full workflow, not a claim that the published sample is a complete training dataset. Full-waveform stages require the missing full-record data or independent reconstruction from source observations. The table-based analyses above are supported by the published package.

Scripts dynamically import earlier scripts by filename: retain their names and root placement. Saved run arguments, threshold files, and selection JSON are the reference for completed experiments; defaults alone are not a complete historical run record.

| Stage | Scripts | Inputs and dependencies |
| --- | --- | --- |
| Catalogue, waveform, phase acquisition | `00`–`05` | SCEDC public services and contributing-network metadata |
| Full preprocessing and QC | `06`–`09` | Raw miniSEED, StationXML, phase-ready tables; `07` produces full HDF5 |
| Grouped manifests | `10`, `11` | Processed summaries and spatiotemporal groups |
| Chronological/Ridgecrest split | `22` | Existing scenario manifest; preserve Ridgecrest holdout |
| Baselines | `45`, `59` | Full HDF5, manifest, train thresholds; `59` imports `40`/`45` and reuses archived Power Gate predictions |
| Earlier dual-risk reference | `60`, `61b` | Selected Cross-Attention Base; `60` imports `45`; `61b` imports `45`/`60` |
| Final CA-URC and gate ablations | `63` | `45`, selected base, thresholds, HDF5, manifest, locked dual-risk pairing reference |
| Grouped statistics/propagation | `69`, `78` | `62` and locked prediction tables |
| Physical lead times | `64`, `70` | Full HDF5 for `64`; stored lead times and predictions for `70` |
| Strict station OOD | `42`, `65`, `71`, `75_*_fixed.py` | Station holdouts; `45`, `52`, `60`, `63`, `65`; separate selection |
| Chronological/frozen Ridgecrest | `68`, `72`, `73` | Chronological manifest; `45`, `60`, `63`, `68`, `72`; frozen checkpoints for `73` |
| Network structure | `74` | `45`, `59`, `62`, full HDF5 and paired references |
| Observation budget | `81` | `45`, `62`, `63`; master manifest before single-budget eligibility filtering |
| Prefix preprocessing/training | `08_preprocess_snapshot_available_inputs.py`, `train_snapshot_experiment.py` | Raw inputs for preprocessing; sidecars, full labels, reviewed exclusions and locked targets for training |

Archival recipe for a grouped correction-head rerun, **only after independently supplying all required full-record label files**. The 81-event sample and four cases are insufficient for this command:

```bash
python 63_train_and_evaluate_cadrg_gate_ablations_A3_A5.py --manifest data/scedc/model_manifests/scenario_t0_5s_k5_linux.csv --h5-root data/scedc/processed_full_v4/events --base-checkpoint runs/final_strong_baselines_reuse_locked/cross_attention/best_model.pt --threshold-json runs/tail_gated_compromise_t0_5s_k5/tail_thresholds_q0.90_t0_5s.json --full-cadrg-predictions runs/cross_attention_dual_risk_previous_grouped_benchmark/locked_dual_risk_predictions.csv --out-dir reproduced/cadrg_gate_ablation_A3_A5
```

This keeps the default full gate suite, including A4's original position and seed convention. Inspect each script's `--help` and archived settings. The full training workflow has not been rerun merely by preparing this repository. Environment differences and stochastic training can affect results.

## Prefix-input cohort

The prefix experiment retains an explicitly reviewed **1,603-event common cohort** from the 1,620-event scenario after 17 exclusions. Its test contains **219 events / 43,800 rows**, distinct from the primary 224-event test. Exclusions are recorded in `prefix_qc_exclusions_17.csv`. Failed eligible stations must not be silently removed, replaced, or filled from full-record inputs.

The snapshot ZIP alone cannot satisfy the cohort preflight because most full-record label files are absent from this release. The following archival recipe requires **all matching full-record source/label HDF5 files** (or a fully reconstructed label-plus-sidecar pair), not just the published sample:

```bash
python train_snapshot_experiment.py --stage audit --manifest data/scedc/model_manifests/scenario_t0_5s_k5_linux.csv --excluded-events-csv prefix_qc_exclusions_17.csv --group-column sequence_group --out-dir reproduced/prefix_common1603
```

Use identical cohort/settings/output-directory arguments for later `--stage base`, `--stage head`, and `--stage evaluate`. The expected-event count remains 1620 because it refers to the scenario before exclusions. Inspect audit failures; do not suppress mismatches or revise the cohort using test performance.

Completed numerical results are in `runs/snapshot_available_validation_common1603/` and are available independently of full waveform downloads. Regenerating physical diagnostics still requires the original station metadata for all analyzed events: `diagnose_physical_conditions.py` defaults to `--stage audit`; `--stage analyze` adds stratified statistics. Original source-bundle notes are in Chinese and describe their historical checks, not checks automatically repeated during publication preparation.

## Figures and validation

Explicit JGR figure scripts are `Fig1_JGR_SE_dataset_distribution.py` and `画图/Fig7_JGR_SE_spatial_case_studies_layout_refined.py`. Other filenames retain `NC` from earlier layouts; this does not imply publication there. Machine-specific argument defaults were made portable in copied sources; see `docs/portable-defaults.json`. Model, loss, selection, and metric logic was not changed by this cleanup. Figure 1 accepts `--relief-raster none` for a flat fallback or an explicit relief GeoTIFF. A fresh Cartopy installation may need basemap downloads.

All 163 copied Python sources passed syntax parsing with Python 3.11.5. The three existing test modules passed all 65 tests. Scripts 69, 70, and 78 were rerun from the staged files; all 18 generated CSV tables matched their archived counterparts at absolute and relative tolerance 1e-12, using the full default 10,000 bootstrap repetitions where applicable. These checks do not constitute full model retraining or reproduction of every figure. See [current validation results](docs/VALIDATION.md).

## Citation and licenses

Code uses the [MIT license](LICENSE). Author-created derived research data and numerical outputs use [CC BY 4.0](DATA_LICENSE.md), with the stated third-party exclusions. Cite the repository version, associated manuscript when available, and original data providers. The software author and dataset curator is **Runping Ma**. `CITATION.cff` records the confirmed software author; this does not establish the manuscript author list. Software and data should be cited separately. No publication or archive DOI is stated until it has been assigned.

See [SCEDC citation guidance](https://scedc.caltech.edu/about/citation.html). Cite SCEDC [10.7909/C3WD3xH1](https://doi.org/10.7909/C3WD3xH1) and SCSN/CI [10.7914/SN/CI](https://doi.org/10.7914/SN/CI) where applicable; other networks require their own citations. A later DOI archive should identify the specific frozen release.


## Download the published data assets

Install or clone this repository, then run from its root:

```bash
python tools/download_data.py --dataset all
python tools/download_data.py --dataset all --check-only
```

Use `--dataset sample` for the 81-event full-record convenience sample or
`--dataset snapshot` for the 1,620 snapshot sidecars. `all` downloads both
published assets only. None of these choices downloads the unpublished full
3,239-event training/label archive.

The helper verifies published SHA256 checksums and resumes by checking already
extracted HDF5 files. Allow space for extracted files and the ZIP being processed.
The two ZIPs are independent; do not concatenate them. Exact asset sizes,
file lists, and hashes are recorded in
[data/RELEASE_DATA_MANIFEST.json](data/RELEASE_DATA_MANIFEST.json).

Release: [v1.0.0](https://github.com/mrp5560/ca-urc-jgr-solid-earth/releases/tag/v1.0.0).
