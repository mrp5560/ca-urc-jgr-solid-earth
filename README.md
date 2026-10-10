# Causal-SeisField and CA-URC

Core training and model-evaluation code for **Cross-Attention Underprediction-Risk Correction (CA-URC)**, prepared for submission to *Journal of Geophysical Research: Solid Earth*.

The current `main` branch retains **23 core and dependency Python files**, together with archived model artifacts, predictions, manifests, and numerical results. This cleanup removes other code without changing the retained scientific algorithms. It does not claim a new training run or a rerun of the historical validation checks.

- [Repository](https://github.com/mrp5560/ca-urc-jgr-solid-earth)
- [Software v1.0.1 archive — DOI 10.5281/zenodo.23273595](https://doi.org/10.5281/zenodo.23273595): the existing complete historical software release. Its contents are not rewritten by the later cleanup of `main`.
- [Dataset — DOI 10.5281/zenodo.23273640](https://doi.org/10.5281/zenodo.23273640): locked predictions, numerical source data, and selected processed waveform data.
- [GitHub v1.0.0 data assets](https://github.com/mrp5560/ca-urc-jgr-solid-earth/releases/tag/v1.0.0).

## Model and benchmark

CA-URC freezes a target-conditioned Cross-Attention Base and applies a nonnegative correction:

```text
prediction = frozen_base + underprediction_risk ** gamma * correction
```

The final model is `A4_under_only` in script `63`. Validation selected epoch 3 and gamma 5 for the primary grouped experiment. Earlier `cadrg` filenames also cover dual-risk models and ablations; newly trained or shifted-domain experiments require their own validation selection.

The primary setting uses a 5 s snapshot, 5 input stations, and 10 non-input target stations. The 1,620 eligible events comprise 1,189 training, 207 validation, and 224 test events. Twenty draws per test event produce 44,800 target predictions. Targets are remaining horizontal PGA/PGV peaks after the snapshot, in log10 units of m/s² and m/s. Metrics average targets within draws, draws within events, and then events; repeated rows are not independent earthquakes.

This is a retrospective snapshot benchmark. Primary inputs were preprocessed offline: full-record detrending and zero-phase filtering can make pre-snapshot samples depend on later raw samples, and archived picks define arrival states. The separate prefix-input experiment truncates raw inputs before preprocessing but retains retrospective picks, archived labels, and cohort-selection limits.

## Installation

Use Python 3.11, activate a fresh environment, and run examples from the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate `.venv` using your operating system's command before installing dependencies. GPU training needs an appropriate PyTorch/CUDA installation. Dependency ranges are installation requirements rather than an exact environment lock for every historical run.

## Data and archived results

`data/scedc/model_manifests/` contains fixed eligibility/split manifests. `runs/` contains selected checkpoints, locked predictions, selection records, and numerical outputs. Published waveform coverage is limited to:

| Data | Scope |
| --- | --- |
| `ca-urc-full-record-sample-81-events.zip` | 81 full-record event HDF5 files, selected by sorted event ID for convenient packaging; not a random or representative training sample |
| `ca-urc-snapshot-1620-events.zip` | 1,620 snapshot-input sidecars: 1,603 ready events and 17 incomplete/excluded events retained for audit |
| Four spatial cases | Event IDs `39464360`, `39111991`, `40865216`, and `40865192` |

The complete **3,239-event full-record archive (approximately 16.97 GiB) is not included**. Snapshot sidecars do not supply the complete full-record target/label archive. Downloading all published data therefore does not enable full retraining or whole-cohort prefix preflight.

The [GitHub v1.0.0 release](https://github.com/mrp5560/ca-urc-jgr-solid-earth/releases/tag/v1.0.0) provides the two independent waveform ZIPs. Verify their SHA256 checksums using [RELEASE_DATA_MANIFEST.json](data/RELEASE_DATA_MANIFEST.json), then extract each independently into the repository root. They restore:

```text
data/scedc/processed_full_v4/events/<event_id>.h5
data/scedc/snapshot_available_t0_5s_k5/events/<event_id>.h5
```

The [Zenodo dataset](https://doi.org/10.5281/zenodo.23273640) distributes its archives in parts. Read its **START_HERE** instructions and use the supplied reassembly tool and checksum manifest to restore and verify the three canonical ZIPs, including `dataset-main-results.zip`. Individual binary parts are not independently extractable ZIPs; the restored ZIP archives are independent datasets.

CSV HDF5 paths are repository-relative. Archived JSON paths and hashes remain historical provenance. Matching source-HDF5 hashes are required by the prefix experiment; do not bypass integrity checks to pair archived sidecars with different label files. The complete historical software archive provides the original broader workflow, while this branch focuses on training and evaluation.

## Core training and evaluation entries

Scripts dynamically import dependencies by exact filename. Retain their names and root placement. Saved run arguments, thresholds, and selection JSON describe completed experiments; defaults alone are not a complete historical run record.

| Task | Entry scripts | Required retained dependencies |
| --- | --- | --- |
| Strong baselines | `59_final_strong_baselines_reuse_locked.py` | `40`, `45`; archived Power Gate predictions |
| Final CA-URC and gate ablations | `63_train_and_evaluate_cadrg_gate_ablations_A3_A5.py` | `45`; selected base, thresholds, and locked dual-risk pairing reference |
| Strict station OOD | `71_train_and_evaluate_caurc_station_ood.py` | `45`, `52`, `60`, `63`, `65`; fixed station-holdout manifests |
| Chronological and frozen Ridgecrest tests | `72_train_and_evaluate_caurc_chronological_2021_2024.py`, `73_evaluate_frozen_caurc_ridgecrest_ood.py` | `45`, `60`, `63`, `68`, `72`; fixed chronological manifests |
| Network-structure ablations | `74_train_and_evaluate_network_structure_ablations.py` | `45`, `59`, `62` |
| Observation-budget experiments | `81_train_evaluate_caurc_observation_budget_grid.py` | `45`, `62`, `63`; master manifest before budget-specific eligibility filtering |
| Locked-prediction analyses | `69_sequence_aware_paired_bootstrap_caurc.py`, `70_caurc_warning_lead_time_analysis.py`, `78_recompute_final_propagation_metrics_caurc_protocol.py` | `62`, locked predictions, stored lead times, and fixed manifests |
| Prefix-input training and testing | `train_snapshot_experiment.py` | `snapshot_experiment_data.py`, `snapshot_input_reader.py`, `source45_baselines.py`, `source63_risk.py` |

The intentionally retained earlier modules are `40_phase1_final_mean_pooling_ablation_suite.py`, `45_phase2_strong_baseline_suite.py`, `52_phase2_train_and_evaluate_station_ood_attention.py`, `60_train_cross_attention_dual_risk_joint_selection.py`, `61b_evaluate_cross_attention_dual_risk_previous_grouped_benchmark.py`, `62_sequence_aware_paired_bootstrap_cadrg.py`, `65_train_and_evaluate_cadrg_station_ood.py`, and `68_train_and_evaluate_cadrg_chronological_ridgecrest.py`. They supply shared implementations, reference benchmarks, or experiment dependencies. The two `source*` files are preserved prefix implementations, not disposable duplicates.

### Final grouped training

Run this archival recipe **only after supplying all matching full-record label files**. The 81-event sample and four cases are insufficient:

```bash
python 63_train_and_evaluate_cadrg_gate_ablations_A3_A5.py --manifest data/scedc/model_manifests/scenario_t0_5s_k5_linux.csv --h5-root data/scedc/processed_full_v4/events --base-checkpoint runs/final_strong_baselines_reuse_locked/cross_attention/best_model.pt --threshold-json runs/tail_gated_compromise_t0_5s_k5/tail_thresholds_q0.90_t0_5s.json --full-cadrg-predictions runs/cross_attention_dual_risk_previous_grouped_benchmark/locked_dual_risk_predictions.csv --out-dir reproduced/cadrg_gate_ablation_A3_A5
```

Preserve the default full gate suite, including A4's original position and seed convention. Inspect each script's `--help` and archived settings. Use fresh output directories; environment differences and stochastic training can affect results.

### Recompute primary numerical analyses

These commands read locked predictions without training or model selection. Bootstrap analyses use 10,000 replicates by default:

```bash
python 69_sequence_aware_paired_bootstrap_caurc.py --manifest data/scedc/model_manifests/scenario_t0_5s_k5_linux.csv --out-dir reproduced/caurc_sequence_aware_bootstrap
python 70_caurc_warning_lead_time_analysis.py --out-dir reproduced/caurc_warning_lead_time
python 78_recompute_final_propagation_metrics_caurc_protocol.py --manifest data/scedc/model_manifests/scenario_t0_5s_k5_linux.csv --out-dir reproduced/propagation_metrics
```

Their archived inputs include `runs/cadrg_gate_ablation_A3_A5/locked_test_ablation_predictions.csv`, `runs/final_strong_baselines_reuse_locked/final_strong_baseline_reused_locked_predictions.csv`, `runs/phase2_warning_lead_time/unique_event_station_lead_times.csv`, and the fixed scenario manifest. Script `70` uses stored physical lead times; `78` applies the final propagation metric/pairing protocol. Preserve all pairing and truth checks.

## Prefix-input cohort and workflow

The prefix experiment uses the reviewed **1,603-event common cohort** after the 17 exclusions in `prefix_qc_exclusions_17.csv`. Its test has **219 events / 43,800 rows**, distinct from the primary 224-event test. Failed eligible stations must not be silently removed, replaced, or filled from full-record inputs.

The following recipe requires all matching full-record source/label HDF5 files, or a fully reconstructed matching label-plus-sidecar pair. The snapshot ZIP alone is insufficient:

```bash
python train_snapshot_experiment.py --stage audit --manifest data/scedc/model_manifests/scenario_t0_5s_k5_linux.csv --excluded-events-csv prefix_qc_exclusions_17.csv --group-column sequence_group --out-dir reproduced/prefix_common1603
```

Use identical cohort/settings/output-directory arguments for the subsequent `--stage base`, `--stage head`, and `--stage evaluate` runs. The expected-event count remains 1620 because it refers to the scenario before exclusions. Inspect audit failures; do not suppress mismatches or revise the cohort using test performance.

The protocol includes source hashes for all five prefix modules, input-content hashes, cohort order, exclusions, thresholds, and scientific settings. Preserve their exact bytes and guards; relocating unchanged files does not justify rewriting an archived protocol ID. Use a new output directory for a fresh experiment. Completed numerical results are in `runs/snapshot_available_validation_common1603/`.

## Historical validation and citation

[VALIDATION.md](docs/VALIDATION.md) records checks for the complete v1.0.0/v1.0.1 publication package. Its 163-source, 65-test, and 18-table results describe that historical package, not fresh checks of this trimmed `main` branch. No new validation or full training rerun is claimed by this documentation update.

The software author and dataset curator is **Runping Ma**. Cite the version used and cite software and data separately through their DOI links above. The existing software DOI identifies the complete v1.0.1 archive, not the later state of `main`.

Code uses the [MIT license](LICENSE). Author-created derived research data and numerical outputs use [CC BY 4.0](DATA_LICENSE.md), subject to its third-party exclusions. Source observations belong to their original providers: cite SCEDC [10.7909/C3WD3xH1](https://doi.org/10.7909/C3WD3xH1), SCSN/CI [10.7914/SN/CI](https://doi.org/10.7914/SN/CI), and other contributing networks as applicable.
