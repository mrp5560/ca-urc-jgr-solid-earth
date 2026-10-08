 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
35_run_section_4_12_generalization_pipeline.py

One-command orchestration for Section 4.12:

4.12.1 Chronological extrapolation to 2021-2024
4.12.2 Ridgecrest sequence-level OOD evaluation

The pipeline keeps the following order strictly separated:

    2010-2018
        -> train base
        -> train tail-risk head

    2019 non-Ridgecrest + 2020
        -> validation / checkpoint selection
        -> gate-power gamma selection

    freeze everything
        -> 2021-2024 chronological test
        -> Ridgecrest OOD test

Required scripts in the same project directory
----------------------------------------------
22_build_clean_time_and_ridgecrest_splits.py
21_train_time_extrapolation_mean_base.py
34_train_time_clean_tail_risk_gated.py
19_scan_gate_power_posthoc.py
20_evaluate_locked_power_gated_test.py

No test result is used to select checkpoint, tail threshold, or gamma.

Example
-------
python 35_run_section_4_12_generalization_pipeline.py ^
  --project-root "F:\\数据集\\世界模型\\causal_seisfield_data_scripts\\causal_seisfield_data_scripts" ^
  --device cuda
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def run_command(
    command: Sequence[str],
    cwd: Path,
) -> None:
    print("\n" + "=" * 100)
    print("RUN:")
    print(
        subprocess.list2cmdline(
            list(
                command
            )
        )
    )
    print("=" * 100)

    subprocess.run(
        list(
            command
        ),
        cwd=str(
            cwd
        ),
        check=True,
    )


def require_file(
    path: Path,
    label: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path.resolve()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        default=".",
    )

    parser.add_argument(
        "--source-manifest",
        default=(
            "data/scedc/model_manifests/"
            "scenario_t0_5s_k5.csv"
        ),
    )

    parser.add_argument(
        "--time-manifest",
        default=(
            "data/scedc/model_manifests/"
            "scenario_t0_5s_k5_time_clean.csv"
        ),
    )

    parser.add_argument(
        "--split-column",
        default="split_time_clean",
    )

    parser.add_argument(
        "--ridgecrest-sequence-group",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--t0-sec",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--input-stations",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--target-stations",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--input-pre-sec",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--base-epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--tail-epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--training-validation-repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--gate-validation-repeats",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--test-repeats",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260713,
    )

    parser.add_argument(
        "--skip-existing",
        default=True,
        help=(
            "Skip a stage if its key expected output already exists. "
            "Use only when resuming the exact same frozen experiment."
        ),
    )

    parser.add_argument(
        "--split-script",
        default="22_build_clean_time_and_ridgecrest_splits.py",
    )

    parser.add_argument(
        "--base-script",
        default="21_train_time_extrapolation_mean_base.py",
    )

    parser.add_argument(
        "--tail-script",
        default="34_train_time_clean_tail_risk_gated_fixed.py",
    )

    parser.add_argument(
        "--gate-scan-script",
        default="19_scan_gate_power_posthoc.py",
    )

    parser.add_argument(
        "--evaluation-script",
        default="20_evaluate_locked_power_gated_test.py",
    )

    parser.add_argument(
        "--base-out-dir",
        default="runs/time_clean_mean_base_t0_5s_k5",
    )

    parser.add_argument(
        "--tail-out-dir",
        default="runs/time_clean_tail_risk_gated_t0_5s_k5",
    )

    parser.add_argument(
        "--gate-out-dir",
        default="runs/time_clean_gate_power_scan_validation_extended",
    )

    parser.add_argument(
        "--chronological-out-dir",
        default="runs/time_clean_locked_test_2021_2024_gamma5p5",
    )

    parser.add_argument(
        "--ridgecrest-out-dir",
        default="runs/time_clean_ridgecrest_ood_gamma5p5",
    )

    args = parser.parse_args()

    root = Path(
        args.project_root
    ).resolve()

    python_executable = Path(
        sys.executable
    ).resolve()

    source_manifest = (
        root
        / args.source_manifest
    )

    time_manifest = (
        root
        / args.time_manifest
    )

    split_script = (
        root
        / args.split_script
    )

    base_script = (
        root
        / args.base_script
    )

    tail_script = (
        root
        / args.tail_script
    )

    gate_scan_script = (
        root
        / args.gate_scan_script
    )

    evaluation_script = (
        root
        / args.evaluation_script
    )

    for path, label in (
        (
            source_manifest,
            "source manifest",
        ),
        (
            split_script,
            "split script",
        ),
        (
            base_script,
            "base-training script",
        ),
        (
            tail_script,
            "tail-risk training script",
        ),
        (
            gate_scan_script,
            "gate-power scan script",
        ),
        (
            evaluation_script,
            "locked evaluation script",
        ),
    ):
        require_file(
            path,
            label,
        )

    base_out = (
        root
        / args.base_out_dir
    )

    tail_out = (
        root
        / args.tail_out_dir
    )

    gate_out = (
        root
        / args.gate_out_dir
    )

    chronological_out = (
        root
        / args.chronological_out_dir
    )

    ridgecrest_out = (
        root
        / args.ridgecrest_out_dir
    )

    # -----------------------------------------------------------------
    # Stage 1: clean chronology + Ridgecrest OOD split.
    # -----------------------------------------------------------------
    split_key_output = time_manifest

    if (
        not args.skip_existing
        or not split_key_output.exists()
    ):
        run_command(
            [
                str(
                    python_executable
                ),
                str(
                    split_script
                ),
                "--manifest",
                str(
                    source_manifest
                ),
                "--output",
                str(
                    time_manifest
                ),
                "--output-column",
                args.split_column,
                "--ridgecrest-sequence-group",
                str(
                    args.ridgecrest_sequence_group
                ),
            ],
            cwd=root,
        )

    require_file(
        time_manifest,
        "chronology-safe manifest",
    )

    # -----------------------------------------------------------------
    # Stage 2: chronology-safe frozen mean base.
    # -----------------------------------------------------------------
    base_checkpoint = (
        base_out
        / "best_model.pt"
    )

    threshold_json = (
        base_out
        / (
            "tail_thresholds_"
            "q0.90_"
            f"t0_{args.t0_sec}s.json"
        )
    )

    if (
        not args.skip_existing
        or not base_checkpoint.exists()
        or not threshold_json.exists()
    ):
        run_command(
            [
                str(
                    python_executable
                ),
                str(
                    base_script
                ),
                "--manifest",
                str(
                    time_manifest
                ),
                "--split-column",
                args.split_column,
                "--train-label",
                "train",
                "--validation-label",
                "validation",
                "--t0-sec",
                str(
                    args.t0_sec
                ),
                "--input-stations",
                str(
                    args.input_stations
                ),
                "--target-stations",
                str(
                    args.target_stations
                ),
                "--input-pre-sec",
                str(
                    args.input_pre_sec
                ),
                "--validation-repeats",
                str(
                    args.training_validation_repeats
                ),
                "--tail-quantile",
                "0.90",
                "--epochs",
                str(
                    args.base_epochs
                ),
                "--seed",
                str(
                    args.seed
                ),
                "--device",
                args.device,
                "--out-dir",
                str(
                    base_out
                ),
            ],
            cwd=root,
        )

    require_file(
        base_checkpoint,
        "chronology-safe base checkpoint",
    )

    require_file(
        threshold_json,
        "training-only tail threshold JSON",
    )

    # -----------------------------------------------------------------
    # Stage 3: train frozen-base tail-risk gate.
    # -----------------------------------------------------------------
    tail_checkpoint = (
        tail_out
        / "best_overall.pt"
    )

    validation_predictions = (
        tail_out
        / "validation_predictions_overall.csv"
    )

    if (
        not args.skip_existing
        or not tail_checkpoint.exists()
        or not validation_predictions.exists()
    ):
        run_command(
            [
                str(
                    python_executable
                ),
                str(
                    tail_script
                ),
                "--manifest",
                str(
                    time_manifest
                ),
                "--split-column",
                args.split_column,
                "--train-label",
                "train",
                "--validation-label",
                "validation",
                "--base-checkpoint",
                str(
                    base_checkpoint
                ),
                "--threshold-json",
                str(
                    threshold_json
                ),
                "--t0-sec",
                str(
                    args.t0_sec
                ),
                "--input-stations",
                str(
                    args.input_stations
                ),
                "--target-stations",
                str(
                    args.target_stations
                ),
                "--input-pre-sec",
                str(
                    args.input_pre_sec
                ),
                "--validation-repeats",
                str(
                    args.training_validation_repeats
                ),
                "--final-validation-repeats",
                str(
                    args.gate_validation_repeats
                ),
                "--epochs",
                str(
                    args.tail_epochs
                ),
                "--seed",
                str(
                    args.seed
                ),
                "--device",
                args.device,
                "--out-dir",
                str(
                    tail_out
                ),
            ],
            cwd=root,
        )

    require_file(
        tail_checkpoint,
        "chronology-safe tail-risk checkpoint",
    )

    require_file(
        validation_predictions,
        "chronological validation predictions",
    )

    # -----------------------------------------------------------------
    # Stage 4: select gamma using ONLY chronological validation.
    # -----------------------------------------------------------------
    best_gate_json = (
        gate_out
        / "best_gate_power.json"
    )

    if (
        not args.skip_existing
        or not best_gate_json.exists()
    ):
        run_command(
            [
                str(
                    python_executable
                ),
                str(
                    gate_scan_script
                ),
                "--predictions",
                str(
                    validation_predictions
                ),
                "--powers",
                (
                    "1.0,1.25,1.5,1.75,"
                    "2.0,2.5,3.0,4.0"
                ),
                "--overall-budget",
                "0.015",
                "--non-tail-budget",
                "0.005",
                "--bias-limit",
                "0.05",
                "--bootstrap-repetitions",
                str(
                    args.bootstrap_repetitions
                ),
                "--seed",
                str(
                    args.seed
                ),
                "--out-dir",
                str(
                    gate_out
                ),
            ],
            cwd=root,
        )

    require_file(
        best_gate_json,
        "selected gate-power JSON",
    )

    selection = json.loads(
        best_gate_json.read_text(
            encoding="utf-8"
        )
    )

    if (
        "selected_gate_power"
        not in selection
    ):
        raise KeyError(
            "best_gate_power.json has no selected_gate_power."
        )

    selected_gamma = float(
        selection[
            "selected_gate_power"
        ]
    )

    print(
        "\n"
        + "#" * 100
    )
    print(
        f"FROZEN gamma selected on chronology-safe validation: "
        f"{selected_gamma:g}"
    )
    print(
        "#" * 100
    )

    # -----------------------------------------------------------------
    # Stage 5: one-shot chronological extrapolation 2021-2024.
    # -----------------------------------------------------------------
    chronological_predictions = (
        chronological_out
        / "locked_test_predictions.csv"
    )

    if (
        not args.skip_existing
        or not chronological_predictions.exists()
    ):
        run_command(
            [
                str(
                    python_executable
                ),
                str(
                    evaluation_script
                ),
                "--training-script",
                str(
                    tail_script
                ),
                "--checkpoint",
                str(
                    tail_checkpoint
                ),
                "--threshold-json",
                str(
                    threshold_json
                ),
                "--manifest",
                str(
                    time_manifest
                ),
                "--split-column",
                args.split_column,
                "--split-name",
                "test",
                "--t0-sec",
                str(
                    args.t0_sec
                ),
                "--input-stations",
                str(
                    args.input_stations
                ),
                "--target-stations",
                str(
                    args.target_stations
                ),
                "--input-pre-sec",
                str(
                    args.input_pre_sec
                ),
                "--gate-power",
                str(
                    selected_gamma
                ),
                "--repeats",
                str(
                    args.test_repeats
                ),
                "--bootstrap-repetitions",
                str(
                    args.bootstrap_repetitions
                ),
                "--seed",
                str(
                    args.seed
                ),
                "--device",
                args.device,
                "--out-dir",
                str(
                    chronological_out
                ),
            ],
            cwd=root,
        )

    require_file(
        chronological_predictions,
        "2021-2024 locked predictions",
    )

    # -----------------------------------------------------------------
    # Stage 6: same frozen model on the complete Ridgecrest OOD split.
    # -----------------------------------------------------------------
    ridgecrest_predictions = (
        ridgecrest_out
        / "locked_test_predictions.csv"
    )

    if (
        not args.skip_existing
        or not ridgecrest_predictions.exists()
    ):
        run_command(
            [
                str(
                    python_executable
                ),
                str(
                    evaluation_script
                ),
                "--training-script",
                str(
                    tail_script
                ),
                "--checkpoint",
                str(
                    tail_checkpoint
                ),
                "--threshold-json",
                str(
                    threshold_json
                ),
                "--manifest",
                str(
                    time_manifest
                ),
                "--split-column",
                args.split_column,
                "--split-name",
                "ridgecrest_ood",
                "--t0-sec",
                str(
                    args.t0_sec
                ),
                "--input-stations",
                str(
                    args.input_stations
                ),
                "--target-stations",
                str(
                    args.target_stations
                ),
                "--input-pre-sec",
                str(
                    args.input_pre_sec
                ),
                "--gate-power",
                str(
                    selected_gamma
                ),
                "--repeats",
                str(
                    args.test_repeats
                ),
                "--bootstrap-repetitions",
                str(
                    args.bootstrap_repetitions
                ),
                "--seed",
                str(
                    args.seed
                ),
                "--device",
                args.device,
                "--out-dir",
                str(
                    ridgecrest_out
                ),
            ],
            cwd=root,
        )

    require_file(
        ridgecrest_predictions,
        "Ridgecrest OOD locked predictions",
    )

    # -----------------------------------------------------------------
    # Frozen-experiment audit.
    # -----------------------------------------------------------------
    audit = {
        "project_root": str(
            root
        ),
        "python_executable": str(
            python_executable
        ),
        "split_column": (
            args.split_column
        ),
        "split_definition": {
            "train": "2010-2018",
            "validation": (
                "2019 non-Ridgecrest + 2020"
            ),
            "test": "2021-2024",
            "ridgecrest_ood": (
                "2019 Ridgecrest sequence group"
            ),
        },
        "base_checkpoint": str(
            base_checkpoint
        ),
        "tail_checkpoint": str(
            tail_checkpoint
        ),
        "tail_threshold_json": str(
            threshold_json
        ),
        "gate_selection_json": str(
            best_gate_json
        ),
        "selected_gate_power": float(
            selected_gamma
        ),
        "chronological_test_directory": str(
            chronological_out
        ),
        "ridgecrest_ood_directory": str(
            ridgecrest_out
        ),
        "test_repeats": int(
            args.test_repeats
        ),
        "bootstrap_repetitions": int(
            args.bootstrap_repetitions
        ),
        "selection_policy": (
            "checkpoint, tail thresholds and gamma selected "
            "without accessing 2021-2024 or Ridgecrest OOD"
        ),
    }

    audit_path = (
        root
        / "runs"
        / "section_4_12_generalization_audit.json"
    )

    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 100
    )
    print(
        "SECTION 4.12 GENERALIZATION PIPELINE COMPLETE"
    )
    print(
        "=" * 100
    )
    print(
        f"Selected gamma     : {selected_gamma:g}"
    )
    print(
        f"2021-2024 results  : {chronological_out.resolve()}"
    )
    print(
        f"Ridgecrest OOD     : {ridgecrest_out.resolve()}"
    )
    print(
        f"Audit              : {audit_path.resolve()}"
    )


if __name__ == "__main__":
    main()
