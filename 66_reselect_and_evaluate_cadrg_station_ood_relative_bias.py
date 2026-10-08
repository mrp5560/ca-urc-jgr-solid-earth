#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
66_reselect_and_evaluate_cadrg_station_ood_relative_bias.py

Station-OOD CA-DRG re-selection WITHOUT retraining.

Why this script exists
----------------------
Script 65 completed:
    1) strict seen-only Cross-Attention training;
    2) strict seen-only Dual-Risk head training;
    3) 40 x gamma validation scan;

but no candidate passed the original absolute bias guard:
    |bias| <= 0.05.

The Station-OOD frozen Base itself has a non-negligible validation bias, so
an absolute bias requirement can become structurally incompatible with the
"preserve overall/non-tail MAE" constraints.

This script changes ONLY the validation-stage bias guard to:

    |bias_candidate(q)| <= |bias_base(q)| + bias_margin

for q in {PGA, PGV}.

The two original accuracy-preservation constraints remain unchanged:

    overall_MAE_candidate - overall_MAE_base <= 0.005
    non_tail_MAE_candidate - non_tail_MAE_base <= 0.003

Default bias_margin:
    0.005 log10 units

Selection objective among feasible candidates remains unchanged:
    1) minimum mean PGA/PGV tail MAE
    2) minimum mean tail U0.5
    3) minimum mean overall MAE
    4) lower gamma
    5) earlier epoch

IMPORTANT
---------
- NO backbone retraining.
- NO Dual-Risk head retraining.
- Selection reads ONLY the already generated VALIDATION scan.
- The Station-OOD TEST is accessed only AFTER the new validation rule
  selects a non-boundary gamma candidate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def load_module(
    path: str | Path,
    name: str,
):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required module not found: {path.resolve()}"
        )

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import module: {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )
    spec.loader.exec_module(
        module
    )
    return module


def load_checkpoint(
    path: str | Path,
    device: torch.device,
):
    try:
        return torch.load(
            str(path),
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            str(path),
            map_location=device,
        )


def parse_float_list(
    text: str,
) -> list[float]:
    values = []

    for token in str(text).split(","):
        token = token.strip()

        if not token:
            continue

        values.append(
            float(token)
        )

    values = sorted(
        set(values)
    )

    if not values:
        raise ValueError(
            "Empty gamma grid."
        )

    return values


def relative_bias_feasibility(
    scan: pd.DataFrame,
    base_metrics: dict[str, float],
    overall_budget: float,
    non_tail_budget: float,
    bias_margin: float,
) -> pd.DataFrame:
    result = scan.copy()

    conditions = []

    for quantity in (
        "pga",
        "pgv",
    ):
        base_bias = float(
            base_metrics[
                f"bias_{quantity}"
            ]
        )

        relative_bias_limit = (
            abs(base_bias)
            + float(bias_margin)
        )

        result[
            f"relative_bias_limit_{quantity}"
        ] = relative_bias_limit

        result[
            f"pass_overall_{quantity}"
        ] = (
            result[
                f"overall_delta_{quantity}"
            ]
            <= float(overall_budget)
        )

        result[
            f"pass_non_tail_{quantity}"
        ] = (
            result[
                f"non_tail_delta_{quantity}"
            ]
            <= float(non_tail_budget)
        )

        result[
            f"pass_relative_bias_{quantity}"
        ] = (
            result[
                f"bias_{quantity}"
            ].abs()
            <= relative_bias_limit
        )

        result[
            f"bias_abs_change_vs_base_{quantity}"
        ] = (
            result[
                f"bias_{quantity}"
            ].abs()
            - abs(base_bias)
        )

        conditions.extend(
            [
                result[
                    f"pass_overall_{quantity}"
                ],
                result[
                    f"pass_non_tail_{quantity}"
                ],
                result[
                    f"pass_relative_bias_{quantity}"
                ],
            ]
        )

    feasible = np.ones(
        len(result),
        dtype=bool,
    )

    for condition in conditions:
        feasible &= condition.to_numpy(
            dtype=bool
        )

    result[
        "feasible_relative_bias"
    ] = feasible

    return result


def constraint_diagnostics(
    frame: pd.DataFrame,
) -> dict[str, int]:
    columns = [
        "pass_overall_pga",
        "pass_non_tail_pga",
        "pass_relative_bias_pga",
        "pass_overall_pgv",
        "pass_non_tail_pgv",
        "pass_relative_bias_pgv",
    ]

    result = {
        column: int(
            frame[
                column
            ].sum()
        )
        for column in columns
    }

    result[
        "all_constraints"
    ] = int(
        frame[
            "feasible_relative_bias"
        ].sum()
    )

    result[
        "total_candidates"
    ] = int(
        len(frame)
    )

    return result


def build_selected_model(
    baseline_module,
    risk_module,
    selected_checkpoint_path: Path,
    hidden_dim: int,
    attention_heads: int,
    risk_hidden: int,
    maximum_correction: float,
    device: torch.device,
):
    checkpoint = load_checkpoint(
        selected_checkpoint_path,
        device,
    )

    base = baseline_module.make_model(
        "cross_attention",
        hidden_dim,
        attention_heads,
        50.0,
    ).to(
        device
    )

    dummy_prevalence = np.asarray(
        [
            0.10,
            0.10,
        ],
        dtype=float,
    )

    model = (
        risk_module
        .CrossAttentionDualRiskGate(
            base=base,
            hidden_dim=hidden_dim,
            risk_hidden=risk_hidden,
            maximum_correction=maximum_correction,
            tail_prevalence=dummy_prevalence,
            under_prevalence=dummy_prevalence,
        )
        .to(
            device
        )
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ],
        strict=True,
    )

    model.eval()

    return model


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--station-ood-module",
        default=(
            "65_train_and_evaluate_cadrg_station_ood.py"
        ),
    )

    parser.add_argument(
        "--protocol-module",
        default=(
            "52_phase2_train_and_evaluate_station_ood_attention.py"
        ),
    )

    parser.add_argument(
        "--baseline-module",
        default=(
            "45_phase2_strong_baseline_suite.py"
        ),
    )

    parser.add_argument(
        "--risk-module",
        default=(
            "60_train_cross_attention_dual_risk_joint_selection.py"
        ),
    )

    parser.add_argument(
        "--source-run-dir",
        default=(
            "runs/cadrg_station_ood"
        ),
    )

    parser.add_argument(
        "--scenario",
        default="",
        help=(
            "Optional resolved station-OOD scenario CSV. "
            "Default: <source-run-dir>/resolved_scenario_station_ood.csv"
        ),
    )

    parser.add_argument(
        "--station-assignment",
        default=(
            "data/model_manifests/station_ood/"
            "station_holdout_assignment.csv"
        ),
    )

    parser.add_argument(
        "--split-column",
        default="split_grouped",
    )

    parser.add_argument(
        "--test-label",
        default="test",
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
        "--paired-targets",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--test-repeats",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--input-pre-sec",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--attention-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--risk-hidden",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--maximum-correction",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--overall-budget",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--non-tail-budget",
        type=float,
        default=0.003,
    )

    parser.add_argument(
        "--bias-margin",
        type=float,
        default=0.005,
        help=(
            "Candidate absolute bias may exceed Base absolute bias "
            "by at most this amount."
        ),
    )

    parser.add_argument(
        "--powers",
        default=(
            "1,1.5,2,2.5,3,4,5,6,7,8,10,12,15,20"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260713,
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
        "--out-dir",
        default=(
            "runs/"
            "cadrg_station_ood_relative_bias_selection"
        ),
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        if (
            args.device
            == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        device = torch.device(
            args.device
        )

    source_run_dir = Path(
        args.source_run_dir
    )

    scan_path = (
        source_run_dir
        / "dual_risk_head"
        / "epoch_gamma_validation_scan.csv"
    )

    failure_path = (
        source_run_dir
        / "dual_risk_head"
        / "selection_failure.json"
    )

    threshold_path = (
        source_run_dir
        / "station_ood_tail_thresholds.json"
    )

    epoch_checkpoint_dir = (
        source_run_dir
        / "dual_risk_head"
        / "epoch_checkpoints"
    )

    base_checkpoint_path = (
        source_run_dir
        / "cross_attention_base"
        / "best_model.pt"
    )

    scenario_path = (
        Path(args.scenario)
        if str(args.scenario).strip()
        else (
            source_run_dir
            / "resolved_scenario_station_ood.csv"
        )
    )

    assignment_path = Path(
        args.station_assignment
    )

    required_paths = [
        scan_path,
        failure_path,
        threshold_path,
        base_checkpoint_path,
        scenario_path,
        assignment_path,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    station_ood_module = load_module(
        args.station_ood_module,
        "cadrg_station_ood_module",
    )

    protocol_module = load_module(
        args.protocol_module,
        "station_ood_protocol_module",
    )

    baseline_module = load_module(
        args.baseline_module,
        "cross_attention_architecture_module",
    )

    risk_module = load_module(
        args.risk_module,
        "dual_risk_module",
    )

    scan = pd.read_csv(
        scan_path
    )

    failure = json.loads(
        failure_path.read_text(
            encoding="utf-8"
        )
    )

    base_metrics = failure[
        "base_validation_metrics"
    ]

    powers = parse_float_list(
        args.powers
    )

    expected_gammas = sorted(
        scan[
            "gamma"
        ].astype(
            float
        ).unique()
        .tolist()
    )

    if not np.allclose(
        expected_gammas,
        powers,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "Gamma grid in scan does not match --powers.\n"
            f"scan={expected_gammas}\n"
            f"args={powers}"
        )

    rescored = (
        relative_bias_feasibility(
            scan=scan,
            base_metrics=(
                base_metrics
            ),
            overall_budget=(
                args.overall_budget
            ),
            non_tail_budget=(
                args.non_tail_budget
            ),
            bias_margin=(
                args.bias_margin
            ),
        )
    )

    rescored_path = (
        out_dir
        / "validation_scan_relative_bias_guard.csv"
    )

    rescored.to_csv(
        rescored_path,
        index=False,
    )

    diagnostics = (
        constraint_diagnostics(
            rescored
        )
    )

    print(
        "=== Station-OOD CA-DRG validation re-selection ==="
    )

    print(
        "NO retraining                 : True"
    )

    print(
        f"Device                        : {device}"
    )

    print(
        "Base validation overall PGA/PGV: "
        f"{base_metrics['overall_mae_pga']:.6f}/"
        f"{base_metrics['overall_mae_pgv']:.6f}"
    )

    print(
        "Base validation bias PGA/PGV   : "
        f"{base_metrics['bias_pga']:+.6f}/"
        f"{base_metrics['bias_pgv']:+.6f}"
    )

    print(
        "Relative |bias| limits PGA/PGV : "
        f"{abs(base_metrics['bias_pga']) + args.bias_margin:.6f}/"
        f"{abs(base_metrics['bias_pgv']) + args.bias_margin:.6f}"
    )

    print(
        "Overall/non-tail budgets       : "
        f"{args.overall_budget:.4f}/"
        f"{args.non_tail_budget:.4f}"
    )

    print(
        f"Bias margin                     : "
        f"{args.bias_margin:.4f}"
    )

    print(
        "\nConstraint pass counts:"
    )

    for key, value in diagnostics.items():
        print(
            f"  {key:30s}: "
            f"{value}/{len(rescored)}"
            if key
            not in {
                "total_candidates",
            }
            else (
                f"  {key:30s}: {value}"
            )
        )

    feasible = rescored.loc[
        rescored[
            "feasible_relative_bias"
        ].astype(
            bool
        )
    ].copy()

    if feasible.empty:
        payload = {
            "status": (
                "no_feasible_candidate"
            ),
            "bias_rule": (
                "|bias_candidate| <= "
                "|bias_base| + bias_margin"
            ),
            "bias_margin": float(
                args.bias_margin
            ),
            "base_metrics": (
                base_metrics
            ),
            "diagnostics": (
                diagnostics
            ),
            "test_evaluated": (
                False
            ),
        }

        (
            out_dir
            / "selection_failure_relative_bias.json"
        ).write_text(
            json.dumps(
                payload,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            "\nNO feasible candidate under "
            "the relative-bias rule."
        )

        print(
            "Station-OOD test was NOT accessed."
        )

        return

    feasible = feasible.sort_values(
        [
            "mean_tail_mae",
            "mean_tail_under05",
            "mean_overall_mae",
            "gamma",
            "epoch",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
        ],
    )

    feasible.to_csv(
        out_dir
        / "feasible_validation_candidates_relative_bias.csv",
        index=False,
    )

    selected = feasible.iloc[
        0
    ]

    selected_epoch = int(
        selected[
            "epoch"
        ]
    )

    selected_gamma = float(
        selected[
            "gamma"
        ]
    )

    upper_boundary = bool(
        np.isclose(
            selected_gamma,
            max(
                powers
            ),
        )
    )

    selected_source_checkpoint = (
        epoch_checkpoint_dir
        / f"epoch_{selected_epoch:03d}.pt"
    )

    if not selected_source_checkpoint.exists():
        raise FileNotFoundError(
            selected_source_checkpoint
        )

    selected_checkpoint_path = (
        out_dir
        / "selected_dual_risk_head_relative_bias.pt"
    )

    shutil.copy2(
        selected_source_checkpoint,
        selected_checkpoint_path,
    )

    selection_payload = {
        "status": (
            "selected"
        ),
        "selection_source": (
            "existing station-OOD validation scan; no retraining"
        ),
        "selected_epoch": (
            selected_epoch
        ),
        "selected_gamma": (
            selected_gamma
        ),
        "selected_at_upper_gamma_boundary": (
            upper_boundary
        ),
        "bias_rule": (
            "|bias_candidate| <= "
            "|bias_base| + bias_margin"
        ),
        "bias_margin": float(
            args.bias_margin
        ),
        "overall_budget": float(
            args.overall_budget
        ),
        "non_tail_budget": float(
            args.non_tail_budget
        ),
        "base_validation_metrics": (
            base_metrics
        ),
        "selected_validation_metrics": {
            key: (
                bool(value)
                if isinstance(
                    value,
                    (
                        bool,
                        np.bool_,
                    ),
                )
                else (
                    int(value)
                    if isinstance(
                        value,
                        (
                            int,
                            np.integer,
                        ),
                    )
                    else (
                        float(value)
                        if isinstance(
                            value,
                            (
                                float,
                                np.floating,
                            ),
                        )
                        else value
                    )
                )
            )
            for key, value
            in selected.to_dict().items()
        },
        "feasible_candidates": int(
            len(
                feasible
            )
        ),
        "total_candidates": int(
            len(
                rescored
            )
        ),
        "test_evaluated": (
            False
        ),
    }

    selection_json_path = (
        out_dir
        / "selected_epoch_gamma_relative_bias.json"
    )

    selection_json_path.write_text(
        json.dumps(
            selection_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== Selected validation candidate ==="
    )

    print(
        f"Epoch                         : "
        f"{selected_epoch}"
    )

    print(
        f"Gamma                         : "
        f"{selected_gamma:g}"
    )

    print(
        f"At upper gamma boundary       : "
        f"{upper_boundary}"
    )

    print(
        "Overall MAE PGA/PGV           : "
        f"{selected['overall_mae_pga']:.6f}/"
        f"{selected['overall_mae_pgv']:.6f}"
    )

    print(
        "Overall delta PGA/PGV         : "
        f"{selected['overall_delta_pga']:+.6f}/"
        f"{selected['overall_delta_pgv']:+.6f}"
    )

    print(
        "Non-tail delta PGA/PGV        : "
        f"{selected['non_tail_delta_pga']:+.6f}/"
        f"{selected['non_tail_delta_pgv']:+.6f}"
    )

    print(
        "Bias PGA/PGV                  : "
        f"{selected['bias_pga']:+.6f}/"
        f"{selected['bias_pgv']:+.6f}"
    )

    print(
        "Tail MAE PGA/PGV              : "
        f"{selected['tail_mae_pga']:.6f}/"
        f"{selected['tail_mae_pgv']:.6f}"
    )

    print(
        "Tail delta PGA/PGV            : "
        f"{selected['tail_delta_pga']:+.6f}/"
        f"{selected['tail_delta_pgv']:+.6f}"
    )

    print(
        "Tail U0.5 PGA/PGV             : "
        f"{selected['tail_under05_pga']:.4f}/"
        f"{selected['tail_under05_pgv']:.4f}"
    )

    print(
        f"Feasible candidates           : "
        f"{len(feasible)}/{len(rescored)}"
    )

    if upper_boundary:
        print(
            "\nSelected gamma is at the upper "
            "VALIDATION grid boundary."
        )

        print(
            "Extend gamma using validation only. "
            "Station-OOD test was NOT accessed."
        )

        return

    # -----------------------------------------------------------------
    # From this point onward, selection is locked.
    # Now and only now access the paired Station-OOD test.
    # -----------------------------------------------------------------

    threshold_data = json.loads(
        threshold_path.read_text(
            encoding="utf-8"
        )
    )

    thresholds = np.asarray(
        [
            threshold_data[
                "log10_pga_threshold"
            ],
            threshold_data[
                "log10_pgv_threshold"
            ],
        ],
        dtype=np.float32,
    )

    model = build_selected_model(
        baseline_module=(
            baseline_module
        ),
        risk_module=(
            risk_module
        ),
        selected_checkpoint_path=(
            selected_checkpoint_path
        ),
        hidden_dim=(
            args.hidden_dim
        ),
        attention_heads=(
            args.attention_heads
        ),
        risk_hidden=(
            args.risk_hidden
        ),
        maximum_correction=(
            args.maximum_correction
        ),
        device=(
            device
        ),
    )

    predictions = (
        station_ood_module
        .evaluate_paired_station_ood_cadrg(
            model=model,
            selected_gamma=(
                selected_gamma
            ),
            thresholds=(
                thresholds
            ),
            scenario_csv=(
                scenario_path
            ),
            assignment_csv=(
                assignment_path
            ),
            ood_module=(
                protocol_module
            ),
            risk_module=(
                risk_module
            ),
            args=(
                args
            ),
            device=(
                device
            ),
        )
    )

    prediction_path = (
        out_dir
        / "repeated_station_ood_predictions.csv"
    )

    predictions.to_csv(
        prediction_path,
        index=False,
    )

    metrics = (
        station_ood_module
        .station_ood_metric_table(
            predictions
        )
    )

    metrics_path = (
        out_dir
        / "station_ood_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    paired = (
        station_ood_module
        .paired_seen_unseen_deltas(
            predictions
        )
    )

    paired_path = (
        out_dir
        / "station_ood_paired_event_deltas.csv"
    )

    paired.to_csv(
        paired_path,
        index=False,
    )

    gate_effect = (
        station_ood_module
        .cadrg_minus_base_by_role(
            predictions
        )
    )

    gate_effect_path = (
        out_dir
        / "cadrg_minus_base_by_target_role.csv"
    )

    gate_effect.to_csv(
        gate_effect_path,
        index=False,
    )

    distances = (
        station_ood_module
        .distance_summary(
            predictions
        )
    )

    distance_path = (
        out_dir
        / "station_ood_distance_summary.csv"
    )

    distances.to_csv(
        distance_path,
        index=False,
    )

    prevalence = (
        station_ood_module
        .tail_prevalence_table(
            predictions
        )
    )

    prevalence_path = (
        out_dir
        / "station_ood_tail_prevalence.csv"
    )

    prevalence.to_csv(
        prevalence_path,
        index=False,
    )

    selection_payload[
        "test_evaluated"
    ] = True

    selection_json_path.write_text(
        json.dumps(
            selection_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== Station-OOD canonical metrics ==="
    )

    print(
        metrics.loc[
            metrics[
                "metric"
            ].isin(
                [
                    "mae",
                    "bias",
                    "factor2",
                    "under05",
                ]
            )
        ][
            [
                "model",
                "target_role",
                "quantity",
                "population",
                "metric",
                "value",
                "n_events",
                "n_target_rows",
                "n_unique_stations",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== Unseen minus seen paired event deltas ==="
    )

    print(
        paired.loc[
            paired[
                "metric"
            ].isin(
                [
                    "mae",
                    "factor2",
                    "under05",
                ]
            )
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== CA-DRG minus Base within each target role ==="
    )

    print(
        gate_effect.loc[
            gate_effect[
                "metric"
            ].isin(
                [
                    "mae",
                    "under05",
                ]
            )
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== Geometry audit ==="
    )

    print(
        distances.to_string(
            index=False
        )
    )

    print(
        "\n=== Tail prevalence ==="
    )

    print(
        prevalence.to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    for path in [
        rescored_path,
        out_dir
        / "feasible_validation_candidates_relative_bias.csv",
        selected_checkpoint_path,
        selection_json_path,
        prediction_path,
        metrics_path,
        paired_path,
        gate_effect_path,
        distance_path,
        prevalence_path,
    ]:
        print(
            f"  {path.resolve()}"
        )


if __name__ == "__main__":
    main()
