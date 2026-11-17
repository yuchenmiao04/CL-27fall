#!/usr/bin/env python3
"""Read-only, CPU-only diagnostics for one frozen-basis development trial.

``tune_frozen_basis.py`` currently saves parameters and diagnostics, not a
numerical core checkpoint. Reconstruct its state by replaying the same cached
fitting-class arrivals and warm starts. Then evaluate one EM mapping on copies
of the final moments. No ViT, raw images, official test split, parameter search,
or deployed learner is involved. Only a new diagnostic JSON is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cl27.cores.shared_basis import SharedPredictorBasis, _log_likelihood
from tune_frozen_basis import (
    DATASETS,
    PARAMETERS,
    array_record,
    canonical_hash,
    file_hash,
    load_cache,
)


def numeric_identity(core):
    """Hash the numerical learner plus configuration/diagnostics without mutation."""
    state = core.state_dict()
    arrays = {
        name: array_record(value)
        for name, value in state.items()
        if isinstance(value, np.ndarray)
    }
    scalar = {
        name: value for name, value in state.items() if not isinstance(value, np.ndarray)
    }
    arrays["weights"] = array_record(core.weights)
    arrays["intercept"] = array_record(core.intercept)
    return canonical_hash({"arrays": arrays, "other": scalar})


def reconstruct_core(arrays, parameters, *, width, stages):
    """Preserve the original stage-wise warm starts; joint refitting differs."""
    unknown = set(parameters) - set(PARAMETERS)
    required = set(PARAMETERS) - {"basis_tolerance"}
    if unknown or not required.issubset(parameters):
        raise ValueError("Expected one complete explicit basis parameter object")
    if isinstance(stages, bool) or not isinstance(stages, int) or not 1 <= stages <= 10:
        raise ValueError("Expected 1--10 completed development stages")
    options = {PARAMETERS[k]: v for k, v in parameters.items()}
    options.setdefault("tol", 1e-6)
    core = SharedPredictorBasis(
        feature_dim=arrays["fit_features"].shape[1], seed=1993, **options
    )
    for task in range(stages):
        mask = (arrays["fit_labels"] >= task * width) & (
            arrays["fit_labels"] < (task + 1) * width
        )
        core.observe(arrays["fit_features"][mask], arrays["fit_labels"][mask])
    return core


def top1(scores):
    """Use the same FP32 CPU tie convention as the development evaluator."""
    import torch

    return torch.topk(
        torch.from_numpy(np.asarray(scores)).float(),
        k=min(5, scores.shape[1]),
        dim=1,
    )[1][:, 0].numpy()


def relative_change(after, before):
    return float(np.linalg.norm(after - before) / max(np.linalg.norm(before), 1e-15))


def diagnose_core(core, holdout_features, holdout_labels, *, width):
    """Evaluate a single constrained M-step on copies; do not update ``core``."""
    before_hash = numeric_identity(core)
    if not core.n_samples:
        raise ValueError("A fitted core is required")
    seen = len(core.classes)
    if core.classes != list(range(seen)):
        raise ValueError("Expected remapped contiguous campaign classes")
    features = np.asarray(holdout_features, dtype=np.float64)
    labels = np.asarray(holdout_labels)
    if features.ndim != 2 or features.shape[1] != core.feature_dim:
        raise ValueError("Invalid holdout feature shape")
    if labels.shape != (len(features),) or labels.dtype.kind not in "iu":
        raise ValueError("Expected integer holdout labels")
    if not len(labels) or labels.min() < 0 or labels.max() >= seen:
        raise ValueError("Holdout must contain only already seen classes")

    centered = core._center(core.G, core.C, core.feature_sum, core.counts, core.n_samples)
    gram, cross, mean_x, frequencies, trace_yy, correction = centered
    covariance = core.Sigma
    likelihood, mean, posterior_covariance = _log_likelihood(
        gram, cross, covariance, core.noise_variance, trace_yy, core.n_samples - 1
    )
    q = (posterior_covariance + mean @ mean.T / seen)
    q = (q + q.T) * 0.5
    values, vectors = np.linalg.eigh(q)
    order = np.argsort(values, kind="stable")[::-1]
    values, vectors = values[order], vectors[:, order]
    cap = min(core.rank, core.feature_dim, max(0, seen - 1))
    next_d = vectors[:, :cap] * np.sqrt(
        np.maximum(values[:cap] - core.epsilon, 0.0)
    )[None, :]
    next_covariance = core.epsilon * np.eye(core.feature_dim) + next_d @ next_d.T
    next_likelihood, next_mean, _ = _log_likelihood(
        gram, cross, next_covariance, core.noise_variance, trace_yy, core.n_samples - 1
    )
    intercept = frequencies - mean_x @ mean
    next_intercept = frequencies - mean_x @ next_mean
    scores = features @ mean + intercept
    next_scores = features @ next_mean + next_intercept
    predictions, next_predictions = top1(scores), top1(next_scores)
    correct, next_correct = predictions == labels, next_predictions == labels
    flipped = predictions != next_predictions

    loadings = core.D
    if loadings.shape[1]:
        basis, singular, _ = np.linalg.svd(loadings, full_matrices=False)
        energies = singular**2
        active = energies > (
            np.finfo(np.float64).eps
            * max(1.0, float(energies.max()))
            * core.feature_dim
        )
        basis = basis[:, active]
        outside = mean - basis @ (basis.T @ mean)
    else:
        energies = np.empty(0)
        outside = mean.copy()
    energy_ratios = energies / core.epsilon
    old = labels < seen - width
    split = {}
    for name, mask in [("all", np.ones(len(labels), dtype=bool)), ("old", old), ("current", ~old)]:
        split[name] = {
            "samples": int(mask.sum()),
            "accuracy_before": float(correct[mask].mean()) if mask.any() else None,
            "accuracy_after_one_em_map": float(next_correct[mask].mean()) if mask.any() else None,
            "prediction_flips": int((flipped & mask).sum()),
            "corrected": int((~correct & next_correct & mask).sum()),
            "broken": int((correct & ~next_correct & mask).sum()),
        }
    gain = float(next_likelihood - likelihood)
    next_value = float(values[cap]) if cap < len(values) else None
    result = {
        "samples": core.n_samples,
        "classes": seen,
        "rank_limit": core.rank,
        "class_contrast_rank_cap": cap,
        "effective_rank": core.effective_rank,
        "epsilon": core.epsilon,
        "noise_variance": core.noise_variance,
        "isotropic_residual_ridge_lambda": core.noise_variance / core.epsilon,
        "final_em_diagnostics": core.diagnostics,
        "one_em_map": {
            "covariance_relative_fro_change": relative_change(next_covariance, covariance),
            "posterior_mean_relative_fro_change": relative_change(next_mean, mean),
            "holdout_score_relative_fro_change": relative_change(next_scores, scores),
            "holdout_score_max_abs_change": float(np.max(np.abs(next_scores - scores))),
            "marginal_log_likelihood_before": float(likelihood),
            "marginal_log_likelihood_after": float(next_likelihood),
            "marginal_log_likelihood_gain": gain,
            "gain_per_response": gain / (seen * (core.n_samples - 1)),
            "q_first_unretained_eigenvalue": next_value,
            "q_first_unretained_over_epsilon": next_value / core.epsilon if next_value is not None else None,
            "q_eigenvalues_above_floor": int(np.sum(values > core.epsilon * (1.0 + 1e-10))),
            "q_retained_eigenvalues": values[:cap].tolist(),
        },
        "loading_energy_over_epsilon": {
            "values_descending": energy_ratios.tolist(),
            "maximum": float(energy_ratios.max()) if len(energy_ratios) else None,
            "median": float(np.median(energy_ratios)) if len(energy_ratios) else None,
            "minimum": float(energy_ratios.min()) if len(energy_ratios) else None,
            "shared_direction_ridge_lambdas": (core.noise_variance / (core.epsilon + energies)).tolist(),
        },
        "posterior_mean_energy_outside_loading_span_fraction": float(
            np.linalg.norm(outside) ** 2 / max(np.linalg.norm(mean) ** 2, 1e-30)
        ),
        "holdout": split,
        "centered_gram_roundoff_correction_fro": correction,
    }
    after_hash = numeric_identity(core)
    if before_hash != after_hash:
        raise RuntimeError("Read-only diagnostic mutated its input learner")
    result["read_only_state_verified"] = True
    result["input_numeric_state_sha256"] = before_hash
    return result


def run(*, cache_dir, dataset, trial_dir, output_json, cpu_threads=4):
    import torch

    if cpu_threads < 1:
        raise ValueError("CPU threads must be positive")
    torch.set_num_threads(cpu_threads)
    trial_dir, output = Path(trial_dir), Path(output_json)
    if output.exists():
        raise FileExistsError("Use a new diagnostic output file")
    result_path, parameters_path = trial_dir / "results.json", trial_dir / "parameters.json"
    trial = json.loads(result_path.read_text())
    parameters = json.loads(parameters_path.read_text())
    if not (
        trial.get("method") == "shared_basis"
        and trial.get("dataset") == dataset
        and trial.get("phase") == "development"
        and trial.get("validation_only") is True
        and trial.get("official_test_read") is False
        and trial.get("evaluation_source") == "training_holdout"
        and trial.get("parameters") == parameters
        and trial.get("seed") == 1993
    ):
        raise ValueError("Use a completed training-holdout development trial")
    core_path = ROOT / "cl27/cores/shared_basis.py"
    if trial.get("code_sha256", {}).get("cl27/cores/shared_basis.py") != file_hash(core_path):
        raise ValueError("Core source differs from the original trial; replay is not exact")
    metadata, arrays = load_cache(cache_dir, dataset)
    if (
        trial.get("cache_identity_sha256") != metadata["cache_identity_sha256"]
        or trial.get("data_fingerprint") != metadata["data_manifest"]["fingerprint"]
    ):
        raise ValueError("Development cache identity differs from the original trial")
    for value in arrays.values():
        value.flags.writeable = False
    stages = trial["metrics"]["completed_tasks"]
    if len(trial["stages"]) != stages or len(trial["task_diagnostics"]) != stages:
        raise ValueError("Original trial has inconsistent completed-stage records")
    began = time.perf_counter()
    width = metadata["protocol"]["increment"]
    core = reconstruct_core(arrays, parameters, width=width, stages=stages)
    heldout = arrays["holdout_labels"] < stages * width
    diagnostic = diagnose_core(
        core, arrays["holdout_features"][heldout], arrays["holdout_labels"][heldout], width=width
    )
    if not np.isclose(
        diagnostic["holdout"]["all"]["accuracy_before"],
        trial["stages"][-1]["accuracy"],
        atol=1e-12,
        rtol=0,
    ):
        raise RuntimeError("Reconstructed final validation accuracy differs from the original trial")
    original_likelihood = trial["task_diagnostics"][-1]["marginal_log_likelihood"]
    if not np.isclose(
        diagnostic["one_em_map"]["marginal_log_likelihood_before"],
        original_likelihood,
        atol=1e-6,
        rtol=1e-10,
    ):
        raise RuntimeError("Reconstructed final likelihood differs from the original trial")
    report = {
        "status": "development_diagnostic_complete",
        "phase": "development",
        "validation_only": True,
        "official_test_read": False,
        "dataset": dataset,
        "seed": 1993,
        "completed_tasks": stages,
        "parameters": parameters,
        "source_state": "Reconstructed original ordered class stream from the verified immutable development cache; original tuner stores no numeric checkpoint",
        "boundary": "CPU-only internal mechanism diagnostic; not formal training, not a formal result, not persistent history supplied to a deployed learner",
        "cache_identity_sha256": metadata["cache_identity_sha256"],
        "trial_results_sha256": file_hash(result_path),
        "trial_parameters_sha256": file_hash(parameters_path),
        "script_sha256": file_hash(Path(__file__)),
        "core_sha256": file_hash(core_path),
        "execution_seconds": time.perf_counter() - began,
        "diagnostic": diagnostic,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "diagnostic_complete": True,
        "dataset": dataset,
        "accuracy_before": diagnostic["holdout"]["all"]["accuracy_before"],
        "one_map_prediction_flips": diagnostic["holdout"]["all"]["prediction_flips"],
        "q_first_unretained_over_epsilon": diagnostic["one_em_map"]["q_first_unretained_over_epsilon"],
    }))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--trial-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
