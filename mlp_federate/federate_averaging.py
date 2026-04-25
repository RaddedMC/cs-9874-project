import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

# Ensure project root is importable when this file is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from mlp_train.train import (
        MultiOutputMLP,
        apply_standardizer,
        build_windowed_samples,
        compute_binary_metrics,
        compute_continuous_metrics,
        impute_and_cast,
        infer_state_columns,
        load_persona_frames,
        resolve_device,
        split_by_time_per_persona,
    )
except ModuleNotFoundError:
    from mlp_train.train import (  # type: ignore
        MultiOutputMLP,
        apply_standardizer,
        build_windowed_samples,
        compute_binary_metrics,
        compute_continuous_metrics,
        impute_and_cast,
        infer_state_columns,
        load_persona_frames,
        resolve_device,
        split_by_time_per_persona,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated averaging across selected MLP checkpoints.")
    parser.add_argument("--checkpoints", nargs="*", default=[], help="Explicit list of checkpoint paths.")
    parser.add_argument("--checkpoint-glob", type=str, default=None, help="Glob pattern for checkpoint discovery.")
    parser.add_argument("--aggregation-method", choices=["average", "weighted"], default="average")
    parser.add_argument("--weights", nargs="*", type=float, default=None, help="Weights for weighted aggregation.")
    parser.add_argument("--allow-single", action="store_true", help="Allow a single checkpoint input.")
    parser.add_argument("--stats-divergence-threshold", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default="mlp_federate/artifacts")
    parser.add_argument("--output-name", type=str, default="federated_model.pt")

    parser.add_argument("--evaluate", action="store_true", help="Run one-step evaluation after aggregation.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--file-name", type=str, default="weeks4_seed42.csv")
    parser.add_argument("--persona", type=str, default=None, help="Override persona used for evaluation data loading.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument("--save-eval-predictions", action="store_true")

    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def discover_checkpoints(explicit: Sequence[str], glob_pattern: str | None) -> List[Path]:
    paths: List[Path] = []

    for p in explicit:
        paths.append(Path(p).expanduser().resolve())

    if glob_pattern:
        for p in sorted(Path().glob(glob_pattern)):
            paths.append(p.expanduser().resolve())

    deduped: List[Path] = []
    seen = set()
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    return deduped


def parse_hidden_dims(value: Any) -> Tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    if isinstance(value, str):
        return tuple(int(v) for v in value.split(",") if str(v).strip())
    raise ValueError(f"Unsupported hidden_dims format: {type(value)}")


def has_persona_dummies(feature_columns: Sequence[str]) -> bool:
    return any(col.startswith("persona_") and col != "persona_label" for col in feature_columns)


def tensor_digest(state_dict: Dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tuple(state_dict[key].shape)).encode("utf-8"))
        hasher.update(state_dict[key].detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Normalize wrapper-specific prefixes from state_dict keys.

    Opacus-wrapped modules commonly save keys like "_module.trunk.0.weight".
    Downstream inference expects plain module keys like "trunk.0.weight".
    """
    if not state_dict:
        return state_dict

    def _strip_prefix(key: str) -> str:
        if key.startswith("_module."):
            return key[len("_module.") :]
        if key.startswith("module."):
            return key[len("module.") :]
        return key

    normalized = {_strip_prefix(k): v for k, v in state_dict.items()}

    # Guard against accidental key collisions after normalization.
    if len(normalized) != len(state_dict):
        raise ValueError("State dict key collision detected while normalizing key prefixes.")

    return normalized


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        checkpoint["model_state_dict"] = normalize_state_dict_keys(checkpoint["model_state_dict"])
    return checkpoint


def validate_compatibility(
    checkpoints: Sequence[Tuple[Path, Dict[str, Any]]],
    stats_divergence_threshold: float,
) -> Dict[str, Any]:
    if not checkpoints:
        raise ValueError("No checkpoints provided.")

    reference_path, ref = checkpoints[0]
    report: Dict[str, Any] = {
        "reference_checkpoint": str(reference_path),
        "compatibility_ok": True,
        "warnings": [],
        "validated_fields": {
            "state_dict_keys": True,
            "state_dict_shapes": True,
            "hidden_dims": True,
            "window_size": True,
            "horizon": True,
            "feature_columns": True,
            "binary_targets": True,
            "continuous_targets": True,
            "persona_feature_mode": True,
        },
        "checked_models": [str(p) for p, _ in checkpoints],
    }

    ref_state = ref["model_state_dict"]
    ref_keys = set(ref_state.keys())
    ref_shapes = {k: tuple(v.shape) for k, v in ref_state.items()}
    ref_args = ref.get("args", {})
    ref_hidden_dims = parse_hidden_dims(ref_args.get("hidden_dims", ""))
    ref_window = int(ref_args.get("window_size", 1))
    ref_horizon = int(ref_args.get("horizon", 1))
    ref_features = list(ref["feature_columns"])
    ref_binary = list(ref["binary_targets"])
    ref_cont = list(ref["continuous_targets"])
    ref_has_persona_dummies = has_persona_dummies(ref_features)

    for path, ckpt in checkpoints[1:]:
        state = ckpt["model_state_dict"]
        keys = set(state.keys())
        if keys != ref_keys:
            missing = sorted(ref_keys - keys)
            extra = sorted(keys - ref_keys)
            raise ValueError(
                f"State dict key mismatch for {path}. Missing={missing[:5]}, Extra={extra[:5]}"
            )

        for key in ref_state.keys():
            if tuple(state[key].shape) != ref_shapes[key]:
                raise ValueError(
                    f"Tensor shape mismatch for {path} on key '{key}': "
                    f"{tuple(state[key].shape)} vs {ref_shapes[key]}"
                )

        args = ckpt.get("args", {})
        hidden_dims = parse_hidden_dims(args.get("hidden_dims", ""))
        window = int(args.get("window_size", 1))
        horizon = int(args.get("horizon", 1))

        if hidden_dims != ref_hidden_dims:
            raise ValueError(f"hidden_dims mismatch: {path} has {hidden_dims}, reference has {ref_hidden_dims}")
        if window != ref_window:
            raise ValueError(f"window_size mismatch: {path} has {window}, reference has {ref_window}")
        if horizon != ref_horizon:
            raise ValueError(f"horizon mismatch: {path} has {horizon}, reference has {ref_horizon}")

        features = list(ckpt["feature_columns"])
        binary_targets = list(ckpt["binary_targets"])
        continuous_targets = list(ckpt["continuous_targets"])

        if features != ref_features:
            raise ValueError(
                f"feature_columns mismatch for {path}. "
                f"Count {len(features)} vs {len(ref_features)}"
            )
        if binary_targets != ref_binary:
            raise ValueError(f"binary_targets mismatch for {path}")
        if continuous_targets != ref_cont:
            raise ValueError(f"continuous_targets mismatch for {path}")

        has_dummies = has_persona_dummies(features)
        if has_dummies != ref_has_persona_dummies:
            raise ValueError(
                f"persona feature mode mismatch for {path}. "
                f"Ref has persona dummies={ref_has_persona_dummies}, this checkpoint has={has_dummies}"
            )

    # Warning-only standardizer drift check.
    ref_stats = ref.get("standardizer_stats", {})
    for path, ckpt in checkpoints[1:]:
        ckpt_stats = ckpt.get("standardizer_stats", {})
        for col, ref_vals in ref_stats.items():
            if col not in ckpt_stats:
                continue
            ref_mean = float(ref_vals.get("mean", 0.0))
            ref_std = float(ref_vals.get("std", 1.0))
            other_mean = float(ckpt_stats[col].get("mean", 0.0))
            other_std = float(ckpt_stats[col].get("std", 1.0))

            mean_delta = abs(other_mean - ref_mean) / (abs(ref_mean) + 1e-8)
            std_delta = abs(other_std - ref_std) / (abs(ref_std) + 1e-8)
            if mean_delta > stats_divergence_threshold or std_delta > stats_divergence_threshold:
                report["warnings"].append(
                    {
                        "type": "standardizer_drift",
                        "checkpoint": str(path),
                        "column": col,
                        "relative_mean_delta": mean_delta,
                        "relative_std_delta": std_delta,
                    }
                )

    return report


def normalized_weights(method: str, num_models: int, weights: Sequence[float] | None) -> np.ndarray:
    if method == "average":
        return np.ones(num_models, dtype=np.float64) / float(num_models)

    if weights is None or len(weights) != num_models:
        raise ValueError("Weighted aggregation requires --weights with one value per checkpoint.")

    w = np.array(weights, dtype=np.float64)
    if np.any(w <= 0):
        raise ValueError("All weights must be positive for weighted aggregation.")

    return w / np.sum(w)


def aggregate_state_dicts(
    state_dicts: Sequence[Dict[str, torch.Tensor]],
    weights: np.ndarray,
) -> Dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("No state dicts provided for aggregation.")

    keys = list(state_dicts[0].keys())
    aggregated: Dict[str, torch.Tensor] = {}

    for key in keys:
        tensors = [sd[key].detach().cpu() for sd in state_dicts]
        if tensors[0].is_floating_point():
            acc = torch.zeros_like(tensors[0], dtype=tensors[0].dtype)
            for t, w in zip(tensors, weights):
                acc = acc + t * float(w)
            aggregated[key] = acc
        else:
            same = all(torch.equal(tensors[0], t) for t in tensors[1:])
            if not same:
                raise ValueError(f"Non-floating tensor '{key}' differs across checkpoints and cannot be averaged.")
            aggregated[key] = tensors[0].clone()

    return aggregated


def ensure_persona_dummy_columns(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_columns:
        if col.startswith("persona_") and col != "persona_label":
            persona_name = col[len("persona_") :]
            out[col] = (out["persona_label"].astype(str) == persona_name).astype(np.float32)
    return out


def choose_split(split_name: str, splits) -> pd.DataFrame:
    if split_name == "train":
        return splits.train
    if split_name == "val":
        return splits.val
    return splits.test


def build_model_from_checkpoint(ckpt: Dict[str, Any], device: torch.device) -> MultiOutputMLP:
    ckpt_args = ckpt.get("args", {})
    feature_columns = list(ckpt["feature_columns"])
    window_size = int(ckpt_args.get("window_size", 1))
    hidden_dims = parse_hidden_dims(ckpt_args.get("hidden_dims", "256,128"))
    dropout = float(ckpt_args.get("dropout", 0.2))

    input_dim = len(feature_columns) * window_size
    model = MultiOutputMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_binary_targets=len(ckpt["binary_targets"]),
        num_continuous_targets=len(ckpt["continuous_targets"]),
    ).to(device)
    return model


def forward_batches(model: MultiOutputMLP, x: np.ndarray, batch_size: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    preds_bin: List[np.ndarray] = []
    preds_cont: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = start + batch_size
            xb = torch.from_numpy(x[start:end]).to(device)
            logits, cont = model(xb)
            preds_bin.append(torch.sigmoid(logits).cpu().numpy())
            preds_cont.append(cont.cpu().numpy())

    yb_prob = np.concatenate(preds_bin, axis=0) if preds_bin else np.zeros((0, 0), dtype=np.float32)
    yc_pred = np.concatenate(preds_cont, axis=0) if preds_cont else np.zeros((0, 0), dtype=np.float32)
    return yb_prob, yc_pred


def evaluate_aggregated_model(
    model: MultiOutputMLP,
    reference_ckpt: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    ckpt_args = reference_ckpt.get("args", {})
    active_persona = args.persona if args.persona is not None else ckpt_args.get("persona", "all")

    train_ratio = float(ckpt_args.get("train_ratio", 0.7))
    val_ratio = float(ckpt_args.get("val_ratio", 0.15))
    window_size = int(ckpt_args.get("window_size", 1))
    horizon = int(ckpt_args.get("horizon", 1))

    feature_columns = list(reference_ckpt["feature_columns"])
    binary_targets = list(reference_ckpt["binary_targets"])
    continuous_targets = list(reference_ckpt["continuous_targets"])
    stats = dict(reference_ckpt.get("standardizer_stats", {}))

    df_raw = load_persona_frames(Path(args.data_dir), args.file_name, active_persona)
    _, inferred_bin, inferred_cont = infer_state_columns(df_raw)
    df_clean = impute_and_cast(df_raw, inferred_bin, inferred_cont)

    splits = split_by_time_per_persona(df_clean, train_ratio=train_ratio, val_ratio=val_ratio)
    split_df = choose_split(args.split, splits)
    split_df = ensure_persona_dummy_columns(split_df, feature_columns)
    split_df = apply_standardizer(split_df, stats)

    x, yb_true, yc_true, personas, timestamps = build_windowed_samples(
        split_df,
        feature_columns,
        binary_targets,
        continuous_targets,
        window_size,
        horizon,
    )

    yb_prob, yc_pred = forward_batches(model, x, args.batch_size, device)

    yb_pred = (yb_prob >= float(args.binary_threshold)).astype(np.float32)
    bin_metrics = compute_binary_metrics(
        torch.from_numpy(np.log(np.clip(yb_prob, 1e-6, 1 - 1e-6) / np.clip(1 - yb_prob, 1e-6, 1 - 1e-6))),
        torch.from_numpy(yb_true),
    )
    cont_metrics = compute_continuous_metrics(torch.from_numpy(yc_pred), torch.from_numpy(yc_true))

    results = {
        "split": args.split,
        "persona": active_persona,
        "num_samples": int(len(x)),
        "binary_threshold": float(args.binary_threshold),
        **bin_metrics,
        **cont_metrics,
    }

    pred_df = pd.DataFrame({"persona_label": personas, "target_timestamp": timestamps})
    for i, col in enumerate(binary_targets):
        pred_df[f"true_{col}"] = yb_true[:, i]
        pred_df[f"pred_{col}_prob"] = yb_prob[:, i]
        pred_df[f"pred_{col}"] = yb_pred[:, i]
    for i, col in enumerate(continuous_targets):
        pred_df[f"true_{col}"] = yc_true[:, i]
        pred_df[f"pred_{col}"] = yc_pred[:, i]

    return {
        "metrics": results,
        "predictions": pred_df,
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    selected_paths = discover_checkpoints(args.checkpoints, args.checkpoint_glob)
    if not selected_paths:
        raise ValueError("No checkpoints selected. Use --checkpoints and/or --checkpoint-glob.")
    if len(selected_paths) < 2 and not args.allow_single:
        raise ValueError("At least 2 checkpoints are required. Use --allow-single to bypass.")

    loaded: List[Tuple[Path, Dict[str, Any]]] = [(p, load_checkpoint(p)) for p in selected_paths]

    compatibility = validate_compatibility(
        loaded,
        stats_divergence_threshold=float(args.stats_divergence_threshold),
    )

    model_weights = normalized_weights(args.aggregation_method, len(loaded), args.weights)
    state_dicts = [ckpt["model_state_dict"] for _, ckpt in loaded]
    aggregated_state = aggregate_state_dicts(state_dicts, model_weights)

    ref_path, ref_ckpt = loaded[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_checkpoint = {
        "model_state_dict": aggregated_state,
        "args": ref_ckpt.get("args", {}),
        "feature_columns": ref_ckpt["feature_columns"],
        "state_columns": ref_ckpt.get("state_columns", []),
        "binary_targets": ref_ckpt["binary_targets"],
        "continuous_targets": ref_ckpt["continuous_targets"],
        "standardizer_stats": ref_ckpt.get("standardizer_stats", {}),
        "federated": True,
        "aggregation_method": args.aggregation_method,
        "source_checkpoints": [str(p) for p, _ in loaded],
        "source_weights": model_weights.tolist(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_checkpoint": str(ref_path),
    }

    checkpoint_path = output_dir / args.output_name
    torch.save(output_checkpoint, checkpoint_path)

    aggregation_report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "aggregation_method": args.aggregation_method,
        "weights": model_weights.tolist(),
        "num_models": len(loaded),
        "source_checkpoints": [str(p) for p, _ in loaded],
        "compatibility": compatibility,
        "tensor_count": len(aggregated_state),
        "federated_checkpoint": str(checkpoint_path),
        "federated_state_digest": tensor_digest(aggregated_state),
    }

    report_path = output_dir / "aggregation_report.json"
    report_path.write_text(json.dumps(aggregation_report, indent=2), encoding="utf-8")

    if args.evaluate:
        device = resolve_device(args.device)
        model = build_model_from_checkpoint(output_checkpoint, device)
        model.load_state_dict(aggregated_state)

        evaluation = evaluate_aggregated_model(model, output_checkpoint, args, device)
        metrics_path = output_dir / "evaluation_metrics.json"
        metrics_path.write_text(json.dumps(evaluation["metrics"], indent=2), encoding="utf-8")

        if args.save_eval_predictions:
            pred_path = output_dir / "evaluation_predictions.csv"
            evaluation["predictions"].to_csv(pred_path, index=False)

        print(f"Saved evaluation metrics: {metrics_path}")
        if args.save_eval_predictions:
            print(f"Saved evaluation predictions: {pred_path}")

    print(f"Saved federated checkpoint: {checkpoint_path}")
    print(f"Saved aggregation report: {report_path}")


if __name__ == "__main__":
    main()
