import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from mlp_train.train import (
        MultiOutputMLP,
        apply_standardizer,
        build_windowed_samples,
        impute_and_cast,
        infer_state_columns,
        load_persona_frames,
        resolve_device,
        split_by_time_per_persona,
    )
except ModuleNotFoundError:
    from train import (  # type: ignore
        MultiOutputMLP,
        apply_standardizer,
        build_windowed_samples,
        impute_and_cast,
        infer_state_columns,
        load_persona_frames,
        resolve_device,
        split_by_time_per_persona,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment with smart-home predictions using a trained multi-output MLP checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, default="mlp_train/artifacts/best_model.pt")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--file-name", type=str, default="weeks4_seed42.csv")
    parser.add_argument("--persona", type=str, default=None, help="Override persona; defaults to training persona from checkpoint.")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--mode", type=str, choices=["one_step", "rollout"], default="one_step")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument("--preview-rows", type=int, default=10)
    parser.add_argument("--output-csv", type=str, default=None)

    parser.add_argument("--rollout-persona", type=str, default=None)
    parser.add_argument("--rollout-start-index", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=24)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def load_checkpoint(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def parse_hidden_dims(value) -> List[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        return [int(v) for v in value.split(",") if str(v).strip()]
    return [256, 128]


def build_model(ckpt: Dict, device: torch.device) -> MultiOutputMLP:
    ckpt_args = ckpt.get("args", {})
    hidden_dims = parse_hidden_dims(ckpt_args.get("hidden_dims", "256,128"))
    dropout = float(ckpt_args.get("dropout", 0.2))
    feature_columns = ckpt["feature_columns"]
    window_size = int(ckpt_args.get("window_size", 1))
    input_dim = len(feature_columns) * window_size

    model = MultiOutputMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_binary_targets=len(ckpt["binary_targets"]),
        num_continuous_targets=len(ckpt["continuous_targets"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


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


def forward_predict(
    model: MultiOutputMLP,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    pred_bin: List[np.ndarray] = []
    pred_cont: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = start + batch_size
            xb = torch.from_numpy(x[start:end]).to(device)
            logits, y_cont = model(xb)
            pred_bin.append(torch.sigmoid(logits).cpu().numpy())
            pred_cont.append(y_cont.cpu().numpy())

    bin_arr = np.concatenate(pred_bin, axis=0) if pred_bin else np.zeros((0, 0), dtype=np.float32)
    cont_arr = np.concatenate(pred_cont, axis=0) if pred_cont else np.zeros((0, 0), dtype=np.float32)
    return bin_arr, cont_arr


def binary_metrics(prob: np.ndarray, y_true: np.ndarray, threshold: float) -> Dict[str, float]:
    if y_true.size == 0:
        return {"binary_accuracy": float("nan"), "binary_macro_f1": float("nan")}
    pred = (prob >= threshold).astype(np.float32)
    accuracy = float((pred == y_true).mean())

    f1_list = []
    for j in range(y_true.shape[1]):
        p = pred[:, j]
        y = y_true[:, j]
        tp = np.sum((p == 1) & (y == 1))
        fp = np.sum((p == 1) & (y == 0))
        fn = np.sum((p == 0) & (y == 1))
        denom = 2 * tp + fp + fn
        f1 = 1.0 if denom == 0 else float((2 * tp) / denom)
        f1_list.append(f1)

    return {
        "binary_accuracy": accuracy,
        "binary_macro_f1": float(np.mean(f1_list)) if f1_list else float("nan"),
    }


def continuous_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {"continuous_mae": float("nan"), "continuous_rmse": float("nan")}
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return {"continuous_mae": mae, "continuous_rmse": rmse}


def inverse_scale_continuous(arr: np.ndarray, cols: Sequence[str], stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    if arr.size == 0:
        return arr
    out = arr.copy()
    for i, col in enumerate(cols):
        if col in stats:
            mean = float(stats[col]["mean"])
            std = float(stats[col]["std"])
            out[:, i] = out[:, i] * std + mean
    return out


def build_prediction_frame(
    personas: Sequence[str],
    timestamps: Sequence[str],
    binary_targets: Sequence[str],
    continuous_targets: Sequence[str],
    yb_true: np.ndarray,
    yc_true: np.ndarray,
    yb_prob: np.ndarray,
    yc_pred: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    out = pd.DataFrame({"persona_label": personas, "target_timestamp": timestamps})

    for i, col in enumerate(binary_targets):
        out[f"true_{col}"] = yb_true[:, i]
        out[f"pred_{col}_prob"] = yb_prob[:, i]
        out[f"pred_{col}"] = (yb_prob[:, i] >= threshold).astype(np.int32)

    for i, col in enumerate(continuous_targets):
        out[f"true_{col}"] = yc_true[:, i]
        out[f"pred_{col}"] = yc_pred[:, i]

    return out


def run_rollout(
    model: MultiOutputMLP,
    split_df: pd.DataFrame,
    feature_columns: Sequence[str],
    binary_targets: Sequence[str],
    continuous_targets: Sequence[str],
    window_size: int,
    horizon: int,
    rollout_persona: str,
    rollout_steps: int,
    rollout_start_index: int,
    threshold: float,
    device: torch.device,
) -> pd.DataFrame:
    if horizon != 1:
        raise ValueError("Rollout mode currently supports horizon=1 only.")

    g = split_df[split_df["persona_label"].astype(str) == rollout_persona].copy()
    if g.empty:
        raise ValueError(f"No rows found for rollout_persona={rollout_persona} in the selected split.")

    g = g.sort_values("timestamp").reset_index(drop=True)
    if rollout_start_index is None:
        rollout_start_index = max(window_size - 1, 0)

    if rollout_start_index < window_size - 1:
        raise ValueError("rollout_start_index is too small for the selected window_size.")

    if rollout_start_index + rollout_steps >= len(g):
        raise ValueError("rollout_start_index + rollout_steps exceeds available rows in selected split/persona.")

    sim = g.copy()
    records: List[Dict] = []

    with torch.no_grad():
        current_idx = rollout_start_index
        for step in range(rollout_steps):
            x_win = sim.loc[current_idx - window_size + 1 : current_idx, feature_columns].to_numpy(dtype=np.float32).reshape(1, -1)
            xb = torch.from_numpy(x_win).to(device)
            logits, pred_cont = model(xb)

            prob = torch.sigmoid(logits).cpu().numpy()[0] if len(binary_targets) else np.zeros((0,), dtype=np.float32)
            cont = pred_cont.cpu().numpy()[0] if len(continuous_targets) else np.zeros((0,), dtype=np.float32)
            pred_bin_hard = (prob >= threshold).astype(np.float32)

            target_idx = current_idx + horizon

            for i, col in enumerate(binary_targets):
                sim.at[target_idx, col] = pred_bin_hard[i]
            for i, col in enumerate(continuous_targets):
                sim.at[target_idx, col] = cont[i]

            row = {
                "persona_label": rollout_persona,
                "target_timestamp": str(sim.at[target_idx, "timestamp"]),
                "target_row_index": int(target_idx),
                "step": int(step + 1),
            }
            for i, col in enumerate(binary_targets):
                row[f"pred_{col}_prob"] = float(prob[i])
                row[f"pred_{col}"] = int(pred_bin_hard[i])
                row[f"true_{col}"] = float(g.at[target_idx, col])
            for i, col in enumerate(continuous_targets):
                row[f"pred_{col}"] = float(cont[i])
                row[f"true_{col}"] = float(g.at[target_idx, col])
            records.append(row)

            current_idx += 1

    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    ckpt = load_checkpoint(Path(args.checkpoint))
    ckpt_args = ckpt.get("args", {})

    train_persona = ckpt_args.get("persona", "all")
    active_persona = args.persona if args.persona is not None else train_persona
    train_ratio = float(ckpt_args.get("train_ratio", 0.7))
    val_ratio = float(ckpt_args.get("val_ratio", 0.15))
    window_size = int(ckpt_args.get("window_size", 1))
    horizon = int(ckpt_args.get("horizon", 1))

    feature_columns: List[str] = ckpt["feature_columns"]
    binary_targets: List[str] = ckpt["binary_targets"]
    continuous_targets: List[str] = ckpt["continuous_targets"]
    stats: Dict[str, Dict[str, float]] = ckpt.get("standardizer_stats", {})

    df_raw = load_persona_frames(Path(args.data_dir), args.file_name, active_persona)
    _, inferred_binary, inferred_continuous = infer_state_columns(df_raw)
    df_clean = impute_and_cast(df_raw, inferred_binary, inferred_continuous)

    splits = split_by_time_per_persona(df_clean, train_ratio=train_ratio, val_ratio=val_ratio)
    split_df = choose_split(args.split, splits)
    split_df = ensure_persona_dummy_columns(split_df, feature_columns)
    split_df = apply_standardizer(split_df, stats)

    model = build_model(ckpt, device)

    if args.mode == "one_step":
        x, yb_true, yc_true_scaled, personas, timestamps = build_windowed_samples(
            split_df,
            feature_columns,
            binary_targets,
            continuous_targets,
            window_size,
            horizon,
        )
        yb_prob, yc_pred_scaled = forward_predict(model, x, args.batch_size, device)

        yc_true = inverse_scale_continuous(yc_true_scaled, continuous_targets, stats)
        yc_pred = inverse_scale_continuous(yc_pred_scaled, continuous_targets, stats)

        b_metrics = binary_metrics(yb_prob, yb_true, args.binary_threshold)
        c_metrics = continuous_metrics(yc_pred, yc_true)
        all_metrics = {**b_metrics, **c_metrics}

        print("Prediction summary")
        for k, v in all_metrics.items():
            print(f"- {k}: {v:.6f}")

        pred_df = build_prediction_frame(
            personas,
            timestamps,
            binary_targets,
            continuous_targets,
            yb_true,
            yc_true,
            yb_prob,
            yc_pred,
            args.binary_threshold,
        )

        preview_rows = max(0, args.preview_rows)
        if preview_rows > 0:
            print("\nPreview")
            print(pred_df.head(preview_rows).to_string(index=False))

        if args.output_csv:
            out_path = Path(args.output_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pred_df.to_csv(out_path, index=False)
            print(f"\nSaved predictions to {out_path}")

        metrics_path = None
        if args.output_csv:
            metrics_path = Path(args.output_csv).with_suffix(".metrics.json")
            metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
            print(f"Saved metrics to {metrics_path}")

    else:
        rollout_persona = args.rollout_persona if args.rollout_persona is not None else active_persona
        if rollout_persona == "all":
            raise ValueError("For rollout mode, provide a single persona via --rollout-persona or --persona.")

        rollout_df = run_rollout(
            model=model,
            split_df=split_df,
            feature_columns=feature_columns,
            binary_targets=binary_targets,
            continuous_targets=continuous_targets,
            window_size=window_size,
            horizon=horizon,
            rollout_persona=rollout_persona,
            rollout_steps=args.rollout_steps,
            rollout_start_index=args.rollout_start_index,
            threshold=args.binary_threshold,
            device=device,
        )

        preview_rows = max(0, args.preview_rows)
        print("Rollout preview")
        if preview_rows > 0:
            print(rollout_df.head(preview_rows).to_string(index=False))

        if args.output_csv:
            out_path = Path(args.output_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            rollout_df.to_csv(out_path, index=False)
            print(f"\nSaved rollout to {out_path}")


if __name__ == "__main__":
    main()
