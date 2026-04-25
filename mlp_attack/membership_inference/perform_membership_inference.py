import argparse
import json
import importlib
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))


def import_train_symbols() -> Dict[str, Any]:
	train_module = importlib.import_module("mlp_train.train")
	required = [
		"MultiOutputMLP",
		"apply_standardizer",
		"build_windowed_samples",
		"impute_and_cast",
		"infer_state_columns",
		"load_persona_frames",
		"resolve_device",
		"split_by_time_per_persona",
	]
	return {name: getattr(train_module, name) for name in required}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run a loss-threshold membership inference attack against a trained MLP checkpoint."
	)
	parser.add_argument(
		"--model_location",
		type=str,
		required=True,
		help="Location of the target model checkpoint (.pt).",
	)
	parser.add_argument(
		"--output_name",
		type=str,
		required=True,
		help="Base name for generated result files.",
	)
	parser.add_argument(
		"--attack_output",
		type=str,
		required=True,
		help="Directory where membership inference results are written.",
	)

	parser.add_argument("--data-dir", type=str, default="data")
	parser.add_argument("--file-name", type=str, default="weeks4_seed42.csv")
	parser.add_argument("--batch-size", type=int, default=512)
	parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
	parser.add_argument("--window-size-override", type=int, default=None)
	parser.add_argument("--horizon-override", type=int, default=None)
	parser.add_argument("--train-ratio-override", type=float, default=None)
	parser.add_argument("--val-ratio-override", type=float, default=None)
	return parser.parse_args()


def parse_hidden_dims(value: object) -> List[int]:
	if isinstance(value, list):
		return [int(v) for v in value]
	if isinstance(value, str):
		return [int(v.strip()) for v in value.split(",") if v.strip()]
	return [256, 128]


def sanitize_filename(name: str) -> str:
	safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
	return safe or "mia_result"


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
	if not state_dict:
		return state_dict

	def strip_prefix(key: str) -> str:
		if key.startswith("_module."):
			return key[len("_module.") :]
		if key.startswith("module."):
			return key[len("module.") :]
		return key

	normalized = {strip_prefix(k): v for k, v in state_dict.items()}
	if len(normalized) != len(state_dict):
		raise ValueError("State dict key collision after prefix normalization.")
	return normalized


def load_checkpoint(path: Path) -> Dict:
	if not path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {path}")
	checkpoint = torch.load(path, map_location="cpu")
	if "model_state_dict" not in checkpoint:
		raise KeyError("Checkpoint is missing 'model_state_dict'.")
	checkpoint["model_state_dict"] = normalize_state_dict_keys(checkpoint["model_state_dict"])
	return checkpoint


def infer_target_persona(checkpoint: Dict, model_path: Path) -> str:
	ckpt_args = checkpoint.get("args", {})
	persona = ckpt_args.get("persona")
	if persona is not None:
		return str(persona)

	known_personas = {
		"commuter",
		"early_shift",
		"gig_driver",
		"hybrid",
		"night_shift",
		"retiree",
		"social",
		"student",
		"traveler",
		"wfh",
		"all",
	}
	for part in model_path.parts:
		if part in known_personas:
			return part

	raise ValueError(
		"Could not infer persona from checkpoint metadata or path. Pass --persona explicitly if needed."
	)


def ensure_persona_dummy_columns(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
	out = df.copy()
	for col in feature_columns:
		if col.startswith("persona_"):
			persona_name = col[len("persona_") :]
			out[col] = (out["persona_label"].astype(str) == persona_name).astype(np.float32)
	return out


def build_model(
	checkpoint: Dict,
	device: torch.device,
	window_size: int,
	model_cls: Callable[..., torch.nn.Module],
) -> torch.nn.Module:
	ckpt_args = checkpoint.get("args", {})
	hidden_dims = parse_hidden_dims(ckpt_args.get("hidden_dims", "256,128"))
	dropout = float(ckpt_args.get("dropout", 0.2))
	feature_columns = checkpoint["feature_columns"]

	model = model_cls(
		input_dim=len(feature_columns) * window_size,
		hidden_dims=hidden_dims,
		dropout=dropout,
		num_binary_targets=len(checkpoint["binary_targets"]),
		num_continuous_targets=len(checkpoint["continuous_targets"]),
	).to(device)

	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()
	return model


def forward_predict(
	model: torch.nn.Module,
	x: np.ndarray,
	batch_size: int,
	device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
	logits_list: List[np.ndarray] = []
	cont_list: List[np.ndarray] = []

	with torch.no_grad():
		for start in range(0, len(x), batch_size):
			end = start + batch_size
			xb = torch.from_numpy(x[start:end]).to(device)
			logits, y_cont = model(xb)
			logits_list.append(logits.cpu().numpy())
			cont_list.append(y_cont.cpu().numpy())

	logits_arr = np.concatenate(logits_list, axis=0) if logits_list else np.zeros((0, 0), dtype=np.float32)
	cont_arr = np.concatenate(cont_list, axis=0) if cont_list else np.zeros((0, 0), dtype=np.float32)
	return logits_arr.astype(np.float32), cont_arr.astype(np.float32)


def bce_with_logits_per_sample(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
	if labels.size == 0:
		return np.zeros((labels.shape[0],), dtype=np.float32)

	x = logits.astype(np.float64)
	y = labels.astype(np.float64)
	max_term = np.maximum(x, 0.0)
	loss = max_term - x * y + np.log1p(np.exp(-np.abs(x)))
	return np.mean(loss, axis=1).astype(np.float32)


def mse_per_sample(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
	if labels.size == 0:
		return np.zeros((labels.shape[0],), dtype=np.float32)
	return np.mean((pred - labels) ** 2, axis=1).astype(np.float32)


def compose_attack_score(binary_loss: np.ndarray, continuous_loss: np.ndarray) -> np.ndarray:
	has_binary = np.any(binary_loss != 0.0)
	has_cont = np.any(continuous_loss != 0.0)

	if has_binary and has_cont:
		return (binary_loss + continuous_loss).astype(np.float32)
	if has_binary:
		return binary_loss.astype(np.float32)
	if has_cont:
		return continuous_loss.astype(np.float32)
	return np.zeros_like(binary_loss, dtype=np.float32)


def best_accuracy_threshold(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray, float]:
	if scores.size == 0:
		raise ValueError("No scores available to compute threshold.")

	unique_scores = np.unique(scores)
	thresholds = np.concatenate(
		[
			[unique_scores[0] - 1e-8],
			(unique_scores[:-1] + unique_scores[1:]) / 2.0,
			[unique_scores[-1] + 1e-8],
		]
	)

	best_acc = -1.0
	best_threshold = float(thresholds[0])
	best_pred = np.zeros_like(labels)

	for thr in thresholds:
		pred_member = (scores <= thr).astype(np.int32)
		acc = float(np.mean(pred_member == labels))
		if acc > best_acc:
			best_acc = acc
			best_threshold = float(thr)
			best_pred = pred_member

	return best_threshold, best_pred, best_acc


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
	tp = int(np.sum((y_true == 1) & (y_pred == 1)))
	tn = int(np.sum((y_true == 0) & (y_pred == 0)))
	fp = int(np.sum((y_true == 0) & (y_pred == 1)))
	fn = int(np.sum((y_true == 1) & (y_pred == 0)))
	return tp, tn, fp, fn


def safe_div(num: float, den: float) -> float:
	if den == 0.0:
		return 0.0
	return float(num / den)


def roc_auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
	pos = scores[y_true == 1]
	neg = scores[y_true == 0]
	if pos.size == 0 or neg.size == 0:
		return float("nan")

	all_scores = np.concatenate([pos, neg])
	ranks = pd.Series(all_scores).rank(method="average").to_numpy(dtype=np.float64)
	n_pos = float(pos.size)
	n_neg = float(neg.size)
	rank_sum_pos = float(np.sum(ranks[: pos.size]))
	u_pos = rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0
	auc_small_score_positive = u_pos / (n_pos * n_neg)
	return float(1.0 - auc_small_score_positive)


def compute_attack_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
	tp, tn, fp, fn = confusion_counts(y_true, y_pred)
	precision = safe_div(tp, tp + fp)
	recall = safe_div(tp, tp + fn)
	f1 = safe_div(2.0 * precision * recall, precision + recall)

	return {
		"attack_accuracy": float(np.mean(y_true == y_pred)),
		"attack_precision": precision,
		"attack_recall": recall,
		"attack_f1": f1,
		"roc_auc": roc_auc_from_scores(y_true, scores),
		"tp": float(tp),
		"tn": float(tn),
		"fp": float(fp),
		"fn": float(fn),
	}


def prepare_split(
	df: pd.DataFrame,
	feature_columns: Sequence[str],
	binary_targets: Sequence[str],
	continuous_targets: Sequence[str],
	standardizer_stats: Dict[str, Dict[str, float]],
	window_size: int,
	horizon: int,
	apply_standardizer_fn: Callable[[pd.DataFrame, Dict[str, Dict[str, float]]], pd.DataFrame],
	build_windowed_samples_fn: Callable[..., Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
	with_dummies = ensure_persona_dummy_columns(df, feature_columns)
	standardized = apply_standardizer_fn(with_dummies, standardizer_stats)
	return build_windowed_samples_fn(
		standardized,
		feature_columns,
		binary_targets,
		continuous_targets,
		window_size,
		horizon,
	)


def main() -> None:
	args = parse_args()
	symbols = import_train_symbols()
	model_cls = symbols["MultiOutputMLP"]
	apply_standardizer_fn = symbols["apply_standardizer"]
	build_windowed_samples_fn = symbols["build_windowed_samples"]
	impute_and_cast_fn = symbols["impute_and_cast"]
	infer_state_columns_fn = symbols["infer_state_columns"]
	load_persona_frames_fn = symbols["load_persona_frames"]
	resolve_device_fn = symbols["resolve_device"]
	split_by_time_per_persona_fn = symbols["split_by_time_per_persona"]

	device = resolve_device_fn(args.device)

	model_path = Path(args.model_location)
	output_dir = Path(args.attack_output)
	output_dir.mkdir(parents=True, exist_ok=True)
	output_base = sanitize_filename(args.output_name)

	checkpoint = load_checkpoint(model_path)
	ckpt_args = checkpoint.get("args", {})

	persona = infer_target_persona(checkpoint, model_path)
	feature_columns: List[str] = [str(c) for c in checkpoint["feature_columns"]]
	binary_targets: List[str] = [str(c) for c in checkpoint.get("binary_targets", [])]
	continuous_targets: List[str] = [str(c) for c in checkpoint.get("continuous_targets", [])]

	window_size = int(args.window_size_override or ckpt_args.get("window_size", 1))
	horizon = int(args.horizon_override or ckpt_args.get("horizon", 1))
	train_ratio = float(args.train_ratio_override or ckpt_args.get("train_ratio", 0.7))
	val_ratio = float(args.val_ratio_override or ckpt_args.get("val_ratio", 0.15))
	standardizer_stats = checkpoint.get("standardizer_stats", {})

	model = build_model(checkpoint, device=device, window_size=window_size, model_cls=model_cls)

	df_raw = load_persona_frames_fn(Path(args.data_dir), args.file_name, persona=persona)
	_, inferred_binary, inferred_cont = infer_state_columns_fn(df_raw)
	df_clean = impute_and_cast_fn(df_raw, binary_cols=inferred_binary, continuous_cols=inferred_cont)
	splits = split_by_time_per_persona_fn(df_clean, train_ratio=train_ratio, val_ratio=val_ratio)

	x_train, yb_train, yc_train, p_train, t_train = prepare_split(
		splits.train,
		feature_columns,
		binary_targets,
		continuous_targets,
		standardizer_stats,
		window_size,
		horizon,
		apply_standardizer_fn,
		build_windowed_samples_fn,
	)
	x_val, yb_val, yc_val, p_val, t_val = prepare_split(
		splits.val,
		feature_columns,
		binary_targets,
		continuous_targets,
		standardizer_stats,
		window_size,
		horizon,
		apply_standardizer_fn,
		build_windowed_samples_fn,
	)
	x_test, yb_test, yc_test, p_test, t_test = prepare_split(
		splits.test,
		feature_columns,
		binary_targets,
		continuous_targets,
		standardizer_stats,
		window_size,
		horizon,
		apply_standardizer_fn,
		build_windowed_samples_fn,
	)

	x_member = np.concatenate([x_train, x_val], axis=0)
	yb_member = np.concatenate([yb_train, yb_val], axis=0)
	yc_member = np.concatenate([yc_train, yc_val], axis=0)
	p_member = list(p_train) + list(p_val)
	t_member = list(t_train) + list(t_val)

	x_non = x_test
	yb_non = yb_test
	yc_non = yc_test
	p_non = list(p_test)
	t_non = list(t_test)

	if x_member.shape[0] == 0 or x_non.shape[0] == 0:
		raise ValueError("Need both member and non-member samples for membership inference attack.")

	member_logits, member_cont = forward_predict(model, x_member, args.batch_size, device)
	non_logits, non_cont = forward_predict(model, x_non, args.batch_size, device)

	member_bin_loss = bce_with_logits_per_sample(member_logits, yb_member)
	member_cont_loss = mse_per_sample(member_cont, yc_member)
	member_score = compose_attack_score(member_bin_loss, member_cont_loss)

	non_bin_loss = bce_with_logits_per_sample(non_logits, yb_non)
	non_cont_loss = mse_per_sample(non_cont, yc_non)
	non_score = compose_attack_score(non_bin_loss, non_cont_loss)

	attack_scores = np.concatenate([member_score, non_score], axis=0)
	attack_labels = np.concatenate(
		[np.ones(member_score.shape[0], dtype=np.int32), np.zeros(non_score.shape[0], dtype=np.int32)],
		axis=0,
	)
	threshold, attack_pred, best_acc = best_accuracy_threshold(attack_scores, attack_labels)
	metrics = compute_attack_metrics(attack_labels, attack_pred, attack_scores)

	sample_df = pd.DataFrame(
		{
			"persona_label": p_member + p_non,
			"target_timestamp": t_member + t_non,
			"membership_true": attack_labels,
			"membership_pred": attack_pred,
			"attack_score": attack_scores,
			"binary_loss": np.concatenate([member_bin_loss, non_bin_loss], axis=0),
			"continuous_loss": np.concatenate([member_cont_loss, non_cont_loss], axis=0),
		}
	)

	metrics_payload = {
		"timestamp_utc": datetime.now(timezone.utc).isoformat(),
		"model_location": str(model_path),
		"output_name": args.output_name,
		"attack_output": str(output_dir),
		"target_persona": persona,
		"window_size": window_size,
		"horizon": horizon,
		"train_ratio": train_ratio,
		"val_ratio": val_ratio,
		"member_count": int(member_score.shape[0]),
		"non_member_count": int(non_score.shape[0]),
		"threshold": float(threshold),
		"threshold_selection_accuracy": float(best_acc),
		"score_mean_member": float(np.mean(member_score)),
		"score_mean_non_member": float(np.mean(non_score)),
		"score_std_member": float(np.std(member_score)),
		"score_std_non_member": float(np.std(non_score)),
		"metrics": metrics,
	}

	metrics_path = output_dir / f"{output_base}_metrics.json"
	samples_path = output_dir / f"{output_base}_samples.csv"
	metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
	sample_df.to_csv(samples_path, index=False)

	print(f"Target model: {model_path}")
	print(f"Target persona: {persona}")
	print(f"Member samples: {member_score.shape[0]} | Non-member samples: {non_score.shape[0]}")
	print(f"Attack threshold: {threshold:.6f}")
	print(f"Attack accuracy: {metrics['attack_accuracy']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
	print(f"Saved metrics: {metrics_path}")
	print(f"Saved samples: {samples_path}")


if __name__ == "__main__":
	main()
