import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


META_COLUMNS = {
	"step_index",
	"week_index",
	"day_index",
	"timestamp",
	"persona_label",
	"season_label",
	"sunrise_minute",
	"sunset_minute",
	"cloudiness",
	"time_of_day_norm",
	"day_of_week",
	"is_weekend",
}

DROP_FEATURE_COLUMNS = {"timestamp", "season_label", "step_index"}


@dataclass
class SplitFrames:
	train: pd.DataFrame
	val: pd.DataFrame
	test: pd.DataFrame


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train a multi-output MLP for smart-home state prediction.")
	parser.add_argument("--data-dir", type=str, default="data", help="Root folder containing persona subfolders.")
	parser.add_argument("--file-name", type=str, default="weeks4_seed42.csv", help="CSV filename to load per persona.")
	parser.add_argument("--persona", type=str, default="all", help="Persona name or 'all'.")
	parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in timesteps (0 for same-step).")
	parser.add_argument("--window-size", type=int, default=1, help="Number of past timesteps in each input sample.")
	parser.add_argument("--train-ratio", type=float, default=0.7)
	parser.add_argument("--val-ratio", type=float, default=0.15)
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--epochs", type=int, default=80)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--dropout", type=float, default=0.2)
	parser.add_argument("--hidden-dims", type=str, default="256,128", help="Comma-separated hidden dimensions.")
	parser.add_argument("--binary-loss-weight", type=float, default=1.0)
	parser.add_argument("--continuous-loss-weight", type=float, default=1.0)
	parser.add_argument("--patience", type=int, default=12)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--output-dir", type=str, default="mlp_train/artifacts")
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
	parser.add_argument("--save-predictions", action="store_true", help="Save per-sample test predictions to CSV.")
	return parser.parse_args()


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
	if device_arg == "cpu":
		return torch.device("cpu")
	if device_arg == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA requested but not available.")
		return torch.device("cuda")
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_persona_frames(data_dir: Path, file_name: str, persona: str) -> pd.DataFrame:
	csv_paths = sorted(data_dir.glob(f"*/{file_name}"))
	if not csv_paths:
		raise FileNotFoundError(f"No CSV files found under {data_dir} matching */{file_name}")

	rows = []
	for csv_path in csv_paths:
		persona_name = csv_path.parent.name
		if persona != "all" and persona_name != persona:
			continue
		df = pd.read_csv(csv_path)
		df["persona_label"] = df.get("persona_label", persona_name)
		rows.append(df)

	if not rows:
		raise ValueError(f"No data loaded for persona selection: {persona}")

	merged = pd.concat(rows, axis=0, ignore_index=True)
	if "timestamp" not in merged.columns:
		raise ValueError("Expected 'timestamp' column is missing.")

	merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
	if merged["timestamp"].isna().any():
		raise ValueError("Some rows have invalid timestamps.")

	merged = merged.sort_values(["persona_label", "timestamp"]).reset_index(drop=True)
	return merged


def infer_state_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
	state_cols = [c for c in df.columns if c not in META_COLUMNS]
	if not state_cols:
		raise ValueError("No state columns inferred. Check dataset schema.")

	binary_cols: List[str] = []
	continuous_cols: List[str] = []
	for col in state_cols:
		s = pd.to_numeric(df[col], errors="coerce").dropna()
		uniq = set(np.unique(s.to_numpy()))
		if uniq.issubset({0.0, 1.0}) and len(uniq) <= 2:
			binary_cols.append(col)
		else:
			continuous_cols.append(col)

	if not binary_cols and not continuous_cols:
		raise ValueError("Could not infer binary or continuous state columns.")

	return state_cols, binary_cols, continuous_cols


def impute_and_cast(df: pd.DataFrame, binary_cols: Sequence[str], continuous_cols: Sequence[str]) -> pd.DataFrame:
	out = df.copy()
	out = out.sort_values(["persona_label", "timestamp"]).reset_index(drop=True)

	fill_cols = [c for c in out.columns if c != "persona_label"]
	for col in fill_cols:
		out[col] = out.groupby("persona_label", sort=False)[col].transform(lambda s: s.ffill().bfill())

	for col in binary_cols:
		out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
		out[col] = (out[col] >= 0.5).astype(np.float32)

	for col in continuous_cols:
		out[col] = pd.to_numeric(out[col], errors="coerce")

	numeric_cols = out.select_dtypes(include=[np.number, "bool"]).columns.tolist()
	for col in numeric_cols:
		out[col] = pd.to_numeric(out[col], errors="coerce")
		if out[col].isna().any():
			out[col] = out[col].fillna(out[col].median())

	out["is_weekend"] = out["is_weekend"].astype(np.float32)
	return out.reset_index(drop=True)


def split_by_time_per_persona(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> SplitFrames:
	if train_ratio <= 0.0 or val_ratio < 0.0 or train_ratio + val_ratio >= 1.0:
		raise ValueError("Require train_ratio > 0, val_ratio >= 0 and train_ratio + val_ratio < 1.")

	train_parts: List[pd.DataFrame] = []
	val_parts: List[pd.DataFrame] = []
	test_parts: List[pd.DataFrame] = []

	for _, group in df.groupby("persona_label", sort=False):
		n = len(group)
		train_end = int(n * train_ratio)
		val_end = train_end + int(n * val_ratio)

		train_parts.append(group.iloc[:train_end])
		val_parts.append(group.iloc[train_end:val_end])
		test_parts.append(group.iloc[val_end:])

	train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
	val_df = pd.concat(val_parts, axis=0).reset_index(drop=True)
	test_df = pd.concat(test_parts, axis=0).reset_index(drop=True)
	return SplitFrames(train=train_df, val=val_df, test=test_df)


def build_feature_columns(df: pd.DataFrame, include_persona: bool) -> List[str]:
	candidates = [c for c in df.columns if c not in DROP_FEATURE_COLUMNS and c != "persona_label"]
	numeric_candidates = [c for c in candidates if pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c])]
	if include_persona:
		numeric_candidates.extend([c for c in df.columns if c.startswith("persona_") and c != "persona_label"])
	return sorted(set(numeric_candidates), key=numeric_candidates.index)


def add_persona_dummies(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
	personas = sorted(train_df["persona_label"].astype(str).unique().tolist())
	dummy_cols = [f"persona_{p}" for p in personas]

	def encode(df: pd.DataFrame) -> pd.DataFrame:
		out = df.copy()
		for p, c in zip(personas, dummy_cols):
			out[c] = (out["persona_label"].astype(str) == p).astype(np.float32)
		return out

	return encode(train_df), encode(val_df), encode(test_df), dummy_cols


def fit_standardizer(train_df: pd.DataFrame, feature_cols: Sequence[str]) -> Dict[str, Dict[str, float]]:
	stats: Dict[str, Dict[str, float]] = {}
	for c in feature_cols:
		if c.startswith("persona_"):
			continue
		s = pd.to_numeric(train_df[c], errors="coerce")
		uniq = set(np.unique(s.dropna().to_numpy()))
		if uniq.issubset({0.0, 1.0}) and len(uniq) <= 2:
			continue
		mean = float(s.mean())
		std = float(s.std(ddof=0))
		if std < 1e-8:
			std = 1.0
		stats[c] = {"mean": mean, "std": std}
	return stats


def apply_standardizer(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
	out = df.copy()
	for c, v in stats.items():
		out[c] = (pd.to_numeric(out[c], errors="coerce") - v["mean"]) / v["std"]
	return out


def build_windowed_samples(
	df: pd.DataFrame,
	feature_cols: Sequence[str],
	binary_targets: Sequence[str],
	continuous_targets: Sequence[str],
	window_size: int,
	horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
	xs: List[np.ndarray] = []
	ys_bin: List[np.ndarray] = []
	ys_cont: List[np.ndarray] = []
	sample_persona: List[str] = []
	sample_timestamp: List[str] = []

	for persona, group in df.groupby("persona_label", sort=False):
		g = group.sort_values("timestamp").reset_index(drop=True)
		x_data = g[feature_cols].to_numpy(dtype=np.float32)
		yb_data = g[binary_targets].to_numpy(dtype=np.float32) if binary_targets else np.zeros((len(g), 0), dtype=np.float32)
		yc_data = g[continuous_targets].to_numpy(dtype=np.float32) if continuous_targets else np.zeros((len(g), 0), dtype=np.float32)

		start = window_size - 1
		end = len(g) - horizon
		for idx in range(start, end):
			x_win = x_data[idx - window_size + 1 : idx + 1].reshape(-1)
			y_index = idx + horizon
			xs.append(x_win)
			ys_bin.append(yb_data[y_index])
			ys_cont.append(yc_data[y_index])
			sample_persona.append(str(persona))
			sample_timestamp.append(str(g.iloc[y_index]["timestamp"]))

	if not xs:
		raise ValueError("No windowed samples generated. Adjust window size/horizon/splits.")

	x_arr = np.stack(xs).astype(np.float32)
	yb_arr = np.stack(ys_bin).astype(np.float32)
	yc_arr = np.stack(ys_cont).astype(np.float32)
	return x_arr, yb_arr, yc_arr, sample_persona, sample_timestamp


class MultiOutputMLP(nn.Module):
	def __init__(
		self,
		input_dim: int,
		hidden_dims: Sequence[int],
		dropout: float,
		num_binary_targets: int,
		num_continuous_targets: int,
	) -> None:
		super().__init__()

		layers: List[nn.Module] = []
		prev = input_dim
		for h in hidden_dims:
			layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
			prev = h
		self.trunk = nn.Sequential(*layers) if layers else nn.Identity()

		self.binary_head = nn.Linear(prev, num_binary_targets) if num_binary_targets > 0 else None
		self.continuous_head = nn.Linear(prev, num_continuous_targets) if num_continuous_targets > 0 else None

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		h = self.trunk(x)
		y_bin = self.binary_head(h) if self.binary_head is not None else torch.zeros((x.shape[0], 0), device=x.device)
		y_cont = self.continuous_head(h) if self.continuous_head is not None else torch.zeros((x.shape[0], 0), device=x.device)
		return y_bin, y_cont


def split_tensor_dataset(x: np.ndarray, y_bin: np.ndarray, y_cont: np.ndarray) -> TensorDataset:
	return TensorDataset(
		torch.from_numpy(x),
		torch.from_numpy(y_bin),
		torch.from_numpy(y_cont),
	)


def compute_binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
	if labels.numel() == 0:
		return {"binary_accuracy": float("nan"), "binary_macro_f1": float("nan")}

	probs = torch.sigmoid(logits)
	preds = (probs >= 0.5).float()
	labels = labels.float()

	accuracy = (preds == labels).float().mean().item()

	f1s: List[float] = []
	for j in range(labels.shape[1]):
		p = preds[:, j]
		y = labels[:, j]
		tp = torch.sum((p == 1) & (y == 1)).item()
		fp = torch.sum((p == 1) & (y == 0)).item()
		fn = torch.sum((p == 0) & (y == 1)).item()
		denom = (2.0 * tp + fp + fn)
		if denom == 0:
			f1 = 1.0
		else:
			f1 = (2.0 * tp) / denom
		f1s.append(float(f1))

	return {
		"binary_accuracy": float(accuracy),
		"binary_macro_f1": float(np.mean(f1s) if f1s else float("nan")),
	}


def compute_continuous_metrics(pred: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
	if labels.numel() == 0:
		return {"continuous_mae": float("nan"), "continuous_rmse": float("nan")}

	abs_err = torch.abs(pred - labels)
	sq_err = (pred - labels) ** 2
	mae = abs_err.mean().item()
	rmse = torch.sqrt(sq_err.mean()).item()
	return {"continuous_mae": float(mae), "continuous_rmse": float(rmse)}


def run_epoch(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	optimizer: torch.optim.Optimizer,
	bce_loss: nn.Module,
	mse_loss: nn.Module,
	binary_weight: float,
	continuous_weight: float,
	train: bool,
) -> Dict[str, float]:
	if train:
		model.train()
	else:
		model.eval()

	total_loss = 0.0
	total_samples = 0
	all_logits: List[torch.Tensor] = []
	all_yb: List[torch.Tensor] = []
	all_yc_hat: List[torch.Tensor] = []
	all_yc: List[torch.Tensor] = []

	for x, yb, yc in loader:
		x = x.to(device)
		yb = yb.to(device)
		yc = yc.to(device)

		with torch.set_grad_enabled(train):
			logits, pred_cont = model(x)

			loss = 0.0
			if yb.shape[1] > 0:
				loss = loss + binary_weight * bce_loss(logits, yb)
			if yc.shape[1] > 0:
				loss = loss + continuous_weight * mse_loss(pred_cont, yc)

			if train:
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

		bs = x.shape[0]
		total_loss += float(loss.item()) * bs
		total_samples += bs
		all_logits.append(logits.detach().cpu())
		all_yb.append(yb.detach().cpu())
		all_yc_hat.append(pred_cont.detach().cpu())
		all_yc.append(yc.detach().cpu())

	logits_cat = torch.cat(all_logits, dim=0) if all_logits else torch.zeros((0, 0))
	yb_cat = torch.cat(all_yb, dim=0) if all_yb else torch.zeros((0, 0))
	yc_hat_cat = torch.cat(all_yc_hat, dim=0) if all_yc_hat else torch.zeros((0, 0))
	yc_cat = torch.cat(all_yc, dim=0) if all_yc else torch.zeros((0, 0))

	metrics = {
		"loss": total_loss / max(total_samples, 1),
		**compute_binary_metrics(logits_cat, yb_cat),
		**compute_continuous_metrics(yc_hat_cat, yc_cat),
	}
	return metrics


def format_metrics(prefix: str, metrics: Dict[str, float]) -> str:
	return (
		f"{prefix} loss={metrics['loss']:.5f} "
		f"bin_acc={metrics['binary_accuracy']:.4f} "
		f"bin_f1={metrics['binary_macro_f1']:.4f} "
		f"cont_mae={metrics['continuous_mae']:.4f} "
		f"cont_rmse={metrics['continuous_rmse']:.4f}"
	)


def save_predictions(
	path: Path,
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	binary_targets: Sequence[str],
	continuous_targets: Sequence[str],
	personas: Sequence[str],
	timestamps: Sequence[str],
) -> None:
	model.eval()
	pred_bin: List[np.ndarray] = []
	pred_cont: List[np.ndarray] = []

	with torch.no_grad():
		for x, _, _ in loader:
			x = x.to(device)
			logits, y_cont = model(x)
			pred_bin.append(torch.sigmoid(logits).cpu().numpy())
			pred_cont.append(y_cont.cpu().numpy())

	bin_arr = np.concatenate(pred_bin, axis=0) if pred_bin else np.zeros((0, len(binary_targets)))
	cont_arr = np.concatenate(pred_cont, axis=0) if pred_cont else np.zeros((0, len(continuous_targets)))

	output = pd.DataFrame({"persona_label": personas, "target_timestamp": timestamps})
	for i, c in enumerate(binary_targets):
		output[f"pred_{c}"] = bin_arr[:, i] if len(bin_arr) else np.array([])
	for i, c in enumerate(continuous_targets):
		output[f"pred_{c}"] = cont_arr[:, i] if len(cont_arr) else np.array([])
	output.to_csv(path, index=False)


def main() -> None:
	args = parse_args()
	set_seed(args.seed)
	device = resolve_device(args.device)

	data_dir = Path(args.data_dir)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	df_raw = load_persona_frames(data_dir=data_dir, file_name=args.file_name, persona=args.persona)
	state_cols, binary_targets, continuous_targets = infer_state_columns(df_raw)
	df_clean = impute_and_cast(df_raw, binary_cols=binary_targets, continuous_cols=continuous_targets)

	splits = split_by_time_per_persona(df_clean, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

	include_persona = False
	feature_cols = build_feature_columns(splits.train, include_persona=include_persona)
	stats = fit_standardizer(splits.train, feature_cols)

	train_df = apply_standardizer(splits.train, stats)
	val_df = apply_standardizer(splits.val, stats)
	test_df = apply_standardizer(splits.test, stats)

	x_train, yb_train, yc_train, _, _ = build_windowed_samples(
		train_df, feature_cols, binary_targets, continuous_targets, args.window_size, args.horizon
	)
	x_val, yb_val, yc_val, _, _ = build_windowed_samples(
		val_df, feature_cols, binary_targets, continuous_targets, args.window_size, args.horizon
	)
	x_test, yb_test, yc_test, persona_test, timestamp_test = build_windowed_samples(
		test_df, feature_cols, binary_targets, continuous_targets, args.window_size, args.horizon
	)

	train_ds = split_tensor_dataset(x_train, yb_train, yc_train)
	val_ds = split_tensor_dataset(x_val, yb_val, yc_val)
	test_ds = split_tensor_dataset(x_test, yb_test, yc_test)

	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
	test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

	hidden_dims = [int(v) for v in args.hidden_dims.split(",") if v.strip()]
	model = MultiOutputMLP(
		input_dim=x_train.shape[1],
		hidden_dims=hidden_dims,
		dropout=args.dropout,
		num_binary_targets=len(binary_targets),
		num_continuous_targets=len(continuous_targets),
	).to(device)

	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=args.learning_rate,
		weight_decay=args.weight_decay,
	)
	bce_loss = nn.BCEWithLogitsLoss()
	mse_loss = nn.MSELoss()

	best_val_loss = float("inf")
	best_state = None
	best_epoch = -1
	epochs_no_improve = 0
	history: List[Dict[str, Dict[str, float]]] = []

	for epoch in range(1, args.epochs + 1):
		train_metrics = run_epoch(
			model,
			train_loader,
			device,
			optimizer,
			bce_loss,
			mse_loss,
			args.binary_loss_weight,
			args.continuous_loss_weight,
			train=True,
		)
		val_metrics = run_epoch(
			model,
			val_loader,
			device,
			optimizer,
			bce_loss,
			mse_loss,
			args.binary_loss_weight,
			args.continuous_loss_weight,
			train=False,
		)

		print(f"Epoch {epoch:03d} | {format_metrics('train', train_metrics)} | {format_metrics('val', val_metrics)}")
		history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

		if val_metrics["loss"] < best_val_loss:
			best_val_loss = val_metrics["loss"]
			best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
			best_epoch = epoch
			epochs_no_improve = 0
		else:
			epochs_no_improve += 1

		if epochs_no_improve >= args.patience:
			print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}).")
			break

	if best_state is None:
		raise RuntimeError("Training did not produce a valid model state.")

	model.load_state_dict(best_state)
	test_metrics = run_epoch(
		model,
		test_loader,
		device,
		optimizer,
		bce_loss,
		mse_loss,
		args.binary_loss_weight,
		args.continuous_loss_weight,
		train=False,
	)

	print(f"Best epoch: {best_epoch}")
	print(format_metrics("test", test_metrics))

	checkpoint_path = output_dir / "best_model.pt"
	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"best_epoch": best_epoch,
			"args": vars(args),
			"feature_columns": feature_cols,
			"state_columns": state_cols,
			"binary_targets": binary_targets,
			"continuous_targets": continuous_targets,
			"standardizer_stats": stats,
		},
		checkpoint_path,
	)

	metrics_path = output_dir / "metrics.json"
	payload = {
		"best_epoch": best_epoch,
		"best_val_loss": best_val_loss,
		"test_metrics": test_metrics,
		"history": history,
		"num_train_samples": int(len(train_ds)),
		"num_val_samples": int(len(val_ds)),
		"num_test_samples": int(len(test_ds)),
		"input_dim": int(x_train.shape[1]),
		"num_binary_targets": len(binary_targets),
		"num_continuous_targets": len(continuous_targets),
	}
	metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

	if args.save_predictions:
		pred_path = output_dir / "test_predictions.csv"
		save_predictions(
			pred_path,
			model,
			test_loader,
			device,
			binary_targets,
			continuous_targets,
			persona_test,
			timestamp_test,
		)

	print(f"Saved checkpoint: {checkpoint_path}")
	print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
	main()
