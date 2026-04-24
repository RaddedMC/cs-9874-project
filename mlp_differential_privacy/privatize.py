import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Import data loading and model functions from train.py
sys.path.insert(0, str(Path(__file__).parent.parent / "mlp_train"))
from train import (
	META_COLUMNS,
	DROP_FEATURE_COLUMNS,
	SplitFrames,
	MultiOutputMLP,
	add_persona_dummies,
	apply_standardizer,
	build_feature_columns,
	build_windowed_samples,
	compute_binary_metrics,
	compute_continuous_metrics,
	fit_standardizer,
	impute_and_cast,
	infer_state_columns,
	load_persona_frames,
	split_by_time_per_persona,
)


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments for DP-SGD or post-hoc privatization."""
	parser = argparse.ArgumentParser(
		description="Apply Differential Privacy via DP-SGD training or post-hoc checkpoint privatization."
	)
	parser.add_argument(
		"--mode",
		type=str,
		default="dp-sgd",
		choices=["dp-sgd", "post-hoc"],
		help="Privacy mode: train with DP-SGD or privatize an existing checkpoint post-hoc.",
	)

	# Data & model arguments (reuse from train.py)
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
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

	# Differential Privacy arguments
	parser.add_argument(
		"--epsilon",
		type=float,
		default=None,
		help="Privacy budget (epsilon). Lower values indicate stricter privacy (typical range: 0.5-10.0).",
	)
	parser.add_argument(
		"--delta",
		type=float,
		default=None,
		help="Failure probability (delta). Typical value: 1e-5 or 1/n where n=dataset size.",
	)
	parser.add_argument(
		"--max-grad-norm",
		type=float,
		default=1.0,
		help="Maximum L2 gradient norm for clipping (required for DP-SGD). Default: 1.0.",
	)
	parser.add_argument(
		"--noise-multiplier",
		type=float,
		default=None,
		help="Noise multiplier for DP-SGD. If not specified, computed from epsilon/delta.",
	)
	parser.add_argument(
		"--input-checkpoint",
		type=str,
		default=None,
		help="Path to an existing checkpoint when --mode post-hoc.",
	)
	parser.add_argument(
		"--posthoc-mechanism",
		type=str,
		default="gaussian",
		choices=["gaussian", "laplace"],
		help="Post-hoc mechanism to apply to checkpoint weights.",
	)
	parser.add_argument(
		"--weight-clip",
		type=float,
		default=1.0,
		help="Element-wise clipping bound C before adding post-hoc noise.",
	)

	# Output directory
	parser.add_argument(
		"--output-dir",
		type=str,
		default="mlp_differential_privacy/artifacts",
		help="Directory to save model checkpoint and metrics.",
	)
	parser.add_argument("--save-predictions", action="store_true", help="Save per-sample test predictions to CSV.")

	return parser.parse_args()


def set_seed(seed: int) -> None:
	"""Set random seeds for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
	"""Resolve device (CPU or CUDA)."""
	if device_arg == "cpu":
		return torch.device("cpu")
	if device_arg == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA requested but not available.")
		return torch.device("cuda")
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_noise_multiplier(
	dataset_size: int, batch_size: int, target_epsilon: float, target_delta: float, epochs: int
) -> float:
	"""
	Compute noise multiplier for DP-SGD to achieve target (epsilon, delta).

	Uses the Opacus recommendation formula based on RDP accounting.
	For a rough estimate, we can use:
		noise_multiplier ≈ sqrt(2 * log(1.25/delta)) / (epsilon * sqrt(dataset_size))

	This is a simplified formula; Opacus will compute exact accounting during training.
	"""
	# Simplified estimate
	num_steps = (dataset_size // batch_size) * epochs
	if num_steps == 0:
		return 1.0

	# Based on RDP accounting formula
	noise_mult = math.sqrt(2 * math.log(1.25 / target_delta)) / (target_epsilon * math.sqrt(num_steps))
	return max(noise_mult, 0.1)  # Ensure minimum noise


def clip_state_dict(state_dict: Dict[str, torch.Tensor], clip_value: float) -> Dict[str, torch.Tensor]:
	"""Clip each tensor element to [-clip_value, clip_value]."""
	if clip_value <= 0:
		raise ValueError("weight-clip must be > 0.")
	return {k: v.clone().clamp(min=-clip_value, max=clip_value) for k, v in state_dict.items()}


def privatize_state_dict_posthoc(
	state_dict: Dict[str, torch.Tensor],
	epsilon: float,
	delta: float,
	mechanism: str,
	clip_value: float,
	seed: int,
) -> Dict[str, torch.Tensor]:
	"""Apply post-hoc noise to a checkpoint state_dict after clipping."""
	if epsilon <= 0:
		raise ValueError("epsilon must be > 0 for post-hoc privatization.")
	if mechanism == "gaussian" and not (0.0 < delta < 1.0):
		raise ValueError("delta must be in (0, 1) for Gaussian post-hoc mechanism.")

	clipped = clip_state_dict(state_dict, clip_value)
	gen = torch.Generator(device="cpu")
	gen.manual_seed(seed)
	rng = np.random.default_rng(seed)

	if mechanism == "gaussian":
		sigma = clip_value * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
		return {
			k: v + torch.randn(v.shape, generator=gen, device=v.device, dtype=v.dtype) * sigma
			for k, v in clipped.items()
		}

	b = clip_value / epsilon
	noisy: Dict[str, torch.Tensor] = {}
	for k, v in clipped.items():
		noise = rng.laplace(loc=0.0, scale=b, size=tuple(v.shape)).astype(np.float32)
		noise_t = torch.from_numpy(noise).to(device=v.device, dtype=v.dtype)
		noisy[k] = v + noise_t
	return noisy


def run_posthoc_privatisation(args: argparse.Namespace) -> None:
	"""Privatize an existing trained checkpoint by adding calibrated noise to weights."""
	if args.input_checkpoint is None:
		raise ValueError("--input-checkpoint is required when --mode post-hoc.")
	if args.epsilon is None:
		raise ValueError("--epsilon is required when --mode post-hoc.")
	if args.delta is None:
		raise ValueError("--delta is required when --mode post-hoc.")

	input_path = Path(args.input_checkpoint)
	if not input_path.exists():
		raise FileNotFoundError(f"Input checkpoint not found: {input_path}")

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
	if "model_state_dict" not in checkpoint:
		raise ValueError("Checkpoint does not contain 'model_state_dict'.")

	priv_state = privatize_state_dict_posthoc(
		state_dict=checkpoint["model_state_dict"],
		epsilon=args.epsilon,
		delta=args.delta,
		mechanism=args.posthoc_mechanism,
		clip_value=args.weight_clip,
		seed=args.seed,
	)

	checkpoint["model_state_dict"] = priv_state
	checkpoint["posthoc_privacy"] = {
		"mechanism": args.posthoc_mechanism,
		"epsilon": args.epsilon,
		"delta": args.delta,
		"weight_clip": args.weight_clip,
	}

	output_path = output_dir / "best_model_posthoc.pt"
	torch.save(checkpoint, output_path)

	metrics_path = output_dir / "metrics_posthoc.json"
	metrics_payload = {
		"mode": "post-hoc",
		"input_checkpoint": str(input_path),
		"output_checkpoint": str(output_path),
		"posthoc_privacy": checkpoint["posthoc_privacy"],
	}
	metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

	print("=" * 80)
	print("POST-HOC PRIVATIZATION COMPLETE")
	print("=" * 80)
	print(f"Mechanism: {args.posthoc_mechanism}")
	print(f"Privacy parameters: ε={args.epsilon:.4f}, δ={args.delta:.2e}")
	print(f"Weight clip: {args.weight_clip}")
	print(f"Saved checkpoint: {output_path}")
	print(f"Saved metadata: {metrics_path}")


def run_epoch_dp(
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
	"""
	Run one epoch with differential privacy (DP-SGD).

	During training, gradients are clipped and noise is injected by the DP-wrapped optimizer.
	During validation/testing, no DP is applied (no gradients).
	"""
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
	"""Format metrics for console output."""
	return (
		f"{prefix} loss={metrics['loss']:.5f} "
		f"bin_acc={metrics['binary_accuracy']:.4f} "
		f"bin_f1={metrics['binary_macro_f1']:.4f} "
		f"cont_mae={metrics['continuous_mae']:.4f} "
		f"cont_rmse={metrics['continuous_rmse']:.4f}"
	)


def get_privacy_budget(privacy_engine: PrivacyEngine, delta: float) -> Tuple[float, float]:
	"""
	Extract (epsilon, delta) from PrivacyEngine.

	The PrivacyEngine's AccountantMixin provides get_epsilon() method.
	"""
	try:
		epsilon = privacy_engine.accountant.get_epsilon(delta)
	except Exception:
		# Fallback if accounting not yet initialized
		epsilon = float("inf")
	return epsilon, delta


def main() -> None:
	"""Main entrypoint for differential privacy workflows."""
	args = parse_args()
	set_seed(args.seed)

	if args.mode == "post-hoc":
		run_posthoc_privatisation(args)
		return

	if args.epsilon is None:
		raise ValueError("--epsilon is required when --mode dp-sgd.")
	if args.delta is None:
		raise ValueError("--delta is required when --mode dp-sgd.")

	device = resolve_device(args.device)

	data_dir = Path(args.data_dir)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	print("=" * 80)
	print("DIFFERENTIAL PRIVACY TRAINING (DP-SGD via Opacus)")
	print("=" * 80)
	print(f"Privacy Budget: ε={args.epsilon:.4f}, δ={args.delta:.2e}")
	print(f"Max Gradient Norm: {args.max_grad_norm}")
	print("=" * 80)

	# Load data
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

	train_ds = TensorDataset(
		torch.from_numpy(x_train),
		torch.from_numpy(yb_train),
		torch.from_numpy(yc_train),
	)
	val_ds = TensorDataset(
		torch.from_numpy(x_val),
		torch.from_numpy(yb_val),
		torch.from_numpy(yc_val),
	)
	test_ds = TensorDataset(
		torch.from_numpy(x_test),
		torch.from_numpy(yb_test),
		torch.from_numpy(yc_test),
	)

	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
	test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

	# Build model
	hidden_dims = [int(v) for v in args.hidden_dims.split(",") if v.strip()]
	model = MultiOutputMLP(
		input_dim=x_train.shape[1],
		hidden_dims=hidden_dims,
		dropout=args.dropout,
		num_binary_targets=len(binary_targets),
		num_continuous_targets=len(continuous_targets),
	).to(device)

	# Create optimizer (will be wrapped by PrivacyEngine)
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=args.learning_rate,
		weight_decay=args.weight_decay,
	)

	# Compute noise multiplier if not provided
	noise_multiplier = args.noise_multiplier
	if noise_multiplier is None:
		noise_multiplier = compute_noise_multiplier(
			dataset_size=len(train_ds),
			batch_size=args.batch_size,
			target_epsilon=args.epsilon,
			target_delta=args.delta,
			epochs=args.epochs,
		)
		print(f"Computed noise multiplier: {noise_multiplier:.6f}")

	# Wrap with PrivacyEngine for DP-SGD
	privacy_engine = PrivacyEngine()
	model, optimizer, train_loader = privacy_engine.make_private(
		module=model,
		optimizer=optimizer,
		data_loader=train_loader,
		noise_multiplier=noise_multiplier,
		max_grad_norm=args.max_grad_norm,
	)

	print(f"PrivacyEngine configured with:")
	print(f"  - Noise multiplier: {noise_multiplier:.6f}")
	print(f"  - Max grad norm (clipping): {args.max_grad_norm:.4f}")
	print(f"  - Batch size: {args.batch_size}")
	print(f"  - Dataset size: {len(train_ds)}")
	print()

	bce_loss = nn.BCEWithLogitsLoss()
	mse_loss = nn.MSELoss()

	best_val_loss = float("inf")
	best_state = None
	best_epoch = -1
	epochs_no_improve = 0
	history: List[Dict[str, object]] = []

	for epoch in range(1, args.epochs + 1):
		train_metrics = run_epoch_dp(
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
		val_metrics = run_epoch_dp(
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

		# Get current privacy budget spent
		epsilon, delta = get_privacy_budget(privacy_engine, args.delta)

		print(
			f"Epoch {epoch:03d} | {format_metrics('train', train_metrics)} | {format_metrics('val', val_metrics)} | "
			f"ε={epsilon:.4f}"
		)
		history.append({
			"epoch": epoch,
			"train": train_metrics,
			"val": val_metrics,
			"epsilon": epsilon,
			"delta": args.delta,
		})

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

	# Load best model for evaluation on test set
	model.load_state_dict(best_state)
	test_metrics = run_epoch_dp(
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

	# Get final privacy budget
	final_epsilon, final_delta = get_privacy_budget(privacy_engine, args.delta)

	print()
	print("=" * 80)
	print(f"Best epoch: {best_epoch}")
	print(format_metrics("test", test_metrics))
	print(f"Final Privacy Budget: ε={final_epsilon:.4f}, δ={final_delta:.2e}")
	print("=" * 80)

	# Save checkpoint
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
			# Privacy metadata
			"epsilon": final_epsilon,
			"delta": final_delta,
			"noise_multiplier": noise_multiplier,
			"max_grad_norm": args.max_grad_norm,
		},
		checkpoint_path,
	)

	# Save metrics with privacy accounting
	metrics_path = output_dir / "metrics.json"
	payload = {
		"best_epoch": best_epoch,
		"best_val_loss": best_val_loss,
		"test_metrics": test_metrics,
		"privacy": {
			"epsilon": final_epsilon,
			"delta": final_delta,
			"noise_multiplier": noise_multiplier,
			"max_grad_norm": args.max_grad_norm,
		},
		"history": history,
		"num_train_samples": int(len(train_ds)),
		"num_val_samples": int(len(val_ds)),
		"num_test_samples": int(len(test_ds)),
		"input_dim": int(x_train.shape[1]),
		"num_binary_targets": len(binary_targets),
		"num_continuous_targets": len(continuous_targets),
	}
	metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

	print(f"Saved checkpoint: {checkpoint_path}")
	print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
	main()
