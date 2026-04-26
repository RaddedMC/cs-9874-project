from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
	import matplotlib.pyplot as plt
except ImportError as exc:
	raise SystemExit(
		"matplotlib is required to generate graphs. Install it with: pip install matplotlib"
	) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"

PRIVATE_TRAIN_FEDERATE_DIR = RESULTS_DIR / "private_train_federate"
TRAIN_PRIVATE_FEDERATE_SERVER_DIR = RESULTS_DIR / "train_private_federate" / "server_output_model"

PLOT_METRICS = [
	("attack_accuracy", "Accuracy"),
	("attack_precision", "Precision"),
	("attack_recall", "Recall"),
	("attack_f1", "F1-Score"),
	("roc_auc", "ROC AUC"),
	("tp", "TP"),
	("tn", "TN"),
	("fp", "FP"),
	("fn", "FN"),
	("threshold_selection_accuracy", "Threshold Selection Accuracy"),
]


def load_attack_results(base_dir: Path, include_clients: bool) -> pd.DataFrame:
	if not base_dir.is_dir():
		raise SystemExit(f"Results directory not found: {base_dir}")

	records: list[dict[str, object]] = []
	for metrics_path in sorted(base_dir.rglob("*_metrics.json")):
		relative = metrics_path.relative_to(base_dir)
		parts = relative.parts
		if len(parts) < 2:
			continue

		if include_clients:
			if parts[0] == "client":
				if len(parts) < 4:
					continue
				label = parts[1].removesuffix("_model")
				epsilon_token = parts[2]
			elif parts[0] == "server":
				label = "server"
				epsilon_token = parts[1]
			else:
				continue
		else:
			label = "server"
			epsilon_token = parts[0]

		if not epsilon_token.startswith("epsilon-"):
			continue

		try:
			epsilon_num = int(epsilon_token.split("-", maxsplit=1)[1])
		except (IndexError, ValueError):
			continue

		with metrics_path.open("r", encoding="utf-8") as handle:
			payload = json.load(handle)

		metrics = payload.get("metrics", {})
		if not isinstance(metrics, dict):
			continue

		record: dict[str, object] = {
			"series": label,
			"epsilon": epsilon_token,
			"epsilon_num": epsilon_num,
			"threshold_selection_accuracy": payload.get("threshold_selection_accuracy"),
		}
		for metric_key, _ in PLOT_METRICS:
			if metric_key in metrics:
				record[metric_key] = metrics[metric_key]

		records.append(record)

	if not records:
		raise SystemExit(f"No metrics JSON files found under: {base_dir}")

	df = pd.DataFrame(records)
	for metric_key, _ in PLOT_METRICS:
		if metric_key in df.columns:
			df[metric_key] = pd.to_numeric(df[metric_key], errors="coerce")

	return df.sort_values(["series", "epsilon_num"]).reset_index(drop=True)


def sanitize_name(name: str) -> str:
	return name.lower().replace(" ", "_").replace("-", "_")


def save_metric_plot(df: pd.DataFrame, metric_key: str, metric_label: str, output_dir: Path, title: str) -> Path:
	fig, ax = plt.subplots(figsize=(10, 6))

	for series, group in df.groupby("series", sort=True):
		points = group[["epsilon_num", metric_key]].dropna().sort_values("epsilon_num")
		if points.empty:
			continue
		ax.plot(points["epsilon_num"], points[metric_key], marker="o", label=series)

	ax.set_title(f"{title}: {metric_label} by epsilon")
	ax.set_xlabel("epsilon")
	ax.set_ylabel(metric_label)
	ax.grid(True, alpha=0.3)

	series_count = int(df["series"].nunique())
	if series_count > 1:
		ax.legend(title="model", bbox_to_anchor=(1.02, 1), loc="upper left")

	fig.tight_layout()

	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / f"mia_{sanitize_name(metric_label)}.png"
	fig.savefig(output_path, dpi=150)
	plt.close(fig)
	return output_path


def create_graphs_for_directory(base_dir: Path, include_clients: bool, title: str) -> list[Path]:
	df = load_attack_results(base_dir=base_dir, include_clients=include_clients)
	output_dir = base_dir / "graphs"

	created: list[Path] = []
	for metric_key, metric_label in PLOT_METRICS:
		if metric_key not in df.columns:
			continue
		created.append(
			save_metric_plot(
				df=df,
				metric_key=metric_key,
				metric_label=metric_label,
				output_dir=output_dir,
				title=title,
			)
		)

	return created


def create_private_graphs_with_overlay() -> list[Path]:
	private_df = load_attack_results(base_dir=PRIVATE_TRAIN_FEDERATE_DIR, include_clients=True)
	train_private_df = load_attack_results(
		base_dir=TRAIN_PRIVATE_FEDERATE_SERVER_DIR,
		include_clients=False,
	)

	train_private_df = train_private_df.copy()
	train_private_df["series"] = "train_private_server"

	combined_df = pd.concat([private_df, train_private_df], ignore_index=True)
	combined_df = combined_df.sort_values(["series", "epsilon_num"]).reset_index(drop=True)

	output_dir = PRIVATE_TRAIN_FEDERATE_DIR / "graphs"
	created: list[Path] = []
	for metric_key, metric_label in PLOT_METRICS:
		if metric_key not in combined_df.columns:
			continue
		created.append(
			save_metric_plot(
				df=combined_df,
				metric_key=metric_key,
				metric_label=metric_label,
				output_dir=output_dir,
				title="private_train_federate (+ train_private overlay)",
			)
		)

	return created


def main() -> None:
	created = []

	created.extend(create_private_graphs_with_overlay())

	created.extend(
		create_graphs_for_directory(
			base_dir=TRAIN_PRIVATE_FEDERATE_SERVER_DIR,
			include_clients=False,
			title="train_private_federate_server_output_model",
		)
	)

	print("Created graph files:")
	for path in created:
		print(path)


if __name__ == "__main__":
	main()
