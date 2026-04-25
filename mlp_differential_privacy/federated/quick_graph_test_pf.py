from pathlib import Path

import pandas as pd

try:
	import matplotlib.pyplot as plt
except ImportError as exc:
	raise SystemExit(
		"matplotlib is required to generate graphs. Install it with: pip install matplotlib"
	) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
FEDERATED_ARTIFACTS_DIR = SCRIPT_DIR.parent / "artifacts" / "federated"
INPUT_CSV = FEDERATED_ARTIFACTS_DIR / "test_pf_metrics.csv"

METRICS = [
	"binary_accuracy",
	"binary_macro_f1",
	"continuous_mae",
	"continuous_rmse",
]


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df["epsilon_num"] = (
		df["epsilon"].astype(str).str.extract(r"epsilon-(\d+)", expand=False).astype(int)
	)
	df["persona"] = df["test_name"].astype(str).str.replace("test_", "", regex=False)

	for metric in METRICS:
		if metric in df.columns:
			df[metric] = pd.to_numeric(df[metric], errors="coerce")

	return df


def save_metric_plot(df: pd.DataFrame, metric: str) -> Path:
	fig, ax = plt.subplots(figsize=(10, 6))

	ordered = df.sort_values(["persona", "epsilon_num"])
	for persona, group in ordered.groupby("persona", sort=True):
		values = group[["epsilon_num", metric]].dropna()
		if values.empty:
			continue
		ax.plot(values["epsilon_num"], values[metric], marker="o", label=persona)

	ax.set_title(f"{metric} by epsilon and persona")
	ax.set_xlabel("epsilon")
	ax.set_ylabel(metric)
	ax.grid(True, alpha=0.3)
	ax.legend(title="persona", bbox_to_anchor=(1.02, 1), loc="upper left")
	fig.tight_layout()

	output_path = FEDERATED_ARTIFACTS_DIR / f"test_pf_{metric}.png"
	fig.savefig(output_path, dpi=150)
	plt.close(fig)
	return output_path


def main() -> None:
	if not INPUT_CSV.is_file():
		raise SystemExit(f"Input CSV not found: {INPUT_CSV}")

	df = pd.read_csv(INPUT_CSV)

	required_cols = {"epsilon", "test_name", *METRICS}
	missing = sorted(required_cols - set(df.columns))
	if missing:
		raise SystemExit(f"Missing expected columns in CSV: {', '.join(missing)}")

	df = normalize_dataframe(df)

	created = []
	for metric in METRICS:
		created.append(save_metric_plot(df, metric))

	print("Created graph files:")
	for path in created:
		print(path)


if __name__ == "__main__":
	main()
