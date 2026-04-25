from pathlib import Path

import pandas as pd

try:
	import matplotlib.pyplot as plt
	from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
	series_by_persona: dict[str, pd.DataFrame] = {}
	for persona, group in ordered.groupby("persona", sort=True):
		values = group[["epsilon_num", metric]].dropna()
		if values.empty:
			continue
		series_by_persona[persona] = values
		ax.plot(values["epsilon_num"], values[metric], marker="o", label=persona)

	if metric in {"continuous_mae", "continuous_rmse"}:
		zoom_ax = inset_axes(ax, width="42%", height="42%", loc="upper right", borderpad=1.1)
		zoom_min = None
		zoom_max = None
		for persona, values in series_by_persona.items():
			zoom_values = values[(values["epsilon_num"] >= 25) & (values["epsilon_num"] <= 30)]
			if zoom_values.empty:
				continue
			zoom_ax.plot(zoom_values["epsilon_num"], zoom_values[metric], marker="o", linewidth=1)
			persona_min = float(zoom_values[metric].min())
			persona_max = float(zoom_values[metric].max())
			zoom_min = persona_min if zoom_min is None else min(zoom_min, persona_min)
			zoom_max = persona_max if zoom_max is None else max(zoom_max, persona_max)

		zoom_ax.set_xlim(25, 30)
		if zoom_min is not None and zoom_max is not None:
			if zoom_min == zoom_max:
				pad = abs(zoom_min) * 0.05 if zoom_min != 0 else 1.0
			else:
				pad = (zoom_max - zoom_min) * 0.12
			zoom_ax.set_ylim(zoom_min - pad, zoom_max + pad)

		zoom_ax.set_title("eps 25-30", fontsize=9)
		zoom_ax.set_xticks([25, 26, 27, 28, 29, 30])
		zoom_ax.tick_params(labelsize=8)
		zoom_ax.grid(True, alpha=0.3)

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
