import csv
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FEDERATED_ARTIFACTS_DIR = SCRIPT_DIR / "artifacts" / "federated"
OUTPUT_CSV = FEDERATED_ARTIFACTS_DIR / "test_pf_metrics.csv"


def epsilon_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    prefix = "epsilon-"
    if name.startswith(prefix):
        suffix = name[len(prefix) :]
        if suffix.isdigit():
            return (int(suffix), name)
    return (10**9, name)


def collect_rows() -> list[dict]:
    rows: list[dict] = []

    for epsilon_dir in sorted(FEDERATED_ARTIFACTS_DIR.glob("epsilon-*"), key=epsilon_sort_key):
        if not epsilon_dir.is_dir():
            continue

        for test_dir in sorted(epsilon_dir.glob("test_*")):
            if not test_dir.is_dir():
                continue

            metrics_path = test_dir / "predictions.metrics.json"
            if not metrics_path.is_file():
                continue

            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)

            row = {
                "epsilon": epsilon_dir.name,
                "test_name": test_dir.name,
                "metrics_file": str(metrics_path.relative_to(FEDERATED_ARTIFACTS_DIR)),
            }
            if isinstance(metrics, dict):
                row.update(metrics)
            rows.append(row)

    return rows


def main() -> None:
    rows = collect_rows()
    if not rows:
        print("No predictions.metrics.json files found.")
        return

    metric_keys = sorted({key for row in rows for key in row.keys()})

    FEDERATED_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metric_keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()