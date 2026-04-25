import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run membership inference attacks for every model listed in targeted_models.json."
	)
	parser.add_argument(
		"--targeted-models",
		type=str,
		default="mlp_attack/targeted_models.json",
		help="Path to the JSON file containing the model attack configuration.",
	)
	parser.add_argument(
		"--attack-script",
		type=str,
		default="mlp_attack/membership_inference/perform_membership_inference.py",
		help="Path to the single-model membership inference script.",
	)
	return parser.parse_args()


def load_models_config(path: Path) -> List[dict]:
	if not path.exists():
		raise FileNotFoundError(f"Targeted models file not found: {path}")

	with path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)

	models = payload.get("models")
	if not isinstance(models, list) or not models:
		raise ValueError(f"No models found in configuration: {path}")

	return models


def run_attack(attack_script: Path, model_entry: dict) -> subprocess.Popen:
	model_location = model_entry.get("model_location")
	output_name = model_entry.get("output_name")
	attack_output = model_entry.get("attack_output")

	if not model_location or not output_name or not attack_output:
		raise ValueError(f"Invalid model entry: {model_entry}")

	attack_output_path = Path(attack_output)
	attack_output_path.mkdir(parents=True, exist_ok=True)

	command = [
		sys.executable,
		str(attack_script),
		"--model_location",
		str(model_location),
		"--output_name",
		str(output_name),
		"--attack_output",
		str(attack_output_path),
	]

	print(f"Running attack for {output_name}")
	print(f"  model_location: {model_location}")
	print(f"  attack_output: {attack_output_path}")
	return subprocess.Popen(command)


def main() -> None:
	args = parse_args()
	models_path = Path(args.targeted_models)
	attack_script = Path(args.attack_script)

	models = load_models_config(models_path)
	if not attack_script.exists():
		raise FileNotFoundError(f"Attack script not found: {attack_script}")

	processes = []
	for index, model_entry in enumerate(models, start=1):
		print(f"[{index}/{len(models)}] Starting attack")
		processes.append(run_attack(attack_script, model_entry))
		if index % 20 == 0:
			for process in processes:
				process.wait()
			processes = []

	for process in processes:
		process.wait()


if __name__ == "__main__":
	main()
