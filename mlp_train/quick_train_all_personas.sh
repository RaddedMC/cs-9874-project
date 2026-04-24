#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS_DIR="$ROOT_DIR/mlp_train/artifacts/train_yourself"
TRAIN_SCRIPT="$ROOT_DIR/mlp_train/train.py"

PERSONAS=(
	commuter
	early_shift
	gig_driver
	hybrid
	night_shift
	retiree
	social
	student
	traveler
	wfh
)

for persona in "${PERSONAS[@]}"; do
	echo "Training persona: $persona"

	python "$TRAIN_SCRIPT" \
		--epochs 80 \
		--batch-size 128 \
		--output-dir "$ARTIFACTS_DIR/$persona" \
		--persona "$persona" &
done

echo "Finished training all listed personas"