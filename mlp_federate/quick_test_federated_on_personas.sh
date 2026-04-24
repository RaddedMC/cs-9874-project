#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS_DIR="$ROOT_DIR/mlp_federate/artifacts"
TEST_SCRIPT="$ROOT_DIR/mlp_train/predict_experiments.py"

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
	echo "Testing train and federate against persona: $persona"

	python "$TEST_SCRIPT" \
		--checkpoint "$ROOT_DIR/mlp_federate/artifacts/federated_model.pt" \
        --mode one_step \
		--persona $persona \
        --split test \
        --preview-rows 3 \
        --output-csv "$ARTIFACTS_DIR/test_$persona/predictions.csv" &
done

echo "Finished testing all listed personas"