#!/usr/bin/env bash
set -euo pipefail

ARTIFACTS_DIR="mlp_train/artifacts"
OUTPUT_DIR="mlp_federate/artifacts"
TRAIN_SCRIPT="mlp_federate/federate_averaging.py"

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

# Build checkpoint paths from PERSONAS list
CHECKPOINTS=()
for persona in "${PERSONAS[@]}"; do
	CHECKPOINTS+=("$ARTIFACTS_DIR/train_yourself/$persona/best_model.pt")
done

python "$TRAIN_SCRIPT" \
    --checkpoints "${CHECKPOINTS[@]}" \
    --output-dir $OUTPUT_DIR \

echo "Finished"