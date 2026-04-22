# Experimenting with Federated Learning and Differential privacy in a Smart Home

## Usage instructions:
1. Install requirements
- Run `pip install -r requirements.txt`

2. Run training of one MLP:
- Choose any persona out of: `commuter, early_shift, gig_driver, hybrid, night_shift, retiree, social, student, teacher, traveler, wfh`
- Run `python mlp_train/train.py --epochs 80 --batch-size 128 --output-dir mlp_train/artifacts/[pick any persona] --persona [pick any persona]`

3. Run training of combined MLP
- Run `python mlp_train/train.py --epochs 80 --batch-size 128 --output-dir mlp_train/artifacts/all`

4. Run predictions on any MLP:
- Run `python mlp_train/predictions.py --checkpoint mlp_train/artifacts/[pick any persona or all]/best_model.pt --mode one_step --split test --preview-rows 3 --output-csv mlp_train/artifacts/[pick any persona or all]/predictions.csv`