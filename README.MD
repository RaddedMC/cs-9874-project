# Experimenting with Federated Learning and Differential privacy in a Smart Home

## Usage instructions:
1. Install requirements
- Optional: Create a Virtual Environment
    - `python -m venv .venv`
    - `source .venv/bin/activate`
- Run `pip install -r requirements.txt`

2. Run training of one MLP:
- Choose any persona out of: `commuter, early_shift, gig_driver, hybrid, night_shift, retiree, social, student, traveler, wfh`
- Run `python mlp_train/train.py --epochs 80 --batch-size 128 --output-dir mlp_train/artifacts/[pick any persona] --persona [pick any persona]`

3. Run training of combined MLP
- Run `python mlp_train/train.py --epochs 80 --batch-size 128 --output-dir mlp_train/artifacts/all`

4. Run predictions on any MLP:
- Run `python mlp_train/predictions.py --checkpoint mlp_train/artifacts/[train_yourself/persona or all]/best_model.pt --persona [persona] --mode one_step --split test --preview-rows 3 --output-csv mlp_train/artifacts/[train_yourself/persona or all]/predictions.csv`
- There are also quick shortcut scripts `quick_test_all_to_personas.sh` and `quick_test_all_personas.sh` to help with this.

5. Run Federated Learning on any set of MLPs
- Run `python mlp_federate/federate_averaging.py --checkpoints [mlp_train/artifacts/train_yourself/[persona]/best_model.pt for all desired personas] --output-dir mlp_federate/artifacts/`
- There is a quick shortcut script `quick_federate_all_personas.sh` to help with this.

6. Test Federated Learning on the persona datasets
- Again I used a quick shortcut script to do this `quick_test_federated_on_personas.sh`

7. Test Federated Learning on all persona datasets
- Run `python mlp_train/predict_experiments.py --checkpoint mlp_federate/artifacts/federated_model.pt --mode one_step --split test --preview-rows 3 --output-csv mlp_federate/artifacts/all/predictions.csv --persona all`

8. Run Differential Privacy in post-hoc mode on the federated model
- Run `python mlp_differential_privacy/privatize.py --mode post-hoc --input-checkpoint mlp_federate/artifacts/federated_model.pt --epsilon [your desired epsilon] --delta 1e-5 --posthoc-mechanism gaussian --weight-clip 1.0 --output-dir mlp_differential_privacy/artifacts/federated/epsilon-[your desired epsilon]`
- You can also run for the full range of epsilon values with `mlp_federate/federated/quick_privatize_federated.py`

9. Test Differential Privacy on the previous model
- I used the script `mlp_federate/federated/quick_test_privatized_federated.py` to create 300 model tests -- one for each Epsilon level and persona
- I then extracted the data out of each test file into a single file with `mlp_federate/federated/quick_collect_test_pf.py` and finally created a graph with `mlp_federate/federated/quick_graph_test_pf.py`

10. Run Differential Privacy in SGD mode on every persona and DP level
- Another script! `mlp_differential_privacy/local_dp/quick_train_ldp.py`

11. Federate the local dp models
- `mlp_differential_privacy/local_dp/quick_federate_ldp.py`

12. Test and graph the federated models
- `mlp_differential_privacy/local_dp/quick_test_ldp.py`
- `mlp_differential_privacy/local_dp/quick_collect_test_ldp.py`
- `mlp_differential_privacy/local_dp/quick_graph_test_ldp.py`