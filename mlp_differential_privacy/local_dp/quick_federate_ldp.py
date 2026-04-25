import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "local_dp" / "models" / "federated"
MODELS_ROOT_DIR = PROJECT_ROOT / "artifacts" / "local_dp" / "models" / "local"

TRAIN_SCRIPT = PROJECT_ROOT.parent / "mlp_federate" / "federate_averaging.py"

PERSONAS = ["commuter", "early_shift", "gig_driver", "hybrid", "night_shift", "retiree", "social", "student", "traveler", "wfh"]

processes = []
for i in range(1, 31):
    epsilon = i
    print(f"Federating models for epsilon {epsilon}")
    output_dir = OUTPUT_ARTIFACTS_DIR / f"epsilon-{epsilon}"
    model_files_list = [f'"{MODELS_ROOT_DIR / f"epsilon-{epsilon}" / persona / "best_model.pt"}"' for persona in PERSONAS]
    command = (
        f'python "{TRAIN_SCRIPT}" '
        f'--checkpoints {" ".join(model_files_list)} '
        f'--output-dir "{output_dir}"'
    )
    # print(command)
    process = subprocess.Popen(command, shell=True)
    processes.append((epsilon, process))