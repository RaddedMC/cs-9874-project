import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_ARTIFACTS_DIR = PROJECT_ROOT / "mlp_differential_privacy" / "artifacts" / "federated"
INPUT_ARTIFACTS_DIR = PROJECT_ROOT / "mlp_federate" / "artifacts"
TRAIN_SCRIPT = PROJECT_ROOT / "mlp_differential_privacy" / "privatize.py"

processes = []
for i in range(21, 31):
    epsilon = i
    print(f"Starting training for epsilon {epsilon}")
    output_dir = OUTPUT_ARTIFACTS_DIR / f"epsilon-{epsilon}"
    command = (
        f'python "{TRAIN_SCRIPT}" '
        f'--mode post-hoc '
        f'--input-checkpoint "{INPUT_ARTIFACTS_DIR / "federated_model.pt"}" '
        f'--epsilon {epsilon} '
        f'--delta 8e-7 '
        f'--posthoc-mechanism gaussian '
        f'--weight-clip 1.0 '
        f'--output-dir "{output_dir}"'
    )
    process = subprocess.Popen(command, shell=True)
    processes.append((epsilon, process))

# Wait for all processes to complete
for epsilon, process in processes:
    process.wait()
    print(f"Completed epsilon {epsilon}")