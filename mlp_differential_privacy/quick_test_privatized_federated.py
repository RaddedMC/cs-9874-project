import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_ARTIFACTS_DIR = PROJECT_ROOT / "mlp_differential_privacy" / "artifacts" / "federated"
TEST_SCRIPT = PROJECT_ROOT / "mlp_train" / "predict_experiments.py"

PERSONAS = ["commuter", "early_shift", "gig_driver", "hybrid", "night_shift", "retiree", "social", "student", "traveler", "wfh"]

processes = []
for persona in PERSONAS:
    for i in range(1, 21):
        epsilon = i/2
        print(f"Starting testing for persona {persona} and epsilon {epsilon}")
        output_dir = OUTPUT_ARTIFACTS_DIR / f"epsilon-{epsilon}" / f"test_{persona}"
        command = (
            f'python "{TEST_SCRIPT}" '
            f'--checkpoint "{OUTPUT_ARTIFACTS_DIR / f"epsilon-{epsilon}" / "best_model_posthoc.pt"}" '
            '--mode one_step '
            f'--persona {persona} '
            '--split test '
            '--preview-rows 3 '
            f'--output-csv "{output_dir}/predictions.csv" '
        )
        process = subprocess.Popen(command, shell=True)
        processes.append((epsilon, process))
    process.wait()

# Wait for all processes to complete
for epsilon, process in processes:
    
    print(f"Completed epsilon {epsilon}")