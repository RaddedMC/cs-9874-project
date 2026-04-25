import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "local_dp" / "tests"
TEST_SCRIPT = PROJECT_ROOT.parent / "mlp_train" / "predict_experiments.py"

PERSONAS = ["commuter", "early_shift", "gig_driver", "hybrid", "night_shift", "retiree", "social", "student", "traveler", "wfh"]

processes = []
for persona in PERSONAS:
    for i in range(1, 31):
        epsilon = i
        print(f"Starting testing for persona {persona} and epsilon {epsilon}")
        model = OUTPUT_ARTIFACTS_DIR.parent / "models" / "federated" / f"epsilon-{epsilon}" / "federated_model.pt"
        output_csv = OUTPUT_ARTIFACTS_DIR / f"epsilon-{epsilon}" / f"test_{persona}" / "predictions.csv"
        command = (
            f'python "{TEST_SCRIPT}" '
            f'--checkpoint "{model}" '
            '--mode one_step '
            f'--persona {persona} '
            '--split test '
            '--preview-rows 3 '
            f'--output-csv "{output_csv}" '
        )
        process = subprocess.Popen(command, shell=True)
        processes.append((epsilon, process))
    process.wait()

# Wait for all processes to complete
for epsilon, process in processes:
    
    print(f"Completed epsilon {epsilon}")