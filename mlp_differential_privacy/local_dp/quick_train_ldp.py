import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "local_dp" / "models" / "local"

TRAIN_SCRIPT = PROJECT_ROOT / "privatize.py"

PERSONAS = ["commuter", "early_shift", "gig_driver", "hybrid", "night_shift", "retiree", "social", "student", "traveler", "wfh"]

processes = []
for persona in PERSONAS:
    for i in range(1, 31):
        epsilon = i
        print(f"Training local model for persona {persona} and epsilon {epsilon}")
        output_dir = OUTPUT_ARTIFACTS_DIR / f"epsilon-{epsilon}" / f"{persona}"
        command = (
            f'python "{TRAIN_SCRIPT}" '
            "--epochs 80 "
            "--batch-size 128 "
            f'--output-dir "{output_dir}" '
            f'--persona "{persona}" '
            '--mode dp-sgd '
            '--delta 8e-7 '
            f'--epsilon {epsilon} '
        )
        process = subprocess.Popen(command, shell=True)
        processes.append((epsilon, process))
        if i % 15 == 0:
            for epsilon, process in processes:
                process.wait()