# run_nasa_artifact.py

import subprocess
import sys

def run_script(script_name):
    print("\n" + "=" * 90)
    print(f"RUNNING: {script_name}")
    print("=" * 90)

    result = subprocess.run(
        [sys.executable, script_name],
        text=True,
        capture_output=True,
    )

    print(result.stdout)

    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with return code {result.returncode}")


if __name__ == "__main__":
    run_script("nasa_sensor_screening.py")
    run_script("nasa_compare_single_fusion.py")

    print("\n" + "=" * 90)
    print("NASA ARTIFACT RUN COMPLETED")
    print("=" * 90)
    print("Step 1: sensor screening completed.")
    print("Step 2: selected single/fusion comparison completed.")