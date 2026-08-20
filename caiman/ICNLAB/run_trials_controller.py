# import subprocess
# import sys
# import os

# SCRIPT_PATH = r"C:\Users\ICNLab\CaImAn_GV\caiman\ICNLAB\test_single_trial_RAM_DISK_5.4_simple.py"

# mode = sys.argv[1]
# folders = sys.argv[2:]

# for folder in folders:
#     print(f"\n=== Processing {folder} ===", flush=True)

#     try:
#         subprocess.run(
#             ["python", SCRIPT_PATH, folder, mode],
#             check=True
#         )

#     except subprocess.CalledProcessError:
#         print(f"CaImAn failed on {folder}", flush=True)
#         continue
import subprocess
import sys
import os
import time
from pathlib import Path

#SCRIPT_PATH = r"C:\Users\ICNLab\CaImAn_GV\caiman\ICNLAB\test_single_trial_RAM_DISK_5.4_simple_resumable.py"

# Get the directory where run_trials_controller.py lives
current_dir = Path(__file__).resolve().parent

# Attach the specific script name and immediately convert to string for subprocess
SCRIPT_PATH = str(current_dir / "volpy_analysis_resumable.py")

mode = sys.argv[1]
fr = sys.argv[2] 
folders = sys.argv[3:]


for i, folder in enumerate(folders):
    print(f"\n=== Processing {folder} ===", flush=True)

    try:
        subprocess.run(
            ["python", SCRIPT_PATH, folder, mode, fr],  # Pass the frame rate as an argument
            check=True
        )

    except subprocess.CalledProcessError:
        print(f"CaImAn failed on {folder}", flush=True)
        continue

    # Rest 1 second between folders (not after last)
    if i < len(folders) - 1:
        print("\n--- Resting for 1 second before next folder ---\n", flush=True)
        time.sleep(1)

