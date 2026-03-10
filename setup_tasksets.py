# setup_tasksets.py
import os
import shutil
import sys

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BENCHMARKS = {
    "automotive": r"task-sets\output\automotive-utilDist\automotive-perDist\1-core\25-task\0-jitter",
    "uunifast":   r"task-sets\output\uunifast-utilDist\uniform-discrete-perDist\1-core\25-task\0-jitter",
}

UTIL_FOLDERS = [
    "0.10-util", "0.20-util", "0.30-util", "0.40-util", "0.50-util",
    "0.60-util", "0.70-util", "0.80-util", "0.90-util", "1.00-util"
]

FILES_PER_UTIL = 5  # ← change this: 1=fast, 5=good, 10=thorough, 100=full

for bench_name, base_path in BENCHMARKS.items():
    out_dir = os.path.join("tasksets", bench_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n── {bench_name.upper()} ──────────────────")

    for util in UTIL_FOLDERS:
        folder = os.path.join(base_path, util, "tasksets")
        if not os.path.exists(folder):
            print(f"  [SKIP] Not found: {folder}")
            continue

        # Skip macOS junk files
        csvfiles = sorted([
            f for f in os.listdir(folder)
            if f.endswith(".csv") and not f.startswith("._")
        ])

        if not csvfiles:
            print(f"  [SKIP] No valid CSVs in: {util}")
            continue

        # Copy up to FILES_PER_UTIL files
        selected = csvfiles[:FILES_PER_UTIL]
        for fname in selected:
            src = os.path.join(folder, fname)
            # Name: automotive/0.50-util_001.csv
            base_fname = fname.replace(".csv", "")
            dst = os.path.join(out_dir, f"{util}_{base_fname}.csv")
            shutil.copy(src, dst)

        print(f"  ✓ {util} → copied {len(selected)} files")

print("\n── Done ───────────────────────────────────────")
print("Run: python main.py --csv-folder tasksets\\automotive")
print("Run: python main.py --csv-folder tasksets\\uunifast")
