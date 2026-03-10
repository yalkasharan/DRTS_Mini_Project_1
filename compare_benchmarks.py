# compare_benchmarks.py
import pickle
import os
import visualization.plots as plots_module
from visualization.plots import plot_benchmark_comparison

plots_module.RESULTS_DIR = "results"
os.makedirs("results", exist_ok=True)

auto_pkl = "results/automotive/results.pkl"
uni_pkl  = "results/uunifast/results.pkl"

if not os.path.exists(auto_pkl):
    print(f"[ERROR] Not found: {auto_pkl}")
    print("  Run: python main.py --csv-folder tasksets\\automotive --output results\\automotive")
    exit(1)

if not os.path.exists(uni_pkl):
    print(f"[ERROR] Not found: {uni_pkl}")
    print("  Run: python main.py --csv-folder tasksets\\uunifast --output results\\uunifast")
    exit(1)

with open(auto_pkl, "rb") as f:
    auto = pickle.load(f)

with open(uni_pkl, "rb") as f:
    uni = pickle.load(f)

print("Generating benchmark comparison plot...")
plot_benchmark_comparison(auto, uni)
print("Done → results/benchmark_comparison.png")
