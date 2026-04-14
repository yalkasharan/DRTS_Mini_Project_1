#!/usr/bin/env python3
"""
DRTS Mini-Project 1: DM vs EDF Scheduling Analysis
02225 Distributed Real-Time Systems

Entry point.  All implementation lives in the sibling packages:

    tasks/       – Task model and CSV I/O         (tasks/generator.py)
    analysis/    – DM RTA and EDF PDC tests        (analysis/dm_wcrt.py,
                                                    analysis/edf_wcrt.py)
    simulation/  – Event-driven simulation engine  (simulation/simulator.py)
    reporting/   – Console tables and figures      (reporting/table.py)
    comparison/  – Analysis pipeline and batch     (comparison/compare.py)

Usage
-----
    cd DRTS_Mini_Project1/Main_code
    python drts_project.py

To analyse a single CSV file programmatically:

    from comparison.compare import analyze_single_taskset
    result = analyze_single_taskset("path/to/taskset.csv",
                                    n_stochastic_runs=500, verbose=True)

To run a batch sweep:

    from comparison.compare import batch_analysis, print_batch_summary
    results = batch_analysis("task-sets/output",
                              "automotive-utilDist", "automotive-perDist")
    print_batch_summary("Automotive", results)
"""

import glob as glob_module
import os
import random
import sys

# ---------------------------------------------------------------------------
# Make packages importable regardless of the working directory.
# All packages (tasks/, analysis/, simulation/, reporting/, comparison/)
# live inside the same directory as this script (Main_code/).
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------
from comparison.compare import (
    analyze_single_taskset,
    batch_analysis,
    print_batch_summary,
)
from reporting.table import (
    print_separator,
    plot_schedulability,
    plot_wcrt_comparison,
    plot_sim_vs_analytical,
    plot_rt_distributions,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Fix random seed for full reproducibility across runs and platforms.
    # All stochastic sampling (random.lognormvariate / random.randint) in
    # simulation/simulator.py uses the Python random module, so seeding here
    # — before any simulation call — makes every result deterministic.
    random.seed(42)

    tasksets_root = os.path.join(_SCRIPT_DIR, "task-sets", "output")

    print_separator()
    print("  DRTS Mini-Project 1: DM vs EDF Scheduling Analysis")
    print_separator()

    # ---- Demo: detailed analysis of one task set ----
    demo_candidates = [
        os.path.join(tasksets_root,
                     "automotive-utilDist", "automotive-perDist",
                     "1-core", "25-task", "0-jitter",
                     "0.50-util", "tasksets", "automotive_0.csv"),
        os.path.join(tasksets_root,
                     "uunifast-utilDist", "uniform-discrete-perDist",
                     "1-core", "25-task", "0-jitter",
                     "0.50-util", "tasksets", "automotive_0.csv"),
    ]

    demo_result = None
    for demo_csv in demo_candidates:
        if os.path.exists(demo_csv):
            print(f"\nDemo Analysis (single task set): {demo_csv}")
            demo_result = analyze_single_taskset(
                demo_csv, n_stochastic_runs=500, verbose=True
            )
            break

    if demo_result is None:
        local = glob_module.glob(os.path.join(_SCRIPT_DIR, "*.csv"))
        if local:
            print(f"\nDemo Analysis: {local[0]}")
            demo_result = analyze_single_taskset(
                local[0], n_stochastic_runs=500, verbose=True
            )
        else:
            print("\nNo task-set CSV found for demo. "
                  "Skipping single-set analysis.")

    # Save per-task-set plots next to this script
    if demo_result:
        print("\nGenerating WCRT comparison plots...")
        plot_wcrt_comparison(demo_result,    output_dir=_SCRIPT_DIR)
        plot_sim_vs_analytical(demo_result,  output_dir=_SCRIPT_DIR)
        print("Generating RT distribution (CDF) plots...")
        plot_rt_distributions(
            demo_result['dm_rt_dist'],
            demo_result['edf_rt_dist'],
            demo_result['tasks'],
            dm_wcrt  = demo_result['dm_wcrt']  if demo_result['dm_sched']  else None,
            edf_wcrt = demo_result['edf_wcrt'] if demo_result['edf_sched'] else None,
            output_dir = _SCRIPT_DIR,
        )

    # ---- Batch analysis ----
    print_separator()
    print("  BATCH ANALYSIS -- All Utilization Levels")
    print_separator()

    results_auto: dict = {}
    results_uni:  dict = {}

    auto_path = os.path.join(tasksets_root,
                             "automotive-utilDist", "automotive-perDist")
    uni_path  = os.path.join(tasksets_root,
                             "uunifast-utilDist", "uniform-discrete-perDist")

    if os.path.exists(auto_path):
        print("\nAutomotive Distribution:")
        results_auto = batch_analysis(
            tasksets_root, "automotive-utilDist", "automotive-perDist"
        )
        print_batch_summary("Automotive", results_auto)
    else:
        print(f"\nAutomotive path not found: {auto_path}")

    if os.path.exists(uni_path):
        print("\nUUniFast Distribution:")
        results_uni = batch_analysis(
            tasksets_root, "uunifast-utilDist", "uniform-discrete-perDist"
        )
        print_batch_summary("UUniFast", results_uni)
    else:
        print(f"\nUUniFast path not found: {uni_path}")

    # ---- Schedulability plot ----
    if results_auto or results_uni:
        print("\nGenerating schedulability plot...")
        plot_schedulability(results_auto, results_uni, output_dir=_SCRIPT_DIR)

    print_separator()
    print("  Done.")
    print_separator()


if __name__ == "__main__":
    main()
