"""
comparison/compare.py
---------------------
High-level analysis pipeline: single task-set analysis and batch sweeps.

Exported symbols
----------------
analyze_single_taskset(csv_file, ...)  -> dict
process_taskset_dir_batch(taskset_dir) -> dict
batch_analysis(base_dir, ...)          -> dict[float, dict]
print_batch_summary(label, results)
"""

import glob as glob_module
import os
import sys

# Allow importing sibling packages when this module is used standalone
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tasks.generator      import load_tasks, calculate_hyperperiod
from analysis.dm_wcrt     import dm_response_time_analysis
from analysis.edf_wcrt    import edf_utilization_test, edf_processor_demand_test
from simulation.simulator import (edf_wcrt_simulation, stochastic_simulation,
                                  collect_rt_distributions)
from reporting.table      import print_separator, print_task_table


# ---------------------------------------------------------------------------
# Single task-set analysis
# ---------------------------------------------------------------------------

def analyze_single_taskset(csv_file: str, n_stochastic_runs: int = 500,
                            verbose: bool = True) -> dict:
    """
    Full analysis of one task set:

    1. DM Response Time Analysis (analytical)
    2. EDF schedulability via PDC
    3. EDF WCRT via hyperperiod simulation
    4. DM stochastic simulation  (log-normal exec times)
    5. EDF stochastic simulation (log-normal exec times)
    6. RT distribution collection for CDF plots

    Parameters
    ----------
    csv_file           : path to the task-set CSV file
    n_stochastic_runs  : Monte Carlo run count  (default 500)
    verbose            : print progress and tables to stdout

    Returns
    -------
    dict with keys:
        tasks, U, H,
        dm_sched, edf_sched,
        dm_wcrt, edf_wcrt,
        dm_sim_max, edf_sim_max,
        dm_avg_missed, edf_avg_missed,
        dm_rt_dist, edf_rt_dist
    """
    tasks = load_tasks(csv_file)
    if not tasks:
        return None

    U = sum(t.wcet / t.period for t in tasks)
    try:
        H = calculate_hyperperiod(tasks)
    except Exception:
        H = None

    if verbose:
        print_separator()
        print(f"  Task Set: {os.path.basename(csv_file)}")
        print(f"  Tasks: {len(tasks)}   U = {U:.4f}   Hyperperiod: {H}")
        print_separator()

    # --- 1. DM Response Time Analysis ---
    if verbose:
        print("\n[1] Deadline Monotonic -- Response Time Analysis (RTA)")
    dm_wcrt  = dm_response_time_analysis(tasks)
    dm_sched = all(v != float('inf') for v in dm_wcrt.values())
    if verbose:
        print(f"    Schedulable under DM: {dm_sched}")
        if dm_sched:
            max_dm = max(dm_wcrt.values())
            util_n = sum(t.wcet / t.deadline for t in tasks)
            print(f"    Max WCRT = {max_dm}  "
                  f"(utilization bound U_n = {util_n:.4f})")

    # --- 2. EDF Schedulability (PDC) ---
    # Using only the utilization test on constrained-deadline sets would
    # over-report schedulability; the PDC is always correct.
    if verbose:
        print("\n[2] EDF Schedulability")
    _, edf_U = edf_utilization_test(tasks)
    edf_sched, edf_reason = edf_processor_demand_test(tasks)
    if verbose:
        print(f"    U = {edf_U:.4f}  PDC: {edf_reason}  "
              f"->  EDF schedulable: {edf_sched}")

    # --- 3. EDF WCRT via hyperperiod simulation ---
    if verbose:
        print("\n[3] EDF WCRT -- Hyperperiod Simulation (Appendix A)")
    if edf_sched:
        if verbose:
            print(f"    Simulating over H = {H} ...")
        edf_wcrt = edf_wcrt_simulation(tasks)
        edf_max  = max(
            (v for v in edf_wcrt.values() if v != float('inf')),
            default=float('inf')
        )
        if verbose:
            print(f"    Max EDF WCRT = {edf_max}")
    else:
        edf_wcrt = {t.id: float('inf') for t in tasks}
        if verbose:
            print("    Skipped (U > 1: not EDF schedulable)")

    # --- 4. DM Stochastic Simulation ---
    if verbose:
        print(f"\n[4] DM Stochastic Simulation "
              f"({n_stochastic_runs} runs, log-normal exec times)")
    dm_sim_max, dm_avg_missed = stochastic_simulation(
        tasks, policy="DM", n_runs=n_stochastic_runs, exec_dist="lognormal"
    )
    if verbose:
        print(f"    Max observed RT = {max(dm_sim_max.values())}")
        print(f"    Avg missed deadlines/run = "
              f"{sum(dm_avg_missed.values()):.2f}")

    # --- 5. EDF Stochastic Simulation ---
    if verbose:
        print(f"\n[5] EDF Stochastic Simulation "
              f"({n_stochastic_runs} runs, log-normal exec times)")
    edf_sim_max, edf_avg_missed = stochastic_simulation(
        tasks, policy="EDF", n_runs=n_stochastic_runs, exec_dist="lognormal"
    )
    if verbose:
        print(f"    Max observed RT = {max(edf_sim_max.values())}")
        print(f"    Avg missed deadlines/run = "
              f"{sum(edf_avg_missed.values()):.2f}")

    # --- 6. Collect full RT distributions for CDF plots ---
    if verbose:
        print("\n[6] Collecting RT distributions for CDF plots (200 runs) ...")
    dm_rt_dist, edf_rt_dist = collect_rt_distributions(tasks, n_runs=200)

    # --- Summary table ---
    if verbose:
        print("\n[Summary Table]")
        print_task_table(
            tasks,
            dm_wcrt=dm_wcrt,
            edf_wcrt=edf_wcrt if edf_sched else None,
            dm_sim=dm_sim_max,
            edf_sim=edf_sim_max,
        )

        # WCRT ratio: simulation / analytical
        if dm_sched:
            print("\n[DM] Analytical WCRT vs. Stochastic Max RT")
            print(f"  {'Task':>5} {'DM-WCRT':>10} "
                  f"{'Sim-MaxRT':>10} {'Ratio':>8}  {'Status'}")
            print("  " + "-" * 42)
            for t in sorted(tasks, key=lambda x: (x.deadline, x.id)):
                dw    = dm_wcrt[t.id]
                sr    = dm_sim_max.get(t.id, 0)
                ratio = sr / dw if dw > 0 else float('nan')
                ok    = "OK" if sr <= dw else "!"
                print(f"  {t.id:>5} {dw:>10} {sr:>10} {ratio:>8.3f}  {ok}")

        if edf_sched:
            print("\n[EDF] Analytical WCRT vs. Stochastic Max RT")
            print(f"  {'Task':>5} {'EDF-WCRT':>10} "
                  f"{'Sim-MaxRT':>10} {'Ratio':>8}  {'Status'}")
            print("  " + "-" * 42)
            for t in sorted(tasks, key=lambda x: (x.deadline, x.id)):
                ew    = edf_wcrt[t.id]
                sr    = edf_sim_max.get(t.id, 0)
                ratio = (sr / ew
                         if ew not in (0, float('inf')) else float('nan'))
                ok    = ("OK"
                         if ew == float('inf') or sr <= ew else "!")
                print(f"  {t.id:>5} {ew:>10} {sr:>10} {ratio:>8.3f}  {ok}")

    return {
        'tasks':          tasks,
        'U':              U,
        'H':              H,
        'dm_sched':       dm_sched,
        'edf_sched':      edf_sched,
        'dm_wcrt':        dm_wcrt,
        'edf_wcrt':       edf_wcrt,
        'dm_sim_max':     dm_sim_max,
        'edf_sim_max':    edf_sim_max,
        'dm_avg_missed':  dm_avg_missed,
        'edf_avg_missed': edf_avg_missed,
        'dm_rt_dist':     dm_rt_dist,
        'edf_rt_dist':    edf_rt_dist,
    }


# ---------------------------------------------------------------------------
# Batch processing (analytical only)
# ---------------------------------------------------------------------------

def process_taskset_dir_batch(taskset_dir: str, n_files: int = 100) -> dict:
    """
    Analytical-only batch processing of all task sets in *taskset_dir*.

    EDF schedulability uses the PDC (identical to the single-set path)
    so that batch numbers remain correct for constrained-deadline sets.
    For the shipped implicit-deadline benchmark the PDC fast-paths to the
    utilization test, so there is no runtime overhead.

    Returns
    -------
    dict with aggregate statistics or None if no CSV files found.
    """
    csv_files = sorted(
        glob_module.glob(os.path.join(taskset_dir, "*.csv"))
    )[:n_files]
    if not csv_files:
        return None

    stats = {
        'total':                0,
        'dm_schedulable':       0,
        'edf_schedulable':      0,      # PDC verdict (authoritative)
        'edf_schedulable_U_only': 0,    # bare U<=1 — kept for cross-check
        'dm_wcrt_ratios':       [],
        'actual_utils':         [],
    }

    for csv_file in csv_files:
        try:
            tasks = load_tasks(csv_file)
            if not tasks:
                continue
        except Exception:
            continue

        stats['total'] += 1
        U = sum(t.wcet / t.period for t in tasks)
        stats['actual_utils'].append(U)

        # DM: exact Response Time Analysis
        dm_wcrt = dm_response_time_analysis(tasks)
        if all(v != float('inf') for v in dm_wcrt.values()):
            stats['dm_schedulable'] += 1

        # EDF: Processor Demand Criterion
        edf_ok_pdc, _ = edf_processor_demand_test(tasks)
        if edf_ok_pdc:
            stats['edf_schedulable'] += 1

        # EDF: bare utilization test (for cross-check / flag column)
        edf_ok_u, _ = edf_utilization_test(tasks)
        if edf_ok_u:
            stats['edf_schedulable_U_only'] += 1

    return stats


def batch_analysis(base_dir: str, util_dist: str, per_dist: str,
                   n_task: str = "25-task",
                   jitter: str = "0-jitter") -> dict:
    """
    Process all utilisation levels for a given distribution configuration.

    Returns
    -------
    dict[float, dict]
        {util_level: stats_dict}
    """
    path = os.path.join(base_dir, util_dist, per_dist,
                        "1-core", n_task, jitter)
    if not os.path.exists(path):
        print(f"  Path not found: {path}")
        return {}

    util_entries = sorted(
        [d for d in os.listdir(path) if d.endswith('-util')]
    )
    all_results: dict = {}

    for entry in util_entries:
        util_val    = float(entry.replace('-util', ''))
        taskset_dir = os.path.join(path, entry, "tasksets")
        if not os.path.exists(taskset_dir):
            continue

        print(f"    util={util_val:.2f} ...", end=' ', flush=True)
        stats = process_taskset_dir_batch(taskset_dir)
        if stats and stats['total'] > 0:
            all_results[util_val] = stats
            pct_dm  = stats['dm_schedulable']  / stats['total'] * 100
            pct_edf = stats['edf_schedulable'] / stats['total'] * 100
            print(f"DM={pct_dm:.0f}%  EDF={pct_edf:.0f}%  "
                  f"(n={stats['total']})")
        else:
            print("no data")

    return all_results


# ---------------------------------------------------------------------------
# Batch summary printer
# ---------------------------------------------------------------------------

def print_batch_summary(label: str, results: dict) -> None:
    """
    Print per-utilisation schedulability summary.

    EDF% uses the PDC verdict (authoritative).  A trailing '*' means the
    bare U<=1 test agreed with PDC for every task set at that level; '!'
    means at least one task set was accepted by U<=1 but rejected by PDC
    (the utilisation test over-reported schedulability).
    """
    if not results:
        print(f"  No results for {label}")
        return

    print(f"\n{label} -- Schedulability Summary")
    print(f"  {'Util':>6} {'N':>5} {'DM%':>7} {'EDF%':>7}  "
          f"{'Dominance':<10}  PDC==U?")
    print("  " + "-" * 50)

    for u in sorted(results.keys()):
        s       = results[u]
        n       = s['total']
        dm_pct  = s['dm_schedulable']  / n * 100
        edf_pct = s['edf_schedulable'] / n * 100
        note    = ("EDF dom." if edf_pct > dm_pct
                   else ("equal" if edf_pct == dm_pct else "DM dom."))
        agree   = (s['edf_schedulable']
                   == s.get('edf_schedulable_U_only', s['edf_schedulable']))
        flag    = "*" if agree else "! (PDC stricter)"
        print(f"  {u:>6.2f} {n:>5} {dm_pct:>6.1f}% {edf_pct:>6.1f}%  "
              f"{note:<10}  {flag}")
