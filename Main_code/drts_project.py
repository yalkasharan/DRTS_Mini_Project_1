#!/usr/bin/env python3
"""
DRTS Mini-Project 1: DM vs EDF Scheduling Analysis
02225 Distributed Real-Time Systems
"""

import csv
import math
import heapq
import random
import os
import glob as glob_module
import sys
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib/numpy not available. Install with: pip install matplotlib numpy")

# ==========================================
# 1. Data Structures
# ==========================================

class Task:
    def __init__(self, t_id, bcet, wcet, period, deadline):
        self.id = t_id
        self.bcet = int(bcet)
        self.wcet = int(wcet)
        self.period = int(period)
        self.deadline = int(deadline)

    def __repr__(self):
        return f"T{self.id}(C=[{self.bcet},{self.wcet}], T={self.period}, D={self.deadline})"


# ==========================================
# 2. Analytical Tools
# ==========================================

def calculate_hyperperiod(tasks):
    """LCM of all task periods."""
    periods = [t.period for t in tasks]
    lcm = periods[0]
    for p in periods[1:]:
        lcm = lcm * p // math.gcd(lcm, p)
    return lcm


def dm_response_time_analysis(tasks):
    """
    Response Time Analysis for Deadline Monotonic scheduling.
    Priority: shorter relative deadline = higher priority.
    Returns {task_id: WCRT} where WCRT = inf means unschedulable.
    Based on Buttazzo Eq. 4.17-4.19.
    """
    sorted_tasks = sorted(tasks, key=lambda x: (x.deadline, x.id))
    wcrt = {}

    for i, task in enumerate(sorted_tasks):
        hp = sorted_tasks[:i]       # higher-priority tasks
        R = task.wcet               # initial guess = WCET

        while True:
            R_new = task.wcet + sum(math.ceil(R / h.period) * h.wcet for h in hp)
            if R_new > task.deadline:
                wcrt[task.id] = float('inf')
                break
            if R_new == R:
                wcrt[task.id] = R
                break
            R = R_new

    return wcrt


def edf_utilization_test(tasks):
    """
    EDF schedulability via utilization bound.
    Necessary and sufficient when Di <= Ti for all tasks.
    Returns (schedulable: bool, U: float).
    """
    U = sum(t.wcet / t.period for t in tasks)
    return U <= 1.0 + 1e-9, U


def edf_processor_demand_test(tasks):
    """
    Processor Demand Criterion (PDC) for constrained-deadline EDF.
    Checks dbf(t) <= t at all deadline checkpoints up to min(H, L*).
    Returns (schedulable: bool, reason: str).
    """
    sched, U = edf_utilization_test(tasks)
    if not sched:
        return False, f"U={U:.4f} > 1.0"

    # For implicit deadlines (Di = Ti), utilization test is sufficient
    if all(t.deadline == t.period for t in tasks):
        return True, f"U={U:.4f} (implicit deadlines)"

    try:
        H = calculate_hyperperiod(tasks)
    except Exception:
        return sched, f"U={U:.4f} (hyperperiod overflow)"

    # Cap to avoid excessive computation
    MAX_L = min(H, 10 * max(t.period for t in tasks))

    checkpoints = set()
    for t in tasks:
        d = t.deadline
        while d <= MAX_L:
            checkpoints.add(d)
            d += t.period

    for L in sorted(checkpoints):
        demand = 0
        for t in tasks:
            if L >= t.deadline:
                demand += math.floor((L + t.period - t.deadline) / t.period) * t.wcet
        if demand > L + 1e-9:
            return False, f"dbf({L})={demand} > {L}"

    return True, f"U={U:.4f}"


# ==========================================
# 3. Event-Driven Simulator
# ==========================================

# Maximum hyperperiod for simulation (safety cap)
MAX_HYPERPERIOD = 10 ** 9   # ~1000 seconds in microseconds


def _event_driven_core(tasks, policy, exec_fn, duration):
    """
    Core event-driven simulation engine.

    policy   : "DM" or "EDF"
    exec_fn  : callable(Task) -> int  (execution time for each job)
    duration : int, simulation end time

    Returns (response_times: {task_id: [rt, ...]}, missed: {task_id: count})
    """
    response_times = {t.id: [] for t in tasks}
    missed = {t.id: 0 for t in tasks}

    # Pre-generate all job releases within [0, duration)
    # Each entry: (release_time, rel_deadline, task_id, abs_deadline, exec_time)
    releases = []
    for task in tasks:
        t = 0
        while t < duration:
            exec_t = exec_fn(task)
            releases.append((t, task.deadline, task.id, t + task.deadline, exec_t))
            t += task.period
    releases.sort()

    rel_ptr = 0
    ready = []          # min-heap of (priority_key, job_dict)
    counter = 0         # unique tiebreaker
    cur_key = None      # priority key of currently running job
    cur_job = None      # currently running job dict
    time = 0

    while rel_ptr < len(releases) or ready or cur_job is not None:
        # Compute next event time
        if rel_ptr < len(releases):
            next_rel = releases[rel_ptr][0]
        else:
            next_rel = float('inf')

        if cur_job is not None:
            next_fin = time + cur_job['remaining']
        else:
            next_fin = float('inf')

        next_t = min(next_rel, next_fin)
        if next_t == float('inf') or next_t > duration:
            break

        # Advance current job's remaining time
        if cur_job is not None:
            cur_job['remaining'] -= next_t - time
        time = next_t

        # Release all jobs whose release_time <= time
        while rel_ptr < len(releases) and releases[rel_ptr][0] <= time:
            rel_time, rel_dl, tid, abs_dl, exec_t = releases[rel_ptr]
            rel_ptr += 1
            if rel_time >= duration:
                continue
            job = {
                'task_id': tid,
                'release': rel_time,
                'abs_dl': abs_dl,
                'rel_dl': rel_dl,
                'remaining': exec_t,
            }
            key = (rel_dl, tid, counter) if policy == "DM" else (abs_dl, tid, counter)
            counter += 1
            heapq.heappush(ready, (key, job))

        # Handle completion of current job
        if cur_job is not None and cur_job['remaining'] <= 0:
            rt = time - cur_job['release']
            response_times[cur_job['task_id']].append(rt)
            if time > cur_job['abs_dl']:
                missed[cur_job['task_id']] += 1
            cur_job = None
            cur_key = None

        # Select/preempt: pick job with lowest priority key
        if ready:
            top_key = ready[0][0]
            if cur_job is None:
                cur_key, cur_job = heapq.heappop(ready)
            elif top_key < cur_key:
                # Preempt current: push back, pop new minimum
                heapq.heappush(ready, (cur_key, cur_job))
                cur_key, cur_job = heapq.heappop(ready)

    return response_times, missed


def edf_wcrt_simulation(tasks):
    """
    Compute EDF WCRTs by simulating with WCET over one hyperperiod (Appendix A method).
    Returns {task_id: WCRT} where WCRT = inf if not schedulable or hyperperiod too large.
    """
    try:
        H = calculate_hyperperiod(tasks)
    except Exception:
        return {t.id: float('inf') for t in tasks}

    if H > MAX_HYPERPERIOD:
        # Fall back to a long but bounded duration
        H = min(H, MAX_HYPERPERIOD)
        print(f"  [Warning] Hyperperiod capped at {H} for EDF WCRT simulation.")

    rt_dict, missed = _event_driven_core(
        tasks, policy="EDF",
        exec_fn=lambda t: t.wcet,
        duration=H
    )

    wcrt = {}
    for tid, rts in rt_dict.items():
        wcrt[tid] = max(rts) if rts else float('inf')
    return wcrt


def dm_wcrt_simulation(tasks):
    """
    Compute DM WCRTs by simulating with WCET over one hyperperiod.
    (Cross-check against RTA.)
    Returns {task_id: WCRT}.
    """
    try:
        H = calculate_hyperperiod(tasks)
    except Exception:
        return {t.id: float('inf') for t in tasks}

    H = min(H, MAX_HYPERPERIOD)

    rt_dict, _ = _event_driven_core(
        tasks, policy="DM",
        exec_fn=lambda t: t.wcet,
        duration=H
    )

    wcrt = {}
    for tid, rts in rt_dict.items():
        wcrt[tid] = max(rts) if rts else float('inf')
    return wcrt


def stochastic_simulation(tasks, policy="DM", n_runs=500, duration=None):
    """
    Monte Carlo simulation with execution times drawn uniformly from [BCET, WCET].

    n_runs   : number of independent simulation runs
    duration : per-run simulation duration (default: hyperperiod, capped)

    Returns:
        max_rt  : {task_id: max observed response time across all runs}
        avg_missed : {task_id: average deadline misses per run}
    """
    if duration is None:
        try:
            H = calculate_hyperperiod(tasks)
            duration = min(H, MAX_HYPERPERIOD)
        except Exception:
            duration = 10 * max(t.period for t in tasks)

    max_rt = {t.id: 0 for t in tasks}
    total_missed = {t.id: 0 for t in tasks}

    def sample_exec(task):
        if task.bcet >= task.wcet:
            return task.wcet
        return random.randint(task.bcet, task.wcet)

    for _ in range(n_runs):
        rt_dict, missed_dict = _event_driven_core(
            tasks, policy=policy,
            exec_fn=sample_exec,
            duration=duration
        )
        for tid in max_rt:
            if rt_dict[tid]:
                max_rt[tid] = max(max_rt[tid], max(rt_dict[tid]))
            total_missed[tid] += missed_dict[tid]

    avg_missed = {tid: total_missed[tid] / n_runs for tid in total_missed}
    return max_rt, avg_missed


# ==========================================
# 4. I/O
# ==========================================

def load_tasks(filename):
    """Load tasks from CSV. Format: TaskID,Jitter,BCET,WCET,Period,Deadline,PE"""
    tasks = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get('TaskID', '').strip()
            if not tid:
                continue
            tasks.append(Task(
                t_id=int(tid),
                bcet=int(row['BCET']),
                wcet=int(row['WCET']),
                period=int(row['Period']),
                deadline=int(row['Deadline'])
            ))
    return tasks


def print_separator(char='=', width=76):
    print(char * width)


def print_task_table(tasks, dm_wcrt=None, edf_wcrt=None,
                     dm_sim=None, edf_sim=None, dm_missed=None, edf_missed=None):
    """Print a formatted comparison table."""
    sorted_tasks = sorted(tasks, key=lambda t: (t.deadline, t.id))

    cols = [("Task", 5), ("WCET", 8), ("Period", 10), ("Deadline", 10)]
    if dm_wcrt:
        cols.append(("DM-WCRT", 10))
    if edf_wcrt:
        cols.append(("EDF-WCRT", 10))
    if dm_sim:
        cols.append(("DM-SimRT", 10))
    if edf_sim:
        cols.append(("EDF-SimRT", 10))
    if dm_missed:
        cols.append(("DM-Miss", 8))
    if edf_missed:
        cols.append(("EDF-Miss", 9))

    header = " | ".join(f"{name:>{width}}" for name, width in cols)
    print(header)
    print("-" * len(header))

    for t in sorted_tasks:
        def fmt(d, tid, width):
            if d is None:
                return f"{'N/A':>{width}}"
            v = d.get(tid, 'N/A')
            if isinstance(v, float):
                s = "INF" if v == float('inf') else f"{v:.1f}"
            else:
                s = str(v)
            return f"{s:>{width}}"

        vals = [
            f"{t.id:>{cols[0][1]}}",
            f"{t.wcet:>{cols[1][1]}}",
            f"{t.period:>{cols[2][1]}}",
            f"{t.deadline:>{cols[3][1]}}",
        ]
        idx = 4
        if dm_wcrt:
            vals.append(fmt(dm_wcrt, t.id, cols[idx][1])); idx += 1
        if edf_wcrt:
            vals.append(fmt(edf_wcrt, t.id, cols[idx][1])); idx += 1
        if dm_sim:
            vals.append(fmt(dm_sim, t.id, cols[idx][1])); idx += 1
        if edf_sim:
            vals.append(fmt(edf_sim, t.id, cols[idx][1])); idx += 1
        if dm_missed:
            vals.append(fmt(dm_missed, t.id, cols[idx][1])); idx += 1
        if edf_missed:
            vals.append(fmt(edf_missed, t.id, cols[idx][1])); idx += 1

        print(" | ".join(vals))


# ==========================================
# 5. Single Task Set Analysis
# ==========================================

def analyze_single_taskset(csv_file, n_stochastic_runs=500, verbose=True):
    """
    Full analysis of one task set:
      1. DM Response Time Analysis
      2. EDF schedulability + WCRT via hyperperiod simulation
      3. DM stochastic simulation
      4. EDF stochastic simulation
    Returns a result dict.
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
    dm_wcrt = dm_response_time_analysis(tasks)
    dm_sched = all(v != float('inf') for v in dm_wcrt.values())
    if verbose:
        print(f"    Schedulable under DM: {dm_sched}")
        if dm_sched:
            max_dm = max(dm_wcrt.values())
            print(f"    Max WCRT = {max_dm}  (utilization bound U_n = {sum(t.wcet/t.deadline for t in tasks):.4f})")

    # --- 2. EDF Schedulability ---
    if verbose:
        print("\n[2] EDF Schedulability")
    edf_sched, edf_U = edf_utilization_test(tasks)
    if verbose:
        print(f"    U = {edf_U:.4f}  ->  EDF schedulable: {edf_sched}")

    # --- 3. EDF WCRT via hyperperiod simulation ---
    if verbose:
        print("\n[3] EDF WCRT -- Hyperperiod Simulation (Appendix A)")
    if edf_sched:
        if verbose:
            print(f"    Simulating over H = {H} ...")
        edf_wcrt = edf_wcrt_simulation(tasks)
        edf_max = max((v for v in edf_wcrt.values() if v != float('inf')), default=float('inf'))
        if verbose:
            print(f"    Max EDF WCRT = {edf_max}")
    else:
        edf_wcrt = {t.id: float('inf') for t in tasks}
        if verbose:
            print("    Skipped (U > 1: not EDF schedulable)")

    # --- 4. DM Stochastic Simulation ---
    if verbose:
        print(f"\n[4] DM Stochastic Simulation ({n_stochastic_runs} runs)")
    dm_sim_max, dm_avg_missed = stochastic_simulation(
        tasks, policy="DM", n_runs=n_stochastic_runs
    )
    if verbose:
        print(f"    Max observed RT = {max(dm_sim_max.values())}")
        print(f"    Avg missed deadlines/run = {sum(dm_avg_missed.values()):.2f}")

    # --- 5. EDF Stochastic Simulation ---
    if verbose:
        print(f"\n[5] EDF Stochastic Simulation ({n_stochastic_runs} runs)")
    edf_sim_max, edf_avg_missed = stochastic_simulation(
        tasks, policy="EDF", n_runs=n_stochastic_runs
    )
    if verbose:
        print(f"    Max observed RT = {max(edf_sim_max.values())}")
        print(f"    Avg missed deadlines/run = {sum(edf_avg_missed.values()):.2f}")

    # --- Print comparison table ---
    if verbose:
        print("\n[Summary Table]")
        print_task_table(
            tasks,
            dm_wcrt=dm_wcrt,
            edf_wcrt=edf_wcrt if edf_sched else None,
            dm_sim=dm_sim_max,
            edf_sim=edf_sim_max,
        )

        # WCRT ratio: simulation / analytical (where both finite)
        if dm_sched:
            print("\n[DM] Analytical WCRT vs. Stochastic Max RT")
            print(f"  {'Task':>5} {'DM-WCRT':>10} {'Sim-MaxRT':>10} {'Ratio':>8}  {'Status':}")
            print("  " + "-" * 42)
            for t in sorted(tasks, key=lambda x: (x.deadline, x.id)):
                dw = dm_wcrt[t.id]
                sr = dm_sim_max.get(t.id, 0)
                ratio = sr / dw if dw > 0 else float('nan')
                ok = "OK" if sr <= dw else "!"
                print(f"  {t.id:>5} {dw:>10} {sr:>10} {ratio:>8.3f}  {ok}")

        if edf_sched:
            print("\n[EDF] Analytical WCRT vs. Stochastic Max RT")
            print(f"  {'Task':>5} {'EDF-WCRT':>10} {'Sim-MaxRT':>10} {'Ratio':>8}  {'Status':}")
            print("  " + "-" * 42)
            for t in sorted(tasks, key=lambda x: (x.deadline, x.id)):
                ew = edf_wcrt[t.id]
                sr = edf_sim_max.get(t.id, 0)
                ratio = sr / ew if ew not in (0, float('inf')) else float('nan')
                ok = "OK" if ew == float('inf') or sr <= ew else "!"
                print(f"  {t.id:>5} {ew:>10} {sr:>10} {ratio:>8.3f}  {ok}")

    return {
        'tasks': tasks,
        'U': U,
        'H': H,
        'dm_sched': dm_sched,
        'edf_sched': edf_sched,
        'dm_wcrt': dm_wcrt,
        'edf_wcrt': edf_wcrt,
        'dm_sim_max': dm_sim_max,
        'edf_sim_max': edf_sim_max,
        'dm_avg_missed': dm_avg_missed,
        'edf_avg_missed': edf_avg_missed,
    }


# ==========================================
# 6. Batch Processing
# ==========================================

def process_taskset_dir_batch(taskset_dir, n_files=100):
    """
    Fast batch processing of task sets in a directory.
    Uses only analytical tests (no simulation) for speed.
    Returns aggregate statistics dict.
    """
    csv_files = sorted(glob_module.glob(os.path.join(taskset_dir, "*.csv")))[:n_files]
    if not csv_files:
        return None

    stats = {
        'total': 0,
        'dm_schedulable': 0,
        'edf_schedulable': 0,
        'dm_wcrt_ratios': [],    # sim_max / dm_wcrt (per task, for schedulable sets)
        'actual_utils': [],
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

        # DM schedulability via RTA
        dm_wcrt = dm_response_time_analysis(tasks)
        if all(v != float('inf') for v in dm_wcrt.values()):
            stats['dm_schedulable'] += 1

        # EDF schedulability via utilization test
        edf_ok, _ = edf_utilization_test(tasks)
        if edf_ok:
            stats['edf_schedulable'] += 1

    return stats


def batch_analysis(base_dir, util_dist, per_dist, n_task="25-task", jitter="0-jitter"):
    """
    Process all utilization levels for a given distribution configuration.
    Returns {util_level: stats_dict}.
    """
    path = os.path.join(base_dir, util_dist, per_dist, "1-core", n_task, jitter)
    if not os.path.exists(path):
        print(f"  Path not found: {path}")
        return {}

    util_entries = sorted([d for d in os.listdir(path) if d.endswith('-util')])
    all_results = {}

    for entry in util_entries:
        util_val = float(entry.replace('-util', ''))
        taskset_dir = os.path.join(path, entry, "tasksets")
        if not os.path.exists(taskset_dir):
            continue

        print(f"    util={util_val:.2f} ...", end=' ', flush=True)
        stats = process_taskset_dir_batch(taskset_dir)
        if stats and stats['total'] > 0:
            all_results[util_val] = stats
            pct_dm = stats['dm_schedulable'] / stats['total'] * 100
            pct_edf = stats['edf_schedulable'] / stats['total'] * 100
            print(f"DM={pct_dm:.0f}%  EDF={pct_edf:.0f}%  (n={stats['total']})")
        else:
            print("no data")

    return all_results


def print_batch_summary(label, results):
    if not results:
        print(f"  No results for {label}")
        return
    print(f"\n{label} -- Schedulability Summary")
    print(f"  {'Util':>6} {'N':>5} {'DM%':>7} {'EDF%':>7}  {'DM>EDF?':}")
    print("  " + "-" * 35)
    for u in sorted(results.keys()):
        s = results[u]
        n = s['total']
        dm_pct = s['dm_schedulable'] / n * 100
        edf_pct = s['edf_schedulable'] / n * 100
        note = "EDF dom." if edf_pct > dm_pct else ("equal" if edf_pct == dm_pct else "DM dom.")
        print(f"  {u:>6.2f} {n:>5} {dm_pct:>6.1f}% {edf_pct:>6.1f}%  {note}")


# ==========================================
# 7. Plotting
# ==========================================

def plot_schedulability(results_auto, results_uni, output_dir="."):
    """Schedulability ratio vs. utilization: DM vs EDF, two distributions."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("DM vs EDF Schedulability -- 25-Task Sets, 0 Jitter", fontsize=14)

    for ax, (results, title) in zip(axes, [
        (results_auto, "Automotive Distribution"),
        (results_uni, "UUniFast Distribution"),
    ]):
        if not results:
            ax.set_title(f"{title}\n(No data)")
            continue

        utils = sorted(results.keys())
        dm_pct = [results[u]['dm_schedulable'] / results[u]['total'] * 100 for u in utils]
        edf_pct = [results[u]['edf_schedulable'] / results[u]['total'] * 100 for u in utils]

        ax.plot(utils, dm_pct,  'b-o', label='DM (RTA)',  linewidth=2, markersize=7)
        ax.plot(utils, edf_pct, 'r-s', label='EDF (U<=1)', linewidth=2, markersize=7)
        ax.fill_between(utils, dm_pct, edf_pct, alpha=0.15, color='green',
                         label='EDF advantage')
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Target Utilization', fontsize=12)
        ax.set_ylabel('Schedulable (%)', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)
        ax.set_xlim(0.05, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, "schedulability_dm_vs_edf.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved plot: {path}")


def plot_wcrt_comparison(result, output_dir="."):
    """
    Per-task WCRT comparison for a single task set:
    DM analytical, EDF analytical, DM stochastic max, EDF stochastic max.
    """
    if not HAS_MATPLOTLIB or result is None:
        return

    tasks = sorted(result['tasks'], key=lambda t: (t.deadline, t.id))
    labels = [f"T{t.id}" for t in tasks]
    x = np.arange(len(tasks))
    width = 0.2

    def norm(val, deadline):
        """Normalize response time by deadline; inf -> nan (hidden in plot)."""
        if val == float('inf') or deadline == 0:
            return float('nan')
        return val / deadline

    dm_wcrt_vals  = [norm(result['dm_wcrt'].get(t.id, 0),  t.deadline) for t in tasks]
    edf_wcrt_vals = [norm(result['edf_wcrt'].get(t.id, 0), t.deadline) for t in tasks]
    dm_sim_vals   = [norm(result['dm_sim_max'].get(t.id, 0),  t.deadline) for t in tasks]
    edf_sim_vals  = [norm(result['edf_sim_max'].get(t.id, 0), t.deadline) for t in tasks]

    fig, ax = plt.subplots(figsize=(max(14, len(tasks) // 2), 6))
    ax.bar(x - 1.5*width, dm_wcrt_vals,  width, label='DM WCRT (RTA)',        color='#2196F3', alpha=0.85)
    ax.bar(x - 0.5*width, edf_wcrt_vals, width, label='EDF WCRT (Sim, WCET)', color='#F44336', alpha=0.85)
    ax.bar(x + 0.5*width, dm_sim_vals,   width, label='DM Stoch. Max RT',     color='#64B5F6', alpha=0.85)
    ax.bar(x + 1.5*width, edf_sim_vals,  width, label='EDF Stoch. Max RT',    color='#EF9A9A', alpha=0.85)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Deadline (y=1)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Task (sorted by deadline)', fontsize=11)
    ax.set_ylabel('Response Time / Deadline', fontsize=11)
    ax.set_title('Normalized WCRT Comparison: DM vs EDF (Analytical & Stochastic)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(1.2, ax.get_ylim()[1]))

    plt.tight_layout()
    path = os.path.join(output_dir, "wcrt_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {path}")


def plot_sim_vs_analytical(result, output_dir="."):
    """
    Scatter plot: stochastic max RT vs. analytical WCRT.
    Points below the diagonal mean simulation stayed within the bound.
    """
    if not HAS_MATPLOTLIB or result is None:
        return

    tasks = result['tasks']
    dm_ok = result['dm_sched']
    edf_ok = result['edf_sched']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Simulation Max RT vs. Analytical WCRT", fontsize=13)

    for ax, (policy, wcrt, sim, ok, color) in zip(axes, [
        ("DM",  result['dm_wcrt'],  result['dm_sim_max'],  dm_ok,  'blue'),
        ("EDF", result['edf_wcrt'], result['edf_sim_max'], edf_ok, 'red'),
    ]):
        if not ok:
            ax.set_title(f"{policy} (not schedulable)")
            ax.axis('off')
            continue

        xs = [wcrt.get(t.id, 0) for t in tasks if wcrt.get(t.id, 0) != float('inf')]
        ys = [sim.get(t.id, 0) for t in tasks if wcrt.get(t.id, 0) != float('inf')]

        if not xs:
            continue

        ax.scatter(xs, ys, color=color, alpha=0.7, s=50, label='tasks')
        lim = max(max(xs), max(ys)) * 1.1
        ax.plot([0, lim], [0, lim], 'k--', linewidth=1, label='y=x (bound)')
        ax.set_xlabel(f'{policy} Analytical WCRT', fontsize=11)
        ax.set_ylabel(f'{policy} Stochastic Max RT', fontsize=11)
        ax.set_title(f'{policy}: Simulation vs Analytical', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "sim_vs_analytical.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {path}")


# ==========================================
# 8. Main
# ==========================================

def main():
    # Determine paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tasksets_root = os.path.join(script_dir, "task-sets", "output")

    print_separator()
    print("  DRTS Mini-Project 1: DM vs EDF Scheduling Analysis")
    print_separator()

    # ---- Demo: Detailed analysis of one task set ----
    # Try a mid-utilization (0.50) automotive task set for the demo
    demo_candidates = [
        os.path.join(tasksets_root, "automotive-utilDist", "automotive-perDist",
                     "1-core", "25-task", "0-jitter", "0.50-util", "tasksets", "automotive_0.csv"),
        os.path.join(tasksets_root, "uunifast-utilDist", "uniform-discrete-perDist",
                     "1-core", "25-task", "0-jitter", "0.50-util", "tasksets", "automotive_0.csv"),
    ]

    demo_result = None
    demo_dir = script_dir

    for demo_csv in demo_candidates:
        if os.path.exists(demo_csv):
            print(f"\nDemo Analysis (single task set): {demo_csv}")
            demo_result = analyze_single_taskset(demo_csv, n_stochastic_runs=500, verbose=True)
            demo_dir = os.path.dirname(demo_csv)
            break

    if demo_result is None:
        # Try any local CSV
        local = glob_module.glob(os.path.join(script_dir, "*.csv"))
        if local:
            print(f"\nDemo Analysis: {local[0]}")
            demo_result = analyze_single_taskset(local[0], n_stochastic_runs=500, verbose=True)
            demo_dir = script_dir
        else:
            print("\nNo task set CSV found for demo. Skipping single-set analysis.")

    # Save per-task-set plots
    if demo_result:
        print("\nGenerating WCRT comparison plots...")
        plot_wcrt_comparison(demo_result, output_dir=script_dir)
        plot_sim_vs_analytical(demo_result, output_dir=script_dir)

    # ---- Batch Analysis ----
    print_separator()
    print("  BATCH ANALYSIS -- All Utilization Levels")
    print_separator()

    results_auto = {}
    results_uni  = {}

    auto_path = os.path.join(tasksets_root, "automotive-utilDist", "automotive-perDist")
    uni_path  = os.path.join(tasksets_root, "uunifast-utilDist",   "uniform-discrete-perDist")

    if os.path.exists(auto_path):
        print("\nAutomotive Distribution:")
        results_auto = batch_analysis(tasksets_root, "automotive-utilDist", "automotive-perDist")
        print_batch_summary("Automotive", results_auto)
    else:
        print(f"\nAutomotive path not found: {auto_path}")

    if os.path.exists(uni_path):
        print("\nUUniFast Distribution:")
        results_uni = batch_analysis(tasksets_root, "uunifast-utilDist", "uniform-discrete-perDist")
        print_batch_summary("UUniFast", results_uni)
    else:
        print(f"\nUUniFast path not found: {uni_path}")

    # ---- Schedulability Plot ----
    if results_auto or results_uni:
        print("\nGenerating schedulability plot...")
        plot_schedulability(results_auto, results_uni, output_dir=script_dir)

    print_separator()
    print("  Done.")
    print_separator()


if __name__ == "__main__":
    main()
