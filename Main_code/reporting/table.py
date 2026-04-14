"""
reporting/table.py
------------------
Console tables and Matplotlib figures for the DRTS Mini-Project 1 toolchain.

All plotting functions are no-ops when matplotlib/numpy are unavailable;
text-output functions always work.

Exported symbols
----------------
print_separator(char, width)
print_task_table(tasks, ...)
plot_schedulability(results_auto, results_uni, output_dir)
plot_wcrt_comparison(result, output_dir)
plot_sim_vs_analytical(result, output_dir)
plot_rt_distributions(dm_rts, edf_rts, tasks, ..., output_dir)
"""

import math
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib/numpy not available. "
          "Install with: pip install matplotlib numpy")


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def print_separator(char: str = '=', width: int = 76) -> None:
    print(char * width)


def print_task_table(tasks, dm_wcrt=None, edf_wcrt=None,
                     dm_sim=None, edf_sim=None,
                     dm_missed=None, edf_missed=None) -> None:
    """Print a formatted per-task comparison table to stdout."""
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

    header = " | ".join(f"{name:>{w}}" for name, w in cols)
    print(header)
    print("-" * len(header))

    def fmt(d, tid, width):
        if d is None:
            return f"{'N/A':>{width}}"
        v = d.get(tid, 'N/A')
        if isinstance(v, float):
            s = "INF" if v == float('inf') else f"{v:.1f}"
        else:
            s = str(v)
        return f"{s:>{width}}"

    for t in sorted_tasks:
        vals = [
            f"{t.id:>{cols[0][1]}}",
            f"{t.wcet:>{cols[1][1]}}",
            f"{t.period:>{cols[2][1]}}",
            f"{t.deadline:>{cols[3][1]}}",
        ]
        idx = 4
        if dm_wcrt:
            vals.append(fmt(dm_wcrt,   t.id, cols[idx][1])); idx += 1
        if edf_wcrt:
            vals.append(fmt(edf_wcrt,  t.id, cols[idx][1])); idx += 1
        if dm_sim:
            vals.append(fmt(dm_sim,    t.id, cols[idx][1])); idx += 1
        if edf_sim:
            vals.append(fmt(edf_sim,   t.id, cols[idx][1])); idx += 1
        if dm_missed:
            vals.append(fmt(dm_missed, t.id, cols[idx][1])); idx += 1
        if edf_missed:
            vals.append(fmt(edf_missed,t.id, cols[idx][1])); idx += 1

        print(" | ".join(vals))


# ---------------------------------------------------------------------------
# Schedulability plot
# ---------------------------------------------------------------------------

def plot_schedulability(results_auto: dict, results_uni: dict,
                        output_dir: str = ".") -> None:
    """
    Schedulability ratio vs. utilisation for both benchmark distributions.

    EDF curve uses the PDC verdict (authoritative for constrained deadlines).
    The legend is labelled 'EDF (PDC)' to reflect this.
    """
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("DM vs EDF Schedulability — 25-Task Sets, 0 Jitter",
                 fontsize=14)

    for ax, (results, title) in zip(axes, [
        (results_auto, "Automotive Distribution"),
        (results_uni,  "UUniFast Distribution"),
    ]):
        if not results:
            ax.set_title(f"{title}\n(No data)")
            continue

        utils   = sorted(results.keys())
        dm_pct  = [results[u]['dm_schedulable']  / results[u]['total'] * 100
                   for u in utils]
        edf_pct = [results[u]['edf_schedulable'] / results[u]['total'] * 100
                   for u in utils]

        ax.plot(utils, dm_pct,  'b-o', label='DM (RTA)',   linewidth=2, markersize=7)
        ax.plot(utils, edf_pct, 'r-s', label='EDF (PDC)',  linewidth=2, markersize=7)
        ax.fill_between(utils, dm_pct, edf_pct,
                        alpha=0.15, color='green', label='EDF advantage')
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


# ---------------------------------------------------------------------------
# Per-task WCRT comparison bar chart
# ---------------------------------------------------------------------------

def plot_wcrt_comparison(result: dict, output_dir: str = ".") -> None:
    """
    Per-task bar chart: DM analytical WCRT, EDF analytical WCRT,
    DM stochastic max RT, EDF stochastic max RT — all normalised by
    task deadline.  Points above y=1 indicate a deadline miss.
    """
    if not HAS_MATPLOTLIB or result is None:
        return

    tasks  = sorted(result['tasks'], key=lambda t: (t.deadline, t.id))
    labels = [f"T{t.id}" for t in tasks]
    x      = np.arange(len(tasks))
    width  = 0.2

    def norm(val, deadline):
        if val == float('inf') or deadline == 0:
            return float('nan')
        return val / deadline

    dm_wcrt_vals  = [norm(result['dm_wcrt'].get(t.id, 0),  t.deadline) for t in tasks]
    edf_wcrt_vals = [norm(result['edf_wcrt'].get(t.id, 0), t.deadline) for t in tasks]
    dm_sim_vals   = [norm(result['dm_sim_max'].get(t.id, 0),  t.deadline) for t in tasks]
    edf_sim_vals  = [norm(result['edf_sim_max'].get(t.id, 0), t.deadline) for t in tasks]

    fig, ax = plt.subplots(figsize=(max(14, len(tasks) // 2), 6))
    ax.bar(x - 1.5*width, dm_wcrt_vals,  width,
           label='DM WCRT (RTA)',        color='#2196F3', alpha=0.85)
    ax.bar(x - 0.5*width, edf_wcrt_vals, width,
           label='EDF WCRT (Sim, WCET)', color='#F44336', alpha=0.85)
    ax.bar(x + 0.5*width, dm_sim_vals,   width,
           label='DM Stoch. Max RT',     color='#64B5F6', alpha=0.85)
    ax.bar(x + 1.5*width, edf_sim_vals,  width,
           label='EDF Stoch. Max RT',    color='#EF9A9A', alpha=0.85)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5,
               label='Deadline (y=1)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Task (sorted by deadline)', fontsize=11)
    ax.set_ylabel('Response Time / Deadline', fontsize=11)
    ax.set_title(
        'Normalized WCRT Comparison: DM vs EDF (Analytical & Stochastic)',
        fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(1.2, ax.get_ylim()[1]))

    plt.tight_layout()
    path = os.path.join(output_dir, "wcrt_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {path}")


# ---------------------------------------------------------------------------
# Simulation vs. analytical scatter plot
# ---------------------------------------------------------------------------

def plot_sim_vs_analytical(result: dict, output_dir: str = ".") -> None:
    """
    Scatter: stochastic max RT vs. analytical WCRT for DM (left) and
    EDF (right).  All points below the y=x diagonal confirm that no
    stochastic run exceeded the analytical bound (Eq. validation).
    """
    if not HAS_MATPLOTLIB or result is None:
        return

    tasks  = result['tasks']
    dm_ok  = result['dm_sched']
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

        xs = [wcrt.get(t.id, 0) for t in tasks
              if wcrt.get(t.id, 0) != float('inf')]
        ys = [sim.get(t.id, 0)  for t in tasks
              if wcrt.get(t.id, 0) != float('inf')]

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


# ---------------------------------------------------------------------------
# Response-time CDF grid
# ---------------------------------------------------------------------------

def plot_rt_distributions(dm_rts: dict, edf_rts: dict, tasks,
                           dm_wcrt=None, edf_wcrt=None,
                           output_dir: str = ".") -> None:
    """
    CDF of response times for DM and EDF for every task (sorted by deadline).

    One sub-plot per task arranged in a grid.  Vertical dashed lines mark
    the analytical WCRT bounds and the task deadline.

    Output file
    -----------
    rt_distributions_cdf.png  — name matches the LaTeX report reference.
    """
    if not HAS_MATPLOTLIB:
        return

    sorted_tasks = sorted(tasks, key=lambda t: (t.deadline, t.id))
    n     = len(sorted_tasks)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.2, nrows * 2.8),
        squeeze=False,
    )
    fig.suptitle(
        "Response-Time CDFs: DM vs EDF (stochastic, log-normal [BCET,WCET])",
        fontsize=12)

    for idx, task in enumerate(sorted_tasks):
        ax          = axes[idx // ncols][idx % ncols]
        dm_samples  = dm_rts.get(task.id, [])
        edf_samples = edf_rts.get(task.id, [])

        for samples, color, label in [
            (dm_samples,  '#2196F3', 'DM'),
            (edf_samples, '#F44336', 'EDF'),
        ]:
            if samples:
                xs = sorted(samples)
                ys = [(i + 1) / len(xs) for i in range(len(xs))]
                ax.step(xs, ys, where='post', color=color,
                        linewidth=1.5, label=label)

        # Analytical WCRT bounds
        if dm_wcrt:
            v = dm_wcrt.get(task.id)
            if v and v != float('inf'):
                ax.axvline(v, color='#1565C0', linestyle='--', linewidth=1,
                           alpha=0.8, label=f'DM-WCRT={v}')
        if edf_wcrt:
            v = edf_wcrt.get(task.id)
            if v and v != float('inf'):
                ax.axvline(v, color='#B71C1C', linestyle='--', linewidth=1,
                           alpha=0.8, label=f'EDF-WCRT={v}')

        ax.axvline(task.deadline, color='black', linestyle=':', linewidth=1,
                   alpha=0.6, label=f'D={task.deadline}')
        ax.set_title(f"T{task.id}  (D={task.deadline})", fontsize=8)
        ax.set_xlabel("Response time", fontsize=7)
        ax.set_ylabel("CDF", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5, loc='lower right')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)

    # Hide unused axes in the last row
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    # Filename matches the \includegraphics reference in main.tex
    path = os.path.join(output_dir, "rt_distributions_cdf.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {path}")
