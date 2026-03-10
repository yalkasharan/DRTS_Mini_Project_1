# analysis/dm_wcrt.py
from math import ceil
from tasks.task_set import Task, TaskSet


def dm_wcrt(taskset: TaskSet) -> list[dict]:
    """
    Deadline Monotonic Response Time Analysis.
    Implements Buttazzo Chapter 4, Section 4.5.2, Figure 4.17.

    Tasks are analysed in DM priority order (sorted by deadline D,
    shortest deadline = highest priority).

    For each task τ_i, the WCRT R_i is the smallest fixed point of:

        R_i^(s) = C_i + sum_{h: P_h > P_i} ceil(R_i^(s-1) / T_h) * C_h

    Initialised with R_i^(0) = sum of C for all tasks up to and including i.
    Iteration stops when R_i^(s) == R_i^(s-1)  (converged)
                      or R_i^(s) >  D_i         (infeasible).

    Parameters
    ----------
    taskset : TaskSet

    Returns
    -------
    List of dicts, one per task in DM priority order:
        {
            'task'        : Task,
            'priority'    : int,      # 1 = highest priority
            'wcrt'        : float,    # worst-case response time
            'schedulable' : bool,     # True if wcrt <= D
            'iterations'  : int,      # number of iterations to converge
        }
    """
    # Sort by deadline ascending — this IS the DM priority assignment
    sorted_tasks = taskset.sorted_by_deadline()
    results = []

    for i, task_i in enumerate(sorted_tasks):
        # Higher-priority tasks = all tasks before i in sorted order
        hp_tasks = sorted_tasks[:i]

        # Initialise R^(0) = sum of WCETs of task_i and all higher-priority tasks
        R = sum(t.C for t in sorted_tasks[:i+1])
        iterations = 0

        while True:
            iterations += 1

            # R^(s) = C_i + sum over higher-priority tasks
            R_new = task_i.C + sum(
                ceil(R / t.T) * t.C
                for t in hp_tasks
            )

            # Converged: fixed point found
            if R_new == R:
                results.append({
                    'task'        : task_i,
                    'priority'    : i + 1,
                    'wcrt'        : R,
                    'schedulable' : R <= task_i.D,
                    'iterations'  : iterations,
                })
                break

            # Diverging: upper bound R > T_i means the busy period never ends
            # (HP tasks' cumulative load fills all CPU time before task_i runs)
            if R_new > task_i.T:
                results.append({
                    'task'        : task_i,
                    'priority'    : i + 1,
                    'wcrt'        : float('inf'),
                    'schedulable' : False,
                    'iterations'  : iterations,
                })
                break

            R = R_new

    return results


def dm_is_schedulable(taskset: TaskSet) -> bool:
    """Returns True only if ALL tasks pass RTA."""
    return all(r['schedulable'] for r in dm_wcrt(taskset))


def print_dm_results(taskset: TaskSet) -> None:
    """Pretty-print the DM RTA results table."""
    results = dm_wcrt(taskset)
    print(f"\n{'='*60}")
    print(f"  DM Response Time Analysis — {taskset.name}")
    print(f"  U={taskset.utilization:.4f}  U_lub={taskset.utilization_bound_dm:.4f}")
    print(f"{'='*60}")
    print(f"  {'Pri':>3}  {'C':>4}  {'D':>4}  {'T':>4}  {'WCRT':>6}  {'Status':>12}")
    print(f"  {'-'*3}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*12}")
    for r in results:
        t = r['task']
        status = "SCHEDULABLE" if r['schedulable'] else "** INFEASIBLE **"
        print(f"  {r['priority']:>3}  {t.C:>4}  {t.D:>4}  {t.T:>4}  "
              f"{r['wcrt']:>6.1f}  {status:>12}")
    print(f"{'='*60}")
    overall = "ALL SCHEDULABLE ✓" if dm_is_schedulable(taskset) else "NOT SCHEDULABLE ✗"
    print(f"  Result: {overall}")
    print(f"{'='*60}\n")
