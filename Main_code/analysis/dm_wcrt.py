"""
analysis/dm_wcrt.py
-------------------
Deadline Monotonic (DM) Worst-Case Response Time Analysis.

Implements the iterative fixed-point Response Time Analysis (RTA) from
Buttazzo, *Hard Real-Time Computing Systems*, Eq. 4.17–4.19.

Exported symbols
----------------
dm_response_time_analysis(tasks) -> dict[int, float]
    Maps task_id -> WCRT.  WCRT == float('inf') means the task misses
    its deadline under DM and the task set is not DM-schedulable.
"""

import math


def dm_response_time_analysis(tasks) -> dict:
    """
    Compute the Worst-Case Response Time for every task under
    Deadline Monotonic (DM) fixed-priority scheduling.

    Priority assignment
    -------------------
    Tasks are sorted by non-decreasing relative deadline; ties are broken
    by task ID (lower ID = higher priority).  This yields the canonical DM
    priority order.

    Algorithm
    ---------
    For each task τ_i in priority order, with higher-priority set hp(i):

        R_i^(0)   = C_i
        R_i^(k+1) = C_i + Σ_{j ∈ hp(i)} ⌈ R_i^(k) / T_j ⌉ · C_j

    Iteration terminates when:
      - R_i^(k+1) == R_i^(k)  →  converged; WCRT = R_i
      - R_i^(k+1) >  D_i      →  unschedulable; WCRT = ∞

    Parameters
    ----------
    tasks : list[Task]

    Returns
    -------
    dict[int, float]
        {task_id: WCRT}  — WCRT is float('inf') for unschedulable tasks.
    """
    sorted_tasks = sorted(tasks, key=lambda t: (t.deadline, t.id))
    wcrt: dict = {}

    for i, task in enumerate(sorted_tasks):
        hp = sorted_tasks[:i]       # strictly higher-priority tasks
        R  = task.wcet              # initial estimate

        while True:
            R_new = task.wcet + sum(
                math.ceil(R / h.period) * h.wcet for h in hp
            )
            if R_new > task.deadline:
                wcrt[task.id] = float('inf')
                break
            if R_new == R:
                wcrt[task.id] = R
                break
            R = R_new

    return wcrt
