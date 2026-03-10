# analysis/edf_wcrt.py
from tasks.task_set import Task, TaskSet


def edf_wcrt(taskset: TaskSet) -> list[dict]:
    """
    Exact EDF Worst-Case Response Time analysis for synchronous periodic tasks
    with D_i <= T_i, via hyperperiod schedule simulation.

    As specified in 02225_DRTS_mini_project.pdf:
      1. Compute H = lcm(T_1, ..., T_n)
      2. Generate all jobs released in [0, H)
      3. Simulate EDF schedule from t=0 to t=H with every job running
         for exactly its WCET (C_i)  — worst-case load assumption
      4. Record finish time f_{i,k} for every job
      5. WCRT_i = max_k (f_{i,k} - r_{i,k})

    Tie-breaking rule: when two jobs share the same absolute deadline,
    the job belonging to the task with the smaller index (position in
    taskset.tasks) wins. This is documented as a design assumption.

    Parameters
    ----------
    taskset : TaskSet

    Returns
    -------
    List of dicts, one per task in original taskset order:
        {
            'task'        : Task,
            'task_index'  : int,
            'wcrt'        : float,
            'schedulable' : bool,   # True if wcrt <= D
            'n_jobs'      : int,    # total jobs analysed
        }
    """
    U = taskset.utilization
    if U > 1.0:
        raise ValueError(f"U={U:.3f} > 1.0 — EDF cannot schedule this task set.")

    H = taskset.hyperperiod
    tasks = taskset.tasks

    # ── Step 1: Generate all jobs released in [0, H) ──────────────────────────
    # Each job: [release, abs_deadline, C, remaining, task_index]
    all_jobs = []
    for i, task in enumerate(tasks):
        r = 0
        while r < H:
            all_jobs.append({
                'r'      : r,
                'd'      : r + task.D,   # absolute deadline
                'C'      : task.C,
                'rem'    : task.C,        # remaining execution (starts at WCET)
                'i'      : i,             # task index for tie-breaking
                'finish' : None,
            })
            r += int(task.T)

    # Sort jobs by release time for efficient scanning
    all_jobs.sort(key=lambda j: j['r'])

    # ── Step 2: Simulate EDF over [0, H] ──────────────────────────────────────
    rptr    = 0          # pointer into all_jobs (next job to release)
    ready   = []         # jobs currently ready to run
    current = None       # job currently on CPU
    time    = 0

    while time <= H:
        # Release all jobs with r <= current time
        while rptr < len(all_jobs) and all_jobs[rptr]['r'] <= time:
            ready.append(all_jobs[rptr])
            rptr += 1

        # If nothing is ready and CPU is idle, jump to next release
        if not ready and current is None:
            if rptr < len(all_jobs):
                time = all_jobs[rptr]['r']
                continue
            else:
                break   # no more jobs

        # EDF dispatch: pick job with earliest absolute deadline
        # Tie-break: smaller task index wins (documented assumption)
        if ready:
            best = min(ready, key=lambda j: (j['d'], j['i']))

            if current is None:
                # CPU is free — start best job
                current = best
                ready.remove(current)
            elif (best['d'], best['i']) < (current['d'], current['i']):
                # Preempt: a more urgent job has arrived
                ready.append(current)
                current = best
                ready.remove(current)

        if current is None:
            time += 1
            continue

        # Calculate how long to run before next event
        # (either a new job releases or current job finishes)
        if rptr < len(all_jobs):
            next_release = all_jobs[rptr]['r']
            run_for = min(current['rem'], next_release - time)
        else:
            run_for = current['rem']

        run_for = max(run_for, 1e-9)   # always advance by at least epsilon

        current['rem'] -= run_for
        time += run_for

        # Job finished?
        if current['rem'] <= 0:
            current['finish'] = time
            current = None

    # ── Step 3: Compute WCRT per task ─────────────────────────────────────────
    wcrt = [0.0] * len(tasks)
    job_counts = [0] * len(tasks)

    for job in all_jobs:
        if job['finish'] is not None:
            rt = job['finish'] - job['r']
            if rt > wcrt[job['i']]:
                wcrt[job['i']] = rt
            job_counts[job['i']] += 1

    results = []
    for i, task in enumerate(tasks):
        results.append({
            'task'       : task,
            'task_index' : i,
            'wcrt'       : wcrt[i],
            'schedulable': wcrt[i] <= task.D,
            'n_jobs'     : job_counts[i],
        })

    return results


def edf_is_schedulable(taskset: TaskSet) -> bool:
    """Returns True only if ALL tasks have WCRT <= D under EDF."""
    return all(r['schedulable'] for r in edf_wcrt(taskset))


def print_edf_results(taskset: TaskSet) -> None:
    """Pretty-print the EDF WCRT results table."""
    results = edf_wcrt(taskset)
    print(f"\n{'='*60}")
    print(f"  EDF WCRT Analysis (Hyperperiod) — {taskset.name}")
    print(f"  U={taskset.utilization:.4f}  H={taskset.hyperperiod}")
    print(f"  Tie-break rule: smaller task index wins on equal deadline")
    print(f"{'='*60}")
    print(f"  {'Idx':>3}  {'C':>4}  {'D':>4}  {'T':>4}  {'WCRT':>6}  "
          f"{'Jobs':>5}  {'Status':>12}")
    print(f"  {'-'*3}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*6}  "
          f"{'-'*5}  {'-'*12}")
    for r in results:
        t = r['task']
        status = "SCHEDULABLE" if r['schedulable'] else "** INFEASIBLE **"
        print(f"  {r['task_index']+1:>3}  {t.C:>4}  {t.D:>4}  {t.T:>4}  "
              f"{r['wcrt']:>6.1f}  {r['n_jobs']:>5}  {status:>12}")
    print(f"{'='*60}")
    overall = "ALL SCHEDULABLE ✓" if edf_is_schedulable(taskset) else "NOT SCHEDULABLE ✗"
    print(f"  Result: {overall}")
    print(f"{'='*60}\n")
