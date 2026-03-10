# simulation/simulator.py
import random
from tasks.task_set import Task, TaskSet

# ── Observer ──────────────────────────────────────────────────────────────────

class Observer:
    """
    Collects statistics during simulation.

    Tracks per task:
    - Maximum observed response time (wcrt)
    - Total deadline misses
    - Full response time distribution (for histogram/box plots)
    """

    def __init__(self, n: int):
        self.n       = n
        self.wcrt    = [0.0] * n          # max observed response time
        self.misses  = [0]   * n          # deadline miss count
        self.rt_dist = [[]   for _ in range(n)]  # all response times

    def record(self, task_idx: int, release: float,
               finish: float, deadline: float) -> None:
        rt = finish - release
        if rt > self.wcrt[task_idx]:
            self.wcrt[task_idx] = rt
        self.rt_dist[task_idx].append(rt)
        if finish > deadline + 1e-9:
            self.misses[task_idx] += 1

    def total_misses(self) -> int:
        return sum(self.misses)

    def __repr__(self):
        lines = ["Observer results:"]
        for i in range(self.n):
            lines.append(
                f"  τ{i+1}: WCRT={self.wcrt[i]:.2f} "
                f"Misses={self.misses[i]} "
                f"Samples={len(self.rt_dist[i])}"
            )
        return "\n".join(lines)


# ── Simulator ─────────────────────────────────────────────────────────────────

def simulate(taskset: TaskSet,
             algorithm:   str   = 'EDF',
             sim_ticks:   int   = 50000,
             bcet_ratio:  float = 0.5,
             bcet_list:   list  = None,
             seed:        int   = 42) -> Observer:
    """
    Discrete-event simulation of DM or EDF scheduling.

    Execution times are drawn uniformly from [BCET_i, WCET_i] where:
        BCET_i = bcet_list[i]   if provided
               = bcet_ratio * C  otherwise

    Parameters
    ----------
    taskset    : TaskSet to simulate
    algorithm  : 'EDF' or 'DM'
    sim_ticks  : simulation duration in time units
    bcet_ratio : BCET = bcet_ratio * C  (default 0.5)
    bcet_list  : per-task BCET values (overrides bcet_ratio)
    seed       : random seed for reproducibility

    Returns
    -------
    Observer with recorded statistics
    """
    assert algorithm in ('EDF', 'DM'), "algorithm must be 'EDF' or 'DM'"

    rng   = random.Random(seed)
    tasks = taskset.tasks
    n     = len(tasks)
    obs   = Observer(n)

    # ── Per-task BCET ─────────────────────────────────────────────────────────
    if bcet_list is not None:
        assert len(bcet_list) == n, \
            f"bcet_list length {len(bcet_list)} != {n} tasks"
        bcets = list(bcet_list)
    else:
        bcets = [bcet_ratio * t.C for t in tasks]

    # ── DM priority ranks (0 = highest priority = shortest deadline) ──────────
    dm_prio = [0] * n
    for rank, i in enumerate(sorted(range(n), key=lambda i: tasks[i].D)):
        dm_prio[i] = rank

    def sched_key(job):
        """Lower value = higher priority."""
        if algorithm == 'EDF':
            return (job['abs_d'],           job['task_idx'])
        else:
            return (dm_prio[job['task_idx']], job['task_idx'])

    # ── Simulation state ──────────────────────────────────────────────────────
    queue        = []          # ready jobs waiting for CPU
    current      = None        # job currently executing
    next_release = [0.0] * n  # next release time per task
    time         = 0.0

    while time < sim_ticks:

        # ── Release all jobs due at current time ──────────────────────────────
        for i, task in enumerate(tasks):
            while next_release[i] <= time + 1e-9:
                c_actual = rng.uniform(bcets[i], task.C)
                queue.append({
                    'r':        next_release[i],
                    'abs_d':    next_release[i] + task.D,
                    'deadline': next_release[i] + task.D,
                    'rem':      c_actual,
                    'task_idx': i,
                })
                next_release[i] += task.T

        # ── Preemption check ──────────────────────────────────────────────────
        if queue:
            best = min(queue, key=sched_key)
            if current is None:
                current = best
                queue.remove(current)
            elif sched_key(best) < sched_key(current):
                queue.append(current)
                current = best
                queue.remove(current)

        # ── Idle CPU: jump to next release ────────────────────────────────────
        if current is None:
            future = [nr for nr in next_release if nr > time + 1e-9]
            time   = min(future) if future else sim_ticks
            continue

        # ── Advance to next event (job finish OR next release) ────────────────
        future_releases = [nr for nr in next_release if nr > time + 1e-9]
        next_rel_time   = min(future_releases) if future_releases else float('inf')

        # Only run until the next release so preemption can fire correctly
        run_for = min(current['rem'], next_rel_time - time)
        run_for = max(run_for, 1e-9)

        current['rem'] -= run_for
        time           += run_for

        # ── Job completion ────────────────────────────────────────────────────
        if current['rem'] <= 1e-9:
            obs.record(
                task_idx = current['task_idx'],
                release  = current['r'],
                finish   = time,
                deadline = current['deadline'],
            )
            current = None

    # ── End of simulation: catch unfinished jobs ──────────────────────────────
    # Job still on CPU that missed its deadline
    if current and time > current['deadline'] + 1e-9:
        obs.misses[current['task_idx']] += 1

    # Jobs still in ready queue that missed their deadline
    for job in queue:
        if time > job['deadline'] + 1e-9:
            obs.misses[job['task_idx']] += 1

    return obs


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_sim_results(taskset: TaskSet, obs: Observer,
                      algorithm: str) -> None:
    """Print simulation results table."""
    tasks = taskset.tasks
    print(f"\n{'='*65}")
    print(f"  Simulation Results — {taskset.name} — {algorithm}")
    print(f"  {'Idx':>3} {'C':>6} {'D':>6} {'T':>6} "
          f"{'SimWCRT':>10} {'Misses':>7} {'Samples':>8}")
    print(f"  {'-'*3} {'-'*6} {'-'*6} {'-'*6} "
          f"{'-'*10} {'-'*7} {'-'*8}")
    for i, task in enumerate(tasks):
        print(f"  {i+1:>3} {task.C:>6} {task.D:>6} {task.T:>6} "
              f"{obs.wcrt[i]:>10.2f} {obs.misses[i]:>7} "
              f"{len(obs.rt_dist[i]):>8}")
    print(f"{'='*65}")
    print(f"  Total deadline misses: {obs.total_misses()}")
    print(f"{'='*65}\n")
