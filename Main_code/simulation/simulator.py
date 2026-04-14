"""
simulation/simulator.py
-----------------------
Event-driven real-time scheduler simulation engine.

A single _event_driven_core() function services both DM and EDF by
parameterising the priority key, ensuring both schedulers share one
rigorously tested preemption logic path.

Exported symbols
----------------
MAX_HYPERPERIOD                          – safety cap for simulation window
edf_wcrt_simulation(tasks)              -> dict[int, float]
dm_wcrt_simulation(tasks)               -> dict[int, float]
stochastic_simulation(tasks, ...)       -> (max_rt, avg_missed)
collect_rt_distributions(tasks, ...)    -> (dm_rts, edf_rts)
"""

import heapq
import math
import os
import random
import sys

# Allow importing sibling packages when this module is used standalone
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tasks.generator import calculate_hyperperiod

# Maximum simulation window (safety cap, ~1 000 s in µs)
MAX_HYPERPERIOD: int = 10 ** 9


# ---------------------------------------------------------------------------
# Core event-driven simulation engine
# ---------------------------------------------------------------------------

def _event_driven_core(tasks, policy: str, exec_fn, duration: int) -> tuple:
    """
    Simulate *tasks* under *policy* ("DM" or "EDF") for *duration* time units.

    Parameters
    ----------
    tasks    : list[Task]
    policy   : "DM" | "EDF"
    exec_fn  : callable(Task) -> int  – execution time sampler
    duration : int – releases are generated for [0, duration)

    Priority keys
    -------------
    DM  : (relative_deadline, task_id, counter)  – fixed priority
    EDF : (absolute_deadline,  task_id, counter)  – dynamic priority

    The three-component key provides deterministic tie-breaking:
    the task_id breaks equal deadline ties; the monotone counter
    disambiguates multiple jobs of the same task at the same instant.

    Boundary fix
    ------------
    Jobs released just before *duration* may finish after it.  The loop
    continues until the ready queue is empty **and** no job is executing,
    so no completion event is silently dropped.

    Returns
    -------
    (response_times, missed)
        response_times : dict[task_id, list[int]]
        missed         : dict[task_id, int]
    """
    response_times = {t.id: [] for t in tasks}
    missed         = {t.id: 0  for t in tasks}

    # Pre-generate all job releases within [0, duration)
    releases = []
    for task in tasks:
        t = 0
        while t < duration:
            exec_t = exec_fn(task)
            releases.append((t, task.deadline, task.id,
                             t + task.deadline, exec_t))
            t += task.period
    releases.sort()

    rel_ptr  = 0
    ready    = []       # min-heap of (priority_key, job_dict)
    counter  = 0        # unique tiebreaker — monotone across entire run
    cur_key  = None
    cur_job  = None
    time     = 0

    while rel_ptr < len(releases) or ready or cur_job is not None:
        # Next event: earliest release or current job completion
        next_rel = releases[rel_ptr][0] if rel_ptr < len(releases) else float('inf')
        next_fin = (time + cur_job['remaining']
                    if cur_job is not None else float('inf'))

        next_t = min(next_rel, next_fin)
        if next_t == float('inf'):
            break

        # Do NOT stop mid-job just because next_t > duration.
        # A job released before duration may legitimately finish after it.
        if next_t > duration and cur_job is None:
            break

        # Advance remaining execution of the running job
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
                'task_id':   tid,
                'release':   rel_time,
                'abs_dl':    abs_dl,
                'rel_dl':    rel_dl,
                'remaining': exec_t,
            }
            key = ((rel_dl, tid, counter) if policy == "DM"
                   else (abs_dl, tid, counter))
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

        # Select next job / preempt if a higher-priority job arrived
        if ready:
            top_key = ready[0][0]
            if cur_job is None:
                cur_key, cur_job = heapq.heappop(ready)
            elif top_key < cur_key:
                heapq.heappush(ready, (cur_key, cur_job))
                cur_key, cur_job = heapq.heappop(ready)

    return response_times, missed


# ---------------------------------------------------------------------------
# Deterministic WCRT simulations (every job runs at WCET)
# ---------------------------------------------------------------------------

def edf_wcrt_simulation(tasks) -> dict:
    """
    Estimate EDF WCRTs by simulating one full hyperperiod with all jobs
    executing at their WCET (Appendix A approach).

    Returns
    -------
    dict[int, float]
        {task_id: WCRT}  —  float('inf') if hyperperiod overflows the cap.
    """
    try:
        H = calculate_hyperperiod(tasks)
    except Exception:
        return {t.id: float('inf') for t in tasks}

    if H > MAX_HYPERPERIOD:
        print(f"  [Warning] Hyperperiod capped at {MAX_HYPERPERIOD} "
              f"for EDF WCRT simulation.")
        H = MAX_HYPERPERIOD

    rt_dict, _ = _event_driven_core(
        tasks, policy="EDF",
        exec_fn=lambda t: t.wcet,
        duration=H,
    )
    return {tid: (max(rts) if rts else float('inf'))
            for tid, rts in rt_dict.items()}


def dm_wcrt_simulation(tasks) -> dict:
    """
    Estimate DM WCRTs by simulating one full hyperperiod with all jobs
    executing at their WCET (cross-check against RTA).

    Returns
    -------
    dict[int, float]
        {task_id: WCRT}
    """
    try:
        H = calculate_hyperperiod(tasks)
    except Exception:
        return {t.id: float('inf') for t in tasks}

    H = min(H, MAX_HYPERPERIOD)

    rt_dict, _ = _event_driven_core(
        tasks, policy="DM",
        exec_fn=lambda t: t.wcet,
        duration=H,
    )
    return {tid: (max(rts) if rts else float('inf'))
            for tid, rts in rt_dict.items()}


# ---------------------------------------------------------------------------
# Execution-time distribution sampler
# ---------------------------------------------------------------------------

def _sample_lognormal(task) -> int:
    """
    Sample one execution time from a log-normal distribution fitted to
    the interval [BCET, WCET].

    Fitting method
    --------------
    BCET and WCET are treated as the 1st and 99th percentiles of the
    underlying normal in log-space (z_{0.99} ≈ 2.3263):

        mu    = (ln(BCET) + ln(WCET)) / 2
        sigma = (ln(WCET) − ln(BCET)) / (2 × 2.3263)

    The sample is clamped to [BCET, WCET] and rounded to the nearest
    integer.  Falls back to WCET when BCET == WCET, and to a discrete
    uniform draw when BCET < 1 (ln(0) is undefined).

    Rationale for log-normal
    ------------------------
    Real software execution times are right-skewed and unimodal.  The
    log-normal distribution is the maximum-entropy choice for a
    multiplicative composition of many independent sub-path durations
    (Central Limit Theorem in log-space) and is well-validated against
    safety-critical embedded benchmarks.
    """
    if task.bcet >= task.wcet:
        return task.wcet
    if task.bcet < 1:
        # ln(0) undefined → fall back to discrete uniform
        return random.randint(0, task.wcet)

    ln_lo = math.log(task.bcet)
    ln_hi = math.log(task.wcet)
    mu    = (ln_lo + ln_hi) / 2.0
    sigma = (ln_hi - ln_lo) / (2.0 * 2.3263)   # z_{0.99}
    sample = random.lognormvariate(mu, sigma)
    return int(min(max(round(sample), task.bcet), task.wcet))


# ---------------------------------------------------------------------------
# Monte Carlo stochastic simulation
# ---------------------------------------------------------------------------

def stochastic_simulation(tasks, policy: str = "DM", n_runs: int = 500,
                           duration: int = None,
                           exec_dist: str = "lognormal") -> tuple:
    """
    Monte Carlo simulation sampling execution times from a chosen distribution.

    Parameters
    ----------
    tasks     : list[Task]
    policy    : "DM" | "EDF"
    n_runs    : number of independent runs  (default: 500)
                Pilot runs at 200/500/1000 showed max-RT estimates
                stabilising to within 1 % relative change by 500 runs.
    duration  : per-run window  (default: one hyperperiod, capped)
    exec_dist : "lognormal" (default) | "uniform"
                lognormal — right-skewed, fitted to [BCET, WCET] at
                             1st/99th percentiles.
                uniform   — discrete uniform over [BCET, WCET]; maximum-
                             entropy baseline when only bounds are known.

    Returns
    -------
    (max_rt, avg_missed)
        max_rt     : dict[task_id, int]   – per-task maximum observed RT
        avg_missed : dict[task_id, float] – per-task average deadline misses
    """
    if duration is None:
        try:
            H = calculate_hyperperiod(tasks)
            duration = min(H, MAX_HYPERPERIOD)
        except Exception:
            duration = 10 * max(t.period for t in tasks)

    max_rt       = {t.id: 0 for t in tasks}
    total_missed = {t.id: 0 for t in tasks}

    if exec_dist == "lognormal":
        def sample_exec(task):
            return _sample_lognormal(task)
    else:
        def sample_exec(task):
            if task.bcet >= task.wcet:
                return task.wcet
            return random.randint(task.bcet, task.wcet)

    for _ in range(n_runs):
        rt_dict, missed_dict = _event_driven_core(
            tasks, policy=policy,
            exec_fn=sample_exec,
            duration=duration,
        )
        for tid in max_rt:
            if rt_dict[tid]:
                max_rt[tid] = max(max_rt[tid], max(rt_dict[tid]))
            total_missed[tid] += missed_dict[tid]

    avg_missed = {tid: total_missed[tid] / n_runs for tid in total_missed}
    return max_rt, avg_missed


# ---------------------------------------------------------------------------
# Full response-time distribution collection for CDF plots
# ---------------------------------------------------------------------------

def collect_rt_distributions(tasks, n_runs: int = 200,
                              duration: int = None,
                              exec_dist: str = "lognormal") -> tuple:
    """
    Run Monte Carlo simulation for both DM and EDF and collect the *full*
    list of observed response times (not just the per-run maximum) for
    every task.

    Fewer runs than stochastic_simulation (200 vs 500) because all
    individual job completions are stored; 200 runs still yields thousands
    of samples per task for smooth CDFs.

    Parameters
    ----------
    tasks     : list[Task]
    n_runs    : number of independent runs  (default: 200)
    duration  : per-run window  (default: one hyperperiod, capped)
    exec_dist : "lognormal" | "uniform"

    Returns
    -------
    (dm_rts, edf_rts)
        dm_rts  : dict[task_id, list[int]]
        edf_rts : dict[task_id, list[int]]
    """
    if duration is None:
        try:
            H = calculate_hyperperiod(tasks)
            duration = min(H, MAX_HYPERPERIOD)
        except Exception:
            duration = 10 * max(t.period for t in tasks)

    dm_rts  = {t.id: [] for t in tasks}
    edf_rts = {t.id: [] for t in tasks}

    if exec_dist == "lognormal":
        def sample_exec(task):
            return _sample_lognormal(task)
    else:
        def sample_exec(task):
            if task.bcet >= task.wcet:
                return task.wcet
            return random.randint(task.bcet, task.wcet)

    for _ in range(n_runs):
        rt_dm,  _ = _event_driven_core(tasks, "DM",  sample_exec, duration)
        rt_edf, _ = _event_driven_core(tasks, "EDF", sample_exec, duration)
        for t in tasks:
            dm_rts[t.id].extend(rt_dm.get(t.id, []))
            edf_rts[t.id].extend(rt_edf.get(t.id, []))

    return dm_rts, edf_rts
