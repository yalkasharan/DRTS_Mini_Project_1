"""
analysis/edf_wcrt.py
--------------------
EDF schedulability tests.

Two complementary tests are provided:

1. edf_utilization_test   – necessary and sufficient for *implicit* deadlines
                            (D_i == T_i);  only necessary for constrained.
2. edf_processor_demand_test – Processor Demand Criterion (PDC / dbf test);
                            necessary and sufficient for *constrained*
                            deadlines (D_i <= T_i).  Fast-paths to the
                            utilization test when all deadlines are implicit,
                            so there is no runtime penalty for homogeneous
                            benchmark sets.

Exported symbols
----------------
edf_utilization_test(tasks)      -> (schedulable: bool, U: float)
edf_processor_demand_test(tasks) -> (schedulable: bool, reason: str)
"""

import math
import os
import sys

# Allow importing sibling packages when this module is used standalone
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tasks.generator import calculate_hyperperiod


# ---------------------------------------------------------------------------
# Utilization bound (Liu & Layland 1973)
# ---------------------------------------------------------------------------

def edf_utilization_test(tasks) -> tuple:
    """
    EDF schedulability via the processor utilization bound.

    Necessary *and* sufficient when D_i == T_i for all tasks.
    Only necessary when D_i < T_i (constrained deadlines).

    Returns
    -------
    (schedulable, U)
        schedulable : bool
        U           : float – total processor utilization
    """
    U = sum(t.wcet / t.period for t in tasks)
    return U <= 1.0 + 1e-9, U


# ---------------------------------------------------------------------------
# Processor Demand Criterion (Baruah et al. 1990)
# ---------------------------------------------------------------------------

def edf_processor_demand_test(tasks) -> tuple:
    """
    EDF schedulability via the Processor Demand Criterion (PDC).

    For implicit-deadline sets (D_i == T_i) the utilization test is
    necessary and sufficient; this function fast-paths to that check.

    For constrained-deadline sets (D_i < T_i) the PDC demand-bound
    function (dbf) is evaluated at every deadline checkpoint L up to
    min(H, 10 · max T_i):

        dbf(L) = Σ_i ⌊(L + T_i − D_i) / T_i⌋ · C_i  ≤  L

    Using only the utilization test on constrained-deadline sets would
    over-report schedulability; this function is always correct.

    Returns
    -------
    (schedulable, reason)
        schedulable : bool
        reason      : str – human-readable explanation for console output
    """
    sched, U = edf_utilization_test(tasks)
    if not sched:
        return False, f"U={U:.4f} > 1.0"

    # Fast-path: implicit deadlines → utilization test is exact
    if all(t.deadline == t.period for t in tasks):
        return True, f"U={U:.4f} (implicit deadlines)"

    try:
        H = calculate_hyperperiod(tasks)
    except Exception:
        return sched, f"U={U:.4f} (hyperperiod overflow)"

    MAX_L = min(H, 10 * max(t.period for t in tasks))

    # Collect all deadline checkpoints in [1, MAX_L]
    checkpoints: set = set()
    for t in tasks:
        d = t.deadline
        while d <= MAX_L:
            checkpoints.add(d)
            d += t.period

    for L in sorted(checkpoints):
        demand = 0
        for t in tasks:
            if L >= t.deadline:
                demand += (
                    math.floor((L + t.period - t.deadline) / t.period)
                    * t.wcet
                )
        if demand > L + 1e-9:
            return False, f"dbf({L})={demand} > {L}"

    return True, f"U={U:.4f}"
