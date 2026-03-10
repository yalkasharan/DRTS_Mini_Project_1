# tasks/task_set.py
from dataclasses import dataclass
from math import gcd
from functools import reduce


# ── Task ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Task:
    """A single periodic real-time task τ_i = (C, D, T)."""
    C: float   # Worst-Case Execution Time (WCET)
    D: float   # Relative Deadline
    T: float   # Period

    def __post_init__(self):
        assert self.C > 0,       f"WCET C must be > 0, got {self.C}"
        assert self.D > 0,       f"Deadline D must be > 0, got {self.D}"
        assert self.T > 0,       f"Period T must be > 0, got {self.T}"
        assert self.D <= self.T, f"Requires D <= T, got D={self.D} T={self.T}"
        assert self.C <= self.D, f"Requires C <= D, got C={self.C} D={self.D}"

    @property
    def utilization(self):
        return self.C / self.T

    def __repr__(self):
        return f"Task(C={self.C}, D={self.D}, T={self.T})"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lcm(a, b):
    return a * b // gcd(int(a), int(b))


# ── TaskSet ───────────────────────────────────────────────────────────────────

class TaskSet:
    """An ordered collection of periodic tasks with shared properties."""

    def __init__(self, tasks, name=""):
        assert len(tasks) >= 2, "Need at least 2 tasks"
        self.tasks = list(tasks)
        self.name  = name

    @property
    def n(self):
        return len(self.tasks)

    @property
    def utilization(self):
        """Total CPU utilization U = sum(C_i / T_i)."""
        return sum(t.utilization for t in self.tasks)

    @property
    def utilization_bound_dm(self):
        """Liu & Layland RM/DM utilization bound: U_lub = n * (2^(1/n) - 1).
        NOTE: This is a sufficient (not necessary) schedulability condition
        for DM only when D_i = T_i (implicit deadlines). For constrained
        deadlines (D_i < T_i), use the full RTA test in dm_wcrt.py instead."""
        n = self.n
        return n * (2 ** (1.0 / n) - 1)

    @property
    def hyperperiod(self):
        """H = lcm(T_1, ..., T_n). Assumes integer periods."""
        periods = [int(t.T) for t in self.tasks]
        return reduce(_lcm, periods)

    @property
    def is_feasible_utilization(self):
        return self.utilization <= 1.0

    @property
    def dm_utilization_test(self):
        """Sufficient condition for DM: U <= U_lub."""
        return self.utilization <= self.utilization_bound_dm

    def validate(self):
        """Print summary and raise ValueError if U > 1.0."""
        U   = self.utilization
        lub = self.utilization_bound_dm
        H   = self.hyperperiod
        print(f"\n{'─'*50}")
        print(f"  TaskSet : {self.name or '(unnamed)'}")
        print(f"  Tasks   : {self.n}")
        print(f"  U       : {U:.4f}")
        print(f"  U_lub   : {lub:.4f}  (DM sufficient bound)")
        print(f"  H       : {H}  (hyperperiod)")
        if U > 1.0:
            raise ValueError(f"U={U:.3f} > 1.0 — infeasible under ANY algorithm.")
        if U > lub:
            print(f"  [WARN]  U > U_lub — DM schedulability NOT guaranteed. Run RTA.")
        if H > 2_000_000:
            print(f"  [WARN]  Hyperperiod H={H:,} is very large.")
        print(f"{'─'*50}")

    def sorted_by_deadline(self):
        """Return tasks sorted by D ascending (DM priority order)."""
        return sorted(self.tasks, key=lambda t: t.D)

    def __len__(self):
        return self.n

    def __iter__(self):
        return iter(self.tasks)

    def __repr__(self):
        tasks_str = "\n    ".join(repr(t) for t in self.tasks)
        return f"TaskSet(name='{self.name}', U={self.utilization:.3f})\n    {tasks_str}"
