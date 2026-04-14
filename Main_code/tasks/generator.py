"""
tasks/generator.py
------------------
Task model and task-set I/O for the DRTS Mini-Project 1 toolchain.

Exported symbols
----------------
Task                          – sporadic-task descriptor
calculate_hyperperiod(tasks)  -> int
load_tasks(filename)          -> list[Task]
"""

import csv
import math


# ---------------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------------

class Task:
    """Sporadic task described by (BCET, WCET, period, deadline)."""

    def __init__(self, t_id: int, bcet: int, wcet: int,
                 period: int, deadline: int) -> None:
        self.id       = int(t_id)
        self.bcet     = int(bcet)
        self.wcet     = int(wcet)
        self.period   = int(period)
        self.deadline = int(deadline)

    def __repr__(self) -> str:
        return (f"T{self.id}(C=[{self.bcet},{self.wcet}], "
                f"T={self.period}, D={self.deadline})")


# ---------------------------------------------------------------------------
# Hyperperiod
# ---------------------------------------------------------------------------

def calculate_hyperperiod(tasks) -> int:
    """Return the least common multiple of all task periods."""
    periods = [t.period for t in tasks]
    lcm = periods[0]
    for p in periods[1:]:
        lcm = lcm * p // math.gcd(lcm, p)
    return lcm


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_tasks(filename: str):
    """
    Load a task set from a CSV file.

    Expected header (columns may appear in any order):
        TaskID, Jitter, BCET, WCET, Period, Deadline, PE

    The Jitter and PE columns are present in the benchmark files but are
    not used by this toolchain (zero-jitter, single-core assumption).

    Returns
    -------
    list[Task]
        Tasks in the order they appear in the file.
    """
    tasks = []
    with open(filename, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tid = row.get('TaskID', '').strip()
            if not tid:
                continue
            tasks.append(Task(
                t_id     = int(tid),
                bcet     = int(row['BCET']),
                wcet     = int(row['WCET']),
                period   = int(row['Period']),
                deadline = int(row['Deadline']),
            ))
    return tasks
