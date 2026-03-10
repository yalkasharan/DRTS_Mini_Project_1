# tasks/generator.py
import random
import csv
import os
from tasks.task_set import Task, TaskSet


# ── The 3 project scenarios ────────────────────────────────────────────────────
#
#  Format: Task(C, D, T)
#  Rules:  D <= T  (constrained deadline)
#          C <= D  (task must be able to finish before its deadline)
#          Non-harmonic periods (no T_i divides T_j evenly across all pairs)
#          At least one task has D < T per scenario
#
#  S1: U=0.736  < U_lub=0.780  → Both DM and EDF schedulable
#  S2: U=0.929  > U_lub=0.780  → DM fails, EDF also fails (dbf > L)
#  S3: U=0.963  > U_lub=0.780  → DM fails, EDF also fails (dbf > L)
#
#  KEY INSIGHT: U ≤ 1.0 guarantees EDF only for implicit deadlines (D=T).
#  With constrained deadlines (D < T), the processor demand bound
#  dbf(L) = Σ max(0, floor((L-D_i)/T_i)+1)*C_i ≤ L must also hold.

S1 = TaskSet([
    Task(C=2, D=5,  T=7),    # U_1 = 0.286  D < T ✓
    Task(C=3, D=7,  T=10),   # U_2 = 0.300  D < T ✓
    Task(C=3, D=14, T=20),   # U_3 = 0.150  D < T ✓
], name="S1")                # Total U = 0.736

S2 = TaskSet([
    Task(C=3, D=5,  T=7),    # U_1 = 0.429  D < T ✓
    Task(C=4, D=8,  T=12),   # U_2 = 0.333  D < T ✓
    Task(C=3, D=12, T=18),   # U_3 = 0.167  D < T ✓
], name="S2")                # Total U = 0.929  dbf(12)=13>12 → EDF fails

S3 = TaskSet([
    Task(C=3, D=5,  T=7),    # U_1 = 0.429  D < T ✓
    Task(C=5, D=9,  T=12),   # U_2 = 0.417  D < T ✓
    Task(C=2, D=11, T=17),   # U_3 = 0.118  D < T ✓
], name="S3")                # Total U = 0.963  dbf(9)=11>9 → EDF fails

ALL_SCENARIOS = {"S1": S1, "S2": S2, "S3": S3}


# ── Random task set generator ─────────────────────────────────────────────────

def generate_taskset(n=3, target_U=0.7, seed=None, name="random"):
    """
    Generate a random task set with utilization close to target_U.

    Parameters
    ----------
    n        : number of tasks
    target_U : desired total utilization (must be < 1.0)
    seed     : random seed for reproducibility
    name     : name label for the TaskSet

    Returns
    -------
    TaskSet
    """
    assert 0 < target_U < 1.0, "target_U must be in (0, 1)"
    rng = random.Random(seed)

    period_pool = [5, 7, 10, 12, 14, 17, 18, 20, 25, 30]

    for _ in range(1000):
        periods = rng.sample(period_pool, n)
        utils   = _uunifast(n, target_U, rng)
        tasks   = []
        valid   = True

        for i in range(n):
            T = periods[i]
            C = max(1, round(utils[i] * T))
            D = rng.randint(C, T)
            try:
                tasks.append(Task(C=C, D=D, T=T))
            except AssertionError:
                valid = False
                break

        if valid and len(tasks) == n:
            return TaskSet(tasks, name=name)

    raise RuntimeError("Could not generate valid task set after 1000 attempts.")


def _uunifast(n, U, rng):
    """
    UUniFast algorithm — unbiased utilization distribution.
    Returns list of n utilization values summing to U.
    """
    utils  = []
    sum_U  = U
    for i in range(1, n):
        next_sum = sum_U * (rng.random() ** (1.0 / (n - i)))
        utils.append(sum_U - next_sum)
        sum_U = next_sum
    utils.append(sum_U)
    return utils


# ── CSV loader ────────────────────────────────────────────────────────────────

def load_taskset_from_csv(filepath: str, name: str = None,
                          deadline_factor: float = 1.0,
                          mix_deadlines: bool = False,
                          wcet_multiplier: bool = False):
    """
    Load a task set from a CSV file.

    deadline_factor  : uniform scale applied as D = max(C, factor * T).
                       Use <1.0 (e.g. 0.8) for constrained deadlines.
                       Default 1.0 preserves the CSV deadline as-is.

    mix_deadlines    : assigns per-task D/T ratios cycling through
                       [0.30..0.90] so DM priority order diverges from
                       T-order.  Works only at high utilisation (U > 0.75).

    wcet_multiplier  : **most accurate for showing DM vs EDF differences**.
                       Sets D_i = clamp(k_i * C_i, C_i+1, T_i) where k_i
                       cycles through [2, 8, 3, 12, 2.5, 6, 4, 15, 2, 5].
                       Because D scales with WCET, slack (D-C) is always
                       comparable to interference regardless of utilisation,
                       so DM != EDF even at U=0.30.

    Auto-detects two formats:

    Format A — Simple:
        C,D,T
        2,5,7

    Format B — Rich (project CSV format):
        TaskID,Jitter,BCET,WCET,Period,Deadline,PE
        0,0,43,432,10000,10000,0

    Parameters
    ----------
    filepath         : path to CSV file
    name             : optional TaskSet name (defaults to filename without extension)
    deadline_factor  : multiply T to get D (clamped to [C, T]). Default 1.0.
    mix_deadlines    : if True, assign per-task varying D/T ratios. Default False.
    wcet_multiplier  : if True, assign D = k*C (clamped to T). Default False.

    Returns
    -------
    (TaskSet, bcet_list)
        bcet_list[i] = BCET for task i (defaults to 0.5*C if not in CSV)
    """
    # Per-task D/T ratio cycle used when mix_deadlines=True.
    # Values spread across [0.30, 0.90] in non-monotone order so that
    # sorting by D does NOT equal sorting by T — this creates genuine
    # DM vs EDF scheduling differences.
    _MIX_FACTORS = [0.30, 0.65, 0.45, 0.80, 0.35, 0.70, 0.50, 0.90, 0.40, 0.60]

    # Per-task WCET multipliers used when wcet_multiplier=True.
    # D_i = clamp(k_i * C_i, C_i+1, T_i).  Because D scales with WCET,
    # slack (D-C) is always proportional to C regardless of T, so
    # interference can push some tasks past their deadline at any U.
    # Non-monotone ordering ensures DM sort (by D) != sort by T.
    _WCET_MULTS = [2.0, 8.0, 3.0, 12.0, 2.5, 6.0, 4.0, 15.0, 2.0, 5.0]
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    if name is None:
        name = os.path.splitext(os.path.basename(filepath))[0]

    tasks     = []
    bcet_list = []

    with open(filepath, newline='', encoding='utf-8') as f:
        reader  = csv.DictReader(f)
        headers = set(reader.fieldnames or [])

        # Auto-detect format
        rich_format = {'WCET', 'Period', 'Deadline'}.issubset(headers)
        simple_fmt  = {'C', 'D', 'T'}.issubset(headers)

        if not rich_format and not simple_fmt:
            raise ValueError(
                f"CSV must have columns (C, D, T) or (WCET, Period, Deadline). "
                f"Found: {list(headers)}"
            )

        for i, row in enumerate(reader, start=1):
            try:
                if rich_format:
                    C    = float(row['WCET'].strip())
                    D    = float(row['Deadline'].strip())
                    T    = float(row['Period'].strip())
                    bcet = float(row['BCET'].strip()) if 'BCET' in row else 0.5 * C
                else:
                    C    = float(row['C'].strip())
                    D    = float(row['D'].strip())
                    T    = float(row['T'].strip())
                    bcet = 0.5 * C

                # Skip rows with C=0
                if C == 0:
                    print(f"  [SKIP] Row {i}: C=0, skipping.")
                    continue

                # Apply deadline scaling
                if wcet_multiplier:
                    # WCET-proportional: D = k*C, so slack ~ C regardless of T.
                    # This makes interference significant at ALL utilisation levels.
                    mult_i = _WCET_MULTS[(i - 1) % len(_WCET_MULTS)]
                    D = max(C + 1, min(T, mult_i * C))
                elif mix_deadlines:
                    # Per-task mixed D/T ratio: cycles through _MIX_FACTORS
                    factor_i = _MIX_FACTORS[(i - 1) % len(_MIX_FACTORS)]
                    D = max(C, min(T, factor_i * T))
                elif deadline_factor != 1.0:
                    D = max(C, min(T, deadline_factor * T))

                tasks.append(Task(C=C, D=D, T=T))
                bcet_list.append(bcet)

            except (ValueError, AssertionError) as e:
                raise ValueError(
                    f"Invalid task on row {i}: {dict(row)} — {e}\n"
                    f"  Check that C <= D <= T for all rows."
                )

    if len(tasks) < 2:
        raise ValueError(
            f"Need at least 2 valid tasks, got {len(tasks)}. "
            f"Check that C <= D <= T for all rows."
        )

    return TaskSet(tasks, name=name), bcet_list


def load_all_csv_from_folder(folder: str,
                              deadline_factor: float = 1.0,
                              mix_deadlines: bool = False,
                              wcet_multiplier: bool = False) -> dict:
    """
    Load all CSV files from a folder.

    deadline_factor  : passed through to load_taskset_from_csv.
    mix_deadlines    : passed through — per-task D/T ratios (works at high U).
    wcet_multiplier  : passed through — D = k*C ratios (works at ANY U).

    Returns
    -------
    dict mapping filename (without .csv) -> (TaskSet, bcet_list)
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    result = {}
    for fname in sorted(os.listdir(folder)):
        if fname.endswith('.csv'):
            path = os.path.join(folder, fname)
            ts, bcet = load_taskset_from_csv(path,
                                              deadline_factor=deadline_factor,
                                              mix_deadlines=mix_deadlines,
                                              wcet_multiplier=wcet_multiplier)
            result[ts.name] = (ts, bcet)
            print(f"  Loaded: {fname}  {ts.n} tasks, "
                  f"U={ts.utilization:.4f}")

    if not result:
        print(f"  [WARN] No CSV files found in {folder}/")

    return result
