# tests/test_cases.py
"""
Validation test cases for the DRTS Mini-Project 1 analytical tools.

Tests are organised in three levels:
  1. Unit tests  — hand-verifiable single-task calculations
  2. Integration — full pipeline on known task sets
  3. Edge cases  — boundary conditions (U=1, U>1, single task, D=T)
"""

import sys
from pathlib import Path

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.task_set import Task, TaskSet
from tasks.generator import S1, S2, S3
from analysis.dm_wcrt import dm_wcrt, dm_is_schedulable
from analysis.edf_wcrt import edf_wcrt, edf_is_schedulable
from simulation.simulator import simulate


# ── Helpers ───────────────────────────────────────────────────────────────────

def passed(name):
    print(f"  PASS  {name}")

def failed(name, msg):
    print(f"  FAIL  {name}: {msg}")
    raise AssertionError(f"{name}: {msg}")


# ── 1. Unit Tests: DM RTA hand calculations ───────────────────────────────────

def test_dm_single_task():
    """Single task: WCRT must equal C (no interference)."""
    ts = TaskSet([Task(C=3, D=8, T=10), Task(C=1, D=10, T=20)], name="unit")
    results = dm_wcrt(ts)
    # Highest priority task has no interference → WCRT = C
    r = results[0]
    assert r['wcrt'] == r['task'].C, f"Expected {r['task'].C}, got {r['wcrt']}"
    passed("test_dm_single_task")


def test_dm_s1_manual():
    """
    S1 manual hand-calculation (Buttazzo Fig 4.17):
    Sorted by D: τ1(C=2,D=5,T=7), τ2(C=3,D=7,T=10), τ3(C=3,D=14,T=20)

    R1 = 2  (no HP tasks)
    R2: R=5 → 3 + ceil(5/7)*2 = 3+2=5  ✓ converged
    R3: R=8 → 3 + ceil(8/7)*2 + ceil(8/10)*3 = 3+4+3=10
        R=10→ 3 + ceil(10/7)*2 + ceil(10/10)*3= 3+4+3=10 ✓ converged
    """
    results = dm_wcrt(S1)
    wcrts = [r['wcrt'] for r in results]
    assert wcrts == [2, 5, 10], f"Expected [2,5,10], got {wcrts}"
    passed("test_dm_s1_manual")


def test_dm_s2_infeasible():
    """
    S2: τ3 is infeasible — the fixed-point iteration diverges past T=18,
    so the correct WCRT is inf (not the first R > D intermediate value).
    """
    results = dm_wcrt(S2)
    assert results[0]['schedulable'] == True,  "S2 τ1 should be schedulable"
    assert results[1]['schedulable'] == True,  "S2 τ2 should be schedulable"
    assert results[2]['schedulable'] == False, "S2 τ3 should be INFEASIBLE"
    assert results[2]['wcrt'] == float('inf'), \
        f"S2 τ3 WCRT should be inf (divergent), got {results[2]['wcrt']}"
    passed("test_dm_s2_infeasible")


def test_dm_s3_infeasible():
    """S3: τ2 and τ3 must be infeasible."""
    results = dm_wcrt(S3)
    assert results[0]['schedulable'] == True,  "S3 τ1 should be schedulable"
    assert results[1]['schedulable'] == False, "S3 τ2 should be INFEASIBLE"
    assert results[2]['schedulable'] == False, "S3 τ3 should be INFEASIBLE"
    passed("test_dm_s3_infeasible")


# ── 2. Unit Tests: EDF WCRT ───────────────────────────────────────────────────

def test_edf_s1_manual():
    """S1 EDF: all schedulable, WCRTs = [2, 5, 10]."""
    results = edf_wcrt(S1)
    wcrts = [r['wcrt'] for r in results]
    assert wcrts == [2, 5, 10], f"Expected [2,5,10], got {wcrts}"
    assert all(r['schedulable'] for r in results), "S1 all EDF should be schedulable"
    passed("test_edf_s1_manual")


def test_edf_s2_infeasible():
    """S2 EDF: τ3 infeasible (WCRT=13 > D=12)."""
    results = edf_wcrt(S2)
    assert results[2]['schedulable'] == False, "S2 τ3 EDF should be INFEASIBLE"
    assert results[2]['wcrt'] == 13, f"S2 τ3 EDF WCRT expected 13, got {results[2]['wcrt']}"
    passed("test_edf_s2_infeasible")


def test_edf_s3_all_infeasible():
    """S3 EDF: all tasks infeasible under tight constrained deadlines."""
    results = edf_wcrt(S3)
    assert not any(r['schedulable'] for r in results), \
        "S3: all EDF tasks should be INFEASIBLE"
    passed("test_edf_s3_all_infeasible")


def test_edf_implicit_deadline():
    """
    EDF with implicit deadlines (D=T): U<=1 guarantees schedulability.
    Task set: τ1(C=2,D=5,T=5), τ2(C=2,D=8,T=8)  U=0.65
    """
    ts = TaskSet([
        Task(C=2, D=5, T=5),
        Task(C=2, D=8, T=8),
    ], name="implicit")
    results = edf_wcrt(ts)
    assert all(r['schedulable'] for r in results), \
        "Implicit deadline task set should be EDF schedulable"
    passed("test_edf_implicit_deadline")


# ── 3. Integration Tests: Simulation ─────────────────────────────────────────

def test_sim_s1_zero_misses():
    """S1: Both DM and EDF simulation must have zero deadline misses."""
    obs_dm  = simulate(S1, algorithm='DM',  sim_ticks=50000, seed=42)
    obs_edf = simulate(S1, algorithm='EDF', sim_ticks=50000, seed=42)
    assert obs_dm.total_misses()  == 0, f"S1 DM misses={obs_dm.total_misses()}"
    assert obs_edf.total_misses() == 0, f"S1 EDF misses={obs_edf.total_misses()}"
    passed("test_sim_s1_zero_misses")


def test_sim_dm_worse_than_edf():
    """
    Implicit-deadline task set where DM RTA proves τ2 infeasible but EDF (U<=1) passes.

    Hand-calculated DM RTA:
      τ1: R=3 ≤ D=5  ✓ schedulable
      τ2: R=3→6→9→9  WCRT=9 > D=8  ✗ INFEASIBLE
    EDF: U=0.975 ≤ 1.0 with D=T → all schedulable.
    """
    ts_dm_fails = TaskSet([
        Task(C=3, D=5, T=5),   # U = 3/5  = 0.600
        Task(C=3, D=8, T=8),   # U = 3/8  = 0.375   total U = 0.975
    ], name="dm_fails_edf_ok")
    # verify analytically first
    dm_res  = dm_wcrt(ts_dm_fails)
    edf_res = edf_wcrt(ts_dm_fails)
    assert dm_res[1]['schedulable']  == False, "DM τ2 should be infeasible"
    assert edf_res[0]['schedulable'] == True,  "EDF τ1 should be schedulable"
    assert edf_res[1]['schedulable'] == True,  "EDF τ2 should be schedulable"

    obs_dm  = simulate(ts_dm_fails, algorithm='DM',  sim_ticks=100000, seed=42)
    obs_edf = simulate(ts_dm_fails, algorithm='EDF', sim_ticks=100000, seed=42)
    assert obs_edf.total_misses() == 0, \
        f"EDF should have 0 misses (U<=1, implicit deadlines), got {obs_edf.total_misses()}"
    assert obs_dm.total_misses() > 0, \
        f"DM should have misses (τ2 WCRT=9 > D=8), got {obs_dm.total_misses()}"
    passed("test_sim_dm_worse_than_edf")


def test_sim_wcrt_positive():
    """Simulated WCRTs must all be positive."""
    obs = simulate(S1, algorithm='EDF', sim_ticks=10000, seed=0)
    assert all(w > 0 for w in obs.wcrt), f"All WCRTs must be > 0: {obs.wcrt}"
    passed("test_sim_wcrt_positive")


def test_sim_distributions_nonempty():
    """Response time distributions must be non-empty for all tasks."""
    obs = simulate(S1, algorithm='EDF', sim_ticks=10000, seed=0)
    assert all(len(d) > 0 for d in obs.rt_dist), "All rt_dist lists must be non-empty"
    passed("test_sim_distributions_nonempty")


# ── 4. Edge Case Tests ────────────────────────────────────────────────────────

def test_utilization_over_1():
    """U > 1.0 must raise ValueError in EDF WCRT tool."""
    try:
        ts = TaskSet([
            Task(C=5, D=6, T=7),
            Task(C=5, D=6, T=7),
        ], name="overload")
        edf_wcrt(ts)
        failed("test_utilization_over_1", "Should have raised ValueError")
    except ValueError:
        passed("test_utilization_over_1")


def test_dm_priority_order():
    """DM results must be sorted by deadline ascending."""
    results = dm_wcrt(S1)
    deadlines = [r['task'].D for r in results]
    assert deadlines == sorted(deadlines), \
        f"DM results not in deadline order: {deadlines}"
    passed("test_dm_priority_order")


def test_wcrt_geq_c():
    """WCRT must always be >= C for every task."""
    for ts in [S1, S2, S3]:
        for r in dm_wcrt(ts):
            assert r['wcrt'] >= r['task'].C, \
                f"WCRT={r['wcrt']} < C={r['task'].C}"
        for r in edf_wcrt(ts):
            assert r['wcrt'] >= r['task'].C, \
                f"WCRT={r['wcrt']} < C={r['task'].C}"
    passed("test_wcrt_geq_c")


# ── Run all tests ─────────────────────────────────────────────────────────────

def run_all_tests():
    print("\n" + "="*55)
    print("  Running all validation tests...")
    print("="*55)

    tests = [
        test_dm_single_task,
        test_dm_s1_manual,
        test_dm_s2_infeasible,
        test_dm_s3_infeasible,
        test_edf_s1_manual,
        test_edf_s2_infeasible,
        test_edf_s3_all_infeasible,
        test_edf_implicit_deadline,
        test_sim_s1_zero_misses,
        test_sim_dm_worse_than_edf,
        test_sim_wcrt_positive,
        test_sim_distributions_nonempty,
        test_utilization_over_1,
        test_dm_priority_order,
        test_wcrt_geq_c,
    ]

    passed_count = 0
    failed_count = 0

    for test in tests:
        try:
            test()
            passed_count += 1
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed_count += 1

    print("="*55)
    print(f"  Results: {passed_count} passed, {failed_count} failed")
    print("="*55 + "\n")
    return failed_count == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
