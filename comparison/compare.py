# comparison/compare.py
from tasks.task_set import TaskSet
from analysis.dm_wcrt import dm_wcrt
from analysis.edf_wcrt import edf_wcrt
from simulation.simulator import simulate, Observer


def run_full_comparison(taskset: TaskSet,
                        sim_ticks: int   = 50000,
                        bcet_ratio: float = 0.5,
                        bcet_list: list   = None,
                        seed: int        = 42) -> dict:
    """
    Run all four tools on a task set and return unified results.

    Returns
    -------
    dict with keys:
        'taskset'    : the TaskSet
        'dm_rta'     : list of dicts from dm_wcrt()
        'edf_wcrt'   : list of dicts from edf_wcrt()
        'obs_dm'     : Observer from DM simulation
        'obs_edf'    : Observer from EDF simulation
        'table'      : list of row dicts (one per task) for printing
    """
    dm_results  = dm_wcrt(taskset)
    edf_results = edf_wcrt(taskset)
    obs_dm      = simulate(taskset, algorithm='DM',
                           sim_ticks=sim_ticks,
                           bcet_ratio=bcet_ratio,
                           bcet_list=bcet_list, seed=seed)
    obs_edf     = simulate(taskset, algorithm='EDF',
                           sim_ticks=sim_ticks,
                           bcet_ratio=bcet_ratio,
                           bcet_list=bcet_list, seed=seed)

    # Build unified table — align by task index
    # dm_results is in DM priority order (sorted by D)
    # edf_results is in original taskset order
    # We present in original taskset order for consistency
    tasks = taskset.tasks

    # Map DM results back to original task index
    dm_by_task = {}
    for r in dm_results:
        for i, t in enumerate(tasks):
            if r['task'] == t:
                dm_by_task[i] = r
                break

    table = []
    for i, task in enumerate(tasks):
        dm_r  = dm_by_task.get(i, {})
        edf_r = edf_results[i]

        dm_wcrt_val  = dm_r.get('wcrt', float('inf'))
        edf_wcrt_val = edf_r['wcrt']
        sim_dm_val   = obs_dm.wcrt[i]
        sim_edf_val  = obs_edf.wcrt[i]

        # Pessimism ratio = theoretical / simulated max
        # Values > 1.0 mean theory is conservative (bound not tight)
        dm_pessimism = (dm_wcrt_val / sim_dm_val
                        if sim_dm_val > 0 and dm_r.get('schedulable')
                        else None)
        edf_pessimism = (edf_wcrt_val / sim_edf_val
                         if sim_edf_val > 0 and edf_r['schedulable']
                         else None)

        table.append({
            'task_idx'      : i + 1,
            'C'             : task.C,
            'D'             : task.D,
            'T'             : task.T,
            'dm_wcrt'       : dm_wcrt_val,
            'dm_schedulable': dm_r.get('schedulable', False),
            'edf_wcrt'      : edf_wcrt_val,
            'edf_schedulable': edf_r['schedulable'],
            'sim_dm_wcrt'   : sim_dm_val,
            'sim_edf_wcrt'  : sim_edf_val,
            'dm_misses'     : obs_dm.misses[i],
            'edf_misses'    : obs_edf.misses[i],
            'dm_pessimism'  : dm_pessimism,
            'edf_pessimism' : edf_pessimism,
        })

    return {
        'taskset'   : taskset,
        'dm_rta'    : dm_results,
        'edf_wcrt'  : edf_results,
        'obs_dm'    : obs_dm,
        'obs_edf'   : obs_edf,
        'table'     : table,
    }


def print_comparison(result: dict) -> None:
    """Print the full side-by-side comparison table."""
    ts    = result['taskset']
    table = result['table']

    U   = ts.utilization
    lub = ts.utilization_bound_dm

    print(f"\n{'='*90}")
    print(f"  FULL COMPARISON — {ts.name}")
    print(f"  U={U:.4f}  U_lub(DM)={lub:.4f}  "
          f"{'U <= U_lub → DM sufficient' if U <= lub else 'U > U_lub → DM NOT guaranteed'}")
    print(f"{'='*90}")
    print(f"  {'τ':>2}  {'C':>3}  {'D':>3}  {'T':>3} │"
          f" {'DM-WCRT':>8} {'Sched':>6} │"
          f" {'EDF-WCRT':>9} {'Sched':>6} │"
          f" {'Sim-DM':>7} {'Sim-EDF':>8} │"
          f" {'DM-Miss':>7} {'EDF-Miss':>8}")
    print(f"  {'-'*2}  {'-'*3}  {'-'*3}  {'-'*3} │"
          f" {'-'*8} {'-'*6} │"
          f" {'-'*9} {'-'*6} │"
          f" {'-'*7} {'-'*8} │"
          f" {'-'*7} {'-'*8}")

    for r in table:
        dm_s  = '✓' if r['dm_schedulable']  else '✗'
        edf_s = '✓' if r['edf_schedulable'] else '✗'
        print(f"  {r['task_idx']:>2}  {r['C']:>3}  {r['D']:>3}  {r['T']:>3} │"
              f" {r['dm_wcrt']:>8.1f} {dm_s:>6} │"
              f" {r['edf_wcrt']:>9.1f} {edf_s:>6} │"
              f" {r['sim_dm_wcrt']:>7.2f} {r['sim_edf_wcrt']:>8.2f} │"
              f" {r['dm_misses']:>7} {r['edf_misses']:>8}")

    print(f"{'='*90}")

    # Pessimism summary
    print(f"\n  Pessimism Ratio (Theoretical WCRT / Max Simulated) — only for schedulable tasks:")
    print(f"  {'τ':>2}  {'DM Pessimism':>14}  {'EDF Pessimism':>14}")
    print(f"  {'-'*2}  {'-'*14}  {'-'*14}")
    for r in table:
        dm_p  = f"{r['dm_pessimism']:.3f}"  if r['dm_pessimism']  is not None else "N/A (infeasible)"
        edf_p = f"{r['edf_pessimism']:.3f}" if r['edf_pessimism'] is not None else "N/A (infeasible)"
        print(f"  {r['task_idx']:>2}  {dm_p:>14}  {edf_p:>14}")
    print(f"{'='*90}\n")


def print_all_comparisons(scenarios: dict, **kwargs) -> dict:
    """Run and print comparisons for all scenarios. Returns all results."""
    all_results = {}
    for name, ts in scenarios.items():
        result = run_full_comparison(ts, **kwargs)
        print_comparison(result)
        all_results[name] = result
    return all_results
