# reporting/table.py


def print_table(all_results: dict) -> None:
    """
    Print a summary comparison table for all analysed scenarios.

    Parameters
    ----------
    all_results : dict mapping scenario name → result dict with keys:
        'taskset', 'table', 'obs_dm', 'obs_edf'
    """
    for name, result in all_results.items():
        ts    = result['taskset']
        table = result['table']
        U     = ts.utilization

        print(f"\n{'='*90}")
        print(f"  RESULTS — {name}  (U={U:.4f})")
        print(f"{'='*90}")
        print(f"  {'τ':>2}  {'C':>4}  {'D':>4}  {'T':>4} │"
              f" {'DM-WCRT':>8} {'Sched':>6} │"
              f" {'EDF-WCRT':>9} {'Sched':>6} │"
              f" {'Sim-DM':>7} {'Sim-EDF':>8} │"
              f" {'DM-Pess':>8} {'EDF-Pess':>9}")
        print(f"  {'-'*2}  {'-'*4}  {'-'*4}  {'-'*4} │"
              f" {'-'*8} {'-'*6} │"
              f" {'-'*9} {'-'*6} │"
              f" {'-'*7} {'-'*8} │"
              f" {'-'*8} {'-'*9}")

        for r in table:
            dm_s  = '✓' if r['dm_schedulable']  else '✗'
            edf_s = '✓' if r['edf_schedulable'] else '✗'
            dm_p  = f"{r['dm_pessimism']:.3f}"  if r['dm_pessimism']  is not None else "   N/A"
            edf_p = f"{r['edf_pessimism']:.3f}" if r['edf_pessimism'] is not None else "   N/A"

            print(f"  {r['task_idx']:>2}  {r['C']:>4}  {r['D']:>4}  {r['T']:>4} │"
                  f" {r['dm_wcrt']:>8.1f} {dm_s:>6} │"
                  f" {r['edf_wcrt']:>9.1f} {edf_s:>6} │"
                  f" {r['sim_dm_wcrt']:>7.2f} {r['sim_edf_wcrt']:>8.2f} │"
                  f" {dm_p:>8} {edf_p:>9}")

        # Summary line
        dm_total_miss  = result['obs_dm'].total_misses()
        edf_total_miss = result['obs_edf'].total_misses()
        print(f"  {'─'*86}")
        print(f"  DM total misses: {dm_total_miss}  │  "
              f"EDF total misses: {edf_total_miss}")
        print(f"{'='*90}\n")
