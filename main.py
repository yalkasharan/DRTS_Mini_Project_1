# main.py
import argparse
import os
import pickle
import sys

# Ensure Unicode output works on Windows (box-drawing, Greek letters, etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from tasks.generator        import ALL_SCENARIOS, load_all_csv_from_folder
from analysis.dm_wcrt       import dm_wcrt
from analysis.edf_wcrt      import edf_wcrt
from simulation.simulator   import simulate
from visualization.plots    import generate_all_plots
from reporting.table        import print_table
import visualization.plots  as plots_module

MAX_SIM_TIME = 1_000_000  # cap hyperperiod for simulation


def run_one(taskset, bcet_list=None, name=""):
    """Run full analysis + simulation on one taskset. Returns result dict."""
    taskset.name = name or taskset.name

    # ── Theory ────────────────────────────────────────────────────────────────
    dm_results  = dm_wcrt(taskset)   # list of dicts in DM priority order
    edf_results = edf_wcrt(taskset)  # list of dicts in original task order

    # Map DM results back to original task index (use 'is' for identity)
    tasks      = taskset.tasks
    dm_by_task = {}
    for r in dm_results:
        for i, t in enumerate(tasks):
            if r['task'] is t:
                dm_by_task[i] = r
                break

    # ── Simulation (stochastic: execution times in [BCET, WCET]) ────────────
    H       = min(taskset.hyperperiod, MAX_SIM_TIME)
    obs_dm  = simulate(taskset, algorithm='DM',  sim_ticks=H,
                       bcet_list=bcet_list)
    obs_edf = simulate(taskset, algorithm='EDF', sim_ticks=H,
                       bcet_list=bcet_list)

    # ── Simulation (worst-case: every job runs for exactly WCET) ──────────
    wcet_list = [t.C for t in tasks]
    obs_dm_wc  = simulate(taskset, algorithm='DM',  sim_ticks=H,
                          bcet_list=wcet_list)
    obs_edf_wc = simulate(taskset, algorithm='EDF', sim_ticks=H,
                          bcet_list=wcet_list)

    # ── Build result table ────────────────────────────────────────────────────
    table = []
    for i, task in enumerate(tasks):
        dm_r    = dm_by_task.get(i, {})
        dm_w    = dm_r.get('wcrt', float('inf'))
        edf_w   = edf_results[i]['wcrt']
        sim_dm  = obs_dm.wcrt[i]  if obs_dm.wcrt[i]  > 0 else None
        sim_edf = obs_edf.wcrt[i] if obs_edf.wcrt[i] > 0 else None

        table.append({
            'task_idx':        i + 1,
            'C':               task.C,
            'D':               task.D,
            'T':               task.T,
            'dm_wcrt':         dm_w,
            'edf_wcrt':        edf_w,
            'sim_dm_wcrt':     sim_dm  or 0,
            'sim_edf_wcrt':    sim_edf or 0,
            'dm_schedulable':  dm_r.get('schedulable', False),
            'edf_schedulable': edf_results[i]['schedulable'],
            'dm_pessimism':    (dm_w  / sim_dm)  if sim_dm  and sim_dm  > 0 else None,
            'edf_pessimism':   (edf_w / sim_edf) if sim_edf and sim_edf > 0 else None,
        })

    return {
        'taskset':    taskset,
        'table':      table,
        'obs_dm':     obs_dm,
        'obs_edf':    obs_edf,
        'obs_dm_wc':  obs_dm_wc,
        'obs_edf_wc': obs_edf_wc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-folder', type=str, default=None,
                        help='Folder of CSV task set files')
    parser.add_argument('--max-files',  type=int, default=None,
                        help='Limit number of CSV files to process')
    parser.add_argument('--output',     type=str, default='results',
                        help='Output folder for plots (default: results)')
    parser.add_argument('--deadline-factor', type=float, default=1.0,
                        help='Scale deadlines: D = max(C, factor*T). '
                             'Use <1.0 (e.g. 0.8) for constrained deadlines '
                             'to expose DM vs EDF differences (default: 1.0)')
    parser.add_argument('--mix-deadlines', action='store_true',
                        help='Assign per-task mixed D/T ratios (cycles through '
                             '0.30-0.90). Ensures DM sort order differs from '
                             'T-order so DM != EDF in plots.')
    parser.add_argument('--wcet-multiplier', action='store_true',
                        help='Set D_i = k_i * C_i (k cycles through '
                             '[2,8,3,12,2.5,6,4,15,2,5]). Deadlines scale '
                             'with WCET so interference is significant at '
                             'ANY utilisation level — most accurate for '
                             'showing real DM vs EDF differences.')
    args = parser.parse_args()

    # ── Point plots module to correct output folder ───────────────────────────
    plots_module.RESULTS_DIR = args.output
    os.makedirs(args.output, exist_ok=True)

    all_results = {}

    if args.csv_folder:
        # ── Benchmark mode ────────────────────────────────────────────────────
        print(f"\nLoading CSVs from: {args.csv_folder}")
        if args.wcet_multiplier:
            print("  [wcet-multiplier] D_i = k_i*C_i with k in [2..15]. "
                  "Interference is significant at any utilisation.")
        elif args.mix_deadlines:
            print("  [mix-deadlines] Assigning per-task varying D/T ratios "
                  "[0.30..0.90] to expose DM vs EDF differences")
        elif args.deadline_factor != 1.0:
            print(f"  [deadline-factor={args.deadline_factor:.2f}] "
                  f"Scaling D = max(C, {args.deadline_factor:.2f}*T) "
                  f"to create constrained-deadline task sets")
        csv_dict = load_all_csv_from_folder(args.csv_folder,
                                            deadline_factor=args.deadline_factor,
                                            mix_deadlines=args.mix_deadlines,
                                            wcet_multiplier=args.wcet_multiplier)

        items = list(csv_dict.items())
        if args.max_files:
            items = items[:args.max_files]
            print(f"  [Limiting to {args.max_files} files]")

        for idx, (name, (taskset, bcet_list)) in enumerate(items):
            print(f"  [{idx+1}/{len(items)}] {name} "
                  f"(U={taskset.utilization:.3f}, H={taskset.hyperperiod:,}) ...")
            try:
                all_results[name] = run_one(taskset, bcet_list, name=name)
            except Exception as e:
                print(f"    [ERROR] Skipped {name}: {e}")
                continue

    else:
        # ── S1 / S2 / S3 mode ─────────────────────────────────────────────────
        print("\nRunning built-in scenarios S1, S2, S3...")
        for name, taskset in ALL_SCENARIOS.items():
            print(f"  Processing {name}...")
            all_results[name] = run_one(taskset, name=name)

    if not all_results:
        print("\n[ERROR] No results generated. Check your CSV folder.")
        return

    # ── Print table ────────────────────────────────────────────────────────────
    print_table(all_results)

    # ── Generate plots ─────────────────────────────────────────────────────────
    generate_all_plots(all_results)

    # ── Save results for benchmark comparison ─────────────────────────────────
    if args.csv_folder:
        pkl_path = os.path.join(args.output, "results.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(all_results, f)
        print(f"  Results saved to: {pkl_path}")

    # ── Print miss summary ────────────────────────────────────────────────────
    for name, result in all_results.items():
        dm_wc  = result['obs_dm_wc'].total_misses()
        edf_wc = result['obs_edf_wc'].total_misses()
        dm_m   = result['obs_dm'].total_misses()
        edf_m  = result['obs_edf'].total_misses()
        u = result['taskset'].utilization
        if dm_wc > 0 or edf_wc > 0 or dm_m > 0 or edf_m > 0:
            print(f"  MISSES — {name} (U={u:.3f}): "
                  f"WCET[DM={dm_wc},EDF={edf_wc}] "
                  f"Rand[DM={dm_m},EDF={edf_m}]")
        else:
            print(f"  No misses — {name} (U={u:.3f})")


if __name__ == '__main__':
    main()
