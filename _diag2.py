import pickle
from tasks.generator import load_taskset_from_csv
from analysis.dm_wcrt import dm_wcrt
from analysis.edf_wcrt import edf_wcrt

with open('results/automotive/results.pkl', 'rb') as f:
    auto = pickle.load(f)

print("=== Automotive: DM vs EDF Theory ===\n")
for name in list(auto.keys())[:8]:
    r = auto[name]
    table = r['table']
    u = r['taskset'].utilization
    print(f"{name}  U={u:.4f}  H={r['taskset'].hyperperiod:,}")
    print(f"  {'t':>2}  {'C':>8}  {'D':>8}  {'T':>8}  {'DM_wcrt':>10}  {'EDF_wcrt':>10}  {'sim_DM':>8}  {'sim_EDF':>8}  {'diff_theo':>10}")
    for row in table:
        diff = row['dm_wcrt'] - row['edf_wcrt']
        print(f"  {row['task_idx']:>2}  {row['C']:>8.0f}  {row['D']:>8.0f}  {row['T']:>8.0f}  {row['dm_wcrt']:>10.2f}  {row['edf_wcrt']:>10.2f}  {row['sim_dm_wcrt']:>8.2f}  {row['sim_edf_wcrt']:>8.2f}  {diff:>+10.2f}")
    # Are D == T for all tasks?
    all_implicit = all(row['D'] == row['T'] for row in table)
    print(f"  implicit deadlines (D==T): {all_implicit}")
    print()

# Also check: what does the csv look like for D vs T?
print("\n=== Check D vs T in raw CSV ===")
ts, _ = load_taskset_from_csv('tasksets/automotive/0.50-util_automotive_0.csv')
for t in ts.tasks[:5]:
    print(f"  C={t.C} D={t.D} T={t.T}  D==T: {t.D == t.T}")
