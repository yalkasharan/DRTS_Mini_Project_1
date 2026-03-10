# visualization/plots.py
"""
All visualisation for the DRTS Mini-Project.

Uses matplotlib + seaborn exclusively.
Every public function saves a PNG to RESULTS_DIR and prints the path.
"""
import os
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                       # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings('ignore', category=UserWarning)

RESULTS_DIR = "results"

# ── Colour palette ────────────────────────────────────────────────────────────
C_DM_TH   = '#2196F3'   # DM  Theoretical  (blue)
C_EDF_TH  = '#F44336'   # EDF Theoretical  (red)
C_DM_SIM  = '#9C27B0'   # DM  Simulated    (purple)
C_EDF_SIM = '#FF9800'   # EDF Simulated    (orange)
C_DEAD    = '#607D8B'   # Deadline marker  (slate-grey)

sns.set_theme(style='darkgrid', palette='muted', font_scale=1.0)
plt.rcParams.update({
    'figure.dpi'       : 150,
    'savefig.dpi'      : 150,
    'savefig.bbox'     : 'tight',
    'figure.facecolor' : '#1E1E2E',
    'axes.facecolor'   : '#2A2A3E',
    'axes.edgecolor'   : '#555577',
    'axes.labelcolor'  : '#CCCCDD',
    'axes.titlecolor'  : '#EEEEFF',
    'xtick.color'      : '#AAAACC',
    'ytick.color'      : '#AAAACC',
    'text.color'       : '#CCCCDD',
    'grid.color'       : '#3A3A5A',
    'grid.linewidth'   : 0.6,
    'legend.facecolor' : '#2A2A3E',
    'legend.edgecolor' : '#555577',
    'legend.labelcolor': '#CCCCDD',
    'font.size'        : 9,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig, filename: str) -> None:
    path = os.path.join(RESULTS_DIR, filename)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: WCRT Comparison  (small n — one subplot per scenario)
# ─────────────────────────────────────────────────────────────────────────────

def plot_wcrt_comparison(all_results: dict) -> None:
    """
    Grouped bar chart: Theoretical WCRT vs Simulated max, one subplot per
    scenario.  Deadline D is drawn as a dashed horizontal line per task.
    Infinite/divergent WCRTs are capped at 5 × max(D) for readability.
    """
    names = list(all_results.keys())
    n     = len(names)
    cols  = min(n, 3)
    rows  = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)
    fig.suptitle('Theoretical vs Simulated WCRT — DM & EDF', fontsize=13, y=1.02)

    bar_w = 0.18
    labels_legend = ['DM Theory', 'EDF Theory', 'DM Sim', 'EDF Sim']
    colors_legend  = [C_DM_TH, C_EDF_TH, C_DM_SIM, C_EDF_SIM]

    for idx, sname in enumerate(names):
        ax     = axes[idx // cols][idx % cols]
        result = all_results[sname]
        table  = result['table']
        xlbls  = [f"t{r['task_idx']}" for r in table]
        x      = np.arange(len(xlbls))

        d_vals  = np.array([r['D'] for r in table], dtype=float)
        y_cap   = 5 * float(np.max(d_vals))

        dm_th   = np.clip([min(r['dm_wcrt'],  y_cap) for r in table], 0, y_cap)
        edf_th  = np.clip([r['edf_wcrt']               for r in table], 0, y_cap)
        dm_sim  = np.array([r['sim_dm_wcrt']             for r in table])
        edf_sim = np.array([r['sim_edf_wcrt']            for r in table])

        offsets = [-1.5, -0.5, 0.5, 1.5]
        for vals, off, col, lbl in zip([dm_th, edf_th, dm_sim, edf_sim],
                                        offsets, colors_legend, labels_legend):
            ax.bar(x + off * bar_w, vals, bar_w, color=col, alpha=0.85,
                   label=lbl, zorder=3)

        # Deadline dashed line per task
        for xi, d in enumerate(d_vals):
            ax.hlines(d, xi - 2 * bar_w, xi + 2 * bar_w,
                      colors=C_DEAD, linewidths=1.5, linestyles='--', zorder=4)

        ax.set_title(f"{sname}  (U={result['taskset'].utilization:.3f})", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, fontsize=8)
        ax.set_ylabel('WCRT (ticks)')
        ax.set_ylim(bottom=0)
        if idx == 0:
            ax.legend(fontsize=7, ncol=2, loc='upper left')

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.tight_layout()
    _save(fig, 'wcrt_comparison.png')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Deadline Misses
# ─────────────────────────────────────────────────────────────────────────────

def plot_deadline_misses(all_results: dict) -> None:
    """
    Grouped bar chart: total deadline misses per scenario.
    Shows worst-case (WCET) and stochastic simulation side by side.
    """
    names  = list(all_results.keys())
    x      = np.arange(len(names))
    bar_w  = 0.18

    dm_wc  = [all_results[s]['obs_dm_wc'].total_misses()  for s in names]
    edf_wc = [all_results[s]['obs_edf_wc'].total_misses() for s in names]
    dm_r   = [all_results[s]['obs_dm'].total_misses()      for s in names]
    edf_r  = [all_results[s]['obs_edf'].total_misses()     for s in names]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.75), 5))
    for vals, off, col, lbl in zip([dm_wc, edf_wc, dm_r, edf_r],
                                    [-1.5, -0.5, 0.5, 1.5],
                                    [C_DM_TH, C_EDF_TH, C_DM_SIM, C_EDF_SIM],
                                    ['DM (WCET)', 'EDF (WCET)', 'DM (rand)', 'EDF (rand)']):
        ax.bar(x + off * bar_w, vals, bar_w, color=col, alpha=0.88,
               label=lbl, zorder=3)
        for xi, v in enumerate(vals):
            if v > 0:
                ax.text(xi + off * bar_w, v + 0.2, str(v),
                        ha='center', va='bottom', fontsize=6, color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Total Deadline Misses')
    ax.set_title('Deadline Misses: Worst-Case (WCET) vs Stochastic Simulation')
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save(fig, 'deadline_misses.png')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Response Time Distributions  (violin + box)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rt_distributions(all_results: dict) -> None:
    """
    Per-scenario violin plot of simulated response times, DM and EDF split.
    Theoretical WCRT shown as a diamond marker.
    Deadline shown as a horizontal dashed line.
    """
    names = list(all_results.keys())
    n     = len(names)
    cols  = min(n, 3)
    rows  = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows), squeeze=False)
    fig.suptitle('Response Time Distributions — DM vs EDF (Simulation)', fontsize=13, y=1.02)

    for idx, sname in enumerate(names):
        ax      = axes[idx // cols][idx % cols]
        result  = all_results[sname]
        table   = result['table']
        obs_dm  = result['obs_dm']
        obs_edf = result['obs_edf']

        records = []
        for r in table:
            ti    = r['task_idx'] - 1
            label = f"t{r['task_idx']}"
            for v in obs_dm.rt_dist[ti]:
                records.append({'Task': label, 'RT': v, 'Alg': 'DM'})
            for v in obs_edf.rt_dist[ti]:
                records.append({'Task': label, 'RT': v, 'Alg': 'EDF'})

        if not records:
            ax.set_title(f"{sname} — no data")
            continue

        df         = pd.DataFrame(records)
        task_order = [f"t{r['task_idx']}" for r in table]

        try:
            sns.violinplot(
                data=df, x='Task', y='RT', hue='Alg',
                order=task_order, hue_order=['DM', 'EDF'],
                palette={'DM': C_DM_SIM, 'EDF': C_EDF_SIM},
                inner='quartile', split=True, linewidth=0.8,
                ax=ax, legend=(idx == 0),
            )
        except Exception:
            # Fallback to box plot if violin fails (e.g. single-value dist)
            sns.boxplot(
                data=df, x='Task', y='RT', hue='Alg',
                order=task_order,
                palette={'DM': C_DM_SIM, 'EDF': C_EDF_SIM},
                ax=ax,
            )

        for xi, r in enumerate(table):
            d_cap  = 3 * r['D']
            dm_th  = min(r['dm_wcrt'],  d_cap)
            edf_th = min(r['edf_wcrt'], d_cap)
            ax.scatter([xi - 0.15], [dm_th],  marker='D', s=45,
                       color=C_DM_TH,  zorder=5,
                       label='DM Theory'  if (idx == 0 and xi == 0) else '')
            ax.scatter([xi + 0.15], [edf_th], marker='D', s=45,
                       color=C_EDF_TH, zorder=5,
                       label='EDF Theory' if (idx == 0 and xi == 0) else '')
            ax.axhline(r['D'], color=C_DEAD, lw=0.9, ls='--', alpha=0.6)

        ax.set_title(f"{sname}  (U={result['taskset'].utilization:.3f})", fontsize=9)
        ax.set_xlabel('')
        ax.set_ylabel('Response Time (ticks)')
        ax.set_xticklabels(task_order, fontsize=8)
        if idx == 0:
            handles, lbls = ax.get_legend_handles_labels()
            ax.legend(handles[:6], lbls[:6], fontsize=7, ncol=2, loc='upper left')
        else:
            leg = ax.get_legend()
            if leg:
                leg.remove()

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.tight_layout()
    _save(fig, 'rt_distributions.png')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Pessimism Ratio
# ─────────────────────────────────────────────────────────────────────────────

def plot_pessimism(all_results: dict) -> None:
    """
    Grouped bar chart of pessimism = Theoretical WCRT / Max Simulated WCRT.
    > 1.0 → bound is conservative (safe).  < 1.0 → simulated exceeded theory.
    """
    names = list(all_results.keys())
    n     = len(names)
    cols  = min(n, 3)
    rows  = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle('Pessimism Ratio  =  Theory WCRT / Max Simulated WCRT', fontsize=12, y=1.02)

    bar_w = 0.35
    for idx, sname in enumerate(names):
        ax    = axes[idx // cols][idx % cols]
        table = all_results[sname]['table']
        xlbls = [f"t{r['task_idx']}" for r in table]
        x     = np.arange(len(xlbls))

        dm_p  = [r['dm_pessimism']  if r['dm_pessimism']  is not None else 0 for r in table]
        edf_p = [r['edf_pessimism'] if r['edf_pessimism'] is not None else 0 for r in table]

        ax.bar(x - bar_w / 2, dm_p,  bar_w, color=C_DM_TH,  alpha=0.85, label='DM',  zorder=3)
        ax.bar(x + bar_w / 2, edf_p, bar_w, color=C_EDF_TH, alpha=0.85, label='EDF', zorder=3)
        ax.axhline(1.0, color='white', lw=1.2, ls='--', label='Tight=1.0')

        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, fontsize=8)
        ax.set_title(sname, fontsize=9)
        ax.set_ylabel('Pessimism Ratio')
        ax.set_ylim(bottom=0)
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.tight_layout()
    _save(fig, 'pessimism.png')


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark aggregated plots (n > 6 scenarios)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_wcrt_aggregated(all_results: dict) -> None:
    """Mean WCRT per scenario vs utilization, with ±1 std shading."""
    rows_data = []
    for name, result in all_results.items():
        table = result['table']
        U     = result['taskset'].utilization
        dm_vals   = [min(r['dm_wcrt'],  r['T']) for r in table]
        edf_vals  = [r['edf_wcrt']               for r in table]
        dms_vals  = [r['sim_dm_wcrt']             for r in table]
        edfs_vals = [r['sim_edf_wcrt']            for r in table]
        rows_data.append({
            'U': U,
            'dm_m': np.mean(dm_vals),   'dm_s': np.std(dm_vals),
            'edf_m': np.mean(edf_vals), 'edf_s': np.std(edf_vals),
            'dms_m': np.mean(dms_vals), 'dms_s': np.std(dms_vals),
            'edfs_m': np.mean(edfs_vals),'edfs_s': np.std(edfs_vals),
        })
    df = pd.DataFrame(rows_data).sort_values('U')

    fig, ax = plt.subplots(figsize=(11, 5))
    for col_m, col_s, color, label in [
        ('dm_m',  'dm_s',  C_DM_TH,   'DM Theory'),
        ('edf_m', 'edf_s', C_EDF_TH,  'EDF Theory'),
        ('dms_m', 'dms_s', C_DM_SIM,  'DM Sim'),
        ('edfs_m','edfs_s',C_EDF_SIM, 'EDF Sim'),
    ]:
        ax.plot(df['U'], df[col_m], 'o-', color=color, lw=2, ms=4, label=label)
        ax.fill_between(df['U'], df[col_m] - df[col_s], df[col_m] + df[col_s],
                         color=color, alpha=0.12)

    ax.set_xlabel('Utilization (U)')
    ax.set_ylabel('Mean WCRT (ticks)')
    ax.set_title('Mean WCRT vs Utilization — Theory vs Simulation (±1 std)')
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save(fig, 'wcrt_comparison.png')


def _plot_deadline_misses_grouped(all_results: dict) -> None:
    """Mean ± std deadline misses grouped by utilization band."""
    records = []
    for result in all_results.values():
        u = result['taskset'].utilization
        records.append({
            'U_band': f"{round(u, 1):.1f}",
            'U':      u,
            'DM_wc':  result['obs_dm_wc'].total_misses(),
            'EDF_wc': result['obs_edf_wc'].total_misses(),
            'DM':     result['obs_dm'].total_misses(),
            'EDF':    result['obs_edf'].total_misses(),
        })
    df  = pd.DataFrame(records)
    grp = df.groupby('U_band').agg(
        DM_wc_m=('DM_wc', 'mean'), DM_wc_s=('DM_wc', 'std'),
        EDF_wc_m=('EDF_wc','mean'), EDF_wc_s=('EDF_wc','std'),
        DM_m=('DM','mean'),         DM_s=('DM','std'),
        EDF_m=('EDF','mean'),       EDF_s=('EDF','std'),
        U_min=('U','min'),
    ).reset_index().sort_values('U_min')

    x     = np.arange(len(grp))
    bar_w = 0.18
    fig, ax = plt.subplots(figsize=(max(9, len(grp) * 0.9), 5))

    for m_col, s_col, off, col, lbl in [
        ('DM_wc_m',  'DM_wc_s',  -1.5, C_DM_TH,   'DM (WCET)'),
        ('EDF_wc_m', 'EDF_wc_s', -0.5, C_EDF_TH,  'EDF (WCET)'),
        ('DM_m',     'DM_s',      0.5, C_DM_SIM,   'DM (rand)'),
        ('EDF_m',    'EDF_s',     1.5, C_EDF_SIM,  'EDF (rand)'),
    ]:
        std = grp[s_col].fillna(0)
        ax.bar(x + off * bar_w, grp[m_col], bar_w,
               yerr=std, capsize=3, color=col, alpha=0.88,
               label=lbl, zorder=3,
               error_kw=dict(ecolor='white', lw=1))

    ax.set_xticks(x)
    ax.set_xticklabels([f"U≈{b}" for b in grp['U_band']], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Deadline Misses (± std across files)')
    ax.set_title('Deadline Misses by Utilization Band — WCET vs Stochastic')
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save(fig, 'deadline_misses.png')


def _plot_rt_by_utilization(all_results: dict) -> None:
    """Seaborn boxplot of EDF response times grouped by utilization band."""
    records = []
    for result in all_results.values():
        u    = result['taskset'].utilization
        band = f"{round(u, 1):.1f}"
        for i in range(result['taskset'].n):
            for val in result['obs_edf'].rt_dist[i]:
                records.append({'U_band': band, 'RT': val, 'U': u})

    if not records:
        print("  [SKIP] No RT distribution data")
        return

    df         = pd.DataFrame(records)
    band_order = sorted(df['U_band'].unique(), key=float)

    fig, ax = plt.subplots(figsize=(max(10, len(band_order) * 0.9), 5))
    sns.boxplot(data=df, x='U_band', y='RT', order=band_order,
                palette='Blues', linewidth=0.8,
                flierprops=dict(marker='.', alpha=0.2, markersize=2),
                ax=ax)
    ax.set_xlabel('Utilization Band')
    ax.set_ylabel('EDF Response Time (ticks)')
    ax.set_title('EDF Response Time Distribution by Utilization')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    _save(fig, 'rt_distributions.png')


def _plot_pessimism_aggregated(all_results: dict) -> None:
    """Scatter: mean pessimism ratio vs utilization with smoothed trend lines."""
    from scipy.ndimage import uniform_filter1d

    rows_data = []
    for name, result in all_results.items():
        table = result['table']
        U     = result['taskset'].utilization
        dm_p  = [r['dm_pessimism']  for r in table if r['dm_pessimism']  is not None]
        edf_p = [r['edf_pessimism'] for r in table if r['edf_pessimism'] is not None]
        rows_data.append({
            'U':       U,
            'DM_pess': np.mean(dm_p)  if dm_p  else np.nan,
            'EDF_pess':np.mean(edf_p) if edf_p else np.nan,
        })

    df = pd.DataFrame(rows_data).sort_values('U')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df['U'], df['DM_pess'],  color=C_DM_TH,  s=30, alpha=0.8, label='DM',  zorder=4)
    ax.scatter(df['U'], df['EDF_pess'], color=C_EDF_TH, s=30, alpha=0.8, label='EDF', zorder=4)

    k = max(3, len(df) // 6)
    for col, color in [('DM_pess', C_DM_TH), ('EDF_pess', C_EDF_TH)]:
        valid = df.dropna(subset=[col])
        if len(valid) > 3:
            smooth = uniform_filter1d(valid[col].values, size=k)
            ax.plot(valid['U'], smooth, color=color, lw=1.8, ls='--', alpha=0.8)

    ax.axhline(1.0, color='white', lw=1.2, ls=':', label='Tight = 1.0')
    ax.set_xlabel('Utilization (U)')
    ax.set_ylabel('Mean Pessimism Ratio')
    ax.set_title('Analytical Pessimism vs Utilization  (Theory / Simulated WCRT)')
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save(fig, 'pessimism.png')


def _plot_miss_rate_vs_utilization(all_results: dict) -> None:
    """Line + scatter: deadline misses vs utilization, WCET and stochastic panels."""
    from scipy.ndimage import uniform_filter1d

    rows = sorted([
        {
            'U':      r['taskset'].utilization,
            'DM_wc':  r['obs_dm_wc'].total_misses(),
            'EDF_wc': r['obs_edf_wc'].total_misses(),
            'DM':     r['obs_dm'].total_misses(),
            'EDF':    r['obs_edf'].total_misses(),
        }
        for r in all_results.values()
    ], key=lambda x: x['U'])
    df = pd.DataFrame(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle('Deadline Misses vs Utilization', fontsize=12)

    k = max(3, len(df) // 6)
    for ax, dm_col, edf_col, dm_c, edf_c, title in [
        (ax1, 'DM_wc', 'EDF_wc', C_DM_TH,  C_EDF_TH,  'Worst-Case (WCET execution)'),
        (ax2, 'DM',    'EDF',    C_DM_SIM, C_EDF_SIM, 'Stochastic (uniform [BCET,WCET])'),
    ]:
        ax.scatter(df['U'], df[dm_col],  color=dm_c,  s=18, alpha=0.6, zorder=4)
        ax.scatter(df['U'], df[edf_col], color=edf_c, s=18, alpha=0.6, zorder=4)
        dm_smooth  = uniform_filter1d(df[dm_col].values,  size=k)
        edf_smooth = uniform_filter1d(df[edf_col].values, size=k)
        ax.plot(df['U'], dm_smooth,  color=dm_c,  lw=2, label='DM')
        ax.plot(df['U'], edf_smooth, color=edf_c, lw=2, label='EDF')
        ax.axvline(1.0, color='white', ls='--', lw=1, label='U=1.0 (EDF limit)')
        ax.set_xlabel('Utilization (U)')
        ax.set_ylabel('Total Deadline Misses')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    _save(fig, 'miss_rate_vs_util.png')


def _plot_schedulability_heatmap(all_results: dict) -> None:
    """Seaborn heatmap: fraction of schedulable tasks per scenario for DM and EDF."""
    names    = list(all_results.keys())
    dm_rate  = [
        sum(1 for r in all_results[n]['table'] if r['dm_schedulable'])
        / max(1, len(all_results[n]['table']))
        for n in names
    ]
    edf_rate = [
        sum(1 for r in all_results[n]['table'] if r['edf_schedulable'])
        / max(1, len(all_results[n]['table']))
        for n in names
    ]

    df_heat = pd.DataFrame({'DM': dm_rate, 'EDF': edf_rate}, index=names).T
    figw    = max(10, len(names) * 0.35)
    fig, ax = plt.subplots(figsize=(figw, 2.5))
    sns.heatmap(df_heat, ax=ax,
                vmin=0, vmax=1, cmap='RdYlGn',
                annot=True, fmt='.0%', annot_kws={'size': 6},
                linewidths=0.3, linecolor='#333355',
                cbar_kws={'label': 'Schedulable %'})
    ax.set_title('Schedulability Rate Heatmap: DM vs EDF')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=9)
    fig.tight_layout()
    _save(fig, 'schedulability_heatmap.png')


def _plot_wcrt_diff_heatmap(all_results: dict) -> None:
    """
    Heatmap of (DM_WCRT − EDF_WCRT) per task per scenario.
    Red = DM worse  |  Blue = EDF worse  |  White = equal.
    """
    records = []
    for name, result in all_results.items():
        u = result['taskset'].utilization
        for r in result['table']:
            dm_w = min(r['dm_wcrt'], r['T'] * 5)
            records.append({
                'Scenario': name,
                'Task':     f"t{r['task_idx']}",
                'U':        u,
                'Diff':     dm_w - r['edf_wcrt'],
            })

    if not records:
        return

    df    = pd.DataFrame(records)
    pivot = df.pivot_table(index='Scenario', columns='Task',
                           values='Diff', aggfunc='mean')

    u_map = {name: r['taskset'].utilization for name, r in all_results.items()}
    pivot = pivot.loc[sorted(pivot.index, key=lambda n: u_map.get(n, 0))]

    figw = max(8,  len(pivot.columns) * 0.5)
    figh = max(5,  len(pivot.index)   * 0.25)
    fig, ax = plt.subplots(figsize=(figw, figh))

    abs_max = max(pivot.abs().max().max(), 1)
    sns.heatmap(pivot, ax=ax,
                center=0, vmin=-abs_max, vmax=abs_max, cmap='coolwarm',
                linewidths=0.2, linecolor='#333355',
                cbar_kws={'label': 'DM WCRT − EDF WCRT  (+ = DM worse)'},
                xticklabels=True, yticklabels=True)
    ax.set_title('WCRT Difference Heatmap  (DM − EDF, per task per scenario)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=6)
    fig.tight_layout()
    _save(fig, 'wcrt_diff_heatmap.png')


# ─────────────────────────────────────────────────────────────────────────────
# Main dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(all_results: dict) -> None:
    n = len(all_results)
    print(f"\nGenerating plots ({n} scenarios)...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if n <= 6:
        plot_wcrt_comparison(all_results)
        plot_deadline_misses(all_results)
        plot_rt_distributions(all_results)
        plot_pessimism(all_results)
    else:
        _plot_wcrt_aggregated(all_results)
        _plot_deadline_misses_grouped(all_results)
        _plot_rt_by_utilization(all_results)
        _plot_pessimism_aggregated(all_results)
        _plot_miss_rate_vs_utilization(all_results)
        _plot_schedulability_heatmap(all_results)
        _plot_wcrt_diff_heatmap(all_results)

    print("All plots saved to results/\n")


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark comparison: Automotive vs UUniFast
# ─────────────────────────────────────────────────────────────────────────────

def plot_benchmark_comparison(auto_results: dict, uni_results: dict) -> None:
    """
    2×2 grid comparing automotive vs UUniFast benchmarks:
    deadline misses (WCET), deadline misses (stochastic), mean WCRT, pessimism.
    """
    def _extract(results):
        rows = []
        for _, r in results.items():
            table = r['table']
            U     = r['taskset'].utilization
            dm_p  = [t['dm_pessimism']  for t in table if t['dm_pessimism']  is not None]
            edf_p = [t['edf_pessimism'] for t in table if t['edf_pessimism'] is not None]
            rows.append({
                'U':           U,
                'dm_miss_wc':  r['obs_dm_wc'].total_misses(),
                'edf_miss_wc': r['obs_edf_wc'].total_misses(),
                'dm_miss':     r['obs_dm'].total_misses(),
                'edf_miss':    r['obs_edf'].total_misses(),
                'dm_wcrt':     np.mean([min(t['dm_wcrt'], t['T']) for t in table]),
                'edf_wcrt':    np.mean([t['edf_wcrt']              for t in table]),
                'dm_pess':     np.mean(dm_p)  if dm_p  else np.nan,
                'edf_pess':    np.mean(edf_p) if edf_p else np.nan,
            })
        return pd.DataFrame(rows).sort_values('U')

    df_a = _extract(auto_results)
    df_u = _extract(uni_results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Benchmark Comparison: Automotive vs UUniFast', fontsize=14)

    panels = [
        (axes[0, 0], 'dm_miss_wc',  'edf_miss_wc',  'Deadline Misses (WCET)'),
        (axes[0, 1], 'dm_miss',      'edf_miss',      'Deadline Misses (Stochastic)'),
        (axes[1, 0], 'dm_wcrt',      'edf_wcrt',      'Mean WCRT (ticks)'),
        (axes[1, 1], 'dm_pess',      'edf_pess',      'Mean Pessimism Ratio'),
    ]

    for ax, dm_col, edf_col, title in panels:
        for df, bname, ls in [(df_a, 'Auto', '-'), (df_u, 'UUni', '--')]:
            ax.plot(df['U'], df[dm_col],  color=C_DM_TH,  ls=ls, lw=2,
                    marker='o', ms=4, label=f'DM {bname}')
            ax.plot(df['U'], df[edf_col], color=C_EDF_TH, ls=ls, lw=2,
                    marker='s', ms=4, label=f'EDF {bname}')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Utilization (U)')
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(bottom=0)

    axes[1, 1].axhline(1.0, color='white', ls=':', lw=1.2, label='Tight = 1.0')

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'benchmark_comparison.png')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
