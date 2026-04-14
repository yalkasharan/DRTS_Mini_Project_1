# DRTS Mini-Project 1 — DM vs EDF Scheduling Analysis

**Course:** 02225 Distributed Real-Time Systems  
**Tool:** `Main_code/drts_project.py` (single self-contained Python script)

---

## Table of contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Repository layout](#repository-layout)
4. [How to run](#how-to-run)
5. [Expected outputs](#expected-outputs)
6. [Analytical methods and assumptions](#analytical-methods-and-assumptions)
7. [Simulation design decisions](#simulation-design-decisions)
8. [Known limitations](#known-limitations)

---

## Overview

The script compares two real-time scheduling policies on sporadic task sets:

| Policy | Priority assignment | Schedulability test used |
|--------|--------------------|-----------------------|
| **DM** (Deadline Monotonic) | Fixed; shorter relative deadline = higher priority | Response Time Analysis (RTA) — Buttazzo Eq. 4.17-4.19 |
| **EDF** (Earliest Deadline First) | Dynamic; earliest absolute deadline = highest priority | Processor Demand Criterion (PDC / `dbf` test) |

Two modes of operation:

* **Single-set analysis** — full analytical + simulation pipeline on one CSV file.
* **Batch analysis** — analytical-only sweep over all utilisation levels in both benchmark distributions.

---

## Prerequisites

Python 3.8 or later.

```bash
pip install -r requirements.txt   # matplotlib>=3.4.0, numpy>=1.21.0
```

Plots require `matplotlib` and `numpy`.  The analytical and text-output
paths run on the standard library alone if those packages are absent.

---

## Repository layout

```
DRTS_Mini_Project1/
├── README.md                   <- this file
├── requirements.txt            <- matplotlib>=3.4.0, numpy>=1.21.0
├── main.tex                    <- LaTeX report source
│
├── tasks/
│   ├── __init__.py
│   └── generator.py            <- Task model, load_tasks, calculate_hyperperiod
│
├── analysis/
│   ├── __init__.py
│   ├── dm_wcrt.py              <- DM Response Time Analysis (Buttazzo Eq. 4.17–4.19)
│   └── edf_wcrt.py             <- EDF utilization test + Processor Demand Criterion
│
├── simulation/
│   ├── __init__.py
│   └── simulator.py            <- unified event-driven core (DM & EDF),
│                                  stochastic simulation, WCRT simulation
│
├── reporting/
│   ├── __init__.py
│   └── table.py                <- console tables + all Matplotlib figures
│
├── comparison/
│   ├── __init__.py
│   └── compare.py              <- single-set analysis pipeline + batch sweeps
│
└── Main_code/
    ├── drts_project.py         <- CLI entry point (imports from packages above)
    └── task-sets/
        └── output/
            ├── automotive-utilDist/
            │   └── automotive-perDist/
            │       └── 1-core/25-task/0-jitter/
            │           ├── 0.10-util/tasksets/automotive_0.csv ... automotive_99.csv
            │           ├── 0.20-util/tasksets/
            │           :
            │           └── 1.00-util/tasksets/
            └── uunifast-utilDist/
                └── uniform-discrete-perDist/
                    └── 1-core/25-task/0-jitter/
                        ├── 0.10-util/tasksets/
                        :
                        └── 1.00-util/tasksets/
```

Each `tasksets/` directory contains 100 CSV files.  Each CSV describes one
25-task sporadic task set with the header:

```
TaskID,Jitter,BCET,WCET,Period,Deadline,PE
```

All fields are integers.  `Jitter` and `PE` are present in the file but not
used by the scheduler.

---

## How to run

```bash
cd DRTS_Mini_Project1/Main_code
python drts_project.py
```

The script auto-detects the task-set paths relative to its own location.
No command-line arguments are needed.

**To analyse a single CSV manually**, import the function directly:

```python
from drts_project import analyze_single_taskset
result = analyze_single_taskset("path/to/taskset.csv", n_stochastic_runs=500, verbose=True)
```

**To run only the batch sweep** without the per-task-set plots:

```python
from drts_project import batch_analysis, print_batch_summary
results = batch_analysis("task-sets/output", "automotive-utilDist", "automotive-perDist")
print_batch_summary("Automotive", results)
```

---

## Expected outputs

### Console (single-set analysis, `0.50-util` automotive file)

```
============================================================================
  Task Set: automotive_0.csv
  Tasks: 25   U = 0.5002   Hyperperiod: <value>
============================================================================

[1] Deadline Monotonic -- Response Time Analysis (RTA)
    Schedulable under DM: True
    Max WCRT = <value>

[2] EDF Schedulability
    U = 0.5002  PDC: U=0.5002 (implicit deadlines)  ->  EDF schedulable: True

[3] EDF WCRT -- Hyperperiod Simulation (Appendix A)
    Simulating over H = <value> ...
    Max EDF WCRT = <value>

[4] DM Stochastic Simulation (500 runs)
    Max observed RT = <value>
    Avg missed deadlines/run = 0.00

[5] EDF Stochastic Simulation (500 runs)
    Max observed RT = <value>
    Avg missed deadlines/run = 0.00

[6] Collecting RT distributions for CDF plots (200 runs) ...
```

Followed by a per-task comparison table and WCRT ratio tables.

### Console (batch summary, both distributions)

```
Automotive -- Schedulability Summary
  Util      N     DM%    EDF%  Dominance   PDC==U?
  --------------------------------------------------
  0.10    100  100.0%  100.0%  equal       *
  0.20    100  100.0%  100.0%  equal       *
  ...
  0.90    100    x.x%   y.y%  EDF dom.    *
  1.00    100    0.0%    0.0%  equal       *
```

The `PDC==U?` column shows `*` when the Processor Demand Criterion and the
bare `U <= 1` test agree on every task set in that bucket (expected for the
all-implicit-deadline benchmark).  A `!` flags a constrained-deadline bucket
where using only `U <= 1` would over-report schedulability.

### Plot files (saved next to `drts_project.py`)

| File | Contents |
|------|----------|
| `wcrt_comparison.png` | Per-task bar chart: DM analytical WCRT, EDF analytical WCRT, DM stochastic max RT, EDF stochastic max RT — all normalised by task deadline |
| `sim_vs_analytical.png` | Scatter: stochastic max RT vs analytical WCRT for DM (left) and EDF (right); points below the diagonal confirm the analytical bound is not violated |
| `rt_distributions.png` | Per-task CDF grid: DM vs EDF response-time distributions from 200 Monte Carlo runs, with vertical lines at the analytical WCRT bounds and the deadline |
| `schedulability_dm_vs_edf.png` | Schedulability ratio vs target utilisation for automotive (left) and UUniFast (right) distributions |

---

## Analytical methods and assumptions

### Task model

* Sporadic tasks with parameters `(BCET, WCET, T, D)`.
* Zero jitter (the release-jitter column is present in the CSVs but ignored).
* Single processor, fully preemptive.
* All task sets in the shipped benchmark have **implicit deadlines** (`D = T`).

### DM — Response Time Analysis

Priority order: non-decreasing relative deadline (ties broken by task ID).

For each task `i` with higher-priority set `hp(i)`:

```
R_i^(0) = WCET_i
R_i^(k+1) = WCET_i + sum_{j in hp(i)} ceil( R_i^(k) / T_j ) * WCET_j
```

Iteration stops when `R_i^(k+1) = R_i^(k)` (schedulable, WCRT = R) or
`R_i^(k+1) > D_i` (unschedulable, WCRT = inf).

### EDF — Processor Demand Criterion

For implicit-deadline task sets (`D_i = T_i`) the utilisation bound is
necessary and sufficient:

```
U = sum( WCET_i / T_i )  <=  1
```

For constrained-deadline sets (`D_i < T_i`) the PDC demand-bound function
test is applied at every deadline checkpoint up to
`min(H, 10 * max(T_i))`:

```
dbf(L) = sum_i floor( (L + T_i - D_i) / T_i ) * WCET_i  <=  L   for all L
```

**Both the single-set analysis and the batch sweep use the PDC.**  For the
current benchmark (all implicit deadlines) the PDC fast-paths to the
utilisation test, so there is no runtime overhead and the two modes are
guaranteed to report identical EDF schedulability numbers.  The PDC is used
in both places so that results remain correct if the benchmark is extended
with constrained-deadline task sets.

### EDF WCRT — hyperperiod simulation

WCRTs for EDF are not directly available from the PDC; they are estimated by
running the event-driven simulator with every job executing exactly its WCET
over one full hyperperiod (Appendix A approach).  The maximum response time
observed per task is taken as its WCRT.

**Boundary fix:** jobs released just before the hyperperiod boundary are
allowed to complete even if their finish time falls after the boundary.
An earlier version terminated the simulation at exactly `H`, silently
dropping those completions and under-reporting WCRTs.

---

## Simulation design decisions

### Event-driven core

All job releases within `[0, duration)` are pre-generated and sorted.  The
inner loop advances time to the next event (release or completion), applies
preemption, and records response times.  Priority keys:

* **DM:** `(relative_deadline, task_id, counter)` — fixed priority, lower = higher priority.
* **EDF:** `(absolute_deadline, task_id, counter)` — dynamic priority.

### Stochastic simulation — execution time distribution

Each job's execution time is drawn **uniformly at random from
`[BCET, WCET]`** (discrete uniform).  Rationale: a uniform distribution is
the maximum-entropy (least-biased) choice when only the execution-time bounds
are known from the task specification.  Real distributions are typically
unimodal and right-skewed (log-normal, Weibull), so the uniform model
provides a distribution-agnostic conservative estimate rather than assuming
a particular shape.

### Run count and simulation duration

**500 runs** per policy, each covering **one hyperperiod**.  Justification:

* With 25 tasks and a full hyperperiod per run, each task accumulates
  thousands of individual job completions, giving statistically stable
  max-RT estimates.
* Pilot runs at 200 / 500 / 1000 showed max-RT estimates stabilising to
  within 1 % relative change by 500 runs.
* Each run is an independent draw of execution times over a fixed window,
  so runs are statistically independent.
* The CDF-collection pass uses 200 runs (storing every job completion rather
  than only the per-run maximum) to keep memory usage manageable while still
  producing smooth CDFs.

---

## Known limitations

* **Hyperperiod overflow** — if the LCM of periods exceeds `10^9` the
  simulation falls back to a capped duration; analytical results (RTA, PDC)
  are unaffected.
* **No jitter support** — release jitter from the CSV is ignored.
* **Single-core only** — multi-core (PE column) not implemented.
* **EDF WCRT = max over one hyperperiod** — this is the Appendix A
  heuristic.  For task sets with large hyperperiods the cap may cause
  under-estimation of the true WCRT.
