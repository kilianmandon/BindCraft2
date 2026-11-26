"""Diff two trees' paired campaigns: per metric, the spread across trajectories in each
tree and the paired difference between them.

    OUT=/scratch/paired dev/parity/paired_stats.py short_hPDL1 long_hPDL1

Reads the per-trajectory losses.csv that paired_campaigns.sh left behind (full precision,
because it runs the CLI through launch_raw.py) and takes each metric's last recorded value
as that trajectory's outcome. Trajectory n of one tree is paired with trajectory n of the
other. |Δ|/sd puts the difference next to the trajectory-to-trajectory spread, which is
the number that says whether a change stays inside the distribution; the p-values are
two-sided on the 24 (or however many) paired differences."""
import csv, glob, os, sys, math
import numpy as np
from scipy import stats
SP = os.environ.get('OUT') or os.path.dirname(os.path.abspath(__file__))

def traj_metrics(run_dir):
    """{trajectory_index: {metric: final value}} from the per-trajectory losses.csv files."""
    out = {}
    index = os.path.join(run_dir, '1_Trajectories', '!_Trajectories.csv')
    if not os.path.exists(index): return out
    order = {}
    for row in csv.DictReader(open(index)):
        order[row['design']] = int(row['trajectory'])
    for path in sorted(glob.glob(os.path.join(run_dir, '1_Trajectories', '*', '*_losses.csv'))):
        name = os.path.basename(os.path.dirname(path))
        rows = list(csv.DictReader(open(path)))
        if not rows: continue
        last = {}
        for row in rows:
            for k, v in row.items():
                if k in ('phase', 'round') or v in ('', None): continue
                last[k] = float(v)
        last['steps'] = len(rows)
        out[order.get(name, -1)] = last
    return out

def design_seconds(run_dir):
    index = os.path.join(run_dir, '1_Trajectories', '!_Trajectories.csv')
    out = {}
    if not os.path.exists(index): return out
    for row in csv.DictReader(open(index)):
        timing = dict(part.split('=') for part in row['Timing'].split(';') if '=' in part)
        out[int(row['trajectory'])] = (float(timing['design']), int(timing.get('compiled', 0)))
    return out

def timing(exp):
    a, b = design_seconds(f'{SP}/runs/{exp}_head'), design_seconds(f'{SP}/runs/{exp}_new')
    keys = sorted(k for k in set(a) & set(b) if not a[k][1] and not b[k][1])
    if not keys: return
    x = np.array([a[k][0] for k in keys]); y = np.array([b[k][0] for k in keys])
    print(f'{"design seconds (warm)":26s} {x.mean():8.3f}±{x.std(ddof=1):<8.3f} {y.mean():8.3f}±{y.std(ddof=1):<8.3f} '
          f'{y.mean()-x.mean():+8.3f} {"":18s} {stats.ttest_rel(y,x).pvalue:6.3f}   ratio {y.mean()/x.mean():.3f}')
    print(f'   trajectories that compiled: head {sorted(k for k in a if a[k][1])}, new {sorted(k for k in b if b[k][1])}')

def paired(exp, metrics=None):
    a, b = traj_metrics(f'{SP}/runs/{exp}_head'), traj_metrics(f'{SP}/runs/{exp}_new')
    keys = sorted(set(a) & set(b))
    if not keys: return None
    names = metrics or sorted(set(a[keys[0]]) & set(b[keys[0]]))
    identical = [k for k in keys if all(a[k][m] == b[k][m] for m in names)]
    print(f'\n### {exp}: {len(keys)} paired trajectories '
          f'({a[keys[0]]["steps"]} rounds each), '
          f'{len(identical)}/{len(keys)} bit-identical on every metric')
    print(f'{"metric":26s} {"head mean±sd":>19s} {"new mean±sd":>19s} {"Δ (95% CI)":>26s} {"|Δ|/sd":>7s} {"t p":>6s} {"W p":>6s}')
    for m in names:
        if m == 'steps': continue
        x = np.array([a[k][m] for k in keys]); y = np.array([b[k][m] for k in keys])
        d = y - x
        tp = stats.ttest_rel(y, x).pvalue if np.any(d) else 1.0
        try: wp = stats.wilcoxon(d).pvalue if np.any(d) else 1.0
        except Exception: wp = float('nan')
        sem = d.std(ddof=1)/math.sqrt(len(d)) if len(d) > 1 else 0.0
        half = stats.t.ppf(0.975, len(d)-1) * sem if len(d) > 1 else 0.0
        pooled = np.r_[x, y].std(ddof=1)
        print(f'{m:26s} {x.mean():8.4f}±{x.std(ddof=1):<8.4f} {y.mean():8.4f}±{y.std(ddof=1):<8.4f} '
              f'{d.mean():+8.4f} [{d.mean()-half:+7.4f},{d.mean()+half:+7.4f}] '
              f'{abs(d.mean())/pooled if pooled else 0:7.2f} {tp:6.3f} {wp:6.3f}')
    # headline: pLDDT = 1 - plddt_loss
    if 'plddt_loss' in ' '.join(names):
        pl = [m for m in names if m.endswith('plddt_loss')][0]
        x = 1 - np.array([a[k][pl] for k in keys]); y = 1 - np.array([b[k][pl] for k in keys])
        print(f'{"pLDDT (1-plddt_loss)":26s} {x.mean():9.4f}±{x.std(ddof=1):<9.4f} {y.mean():9.4f}±{y.std(ddof=1):<9.4f} '
              f'{(y-x).mean():+8.4f}')
    print(f'\n  distribution over the {len(keys)} trajectories (new tree), '
          f'which is what any quality claim has to be measured against:')
    print(f'  {"metric":26s} {"mean":>9s} {"sd":>9s} {"min":>9s} {"median":>9s} {"max":>9s}')
    for m in names:
        if m == 'steps': continue
        y = np.array([b[k][m] for k in keys])
        print(f'  {m:26s} {y.mean():9.4f} {y.std(ddof=1):9.4f} {y.min():9.4f} '
              f'{np.median(y):9.4f} {y.max():9.4f}')
    print()
    timing(exp)
    return keys

for exp in sys.argv[1:] or sorted({os.path.basename(p).removesuffix('_head').removesuffix('_new')
                                   for p in glob.glob(f'{SP}/runs/*_*')}):
    if paired(exp) is None:
        print(f'\n### {exp}: no paired trajectories (one tree has not run)')
