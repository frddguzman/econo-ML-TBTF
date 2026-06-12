"""Merge the 5-seed (26462-26466) and 10-seed (26467-26476) Phase 0.5 sweeps,
write a 15-seed CSV, and produce summary tables under three statistical lenses:
  - Mean +/- std (the standard reporting)
  - Median + IQR (Q3-Q1) (robust for bimodal/heavy-tailed cells)
  - Champion-fraction (% of seeds with max_ten > 100; directly interpretable)

Plus the same decision-criteria scan as the original Phase 0.5 script, recomputed
on 15 seeds, so we can see whether the bimodality replicated or resolved.
"""
import os
import csv
import math
from collections import defaultdict

PROJ = os.path.dirname(os.path.abspath(__file__))
P5    = os.path.join(PROJ, 'sweep_w58_cv_phase05_5seed.csv')
PEXTRA= os.path.join(PROJ, 'sweep_w58_cv_phase05_extra_raw.csv')
P15   = os.path.join(PROJ, 'sweep_w58_cv_phase05_15seed.csv')

CVS  = [0.5, 0.7, 0.85, 1.0]
ETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CHAMPION_THRESHOLD = 100   # max_ten >= this counts as champion-hub seed


def load_rows(path):
    with open(path) as f:
        rd = csv.DictReader(f)
        return list(rd)


def percentile(xs_sorted, p):
    """Linear-interpolation percentile (numpy default)."""
    n = len(xs_sorted)
    if n == 0: return float('nan')
    if n == 1: return float(xs_sorted[0])
    pos = p / 100.0 * (n - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 >= n: return float(xs_sorted[lo])
    return xs_sorted[lo] * (1 - frac) + xs_sorted[lo + 1] * frac


def stats(xs):
    """Return (mean, std, q1, median, q3, iqr) for a list of numbers."""
    xs = [float(x) for x in xs]
    n = len(xs)
    if n == 0:
        return (float('nan'),) * 6
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / max(1, n - 1) if n >= 2 else 0
    s = math.sqrt(var)
    xs_s = sorted(xs)
    q1 = percentile(xs_s, 25)
    md = percentile(xs_s, 50)
    q3 = percentile(xs_s, 75)
    return m, s, q1, md, q3, q3 - q1


def main():
    rows5  = load_rows(P5)
    rows10 = load_rows(PEXTRA)
    print(f'Loaded {len(rows5)} rows from 5-seed CSV, {len(rows10)} from extra-10 CSV')

    # Sanity: schemas match
    if rows5 and rows10:
        s5 = set(rows5[0].keys()); s10 = set(rows10[0].keys())
        if s5 != s10:
            print(f'  WARNING: schema mismatch. 5-only: {s5-s10}. 10-only: {s10-s5}')

    # Merge
    keys = list(rows5[0].keys()) if rows5 else list(rows10[0].keys())
    all_rows = rows5 + rows10
    with open(P15, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, '') for k in keys})
    print(f'Wrote {len(all_rows)} rows -> {P15}')
    print()

    # Group by (cv, eta)
    by_cell = defaultdict(list)
    for r in all_rows:
        try:
            cv = float(r['cv']); eta = float(r['eta'])
        except (KeyError, ValueError):
            continue
        by_cell[(round(cv, 3), round(eta, 3))].append(r)

    # ─────────────────────────────────────────────────────────────────────
    # Table 1: Total bk by (cv, eta) — mean and median, claim 3 verdict
    # ─────────────────────────────────────────────────────────────────────
    print('=== Table 1: Total bk by cv x eta (15-seed mean / median) ===')
    print()
    for stat_label in ('mean', 'median'):
        print(f'-- {stat_label} of total_bk --')
        print(f'{"cv":>5} | ' + ' | '.join(f'{e:>6.1f}' for e in ETAS) + ' | min eta | min val | bk@eta=0 | claim 3')
        print('-' * 145)
        for cv in CVS:
            vals = []
            for eta in ETAS:
                cells = by_cell.get((cv, eta), [])
                bks = [int(r['total_bk']) for r in cells]
                m, s, q1, md, q3, iqr = stats(bks)
                v = m if stat_label == 'mean' else md
                vals.append(v)
            bk0 = vals[0]
            min_idx = vals.index(min(vals))
            min_eta = ETAS[min_idx]
            min_val = vals[min_idx]
            delta = min_val - bk0
            verdict = f'YES (eta*={min_eta:.1f}, d={delta:+.0f})' if delta < 0 else 'NO'
            print(f'{cv:>5.2f} | ' + ' | '.join(f'{v:>6.0f}' for v in vals) + f' | {min_eta:>6.1f} | {min_val:>7.0f} | {bk0:>8.0f} | {verdict}')
        print()

    # ─────────────────────────────────────────────────────────────────────
    # Table 2: max_ten — mean+/-std vs median+IQR, with champion-fraction
    # ─────────────────────────────────────────────────────────────────────
    print(f'=== Table 2: max_ten lens comparison per (cv, eta) — 15 seeds, champion threshold = {CHAMPION_THRESHOLD} ===')
    print()
    print(f'{"cv":>5} | {"eta":>4} | {"mean":>6} | {"std":>5} | {"std%":>6} | '
          f'{"median":>7} | {"Q1":>5} | {"Q3":>5} | {"IQR":>5} | {"champ%":>7} | {"interpretation":>25}')
    print('-' * 145)
    for cv in CVS:
        for eta in ETAS:
            cells = by_cell.get((cv, eta), [])
            if not cells: continue
            mts = [int(r['max_ten']) for r in cells]
            m, s, q1, md, q3, iqr = stats(mts)
            std_pct = 100 * s / m if m else 0
            n_champ = sum(1 for x in mts if x >= CHAMPION_THRESHOLD)
            champ_pct = 100 * n_champ / len(mts)
            # Heuristic interpretation
            if std_pct < 25:
                interp = 'tight unimodal'
            elif champ_pct == 0:
                interp = 'tight unimodal'
            elif champ_pct == 100:
                interp = 'all-champion mode'
            elif 20 <= champ_pct <= 80 and std_pct > 50:
                interp = 'BIMODAL'
            elif champ_pct < 20:
                interp = 'mostly churn + few champions'
            elif champ_pct > 80:
                interp = 'mostly champion + few churn'
            else:
                interp = 'mixed'
            print(f'{cv:>5.2f} | {eta:>4.1f} | {m:>6.1f} | {s:>5.1f} | {std_pct:>5.1f}% | '
                  f'{md:>7.1f} | {q1:>5.1f} | {q3:>5.1f} | {iqr:>5.1f} | {champ_pct:>6.1f}% | {interp:>25}')
    print()

    # ─────────────────────────────────────────────────────────────────────
    # Table 3: Decision criteria scan (15-seed) — mean and median variants
    # ─────────────────────────────────────────────────────────────────────
    print('=== Table 3: Decision criteria scan @ 15 seeds (claim3 + fisc>0 + hub_ok + low_noise) ===')
    print()
    print(f'{"cv":>5} | {"eta":>4} | {"bk_mean":>7} | {"d_mean":>7} | {"bk_med":>7} | {"d_med":>7} | '
          f'{"fisc_med":>8} | {"max_ten_med":>11} | {"std%":>6} | {"champ%":>7} | {"flags (medians)":>30}')
    print('-' * 165)
    for cv in CVS:
        bk0_cells = [int(r['total_bk']) for r in by_cell.get((cv, 0.0), [])]
        bk0_mean = sum(bk0_cells) / len(bk0_cells) if bk0_cells else 0
        bk0_med  = percentile(sorted(bk0_cells), 50) if bk0_cells else 0
        for eta in ETAS:
            if eta == 0.0: continue
            cells = by_cell.get((cv, eta), [])
            if not cells: continue
            bks = [int(r['total_bk']) for r in cells]
            fisc = [int(r['fiscal_deaths']) for r in cells]
            mts  = [int(r['max_ten']) for r in cells]
            m_bk, _, _, md_bk, _, _ = stats(bks)
            d_mean = m_bk - bk0_mean
            d_med  = md_bk - bk0_med
            _, _, _, md_fi, _, _ = stats(fisc)
            m_mt, s_mt, _, md_mt, _, _ = stats(mts)
            std_pct = 100 * s_mt / m_mt if m_mt else 0
            n_champ = sum(1 for x in mts if x >= CHAMPION_THRESHOLD)
            champ_pct = 100 * n_champ / len(mts)
            # Flags computed on MEDIAN
            flags = []
            if d_med < 0:                            flags.append('claim3')
            if md_fi > 0:                            flags.append('fisc>0')
            if 15 <= md_mt <= 80:                    flags.append('hub_ok')
            if std_pct < 50 and champ_pct < 20:      flags.append('low_noise')
            tag = ' <<<' if len(flags) == 4 else ''
            print(f'{cv:>5.2f} | {eta:>4.1f} | {m_bk:>7.0f} | {d_mean:>+7.0f} | {md_bk:>7.0f} | {d_med:>+7.0f} | '
                  f'{md_fi:>8.0f} | {md_mt:>5.0f}+/-{s_mt:>5.1f} | {std_pct:>5.1f}% | {champ_pct:>6.1f}% | {",".join(flags):>27}{tag}')
    print()


if __name__ == '__main__':
    main()
