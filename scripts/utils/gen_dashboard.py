import pandas as pd, numpy as np, json, sys, os
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

# Project root (the repo directory itself).
PROJ = Path(__file__).resolve().parent
OUT = str(PROJ / 'dashboard.html')

# Phase 1.5 candidate-baseline dashboard — 27-cell grid.
# Tag schema: <cand>_<regime>_<eta> e.g. a_st_e01.
# All sims at seed 26474 (maximin-selected), single-seed visualization.
# Output / simulation data directory. Override via TBTF_SIM_DIR env var.
# Default: a sibling 'Simulations/' folder next to the repo root.
SIM_DIR = os.environ.get(
    'TBTF_SIM_DIR',
    str(Path(__file__).resolve().parent.parent / 'Simulations'),
)

# Dimension definitions (authoritative — matches dump_dashboard_runs.DASH27_CONFIGS)
CAND_DEFS = [   # tag, basis, omega, label, base_color
    # Post-2026-05-08 cleanup: dropped C3 family + w58_g04 (supervisor committed to w58 cv=0
    # canonical). Added _s62 variants (single-seed dumps at SEED=26462, the multi-seed pool
    # median by total_bk) so dashboard visualization aligns with thesis multi-seed methodology.
    # Original SEED=26474 cells preserved alongside _s62 cells for direct comparison.
    ('bsl',     'equity', 0.50, 'baseline ω=0.50',                '#3498db'),
    ('a',       'equity', 0.53, '[A] equity ω=0.53',              '#27ae60'),
    ('w55',     'equity', 0.55, '[ω55] equity ω=0.55',            '#e67e22'),
    # w58 family: ω=0.58 at default μ=0.7. Canonical thesis cell.
    ('w58',     'equity', 0.58, '[w58 γ=0.10] ω=0.58',            '#c0392b'),
    # w58 low-τ variants (τ→0 crossover finding §14.2). Fund regime preserves interior min at τ ≤ 1e-5.
    ('w58_t6',  'equity', 0.58, '[w58 τ=1e-6] ω=0.58 low-τ',      '#922b21'),
    ('w58_t5',  'equity', 0.58, '[w58 τ=1e-5] ω=0.58 low-τ',      '#a93226'),
    # _s62 variants: same configurations but dumped at SEED=26462 (thesis median seed).
    # Allows side-by-side comparison of single-seed visualizations between dashboard
    # convention (26474) and thesis convention (26462).
    ('w58_s62',     'equity', 0.58, '[w58 γ=0.10] ω=0.58 — SEED=26462',     '#8e44ad'),
    ('w58_t6_s62',  'equity', 0.58, '[w58 τ=1e-6] ω=0.58 low-τ — SEED=26462', '#7d3c98'),
    ('w58_t5_s62',  'equity', 0.58, '[w58 τ=1e-5] ω=0.58 low-τ — SEED=26462', '#6c3483'),
]
REGIME_DEFS = [  # tag, fiscal_regime, label, hue_shift_deg
    ('nt', 'none',            'no tax',          0),
    ('st', 'socialized_tax',  'socialized',      0),
    ('rf', 'resolution_fund', 'resolution fund', 0),
]
ETA_DEFS = [    # tag, value, label
    ('e0',   0.00, 'η=0'),
    ('e01',  0.10, 'η=0.1'),
    ('e085', 0.85, 'η=0.85'),
]

def _color_for(cand_color, regime_idx, eta_idx):
    """Mix the cand base color with regime+eta variation by lightness/saturation."""
    # Convert hex to RGB then to HSL-ish lightness adjustment. Simple mix.
    h = cand_color.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    # Lightness shift: regime 0/1/2 → -25/0/+25 in luminance
    # Eta     shift: 0/1/2 → solid/lighter/lightest variant
    light_pct = (regime_idx - 1) * 0.18 + (eta_idx) * 0.06
    if light_pct >= 0:
        r = int(r + (255 - r) * light_pct)
        g = int(g + (255 - g) * light_pct)
        b = int(b + (255 - b) * light_pct)
    else:
        r = int(r * (1 + light_pct))
        g = int(g * (1 + light_pct))
        b = int(b * (1 + light_pct))
    r,g,b = max(0,min(255,r)), max(0,min(255,g)), max(0,min(255,b))
    return f'#{r:02x}{g:02x}{b:02x}'

FILES, RUNS, COLS_JS, LABELS_JS = {}, [], {}, {}
_n_skipped = 0
for ci, (c_tag, basis, omega, c_label, c_color) in enumerate(CAND_DEFS):
    for ri, (r_tag, fiscal, r_label, _) in enumerate(REGIME_DEFS):
        for ei, (e_tag, e_val, e_label) in enumerate(ETA_DEFS):
            tag = f'{c_tag}_{r_tag}_{e_tag}'
            csv_path = SIM_DIR + f'\\dash_{tag}.csv'
            # Skip missing CSVs gracefully (e.g. BA m=5 cands only have st × e01).
            # Cells without data fall through to graceful-degrade in JS.
            if not os.path.exists(csv_path):
                _n_skipped += 1
                continue
            FILES[tag]    = csv_path
            RUNS.append(tag)
            COLS_JS[tag]  = _color_for(c_color, ri, ei)
            LABELS_JS[tag]= f'{c_label} · {r_label} · {e_label}'
if _n_skipped:
    print(f'[FILES] skipped {_n_skipped} missing cells (e.g. BA m=5 cands only dumped at st × η=0.1)')

# Per-dimension chip+pill metadata (used by JS to build the dim controls)
CAND_TAGS   = [c[0] for c in CAND_DEFS]
REGIME_TAGS = [r[0] for r in REGIME_DEFS]
ETA_TAGS    = [e[0] for e in ETA_DEFS]
CAND_LABELS   = {c[0]: c[3] for c in CAND_DEFS}
REGIME_LABELS = {r[0]: r[2] for r in REGIME_DEFS}
ETA_LABELS    = {e[0]: e[2] for e in ETA_DEFS}
CAND_COLORS   = {c[0]: c[4] for c in CAND_DEFS}
REGIME_COLORS = {'nt':'#2c3e50', 'st':'#7f8c8d', 'rf':'#95a5a6'}  # dim, neutral
ETA_COLORS    = {'e0':'#5c6b6b', 'e01':'#7fa8a8', 'e085':'#a3c8c8'}  # cool gradient

EMBED = ['time','bankruptcy','bankruptcies_shock','bankruptcies_rationing',
         'bankruptcies_repay','bankruptcies_contagion','bankruptcies_fiscal',
         'fire_sale_survivors','equity','B','rationing','bailout_bill',
         'bailout_count','fitness','num_banks','loans',
         # interest rate is TBTF-influenced (eq. 6 has p_j, b_j, η in denominator)
         'interest_rate',
         # hub tracking columns (present after re-running simulations)
         'best_lender_clients','best_lender','best_lender_fitness','best_lender_equity',
         'equity_lenders','prob_bankruptcy','num_loans',
         # borrower-side TBTF reframe (§1.6 of updated_contextv3.md)
         'top_A_bank','top_A_value','best_lender_generation','previous_hub_alive']

dfs = {}
for k, f in FILES.items():
    df = pd.read_csv(f).apply(pd.to_numeric, errors='coerce')
    n  = df['num_banks'].replace(0, np.nan)
    df['avg_eq']    = df['equity'] / n
    df['fisc_mult'] = df['bankruptcies_fiscal'] / (df['bailout_bill']/n).replace(0,np.nan)
    df['npl']       = df['B'] / df['loans'].replace(0, np.nan)
    df['rat_mort']  = df['bankruptcies_rationing'] / df['rationing'].replace(0, np.nan)
    # hub derived series (safe when columns absent — falls back to NaN → null in JS)
    if 'best_lender_clients' in df.columns:
        df['hub_share']   = df['best_lender_clients'] / n
    else:
        df['hub_share']   = np.nan
    if 'best_lender' in df.columns:
        df['hub_id_norm'] = df['best_lender'] / n
    else:
        df['hub_id_norm'] = np.nan
    dfs[k] = df

EMBED_DERIVED = ['avg_eq','fisc_mult','npl','rat_mort','hub_share','hub_id_norm']

# ── Bank detail per-run (LAZY-LOAD via per-cand JSON files) ─────────────────
# Each run's per-(bank, period) CSV is converted to a SEPARATE dashboard_bank_<k>.json
# alongside dashboard.html. JS fetches them on-demand when the Bank Detail tab opens
# (one fetch per cand, cached client-side after first load).
#
# Incremental: skip JSON regeneration if it's newer than the source CSV. This means
# adding a new cand only re-processes that cand (~3 sec) instead of all 12 (~30 sec).
#
# Requires running a local HTTP server (`python -m http.server`) — browsers block
# fetch() from file:// origins.
import os, json as _json
BANK_FILES = {k: SIM_DIR + f'\\bank_detail_{k}.csv' for k in RUNS}
PROJECT_ROOT = os.path.dirname(os.path.abspath(OUT))
JS_BANK_DATA_AVAILABLE = []
n_rebuilt, n_cached, n_missing = 0, 0, 0
for k in RUNS:
    bf = BANK_FILES.get(k, '')
    if not bf or not os.path.exists(bf):
        n_missing += 1
        continue
    out_json = os.path.join(PROJECT_ROOT, f'dashboard_bank_{k}.json')
    JS_BANK_DATA_AVAILABLE.append(k)
    if os.path.exists(out_json) and os.path.getmtime(out_json) >= os.path.getmtime(bf):
        n_cached += 1
        continue
    bdf = pd.read_csv(bf)
    for col in ('equity', 'fitness', 'p_j', 'b_j', 'interest_rate', 'loan'):
        if col in bdf.columns:
            bdf[col] = bdf[col].round(3)
    # Replace NaN with None so json.dump emits 'null' (valid JSON) instead of
    # the literal 'NaN' (invalid JSON; JSON.parse() rejects it). NaN appears in
    # interest_rate when a bank has no active borrowers and similar columns.
    bdf = bdf.astype(object).where(bdf.notna(), None)
    rows = bdf.to_dict(orient='records')
    with open(out_json, 'w', encoding='utf-8') as f:
        _json.dump(rows, f, separators=(',', ':'))
    n_rebuilt += 1
print(f'bank-data JSONs: rebuilt {n_rebuilt}, cached {n_cached}, missing {n_missing}, '
      f'available {len(JS_BANK_DATA_AVAILABLE)}/{len(RUNS)}')
JS_BANK_DATA_AVAILABLE_LIST = json.dumps(JS_BANK_DATA_AVAILABLE)
ALL_EMBED = EMBED + EMBED_DERIVED

def s(v):
    if v is None: return 'null'
    if isinstance(v, float) and np.isnan(v): return 'null'
    return str(round(float(v), 5))

def series_js(arr):
    return '[' + ','.join(s(v) for v in arr) + ']'

js_data_parts = []
for k in RUNS:
    df = dfs[k]
    cols_str = []
    for col in ALL_EMBED:
        if col in df.columns:
            cols_str.append(f'"{col}":{series_js(df[col])}')
    js_data_parts.append(f'"{k}":{{{",".join(cols_str)}}}')
JS_DATA = '{' + ','.join(js_data_parts) + '}'

def nan0(v):
    return 0 if (v is None or (isinstance(v,float) and np.isnan(v))) else v

def _rle_runs(keys):
    if not keys: return []
    runs=[]; cur=keys[0]; start=0
    for i in range(1, len(keys)):
        if keys[i] != cur:
            runs.append(i-start); cur=keys[i]; start=i
    runs.append(len(keys)-start)
    return runs

def _compute_lender_max(df):
    """RLE on composite (best_lender, best_lender_generation) key."""
    if 'best_lender' not in df.columns or 'best_lender_generation' not in df.columns:
        return 0
    bl = df['best_lender'].fillna(-1).astype(int).tolist()
    bg = df['best_lender_generation'].fillna(-1).astype(int).tolist()
    keys = [(b, g) for b, g in zip(bl, bg) if b >= 0]
    runs = _rle_runs(keys)
    return max(runs) if runs else 0

def _compute_lender_avg(df):
    if 'best_lender' not in df.columns: return 0.0
    bl = df['best_lender'].fillna(-1).astype(int).tolist()
    if 'best_lender_generation' in df.columns:
        bg = df['best_lender_generation'].fillna(-1).astype(int).tolist()
        keys = [(b, g) for b, g in zip(bl, bg) if b >= 0]
    else:
        keys = [(b, 0) for b in bl if b >= 0]
    runs = _rle_runs(keys)
    return float(sum(runs))/len(runs) if runs else 0.0

def _compute_borrower_a_max(df):
    if 'top_A_bank' not in df.columns: return 0
    keys = [b for b in df['top_A_bank'].fillna(-1).astype(int).tolist() if b >= 0]
    runs = _rle_runs(keys)
    return max(runs) if runs else 0

def _compute_borrower_a_avg(df):
    if 'top_A_bank' not in df.columns: return 0.0
    keys = [b for b in df['top_A_bank'].fillna(-1).astype(int).tolist() if b >= 0]
    runs = _rle_runs(keys)
    return float(sum(runs))/len(runs) if runs else 0.0

summ = {}
for k in RUNS:
    df = dfs[k]
    bb = df['bailout_bill'] > 0
    fm = df['fisc_mult'][bb]
    summ[k] = {
        'tot_bkr':   int(df['bankruptcy'].sum()),
        'tot_fisc':  int(df['bankruptcies_fiscal'].sum()),
        'tot_rat':   int(df['bankruptcies_rationing'].sum()),
        'tot_cntg':  int(df['bankruptcies_contagion'].sum()),
        'tot_shock': int(df['bankruptcies_shock'].sum()),
        'tot_repay': int(df['bankruptcies_repay'].sum()),
        'mean_eq':   round(float(df['equity'].mean()), 1),
        'std_eq':    round(float(df['equity'].std()), 1),
        'mean_zomb': round(float(df['fire_sale_survivors'].mean()), 1),
        'tot_bill':  round(float(df['bailout_bill'].sum()), 0),
        'tot_bcount':int(df['bailout_count'].sum() if 'bailout_count' in df else 0),
        'max_fisc_m':round(nan0(float(fm.max()) if len(fm) else 0), 1),
        'mean_npl':  round(nan0(float(df['npl'].mean())), 4),
        'lender_max':    _compute_lender_max(df),
        'lender_avg':    round(_compute_lender_avg(df), 3),
        'borrower_a_max':_compute_borrower_a_max(df),
        'borrower_a_avg':round(_compute_borrower_a_avg(df), 3),
        'tot_zomb':  int(df['fire_sale_survivors'].sum()),
    }
JS_SUMM = json.dumps(summ)

# ── Sweeps tab: aggregate sweep CSVs into one JS variable ─────────────────────
# Loads the four Phase 1 grid + sweep CSVs, normalises columns (the two grid
# files lack an 'omega' column), aggregates seed-mean and seed-std per
# (basis, inertia, omega, eta) cell, embeds as JS variable SWEEP_DATA.
SWEEP_FILES = [
    (SIM_DIR + r'\phase1_grid_omega_0,5.csv',                {'omega': 0.50}),
    (SIM_DIR + r'\phase1_grid_omega_0,55.csv',               {'omega': 0.55}),
    (SIM_DIR + r'\phase1_5_omega_sweep.csv',                 {}),
    (SIM_DIR + r'\phase1_5_omega_sweep_bilateral_l0.csv',    {}),
    (SIM_DIR + r'\phase1_5_omega_sweep_equity_l0.csv',       {}),
]
SWEEP_METRICS = [
    'total_bk', 'contagion', 'shock', 'rationing', 'repay', 'fiscal', 'zombies',
    'lender_avg_tenure', 'lender_max_tenure',
    'borrower_a_avg_tenure', 'borrower_a_max_tenure',
    'n_transitions', 'mortality_frac', 'clamp_fraction',
]
import os as _os_sweep
sweep_dfs = []
for path, override in SWEEP_FILES:
    if not _os_sweep.path.exists(path):
        print(f"  [WARN] sweep CSV missing: {path}")
        continue
    s = pd.read_csv(path)
    for col, val in override.items():
        s[col] = val
    sweep_dfs.append(s[['seed', 'basis', 'inertia', 'omega', 'eta'] + SWEEP_METRICS])
if sweep_dfs:
    sweep_df = pd.concat(sweep_dfs, ignore_index=True)
    grouped = sweep_df.groupby(['basis', 'inertia', 'omega', 'eta'])
    sweep_records = []
    for (basis, inertia, omega, eta), grp in grouped:
        rec = {'basis': basis, 'inertia': float(inertia),
               'omega': float(omega), 'eta': float(eta),
               'n_seeds': int(len(grp))}
        for m in SWEEP_METRICS:
            mean_v = float(grp[m].mean())
            std_v  = float(grp[m].std()) if len(grp) > 1 else 0.0
            rec[m + '_mean'] = mean_v if not (np.isnan(mean_v) or np.isinf(mean_v)) else 0.0
            rec[m + '_std']  = std_v  if not (np.isnan(std_v)  or np.isinf(std_v))  else 0.0
        sweep_records.append(rec)
    JS_SWEEP_DATA = json.dumps(sweep_records)
    print(f"SWEEP_DATA: {len(sweep_records)} (basis, inertia, ω, η) cells")
else:
    JS_SWEEP_DATA = '[]'
    print("  [WARN] no sweep CSVs loaded; SWEEP_DATA empty")

# ── Thesis-reproduction sweeps (Lehman / ρ / ρ=0.1) ───────────────────────────
def _load_thesis_sweep(path, group_keys, metrics=('total_bk','contagion')):
    if not os.path.exists(path):
        print(f"  [WARN] thesis sweep CSV missing: {path}")
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    agg_map = {f'{m}_mean': (m, 'mean') for m in metrics}
    agg_map.update({f'{m}_std': (m, 'std') for m in metrics})
    g = df.groupby(group_keys, sort=False).agg(**agg_map).reset_index()
    # Replace NaN std with 0 (single-seed cells)
    for m in metrics:
        col = f'{m}_std'
        if col in g.columns:
            g[col] = g[col].fillna(0.0)
    return g.to_dict(orient='records')

def _load_thesis_per_cand(stem, group_keys, metrics):
    """Load thesis_<stem>.csv (baseline) + thesis_<stem>_<cand>.csv for each
    candidate. Returns {cand_tag: [records...]}. Missing files graceful-degrade.
    Phase 0 cleanup: removed BA m=3 loading (BA path supervisor-rejected)."""
    # cand_tag → file suffix (after thesis_<stem>)
    cand_files = [
        ('bsl',     ''),
        ('a',       '_a'),
        ('w55',     '_w55'),
        ('w58',     '_w58'),
        ('w58_t6',  '_w58_t6'),
        ('w58_t5',  '_w58_t5'),
        # _s62 cands: single-seed dumps (SEED=26462) for visualization-only.
        # No multi-seed thesis-sweep data — they fall back to baseline in Sweep tab.
        # ('w58_s62', no thesis_sweep file),
        # ('w58_t5_s62', no thesis_sweep file),
        # ('w58_t6_s62', no thesis_sweep file),
    ]
    out = {}
    for cand_tag, suffix in cand_files:
        recs = _load_thesis_sweep(
            SIM_DIR + f'\\thesis_{stem}{suffix}.csv',
            group_keys, metrics=metrics)
        if recs:
            out[cand_tag] = recs
    return out

_thesis_metrics = ('total_bk', 'contagion', 'shock', 'rationing', 'repay', 'zombies')
LEHMAN_DATA      = _load_thesis_per_cand('lehman', ['fiscal_regime', 'eta'], _thesis_metrics)
RHO_DATA         = _load_thesis_per_cand('rho',    ['rho'],                  _thesis_metrics)
RHO01_DATA       = _load_thesis_per_cand('rho01',  ['fiscal_regime', 'eta'], _thesis_metrics)
def _safe_dump(obj):
    return json.dumps(obj, default=lambda x: None if (x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))) else x)
JS_LEHMAN_DATA      = _safe_dump(LEHMAN_DATA)
JS_RHO_DATA         = _safe_dump(RHO_DATA)
JS_RHO01_DATA       = _safe_dump(RHO01_DATA)
# BA m=3 datasets retired in Phase 0 — empty objects so existing JS toggles fall back.
JS_LEHMAN_DATA_BA3  = '{}'
JS_RHO_DATA_BA3     = '{}'
JS_RHO01_DATA_BA3   = '{}'
def _count(d): return ' / '.join(f'{k}:{len(v)}' for k, v in d.items()) or 'empty'
print(f"LEHMAN_DATA: {_count(LEHMAN_DATA)}")
print(f"RHO_DATA:    {_count(RHO_DATA)}")
print(f"RHO01_DATA:  {_count(RHO01_DATA)}")

# ── Topology snapshots (Brini Fig. 3 analogue) ────────────────────────────────
# Loads topology_<cand>_st_e01.json (per-cand, three snapshot timepoints t=0/500/999)
# and precomputes 2D layouts via networkx.spring_layout. The layout is held fixed
# across the three snapshots per cand so node positions stay comparable.
try:
    import networkx as nx
    _has_nx = True
except ImportError:
    _has_nx = False
TOPOLOGY_DATA = {}
if _has_nx:
    for cand_tag in (c[0] for c in CAND_DEFS):
        topo_path = SIM_DIR + f'\\topology_{cand_tag}_st_e01.json'
        if not os.path.exists(topo_path):
            continue
        with open(topo_path, 'r', encoding='utf-8') as f:
            snaps = json.load(f)
        # Use the union of edges across all snapshots to fix node positions
        G = nx.DiGraph()
        all_bank_ids = set()
        for snap in snaps.values():
            for b in snap.get('banks', []):
                all_bank_ids.add(int(b['id']))
            for e in snap.get('edges', []):
                G.add_edge(int(e['borrower']), int(e['lender']))
        for bid in all_bank_ids:
            G.add_node(bid)
        # Spring layout, deterministic seed for reproducibility
        try:
            pos = nx.spring_layout(G, seed=42, k=0.6, iterations=80)
        except Exception:
            pos = {n: (0, 0) for n in G.nodes()}
        # Per-snapshot view: list of nodes (with x/y/equity/is_hub) and edges
        TOPOLOGY_DATA[cand_tag] = {}
        for t_str, snap in snaps.items():
            nodes = []
            for b in snap.get('banks', []):
                bid = int(b['id'])
                xy = pos.get(bid, (0.0, 0.0))
                nodes.append({
                    'id': bid,
                    'x': round(float(xy[0]), 4),
                    'y': round(float(xy[1]), 4),
                    'equity': float(b.get('equity', 0.0)),
                    'is_hub': bool(b.get('is_hub', False)),
                    'clients': int(b.get('clients', 0)),
                })
            edges = [
                {'borrower': int(e['borrower']),
                 'lender':   int(e['lender']),
                 'loan':     float(e.get('loan', 0.0))}
                for e in snap.get('edges', [])
            ]
            TOPOLOGY_DATA[cand_tag][t_str] = {'nodes': nodes, 'edges': edges}
JS_TOPOLOGY_DATA = json.dumps(TOPOLOGY_DATA, separators=(',', ':'))
print(f"TOPOLOGY_DATA: {_count(TOPOLOGY_DATA)}")

# ── Seed-robustness summary (30-seed × 3-regime × 4-cand at η=0.1) ─────────────
# Addresses the "max=24 vs 8 — seed artifact or regime property?" question.
ROBUSTNESS_DATA = {}
_robustness_path = SIM_DIR + r'\seed_robustness_eta01.csv'
if os.path.exists(_robustness_path):
    rdf = pd.read_csv(_robustness_path)
    grp = rdf.groupby(['cand', 'regime']).agg(
        max_tenure_mean=('max_tenure', 'mean'), max_tenure_std=('max_tenure', 'std'),
        max_tenure_max=('max_tenure', 'max'),
        avg_tenure_mean=('avg_tenure', 'mean'), avg_tenure_std=('avg_tenure', 'std'),
        n_seeds=('seed', 'count'),
    ).reset_index()
    for _, r in grp.iterrows():
        ROBUSTNESS_DATA.setdefault(r['cand'], {})[r['regime']] = {
            'max_mean': round(float(r['max_tenure_mean']), 2),
            'max_std':  round(float(r['max_tenure_std']) if pd.notna(r['max_tenure_std']) else 0.0, 2),
            'max_max':  int(r['max_tenure_max']),
            'avg_mean': round(float(r['avg_tenure_mean']), 2),
            'avg_std':  round(float(r['avg_tenure_std']) if pd.notna(r['avg_tenure_std']) else 0.0, 2),
            'n_seeds':  int(r['n_seeds']),
        }
    print(f"ROBUSTNESS_DATA: {_count(ROBUSTNESS_DATA)} (n_seeds per cell ≈ 30)")
else:
    print(f"  [WARN] seed_robustness CSV missing: {_robustness_path}")
JS_ROBUSTNESS_DATA = json.dumps(ROBUSTNESS_DATA)

# ── Build HTML ─────────────────────────────────────────────────────────────────
html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>TBTF Signal Analysis — Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#0f0f0f; --s1:#161616; --s2:#1e1e1e; --s3:#262626;
  --border:#2a2a2a; --border2:#333;
  --ink:#e2ddd6; --muted:#888; --faint:#444;
  --red:#e74c3c; --blue:#3498db; --purple:#9b59b6;
  --green:#27ae60; --teal:#1abc9c; --amber:#f39c12;
}
*{margin:0;padding:0;box-sizing:border-box;}
html,body{height:100%;background:var(--bg);color:var(--ink);
  font-family:"IBM Plex Mono",monospace;font-size:13px;line-height:1.55;}

/* ── HEADER ── */
.hdr{
  display:flex;align-items:center;justify-content:space-between;
  padding:1.1rem 2rem;border-bottom:1px solid var(--border2);
  background:var(--s1);position:sticky;top:0;z-index:100;
}
.hdr-left h1{font-size:.95rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;}
.hdr-left .sub{font-size:.68rem;color:var(--muted);margin-top:.15rem;letter-spacing:.04em;}
.run-chips{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;}
/* CURRENT VIEW chips (single-run focus) */
.cv-label{font-size:.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;margin-right:.25rem;}
.cv-chip{display:inline-flex;align-items:center;gap:.35rem;padding:.3rem .55rem;border:1px solid var(--border2);background:var(--s2);color:var(--ink);font-family:inherit;font-size:.7rem;cursor:pointer;border-radius:2px;letter-spacing:.04em;}
.cv-chip:hover{background:var(--s3);}
.cv-chip .cv-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
.cv-chip::after{content:' ▾';color:var(--muted);font-size:.6rem;margin-left:.2rem;}
.cv-popover{position:absolute;background:var(--s2);border:1px solid #555;padding:.3rem;display:none;flex-direction:column;gap:.2rem;z-index:1000;min-width:200px;border-radius:2px;box-shadow:0 6px 24px rgba(0,0,0,.5);}
.cv-popover.show{display:flex;}
.cv-popover button{background:var(--s3);border:1px solid var(--border2);color:var(--ink);font-family:inherit;font-size:.68rem;padding:.35rem .5rem;text-align:left;cursor:pointer;border-radius:2px;}
.cv-popover button:hover{background:#3a3a3a;}
.cv-popover button.current{border-color:#fff;color:#fff;}
/* Phase 0: overview cells multi-select popover */
.ov-cells-popover{position:absolute;background:var(--s2);border:1px solid #555;padding:0;display:none;z-index:1000;min-width:300px;max-height:520px;overflow-y:auto;border-radius:2px;box-shadow:0 6px 24px rgba(0,0,0,.5);}
.ov-cells-popover.show{display:block;}
.ov-cells-popover .ov-cand-hdr{padding:.3rem .55rem;color:#bbb;font-size:.65rem;font-weight:600;letter-spacing:.04em;border-top:1px solid #333;background:#1a1a1a;}
.ov-cells-popover .ov-cand-hdr:first-child{border-top:none;}
.ov-cells-popover label{display:flex;align-items:center;gap:.45rem;padding:.18rem .65rem;cursor:pointer;font-size:.66rem;color:var(--ink);}
.ov-cells-popover label:hover{background:#2a2a2a;}
.ov-cells-popover input[type=checkbox]{margin:0;cursor:pointer;}
.single-run-label{font-size:.65rem;color:var(--muted);padding:.3rem .65rem;background:var(--s2);border-left:2px solid currentColor;margin-bottom:.6rem;letter-spacing:.04em;}
.chip{
  display:inline-flex;align-items:center;gap:.35rem;
  padding:.25rem .6rem;border:1px solid;border-radius:2px;
  font-size:.65rem;font-family:inherit;letter-spacing:.05em;text-transform:uppercase;
  cursor:pointer;background:transparent;transition:opacity .15s;
}
.chip.off{opacity:.25;filter:grayscale(1);}
.chip-dot{width:7px;height:7px;border-radius:50%;display:inline-block;}

/* ── TABS ── */
.tabs{
  display:flex;padding:0 2rem;border-bottom:1px solid var(--border2);
  background:var(--s1);gap:.1rem;overflow-x:auto;scrollbar-width:none;
}
.tabs::-webkit-scrollbar{display:none;}
.tab{
  padding:.55rem 1.1rem;font-size:.68rem;font-family:inherit;
  letter-spacing:.08em;text-transform:uppercase;
  background:none;border:none;border-bottom:2px solid transparent;
  color:var(--muted);cursor:pointer;white-space:nowrap;transition:color .12s;
}
.tab:hover{color:var(--ink);}
.tab.active{color:#fff;border-bottom-color:#fff;}

/* ── PANELS ── */
.panel{display:none;padding:1.6rem 2rem 3rem;}
.panel.active{display:block;}

/* ── CONTROLS ── */
.ctrl-bar{
  display:flex;flex-wrap:wrap;gap:.7rem;align-items:center;
  padding:.7rem .9rem;background:var(--s2);border:1px solid var(--border);
  margin-bottom:1.2rem;
}
.ctrl-label{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;}
select,input[type=range]{
  font-family:inherit;font-size:.75rem;background:var(--s3);
  color:var(--ink);border:1px solid var(--border2);padding:.28rem .5rem;
  cursor:pointer;border-radius:2px;
}
select:focus{outline:1px solid #555;}
.seg-btn{
  padding:.28rem .7rem;font-size:.65rem;font-family:inherit;
  letter-spacing:.06em;text-transform:uppercase;
  background:var(--s3);border:1px solid var(--border2);
  color:var(--muted);cursor:pointer;transition:all .12s;
}
.seg-btn.on{background:var(--s2);border-color:#666;color:var(--ink);}

/* ── CHART BOXES ── */
.chart-box{
  background:var(--s1);border:1px solid var(--border);
  padding:.9rem;margin-bottom:1rem;
}
.chart-title{
  font-size:.68rem;color:var(--muted);text-transform:uppercase;
  letter-spacing:.07em;margin-bottom:.5rem;
}
.chart-wrap{width:100%;min-height:60px;}

/* ── STAT CARDS ── */
.stat-row{display:grid;grid-template-columns:repeat(6,1fr);gap:.5rem;margin-bottom:1.2rem;}
@media(max-width:900px){.stat-row{grid-template-columns:repeat(3,1fr);}}
.stat-card{
  background:var(--s2);border:1px solid var(--border);
  padding:.7rem .8rem;cursor:pointer;transition:border-color .15s;
  position:relative;overflow:hidden;
}
.stat-card:hover{border-color:#555;}
.stat-card .run-stripe{
  position:absolute;top:0;left:0;right:0;height:2px;
}
.stat-card .run-lbl{font-size:.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-top:.2rem;}
.stat-card .big{font-size:1.2rem;font-weight:700;margin:.3rem 0 .15rem;line-height:1;}
.stat-card .detail{font-size:.62rem;color:var(--muted);}
.stat-card.hi .big{color:var(--red);}
.stat-card.lo .big{color:var(--teal);}

/* ── TWO / THREE COLS ── */
.two{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;}
.three{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.8rem;}
@media(max-width:700px){.two,.three{grid-template-columns:1fr;}}

/* ── LEGEND ── */
.leg{display:flex;flex-wrap:wrap;gap:.5rem 1.2rem;margin:.4rem 0 .6rem;font-size:.67rem;color:var(--muted);}
.leg-item{display:flex;align-items:center;gap:.4rem;}
.leg-dot{width:10px;height:2px;border-radius:1px;display:inline-block;}

/* ── CALLOUT ── */
.callout{
  border:1px solid var(--border2);padding:.6rem .9rem;
  font-size:.75rem;color:var(--muted);margin:.8rem 0;
  background:var(--s2);line-height:1.6;
}
.callout.hi{border-color:#5a2020;background:#1a0f0f;color:#c07070;}
.callout.ok{border-color:#1a4a2a;background:#0c1810;color:#70b080;}

/* ── CCF ── */
.ccf-peak{
  display:inline-block;padding:.3rem .7rem;background:var(--s3);
  border:1px solid var(--border2);font-size:.75rem;margin-top:.5rem;
}
.ccf-peak span{color:#fff;font-weight:700;}

/* ── DECOMP TABS ── */
.d-tabs{display:flex;gap:.3rem;margin-bottom:.8rem;flex-wrap:wrap;}
.d-tab{
  padding:.25rem .65rem;font-size:.65rem;font-family:inherit;
  text-transform:uppercase;letter-spacing:.05em;
  background:var(--s3);border:1px solid var(--border2);color:var(--muted);
  cursor:pointer;border-radius:2px;transition:all .12s;
}
.d-tab.active{background:var(--s2);border-color:#666;color:var(--ink);}

/* ── RADAR WRAPPER ── */
.radar-wrap{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;}
@media(max-width:700px){.radar-wrap{grid-template-columns:1fr;}}

/* ── OVERLAY SLOTS ── */
.ov-slot{
  display:flex;align-items:center;gap:.5rem;padding:.4rem .7rem;
  background:var(--s2);border:1px solid var(--border);margin-bottom:.35rem;
  flex-wrap:wrap;
}
.ov-color-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;}
.ov-axis-btn{
  padding:.2rem .5rem;font-size:.62rem;font-family:inherit;text-transform:uppercase;
  letter-spacing:.05em;background:var(--s3);border:1px solid var(--border2);
  color:var(--muted);cursor:pointer;border-radius:2px;min-width:28px;text-align:center;
}
.ov-axis-btn.right{border-color:#888;color:var(--ink);}
.ov-remove{
  padding:.2rem .45rem;font-size:.75rem;background:none;border:none;
  color:#666;cursor:pointer;margin-left:auto;
}
.ov-remove:hover{color:var(--red);}

/* ── HUB / BANK STAT CHIPS ── */
.mini-cards{display:flex;gap:.5rem;margin-bottom:.9rem;flex-wrap:wrap;}
.mini-card{
  background:var(--s2);border:1px solid var(--border);
  padding:.55rem .8rem;min-width:90px;
}
.mini-card .lbl{font-size:.58rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;}
.mini-card .val{font-size:1.1rem;font-weight:700;margin:.2rem 0 .1rem;line-height:1;}
.mini-card .sub{font-size:.6rem;color:var(--muted);}

/* ── METRIC CHIPS ── */
.metric-chips{display:flex;flex-wrap:wrap;gap:.3rem;margin-bottom:.8rem;}
.metric-chip{
  padding:.22rem .55rem;font-size:.62rem;font-family:inherit;
  text-transform:uppercase;letter-spacing:.05em;
  background:var(--s3);border:1px solid var(--border2);color:var(--muted);
  cursor:pointer;border-radius:2px;transition:all .12s;
}
.metric-chip.on{border-color:#888;color:var(--ink);background:var(--s2);}

/* ── ctrl-group ── */
.ctrl-group{display:flex;align-items:center;gap:.4rem;}
</style>
</head>
<body>

<!-- HEADER -->
<div class="hdr">
  <div class="hdr-left">
    <h1>TBTF Signal Analysis</h1>
    <div class="sub">3 cands &nbsp;×&nbsp; 3 fiscal regimes &nbsp;×&nbsp; 3 η &nbsp;·&nbsp; interbank ABM &nbsp;·&nbsp; N=50 · T=1000 · seed 26474 · 27 cells</div>
  </div>
  <div class="run-chips" id="cv-chips">
    <span class="cv-label">CURRENT VIEW</span>
    <button class="cv-chip" id="chip-cand"   onclick="openChipMenu('cand', this, event)"></button>
    <button class="cv-chip" id="chip-regime" onclick="openChipMenu('regime', this, event)"></button>
    <button class="cv-chip" id="chip-eta"    onclick="openChipMenu('eta', this, event)"></button>
  </div>
</div>

<!-- Popover (single shared) for chip menus -->
<div class="cv-popover" id="cv-popover"></div>

<!-- TABS -->
<nav class="tabs">
  <button class="tab active" onclick="showTab('overview')">Overview</button>
  <button class="tab" onclick="showTab('sweeps')">Sweeps</button>
  <button class="tab" onclick="showTab('hub')">Hub Tracker</button>
  <button class="tab" onclick="showTab('overlay')">Mix &amp; Match</button>
  <button class="tab" onclick="showTab('timeseries')">Time Series</button>
  <button class="tab" onclick="showTab('decomp')">Decomposition</button>
  <button class="tab" onclick="showTab('regcmp')">Regime Cmp</button>
  <button class="tab" onclick="showTab('fiscal')">Fiscal Channel</button>
  <button class="tab" onclick="showTab('ccf')">CCF Explorer</button>
  <button class="tab" onclick="showTab('bank')">Bank Detail</button>
</nav>

<!-- ══════════════════════════════════════════════════════════════════════════ -->
<!-- PANEL: OVERVIEW -->
<!-- ══════════════════════════════════════════════════════════════════════════ -->
<div id="panel-overview" class="panel active">
  <!-- Phase 0: 72-card grid replaced with multi-select dropdown. Default = chip's
       compareRow (8 cands at chip's regime/eta). User can toggle other cells in popover. -->
  <div class="ctrl-bar">
    <div class="ctrl-group">
      <span class="ctrl-label">Cells</span>
      <button class="seg-btn" id="ov-cells-btn" onclick="openOverviewMenu(this, event)">
        <span id="ov-cells-count">8</span> selected ▾
      </button>
      <button class="seg-btn" onclick="ovResetToRow()">Chip row</button>
      <button class="seg-btn" onclick="ovSelectAll()">All</button>
      <button class="seg-btn" onclick="ovSelectNone()">None</button>
    </div>
  </div>
  <div class="ov-cells-popover" id="ov-cells-popover"></div>
  <div id="stat-row" class="stat-row"></div>

  <div class="radar-wrap">
    <div class="chart-box">
      <div class="chart-title">Policy performance radar — normalized (outer = worse)</div>
      <div id="chart-radar" style="height:340px"></div>
    </div>
    <div class="chart-box">
      <div class="chart-title">Cumulative bankruptcies by cause</div>
      <div id="chart-bkr-bar" style="height:340px"></div>
    </div>
  </div>

  <div class="chart-box">
    <div class="chart-title">Cumulative bailout bill vs. total fiscal deaths</div>
    <div id="chart-scatter" style="height:280px"></div>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════════ -->
<!-- PANEL: TIME SERIES -->
<!-- ══════════════════════════════════════════════════════════════════════════ -->
<div id="panel-timeseries" class="panel">
  <div class="ctrl-bar">
    <div class="ctrl-group">
      <span class="ctrl-label">Variable</span>
      <select id="var-select" onchange="renderTS()"></select>
    </div>
    <div class="ctrl-group">
      <span class="ctrl-label">Smooth</span>
      <select id="smooth-select" onchange="renderTS()">
        <option value="1">Off</option>
        <option value="5">5-period MA</option>
        <option value="10" selected>10-period MA</option>
        <option value="20">20-period MA</option>
      </select>
    </div>
  </div>

  <div class="chart-box">
    <div class="chart-title" id="ts-title">—</div>
    <div id="chart-ts" style="height:420px"></div>
  </div>

  <div class="chart-box">
    <div class="chart-title">Period-by-period difference vs. η=0.85 social (baseline)</div>
    <div id="chart-diff" style="height:240px"></div>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════════ -->
<!-- PANEL: DECOMPOSITION -->
<!-- ══════════════════════════════════════════════════════════════════════════ -->
<div id="panel-decomp" class="panel">
  <!-- Phase 0: 72 d-tab pills replaced with a single dropdown for navigation. -->
  <div class="ctrl-bar">
    <div class="ctrl-group">
      <span class="ctrl-label">Cell</span>
      <select id="decomp-select" onchange="onDecompSelect(this.value)" style="min-width:380px"></select>
    </div>
  </div>
  <div class="chart-box">
    <div class="chart-title" id="decomp-title">Bankruptcy decomposition</div>
    <div id="chart-decomp" style="height:380px"></div>
  </div>
  <div class="chart-box">
    <div class="chart-title">Decomposition share comparison — cumulative % across all runs</div>
    <div id="chart-decomp-share" style="height:240px"></div>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════════ -->
<!-- PANEL: REGIME COMPARISON (Phase 0 — professor's request) -->
<!-- ══════════════════════════════════════════════════════════════════════════ -->
<div id="panel-regcmp" class="panel">
  <div class="ctrl-bar">
    <div class="ctrl-group">
      <span class="ctrl-label">Metric</span>
      <select id="regcmp-var" onchange="renderRegCmp()"></select>
    </div>
    <div class="ctrl-group">
      <span class="ctrl-label">Smooth</span>
      <select id="regcmp-smooth" onchange="renderRegCmp()">
        <option value="1">Off</option>
        <option value="5">5-period MA</option>
        <option value="10" selected>10-period MA</option>
        <option value="20">20-period MA</option>
      </select>
    </div>
    <div class="ctrl-group">
      <span class="ctrl-label">Cand</span>
      <span class="single-run-label" id="regcmp-cand"></span>
      <span class="ctrl-label" style="margin-left:.6rem">η</span>
      <span class="single-run-label" id="regcmp-eta"></span>
    </div>
  </div>

  <div class="chart-box">
    <div class="chart-title" id="regcmp-title">Metric across 3 fiscal regimes</div>
    <div id="chart-regcmp" style="height:360px"></div>
  </div>

  <div class="chart-box" style="margin-top:.8rem">
    <div class="chart-title">Hub centrality across 3 regimes — lender ID, client count, hub fitness (professor's stacked view)</div>
    <div id="chart-regcmp-id"  style="height:200px"></div>
    <div id="chart-regcmp-cli" style="height:200px"></div>
    <div id="chart-regcmp-fit" style="height:200px"></div>
  </div>

  <div class="callout" style="margin-top:.6rem;font-size:.71rem">
    Top chart: chosen metric over time, one line per fiscal regime, holding the chipped cand and η fixed —
    pivots TS's "compare cands at one regime" into "compare 3 regimes at one cand".
    Bottom: the centrality triplet (best_lender ID, num_clients, fitness Φ) the supervisor specifically
    asked to see together. Each sub-chart shows all 3 regimes overlaid so the network-centrality
    differences (or lack thereof) are visible at a glance.
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════════ -->
<!-- PANEL: FISCAL -->
<!-- ══════════════════════════════════════════════════════════════════════════ -->
<div id="panel-fiscal" class="panel">
  <div class="callout hi">
    <strong>Fiscal cascade mechanism:</strong> bailout → per-bank tax on survivors → near-zero equity banks fail → more bailouts.
    Only active under <em>socialized ex-post tax</em> regime.
  </div>
  <div class="ctrl-bar">
    <div class="ctrl-group">
      <span class="ctrl-label">Clip bill at</span>
      <select id="bill-clip" onchange="renderFiscal()">
        <option value="0">No clip</option>
        <option value="60" selected>60</option>
        <option value="200">200</option>
      </select>
    </div>
  </div>
  <div class="chart-box">
    <div class="chart-title">Bailout bill per period</div>
    <div id="chart-bill" style="height:220px"></div>
  </div>
  <div class="two">
    <div class="chart-box">
      <div class="chart-title">Fiscal cascade deaths per period</div>
      <div id="chart-fisc-deaths" style="height:220px"></div>
    </div>
    <div class="chart-box">
      <div class="chart-title">Fiscal multiplier (fiscal deaths / per-bank tax unit) — clip 50</div>
      <div id="chart-fisc-mult" style="height:220px"></div>
    </div>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════════ -->
<!-- PANEL: CCF EXPLORER -->
<!-- ══════════════════════════════════════════════════════════════════════════ -->
<div id="panel-ccf" class="panel">
  <div class="ctrl-bar">
    <div class="ctrl-group">
      <span class="ctrl-label">X (cause)</span>
      <select id="ccf-x" onchange="renderCCF()"></select>
    </div>
    <div class="ctrl-group">
      <span class="ctrl-label">leads →</span>
    </div>
    <div class="ctrl-group">
      <span class="ctrl-label">Y (effect)</span>
      <select id="ccf-y" onchange="renderCCF()"></select>
    </div>
    <div class="ctrl-group">
      <span class="ctrl-label">Max lag</span>
      <select id="ccf-lag" onchange="renderCCF()">
        <option value="10">±10</option>
        <option value="20" selected>±20</option>
        <option value="40">±40</option>
      </select>
    </div>
  </div>

  <div class="chart-box">
    <div class="chart-title">Cross-correlation function — all active runs overlaid</div>
    <div id="chart-ccf" style="height:340px"></div>
  </div>

  <div id="ccf-peaks"></div>

  <div class="callout ok" style="margin-top:.8rem">
    <strong>Reading CCF:</strong> positive lag k = X leads Y by k periods.
    Peak at k=0 → contemporaneous; k=+1 → X predicts Y one period ahead.
    Causal chain: zombies → rationing (+1) → bad debt (+2–3) → contagion (0).
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════════ -->
<!-- PANEL: MIX & MATCH OVERLAY -->
<!-- ══════════════════════════════════════════════════════════════════════════ -->
<div id="panel-overlay" class="panel">
  <div class="ctrl-bar">
    <div class="ctrl-group">
      <span class="ctrl-label">Viewing</span>
      <span class="single-run-label" id="ov-run"></span>
    </div>
    <div class="ctrl-group">
      <span class="ctrl-label">Smooth</span>
      <select id="ov-smooth" onchange="renderOverlay()">
        <option value="1">Off</option>
        <option value="5">5-period</option>
        <option value="10" selected>10-period</option>
        <option value="20">20-period</option>
      </select>
    </div>
    <button class="seg-btn on" onclick="addOverlaySlot()" style="margin-left:.3rem">+ Add variable</button>
    <div class="ctrl-group" style="margin-left:.6rem">
      <span class="ctrl-label">Left axis</span>
      <button class="seg-btn on" id="ov-log-left" onclick="toggleOvLog('left')">Linear</button>
    </div>
    <div class="ctrl-group">
      <span class="ctrl-label">Right axis</span>
      <button class="seg-btn on" id="ov-log-right" onclick="toggleOvLog('right')">Linear</button>
    </div>
  </div>

  <div id="ov-slots"></div>

  <div class="chart-box" style="margin-top:.5rem">
    <div class="chart-title">Mix &amp; Match Overlay — left axis · right axis</div>
    <div id="chart-overlay" style="height:440px"></div>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════════ -->
<!-- PANEL: HUB TRACKER -->
<!-- ══════════════════════════════════════════════════════════════════════════ -->
<div id="panel-hub" class="panel">
  <div class="ctrl-bar">
    <div class="ctrl-group">
      <span class="ctrl-label">Viewing</span>
      <span class="single-run-label" id="hub-run"></span>
    </div>
    <div class="ctrl-group">
      <span class="ctrl-label">Hub window (rolling mode)</span>
      <select id="hub-window" onchange="renderHub()">
        <option value="1">Raw (per-period)</option>
        <option value="5">5-period</option>
        <option value="10" selected>10-period</option>
        <option value="20">20-period</option>
        <option value="50">50-period</option>
      </select>
    </div>
    <div class="ctrl-group">
      <span class="ctrl-label">Smooth fitness/equity</span>
      <select id="hub-smooth" onchange="renderHub()">
        <option value="1" selected>Off</option>
        <option value="5">5-period</option>
        <option value="10">10-period</option>
      </select>
    </div>
  </div>

  <div class="mini-cards" id="hub-stats"></div>

  <div class="chart-box">
    <div class="chart-title">Hub bank ID (normalized, step) &nbsp;·&nbsp; Hub individual fitness (dotted, right) &nbsp;·&nbsp; Lines = hub turnovers</div>
    <div id="chart-hub" style="height:360px"></div>
  </div>
  <div class="chart-box">
    <div class="chart-title">Hub client count &nbsp;·&nbsp; Hub individual equity (right axis)</div>
    <div id="chart-hub2" style="height:270px"></div>
  </div>

  <div class="chart-box">
    <div class="chart-title">Network topology snapshots — Brini Fig. 3 analogue · chip-focused cand · socialized · η=0.1</div>
    <div id="chart-topology" style="height:auto;display:grid;grid-template-columns:repeat(3,1fr);gap:6px"></div>
    <div style="font-size:.65rem;color:#888;text-align:center;margin-top:.5rem">
      Cand follows the chip (top-right). Three panels = t=0 / t=500 / t=999. Force-directed layout (positions fixed across the three snapshots so hub emergence is visible). Uniform light-blue nodes sized by in-degree; arrows point borrower → lender. The hub stands out by size alone.
    </div>
  </div>

  <div class="callout" id="hub-note" style="margin-top:.6rem;font-size:.71rem">
    Hub bank ID step function tracks which specific bank (by index 0–N−1) dominates lending at each period.
    Each vertical line marks a hub turnover — the moment a different bank becomes the top lender.
    Fitness and equity shown are the hub bank's individual values (not population averages).
    <br><em id="hub-data-note" style="color:#c07070"></em>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════════ -->
<!-- PANEL: SWEEPS (Phase 1.5 trade-off curves) -->
<!-- ══════════════════════════════════════════════════════════════════════════ -->
<div id="panel-sweeps" class="panel">
  <!-- Mode selector — single source of truth for what's displayed -->
  <div class="ctrl-bar" style="gap:.4rem;flex-wrap:wrap">
    <button class="seg-btn on" id="smode-lehman"  onclick="setSweepMode('lehman')">Lehman · η × 3 regimes (C3)</button>
    <button class="seg-btn"    id="smode-rho"     onclick="setSweepMode('rho')">ρ-sweep · η=0 no-tax (C1+C2)</button>
    <button class="seg-btn"    id="smode-rho01"   onclick="setSweepMode('rho01')">ρ=0.1 secondary (C3b)</button>
    <button class="seg-btn"    id="smode-omega"   onclick="setSweepMode('omega')">ω-sweep (basis × inertia)</button>
    <button class="seg-btn"    id="smode-cells"   onclick="setSweepMode('cells')">27-cell snapshot</button>
  </div>
  <!-- Algorithm toggle retired in Phase 0 (BA path supervisor-rejected). Boltzmann only. -->
  <!-- The BA3 datasets are still wired as empty objects in JS for backward-compat. -->

  <!-- Per-mode inline controls — populated by setSweepMode -->
  <div class="ctrl-bar" id="sweep-controls" style="gap:.5rem;font-size:.65rem;min-height:1.4rem"></div>
  <!-- Mode-specific chart container — populated by setSweepMode -->
  <div id="sweep-panels"></div>
  <div class="callout" id="sweeps-note" style="font-size:.71rem">
    <b>Lehman / ρ-sweep / ρ=0.1:</b> 5-seed mean ±1 std at ω=0.50, reproducing thesis figs from claim_3.tex / claim_1.tex / claim_2.tex.
    <b>ω-sweep:</b> reuses Phase 1 / Phase 1.5 grids — multi-seed across (basis, inertia) at varying ω.
    <b>27-cell:</b> single-seed (26474) snapshot of the 3 candidates × 3 regimes × 3 ηs grid.
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════════ -->
<!-- PANEL: BANK DETAIL -->
<!-- ══════════════════════════════════════════════════════════════════════════ -->
<div id="panel-bank" class="panel">
  <div class="ctrl-bar">
    <div class="ctrl-group">
      <span class="ctrl-label">Viewing</span>
      <span class="single-run-label" id="bank-run"></span>
    </div>
    <div class="ctrl-group">
      <span class="ctrl-label">Bank ID</span>
      <select id="bank-id" onchange="renderBank()"></select>
    </div>
  </div>

  <div class="metric-chips" id="bank-metric-chips"></div>

  <div class="chart-box">
    <div class="chart-title" id="bank-chart-title">Per-bank time series</div>
    <div id="chart-bank" style="height:420px"></div>
  </div>

  <div class="callout" id="bank-note" style="margin-top:.6rem;font-size:.71rem">
    Tracks one bank by ID across all T periods: equity, fitness (Φ), survival prob. (p_j), loan received, clients served, and hub flag.
    Bank IDs are persistent within a run; replacement banks inherit the original ID.
    <br><em id="bank-data-note" style="color:#c07070"></em>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════════ -->
<!-- SCRIPT -->
<!-- ══════════════════════════════════════════════════════════════════════════ -->
<script>
// ── Data ──────────────────────────────────────────────────────────────────────
""" + f"""const DATA = {JS_DATA};
const SUMM = {JS_SUMM};
// Lazy-loaded per cand on Bank Detail tab open. Requires HTTP server (fetch).
const BANK_DATA = {{}};
const BANK_DATA_AVAILABLE = {JS_BANK_DATA_AVAILABLE_LIST};
const BANK_DATA_LOADING = {{}};   // run -> Promise (de-dup concurrent loads)
async function loadBankDataIfNeeded(run) {{
  if (BANK_DATA[run] !== undefined) return BANK_DATA[run];
  if (!BANK_DATA_AVAILABLE.includes(run)) {{
    BANK_DATA[run] = []; return [];
  }}
  if (BANK_DATA_LOADING[run]) return BANK_DATA_LOADING[run];
  BANK_DATA_LOADING[run] = (async () => {{
    try {{
      // cache: 'no-store' bypasses conditional 304 dance — Python's http.server
      // returns 304 with empty body which made fetch().json() throw.
      // Once loaded, BANK_DATA[run] caches in JS memory so per-tab repeats are free.
      const r = await fetch(`dashboard_bank_${{run}}.json`, {{ cache: 'no-store' }});
      if (!r.ok) throw new Error(`HTTP ${{r.status}}`);
      BANK_DATA[run] = await r.json();
    }} catch (e) {{
      console.warn(`Failed to load bank data for ${{run}}:`, e);
      BANK_DATA[run] = [];
    }}
    return BANK_DATA[run];
  }})();
  return BANK_DATA_LOADING[run];
}}
const SWEEP_DATA = {JS_SWEEP_DATA};
const LEHMAN_DATA      = {JS_LEHMAN_DATA};
const RHO_DATA         = {JS_RHO_DATA};
const RHO01_DATA       = {JS_RHO01_DATA};
const LEHMAN_DATA_BA3  = {JS_LEHMAN_DATA_BA3};
const RHO_DATA_BA3     = {JS_RHO_DATA_BA3};
const RHO01_DATA_BA3   = {JS_RHO01_DATA_BA3};
// Algorithm toggle for Sweeps modes — switches dataset family, never overlays
let currentSweepAlgo = 'boltz';   // 'boltz' | 'ba3'
function _algoMap(boltz, ba3) {{
  return (currentSweepAlgo === 'ba3') ? ba3 : boltz;
}}
function setSweepAlgo(a) {{
  currentSweepAlgo = a;
  document.querySelectorAll('.sweep-algo-btn').forEach(btn => {{
    btn.classList.toggle('on', btn.dataset.algo === a);
  }});
  if (typeof renderSweeps === 'function') renderSweeps();
}}
const TOPOLOGY_DATA = {JS_TOPOLOGY_DATA};
const ROBUSTNESS_DATA = {JS_ROBUSTNESS_DATA};
const RUNS    = {json.dumps(RUNS)};
const COLORS  = {json.dumps(COLS_JS)};
const LABELS  = {json.dumps(LABELS_JS)};
const CAND_TAGS    = {json.dumps(CAND_TAGS)};
const REGIME_TAGS  = {json.dumps(REGIME_TAGS)};
const ETA_TAGS     = {json.dumps(ETA_TAGS)};
const CAND_LABELS    = {json.dumps(CAND_LABELS)};
const REGIME_LABELS  = {json.dumps(REGIME_LABELS)};
const ETA_LABELS     = {json.dumps(ETA_LABELS)};
const CAND_COLORS    = {json.dumps(CAND_COLORS)};
const REGIME_COLORS  = {json.dumps(REGIME_COLORS)};
const ETA_COLORS     = {json.dumps(ETA_COLORS)};
""" + r"""

// RUNS, COLORS, LABELS, CAND_TAGS, REGIME_TAGS, ETA_TAGS, etc. are
// interpolated from Python at the f-string above. Do not redefine.
const SIG_LABELS = {
  time:'Period',
  bankruptcy:'Total bankruptcies',
  bankruptcies_shock:'Shock deaths',
  bankruptcies_rationing:'Rationing deaths',
  bankruptcies_repay:'Repayment deaths',
  bankruptcies_contagion:'Contagion deaths',
  bankruptcies_fiscal:'Fiscal cascade deaths',
  fire_sale_survivors:'Zombie count',
  equity:'System equity',
  B:'Bad debt (B)',
  rationing:'Credit rationing',
  bailout_bill:'Bailout bill',
  bailout_count:'Bailout events',
  fitness:'Avg fitness (Φ)',
  num_banks:'Active banks',
  loans:'Avg loans',
  interest_rate:'Avg interest rate (TBTF-influenced via eq. 6)',
  avg_eq:'Avg equity per bank',
  fisc_mult:'Fiscal multiplier',
  npl:'NPL ratio',
  rat_mort:'Rationing mortality',
  best_lender_clients:'Hub client count',
  best_lender:'Hub bank ID',
  best_lender_fitness:'Hub fitness (Φ)',
  best_lender_equity:'Hub equity',
  equity_lenders:'Avg lender equity',
  prob_bankruptcy:'Avg default prob (p_j)',
  num_loans:'Loan count',
  hub_share:'Hub market share',
  hub_id_norm:'Hub ID (normalized)',
  // Borrower-side TBTF reframe (§1.6)
  top_A_bank:'Top borrower (largest A) bank ID',
  top_A_value:'Top borrower asset value',
  best_lender_generation:'Hub generation (composite-key turnover)',
  previous_hub_alive:'Previous hub alive (1=fitness churn, 0=mortality)',
};

// ── State ─────────────────────────────────────────────────────────────────────
// Chip-driven focus is the single source of truth. Multi-run views default to the
// "compare row": at the chip-focused (regime, eta), show all 3 candidates.
let currentFocus = { cand: 'a', regime: 'st', eta: 'e01' };
function focusTag() { return currentFocus.cand + '_' + currentFocus.regime + '_' + currentFocus.eta; }
function compareRow() {
  return CAND_TAGS.map(c => c + '_' + currentFocus.regime + '_' + currentFocus.eta);
}

let crisisOn   = true;
let crisisFOn  = true;
let activeDecomp = focusTag();
// Phase 0: overview multi-select state. null = auto-follow chip's compareRow.
// Set to a Set<tag> when user manually toggles cells in the popover.
let overviewSelected = null;

// ── Plotly base layout ────────────────────────────────────────────────────────
const DL = {
  paper_bgcolor:'#161616', plot_bgcolor:'#161616',
  font:{family:'"IBM Plex Mono",monospace', color:'#e2ddd6', size:10},
  margin:{l:52, r:14, t:28, b:44},
  xaxis:{gridcolor:'#222', linecolor:'#333', tickfont:{size:9}},
  yaxis:{gridcolor:'#222', linecolor:'#333', tickfont:{size:9}},
  legend:{bgcolor:'#1e1e1e', bordercolor:'#333', borderwidth:1, font:{size:9}},
  hovermode:'x unified',
  hoverlabel:{bgcolor:'#1e1e1e', bordercolor:'#333', font:{family:'"IBM Plex Mono",monospace', size:10}},
};
const CFG = {responsive:true, displayModeBar:true, modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d'],
             displaylogo:false};

function mkLayout(extra){
  return Object.assign({}, DL, extra,
    {xaxis: Object.assign({},DL.xaxis, extra && extra.xaxis),
     yaxis: Object.assign({},DL.yaxis, extra && extra.yaxis)});
}

// Phase 0: Plotly legend-toggle persistence helper. When a user clicks a legend
// item to hide a trace ("legendonly"), then triggers a re-render (metric switch,
// chip change, etc.), Plotly.react rebuilds traces fresh and the visibility state
// is lost. pReact/pNewPlot capture the old visibility by trace name and re-apply
// it to the new traces before delegating to Plotly. Trace identity is matched by
// `name` (which is what shows up in the legend).
function _captureVis(divId) {
  const div = document.getElementById(divId);
  if (!div || !div.data) return null;
  const map = {};
  div.data.forEach(t => {
    if (t.name !== undefined && t.visible !== undefined) map[t.name] = t.visible;
  });
  return map;
}
function _applyVis(traces, vis) {
  if (!vis) return traces;
  return traces.map(t => {
    if (t.name !== undefined && vis[t.name] !== undefined) {
      return Object.assign({}, t, { visible: vis[t.name] });
    }
    return t;
  });
}
function pReact(divId, traces, layout, cfg) {
  const vis = _captureVis(divId);
  return Plotly.react(divId, _applyVis(traces, vis), layout, cfg);
}
function pNewPlot(divId, traces, layout, cfg) {
  const vis = _captureVis(divId);
  return Plotly.newPlot(divId, _applyVis(traces, vis), layout, cfg);
}

// ── Utility ───────────────────────────────────────────────────────────────────
function ma(arr, w) {
  if (w <= 1) return arr;
  return arr.map((_, i) => {
    const s = Math.max(0, i-w+1), e = i+1;
    const sub = arr.slice(s, e).filter(v => v !== null);
    return sub.length ? sub.reduce((a,b)=>a+b,0)/sub.length : null;
  });
}

// activeRunsList → compareRow alias for backwards-compat with multi-run renderers
function activeRunsList() { return compareRow(); }

function crisisShape() {
  // Crisis-mark stripped in Phase 0: was bsl-specific (equity erosion at t~199);
  // doesn't generalise to c3/w58 cands. Toggle buttons remain as no-ops for HTML compat.
  return [];
  // legacy: [{type:'rect', xref:'x', yref:'paper', x0:195, x1:210,
  //  y0:0, y1:1, fillcolor:'rgba(200,60,60,0.07)', line:{width:0}}];
}

// ── Tab switching ─────────────────────────────────────────────────────────────
function showTab(name) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('panel-'+name).classList.add('active');
  event.currentTarget.classList.add('active');
  if (name === 'overview')    renderOverview();
  if (name === 'sweeps')      { if (!currentSweepMode) setSweepMode('lehman'); else renderSweeps(); }
  if (name === 'timeseries')  renderTS();
  if (name === 'decomp')      { renderDecomp(); renderDecompShare(); }
  if (name === 'regcmp')      renderRegCmp();
  if (name === 'fiscal')      renderFiscal();
  if (name === 'ccf')         renderCCF();
  if (name === 'overlay')     renderOverlay();
  if (name === 'hub')         renderHub();
  if (name === 'bank')        { (async () => { await buildBankIdSelect(); await renderBank(); })(); }
}

// ── Run chips ─────────────────────────────────────────────────────────────────
// ── Dimension UI: chips (focus) + pill rows (filter) ────────────────────────
const DIM_TAGS    = { cand: CAND_TAGS,    regime: REGIME_TAGS,    eta: ETA_TAGS };
const DIM_LABELS  = { cand: CAND_LABELS,  regime: REGIME_LABELS,  eta: ETA_LABELS };
const DIM_COLORS  = { cand: CAND_COLORS,  regime: REGIME_COLORS,  eta: ETA_COLORS };

function updateChipDisplay() {
  for (const dim of ['cand', 'regime', 'eta']) {
    const tag = currentFocus[dim];
    const el  = document.getElementById('chip-' + dim);
    const color = DIM_COLORS[dim][tag];
    el.innerHTML = (dim === 'cand'
      ? `<span class="cv-dot" style="background:${color}"></span>`
      : '') + DIM_LABELS[dim][tag];
  }
}
function setFocus(dim, tag) {
  currentFocus[dim] = tag;
  updateChipDisplay();
  closeChipMenu();
  // Reset overview manual selection so it auto-follows the new chip's compareRow.
  overviewSelected = null;
  repaintSingleRunPanels();
  repaintMultiRunPanels();   // compareRow() depends on currentFocus.regime + currentFocus.eta
  refreshOverviewCount();
}
function openChipMenu(dim, anchorEl, ev) {
  if (ev) ev.stopPropagation();
  const pop = document.getElementById('cv-popover');
  pop.innerHTML = '';
  for (const tag of DIM_TAGS[dim]) {
    const btn = document.createElement('button');
    btn.textContent = DIM_LABELS[dim][tag];
    if (tag === currentFocus[dim]) btn.classList.add('current');
    btn.onclick = (e) => { e.stopPropagation(); setFocus(dim, tag); };
    pop.appendChild(btn);
  }
  // Position below the anchor
  const r = anchorEl.getBoundingClientRect();
  pop.style.top  = (r.bottom + window.scrollY + 4) + 'px';
  pop.style.left = (r.left + window.scrollX) + 'px';
  pop.classList.add('show');
  setTimeout(() => document.addEventListener('click', closeChipMenu, { once: true }), 0);
}
function closeChipMenu() {
  document.getElementById('cv-popover').classList.remove('show');
}

// buildPillRows / togglePill removed — chips are the only navigation now.

function repaintSingleRunPanels() {
  const panel = document.querySelector('.panel.active');
  if (!panel) return;
  const id = panel.id.replace('panel-', '');
  if (id === 'hub')     renderHub();
  if (id === 'overlay') renderOverlay();
  if (id === 'bank')    { (async () => { await buildBankIdSelect(); await renderBank(); })(); }
  if (id === 'sweeps')  renderSweeps();   // some sub-views read currentFocus
}
function repaintMultiRunPanels() {
  const panel = document.querySelector('.panel.active');
  if (!panel) return;
  const id = panel.id.replace('panel-', '');
  if (id === 'overview')   renderOverview();
  if (id === 'timeseries') renderTS();
  // Sync activeDecomp to chip on chip change so the displayed cell follows the chip.
  // User can still override by clicking a different decomp pill.
  if (id === 'decomp')     { activeDecomp = focusTag(); renderDecomp(); renderDecompShare();
                             const sel = document.getElementById('decomp-select'); if (sel) sel.value = activeDecomp; }
  if (id === 'regcmp')     renderRegCmp();
  if (id === 'fiscal')     renderFiscal();
  if (id === 'ccf')        renderCCF();
  if (id === 'sweeps')     renderSweeps();
}
// Backwards-compat for any code that still calls these
function refreshActive() { repaintSingleRunPanels(); repaintMultiRunPanels(); }
function toggleRun(r)    { /* legacy single-run toggle; no-op under new model */ }

// ── OVERVIEW ──────────────────────────────────────────────────────────────────
// Phase 0: 72-card grid → multi-select dropdown. Default = chip's compareRow (8 cells
// at chip's regime/eta across all 8 cands). User can toggle any of the 72 cells via
// the popover. Setting to null auto-follows the chip; manual selection persists until
// chip change (which resets to null per setFocus()).

function getOverviewCells() {
  if (overviewSelected !== null) return [...overviewSelected].filter(t => RUNS.includes(t));
  return compareRow().filter(t => RUNS.includes(t));
}
function refreshOverviewCount() {
  const el = document.getElementById('ov-cells-count');
  if (el) el.textContent = String(getOverviewCells().length);
}
function ovResetToRow() {
  overviewSelected = null;
  refreshOverviewCount();
  renderOverview();
  // Re-render popover if open so checkboxes reflect new state
  const pop = document.getElementById('ov-cells-popover');
  if (pop && pop.classList.contains('show')) buildOverviewMenu();
}
function ovSelectAll() {
  overviewSelected = new Set(RUNS);
  refreshOverviewCount();
  renderOverview();
  const pop = document.getElementById('ov-cells-popover');
  if (pop && pop.classList.contains('show')) buildOverviewMenu();
}
function ovSelectNone() {
  overviewSelected = new Set();
  refreshOverviewCount();
  renderOverview();
  const pop = document.getElementById('ov-cells-popover');
  if (pop && pop.classList.contains('show')) buildOverviewMenu();
}
function buildOverviewMenu() {
  const pop = document.getElementById('ov-cells-popover');
  if (!pop) return;
  const sel = new Set(getOverviewCells());
  const html = [];
  CAND_TAGS.forEach(c => {
    html.push(`<div class="ov-cand-hdr" style="color:${CAND_COLORS[c]}">${CAND_LABELS[c]}</div>`);
    REGIME_TAGS.forEach(rt => {
      ETA_TAGS.forEach(et => {
        const tag = `${c}_${rt}_${et}`;
        if (!RUNS.includes(tag)) return;
        const checked = sel.has(tag) ? 'checked' : '';
        html.push(
          `<label><input type="checkbox" ${checked} onchange="ovToggleCell('${tag}', this.checked)">` +
          `<span>${REGIME_LABELS[rt]} · ${ETA_LABELS[et]}</span></label>`);
      });
    });
  });
  pop.innerHTML = html.join('');
}
function ovToggleCell(tag, on) {
  // Materialize overviewSelected from current effective set on first manual edit
  if (overviewSelected === null) overviewSelected = new Set(getOverviewCells());
  if (on) overviewSelected.add(tag);
  else    overviewSelected.delete(tag);
  refreshOverviewCount();
  renderOverview();
}
function openOverviewMenu(anchor, ev) {
  if (ev) ev.stopPropagation();
  const pop = document.getElementById('ov-cells-popover');
  if (!pop) return;
  buildOverviewMenu();
  const r = anchor.getBoundingClientRect();
  pop.style.top  = (r.bottom + window.scrollY + 4) + 'px';
  pop.style.left = (r.left + window.scrollX) + 'px';
  pop.classList.add('show');
  setTimeout(() => document.addEventListener('click', closeOverviewMenu, { once: true }), 0);
}
function closeOverviewMenu(ev) {
  const pop = document.getElementById('ov-cells-popover');
  if (!pop) return;
  // Don't close when clicking inside the popover (so checkbox toggles work)
  if (ev && pop.contains(ev.target)) {
    setTimeout(() => document.addEventListener('click', closeOverviewMenu, { once: true }), 0);
    return;
  }
  pop.classList.remove('show');
}

function renderOverview() {
  // Stat cards — only show selected cells (default = chip's compareRow).
  refreshOverviewCount();
  const cells = getOverviewCells();
  const row = document.getElementById('stat-row');
  row.innerHTML = '';
  const tots = cells.map(r => SUMM[r] ? SUMM[r].tot_bkr : 0);
  const minT = Math.min(...tots), maxT = Math.max(...tots);
  cells.forEach(r => {
    const s = SUMM[r]; if (!s) return;
    const cls = s.tot_bkr === minT ? ' lo' : s.tot_bkr === maxT ? ' hi' : '';
    row.innerHTML += `
    <div class="stat-card${cls}">
      <div class="run-stripe" style="background:${COLORS[r]}"></div>
      <div class="run-lbl" style="color:${COLORS[r]};margin-top:.4rem">${LABELS[r]}</div>
      <div class="big">${s.tot_bkr.toLocaleString()}</div>
      <div class="detail">total deaths</div>
      <div class="detail" style="margin-top:.3rem">fiscal: ${s.tot_fisc.toLocaleString()} &nbsp;|&nbsp; bill: ${s.tot_bill.toLocaleString()}</div>
    </div>`;
  });

  // Radar — 6 metrics, normalized over selected cells
  const metrics = ['tot_bkr','tot_fisc','tot_cntg','mean_zomb','tot_bill','std_eq'];
  const mLabels = ['Total deaths','Fiscal deaths','Contagion<br>deaths','Avg zombies','Bailout cost','Equity<br>volatility'];
  const ranges  = metrics.map(m => {
    const vals = cells.map(r => SUMM[r] ? SUMM[r][m] : 0);
    return [Math.min(...vals), Math.max(...vals)];
  });
  const radarTraces = cells.map(r => {
    const vals = metrics.map((m,i) => {
      const [lo,hi] = ranges[i];
      return hi>lo ? (SUMM[r][m]-lo)/(hi-lo) : 0.5;
    });
    vals.push(vals[0]); // close polygon
    const lbls = [...mLabels, mLabels[0]];
    return {type:'scatterpolar', r:vals, theta:lbls, fill:'toself',
      name:LABELS[r], line:{color:COLORS[r], width:1.5},
      fillcolor:COLORS[r].replace(')',',0.08)').replace('rgb','rgba')
        .replace('#e74c3c','rgba(231,76,60,0.08)').replace('#3498db','rgba(52,152,219,0.08)')
        .replace('#9b59b6','rgba(155,89,182,0.08)').replace('#27ae60','rgba(39,174,96,0.08)')
        .replace('#1abc9c','rgba(26,188,156,0.08)').replace('#f39c12','rgba(243,156,18,0.08)'),
      opacity: 0.9};
  });
  const radarLayout = {
    paper_bgcolor:'#161616', polar:{
      bgcolor:'#161616',
      radialaxis:{visible:true, range:[0,1.05], tickfont:{size:8}, gridcolor:'#2a2a2a', linecolor:'#333'},
      angularaxis:{tickfont:{size:8.5}, gridcolor:'#2a2a2a', linecolor:'#333'}},
    font:{family:'"IBM Plex Mono",monospace', color:'#e2ddd6', size:9},
    legend:{bgcolor:'#1e1e1e', bordercolor:'#333', borderwidth:1, font:{size:9}},
    margin:{l:40,r:40,t:30,b:40}, showlegend:true
  };
  pReact('chart-radar', radarTraces, radarLayout, CFG);

  // Cumulative bar chart — uses the same selected cells as stat cards.
  const decomp_keys = ['tot_shock','tot_rat','tot_repay','tot_cntg','tot_fisc'];
  const decomp_lbls = ['Shock','Rationing','Repayment','Contagion','Fiscal'];
  const dcols = ['#6b7280','#d97706','#06b6d4','#8b5cf6','#ef4444'];
  const bkrTraces = decomp_keys.map((k,i) => ({
    type:'bar', name:decomp_lbls[i],
    x: cells.map(r => LABELS[r]),
    y: cells.map(r => SUMM[r] ? SUMM[r][k] : 0),
    marker:{color:dcols[i]}, opacity:0.88,
  }));
  pReact('chart-bkr-bar', bkrTraces, mkLayout({
    barmode:'stack',
    xaxis:{tickfont:{size:8.5}, automargin:true},
    yaxis:{title:{text:'Deaths', font:{size:9}}},
    legend:{traceorder:'normal'},
    margin:{l:52,r:14,t:10,b:80},
  }), CFG);

  // Scatter: bill vs fiscal deaths (bubble = mean zombies) — same selected cells.
  const scatterTrace = {
    type:'scatter', mode:'markers+text',
    x: cells.map(r => SUMM[r] ? SUMM[r].tot_bill : 0),
    y: cells.map(r => SUMM[r] ? SUMM[r].tot_fisc : 0),
    text: cells.map(r => LABELS[r]),
    textposition: 'top center',
    textfont:{size:8},
    marker:{
      size: cells.map(r => 8 + (SUMM[r] ? SUMM[r].mean_zomb : 0)/8),
      color: cells.map(r => COLORS[r]),
      line:{color:'#333', width:1},
    },
    hovertemplate: '<b>%{text}</b><br>Bill: %{x:,.0f}<br>Fiscal deaths: %{y:,}<extra></extra>',
  };
  Plotly.react('chart-scatter', [scatterTrace], mkLayout({
    xaxis:{title:{text:'Cumulative bailout bill', font:{size:9}}},
    yaxis:{title:{text:'Cumulative fiscal deaths', font:{size:9}}},
    margin:{l:60,r:14,t:14,b:52},
    hovermode:'closest',
  }), CFG);
}

// ── TIME SERIES ───────────────────────────────────────────────────────────────
function buildVarSelect() {
  const sel = document.getElementById('var-select');
  // Phase 0: aligned with Mix & Match (full SIG_LABELS catalogue minus 'time').
  // Previously hardcoded 17 metrics; now exposes all 33 series including hub-tracker
  // metrics (best_lender, best_lender_fitness, top_A_bank, etc.) and interest_rate.
  const vars = Object.keys(SIG_LABELS).filter(k => k !== 'time');
  // Default selection: keep 'bankruptcy' as the headline metric.
  vars.forEach(v => {
    const o = document.createElement('option');
    o.value = v; o.text = SIG_LABELS[v] || v;
    if (v === 'bankruptcy') o.selected = true;
    sel.appendChild(o);
  });
}

function getSmooth() { return parseInt(document.getElementById('smooth-select').value); }

function renderTS() {
  const varKey = document.getElementById('var-select').value;
  const w = getSmooth();
  const runs = activeRunsList();
  document.getElementById('ts-title').textContent = (SIG_LABELS[varKey]||varKey) + ' — all active runs';

  const traces = runs.map(r => ({
    x: DATA[r].time,
    y: ma(DATA[r][varKey]||[], w),
    name: LABELS[r],
    type:'scatter', mode:'lines',
    line:{color:COLORS[r], width:1.8},
  }));

  const layout = mkLayout({
    shapes: [],   // crisis annotation stripped in Phase 0 (was bsl-specific)
    annotations: [],
    margin:{l:55,r:14,t:14,b:44},
    xaxis:{title:{text:'Period', font:{size:9}}},
    yaxis:{title:{text:SIG_LABELS[varKey]||varKey, font:{size:9}}},
  });
  pReact('chart-ts', traces, layout, CFG);

  // Diff vs baseline (high-η baseline cell at ω=0.50, socialized)
  const baseTag = 'bsl_st_e085';
  const base = (DATA[baseTag] && DATA[baseTag][varKey]) || [];
  const diffTraces = runs.filter(r=>r!==baseTag).map(r => {
    const ser = DATA[r][varKey] || [];
    const diff = ser.map((v,i) => (v!==null && base[i]!==null) ? v-base[i] : null);
    return {
      x: DATA[r].time, y: ma(diff, w), name: LABELS[r],
      type:'scatter', mode:'lines', line:{color:COLORS[r], width:1.5},
    };
  });
  const zeroLine = {x:[0,999], y:[0,0], type:'scatter', mode:'lines',
    line:{color:'#444', width:1, dash:'dot'}, showlegend:false};
  pReact('chart-diff', [zeroLine, ...diffTraces], mkLayout({
    shapes: crisisOn ? crisisShape() : [],
    margin:{l:55,r:14,t:10,b:44},
    xaxis:{title:{text:'Period', font:{size:9}}},
    yaxis:{title:{text:'Δ vs η=0.85 social', font:{size:9}}},
  }), CFG);
}

function toggleCrisis() {
  crisisOn = !crisisOn;
  document.getElementById('btn-crisis').classList.toggle('on', crisisOn);
  renderTS();
}

// ── DECOMPOSITION ─────────────────────────────────────────────────────────────
const DECOMP_CATS = [
  {key:'bankruptcies_shock',     color:'#6b7280', name:'Shock'},
  {key:'bankruptcies_rationing', color:'#d97706', name:'Rationing'},
  {key:'bankruptcies_repay',     color:'#06b6d4', name:'Repayment'},
  {key:'bankruptcies_contagion', color:'#8b5cf6', name:'Contagion'},
  {key:'bankruptcies_fiscal',    color:'#ef4444', name:'Fiscal'},
];

function buildDecompTabs() {
  // Phase 0: dropdown replaces the 72-pill grid.
  // Options grouped by cand for readability via <optgroup>.
  const sel = document.getElementById('decomp-select');
  if (!sel) return;
  sel.innerHTML = '';
  CAND_TAGS.forEach(c => {
    const grp = document.createElement('optgroup');
    grp.label = CAND_LABELS[c];
    REGIME_TAGS.forEach(rt => {
      ETA_TAGS.forEach(et => {
        const tag = `${c}_${rt}_${et}`;
        if (!RUNS.includes(tag)) return;
        const o = document.createElement('option');
        o.value = tag;
        o.textContent = `${REGIME_LABELS[rt]} · ${ETA_LABELS[et]}`;
        if (tag === activeDecomp) o.selected = true;
        grp.appendChild(o);
      });
    });
    sel.appendChild(grp);
  });
}
function onDecompSelect(tag) {
  activeDecomp = tag;
  renderDecomp();
  renderDecompShare();
}

function renderDecomp() {
  const r = activeDecomp;
  document.getElementById('decomp-title').textContent = 'Bankruptcy decomposition — ' + LABELS[r];
  const t = DATA[r].time;
  const traces = DECOMP_CATS.map(cat => ({
    x: t, y: DATA[r][cat.key],
    name: cat.name, stackgroup:'one',
    fillcolor: cat.color+'cc',
    line:{color:cat.color, width:0.5},
    type:'scatter', mode:'none',
    hovertemplate:'%{y:.0f}<extra>'+cat.name+'</extra>',
  }));
  pReact('chart-decomp', traces, mkLayout({
    margin:{l:52,r:14,t:10,b:44},
    xaxis:{title:{text:'Period', font:{size:9}}},
    yaxis:{title:{text:'Deaths', font:{size:9}}},
  }), CFG);
}

function renderDecompShare() {
  const runs = activeRunsList();
  const total = runs.map(r => SUMM[r].tot_bkr);
  const traces = DECOMP_CATS.map(cat => {
    const keyMap = {
      bankruptcies_shock:'tot_shock', bankruptcies_rationing:'tot_rat',
      bankruptcies_repay:'tot_repay', bankruptcies_contagion:'tot_cntg',
      bankruptcies_fiscal:'tot_fisc'
    };
    return {
      type:'bar', name:cat.name,
      x: runs.map(r=>LABELS[r]),
      y: runs.map((r,i) => total[i] > 0 ? SUMM[r][keyMap[cat.key]]/total[i]*100 : 0),
      marker:{color:cat.color}, opacity:0.88,
    };
  });
  pReact('chart-decomp-share', traces, mkLayout({
    barmode:'stack',
    xaxis:{tickfont:{size:8.5}, automargin:true},
    yaxis:{title:{text:'Share (%)', font:{size:9}}, range:[0,100]},
    margin:{l:52,r:14,t:10,b:80},
  }), CFG);
}

// ── REGIME CMP (Phase 0) ──────────────────────────────────────────────────────
// Pivots TS's "compare cands at one regime" into "compare 3 regimes at one cand".
// Plus the professor's stacked centrality view (lender ID + clients + fitness)
// across the 3 regimes — the metrics he specifically asked to see together.
function buildRegCmpVarSelect() {
  const sel = document.getElementById('regcmp-var');
  if (!sel) return;
  sel.innerHTML = '';
  const vars = Object.keys(SIG_LABELS).filter(k => k !== 'time');
  vars.forEach(v => {
    const o = document.createElement('option');
    o.value = v; o.text = SIG_LABELS[v] || v;
    if (v === 'bankruptcy') o.selected = true;
    sel.appendChild(o);
  });
}

const REGIME_TRACE_COLORS = {nt: '#3498db', st: '#e74c3c', rf: '#27ae60'};

function renderRegCmp() {
  const cand = currentFocus.cand;
  const eta  = currentFocus.eta;
  const candEl = document.getElementById('regcmp-cand');
  const etaEl  = document.getElementById('regcmp-eta');
  if (candEl) candEl.textContent = CAND_LABELS[cand];
  if (etaEl)  etaEl.textContent  = ETA_LABELS[eta];

  const varSel = document.getElementById('regcmp-var');
  const smSel  = document.getElementById('regcmp-smooth');
  if (!varSel) return;
  const varKey = varSel.value || 'bankruptcy';
  const w = parseInt((smSel && smSel.value) || '10');

  // Top chart: chosen metric across 3 regimes
  document.getElementById('regcmp-title').textContent =
    (SIG_LABELS[varKey] || varKey) + ` — ${CAND_LABELS[cand]} · ${ETA_LABELS[eta]}`;
  const topTraces = REGIME_TAGS.map(rt => {
    const tag = `${cand}_${rt}_${eta}`;
    const series = (DATA[tag] && DATA[tag][varKey]) || [];
    const t = (DATA[tag] && DATA[tag].time) || [];
    return {
      x: t, y: ma(series, w),
      name: REGIME_LABELS[rt],
      type: 'scatter', mode: 'lines',
      line: {color: REGIME_TRACE_COLORS[rt], width: 1.8},
    };
  });
  pReact('chart-regcmp', topTraces, mkLayout({
    margin: {l: 55, r: 14, t: 14, b: 44},
    xaxis: {title: {text: 'Period', font: {size: 9}}},
    yaxis: {title: {text: SIG_LABELS[varKey] || varKey, font: {size: 9}}},
  }), CFG);

  // Bottom: 3 stacked sub-charts of the centrality triplet (id, clients, fitness),
  // each overlaying the 3 regimes.
  const triplet = [
    {key: 'best_lender',          divId: 'chart-regcmp-id',  title: 'Hub bank ID',           shape: 'hv', smooth: false},
    {key: 'best_lender_clients',  divId: 'chart-regcmp-cli', title: 'Hub client count',     shape: 'linear', smooth: true},
    {key: 'best_lender_fitness',  divId: 'chart-regcmp-fit', title: 'Hub fitness (Φ)',      shape: 'linear', smooth: true},
  ];
  triplet.forEach(({key, divId, title, shape, smooth}) => {
    const traces = REGIME_TAGS.map(rt => {
      const tag = `${cand}_${rt}_${eta}`;
      const series = (DATA[tag] && DATA[tag][key]) || [];
      const t = (DATA[tag] && DATA[tag].time) || [];
      return {
        x: t,
        y: smooth ? ma(series, w) : series,
        name: REGIME_LABELS[rt],
        type: 'scatter', mode: 'lines',
        line: {color: REGIME_TRACE_COLORS[rt], width: 1.4, shape},
      };
    });
    pReact(divId, traces, mkLayout({
      margin: {l: 55, r: 14, t: 22, b: 36},
      title: {text: title, font: {size: 10, color: '#bbb'}, x: 0, xanchor: 'left'},
      xaxis: {title: {text: 'Period', font: {size: 9}}},
      yaxis: {title: {text: SIG_LABELS[key] || key, font: {size: 9}}},
    }), CFG);
  });
}

// ── FISCAL ────────────────────────────────────────────────────────────────────
function renderFiscal() {
  const runs = activeRunsList();
  const clip = parseInt(document.getElementById('bill-clip').value);
  const shapes = crisisFOn ? crisisShape() : [];

  const billTraces = runs.map(r => {
    let y = DATA[r].bailout_bill || [];
    if (clip > 0) y = y.map(v => v===null?null:Math.min(v,clip));
    return {x:DATA[r].time, y, name:LABELS[r], type:'scatter', mode:'lines',
      line:{color:COLORS[r], width:1.5}};
  });
  pReact('chart-bill', billTraces, mkLayout({
    shapes, margin:{l:52,r:14,t:10,b:44},
    xaxis:{title:{text:'Period', font:{size:9}}},
    yaxis:{title:{text:'Bill'+(clip?` (clip ${clip})`:'')+' per period', font:{size:9}}},
  }), CFG);

  const fiscTraces = runs.map(r => ({
    x:DATA[r].time, y:DATA[r].bankruptcies_fiscal||[],
    name:LABELS[r], type:'scatter', mode:'lines', line:{color:COLORS[r], width:1.5},
  }));
  pReact('chart-fisc-deaths', fiscTraces, mkLayout({
    shapes, margin:{l:52,r:14,t:10,b:44},
    xaxis:{title:{text:'Period', font:{size:9}}},
    yaxis:{title:{text:'Fiscal deaths', font:{size:9}}},
  }), CFG);

  const multTraces = runs.map(r => {
    const y = (DATA[r].fisc_mult||[]).map(v => v===null?null:Math.min(v,50));
    return {x:DATA[r].time, y, name:LABELS[r], type:'scatter', mode:'lines',
      line:{color:COLORS[r], width:1.5}};
  });
  pReact('chart-fisc-mult', multTraces, mkLayout({
    shapes, margin:{l:52,r:14,t:10,b:44},
    xaxis:{title:{text:'Period', font:{size:9}}},
    yaxis:{title:{text:'Multiplier (clip 50)', font:{size:9}}},
  }), CFG);
}

function toggleCrisisF() {
  crisisFOn = !crisisFOn;
  document.getElementById('btn-crisis-f').classList.toggle('on', crisisFOn);
  renderFiscal();
}

// ── CCF EXPLORER ──────────────────────────────────────────────────────────────
function buildCCFSelects() {
  const ccf_vars = ['fire_sale_survivors','rationing','B',
    'bankruptcies_fiscal','bankruptcies_contagion','bankruptcies_rationing',
    'bailout_bill','equity','fitness','num_banks','avg_eq'];
  const defaults = {x:'fire_sale_survivors', y:'rationing'};
  ['ccf-x','ccf-y'].forEach(id => {
    const sel = document.getElementById(id);
    ccf_vars.forEach(v => {
      const o = document.createElement('option');
      o.value=v; o.text=SIG_LABELS[v]||v;
      if (v===defaults[id.split('-')[1]]) o.selected=true;
      sel.appendChild(o);
    });
  });
}

function computeCCF(x, y, maxLag) {
  const n = x.length;
  const xm = x.reduce((a,b)=>a+(b||0),0)/n;
  const ym = y.reduce((a,b)=>a+(b||0),0)/n;
  const xs = Math.sqrt(x.reduce((a,v)=>a+((v||0)-xm)**2,0)/n)||1;
  const ys = Math.sqrt(y.reduce((a,v)=>a+((v||0)-ym)**2,0)/n)||1;
  const xn = x.map(v=>((v||0)-xm)/xs);
  const yn = y.map(v=>((v||0)-ym)/ys);
  const lags=[], ccf=[];
  for(let k=-maxLag; k<=maxLag; k++){
    let s=0,c=0;
    for(let t=0;t<n;t++){const t2=t+k;if(t2>=0&&t2<n){s+=xn[t]*yn[t2];c++;}}
    lags.push(k); ccf.push(c>0?s/c:0);
  }
  return {lags, ccf};
}

function renderCCF() {
  const xKey = document.getElementById('ccf-x').value;
  const yKey = document.getElementById('ccf-y').value;
  const maxLag = parseInt(document.getElementById('ccf-lag').value);
  const runs = activeRunsList();

  const traces = runs.map(r => {
    const {lags, ccf} = computeCCF(DATA[r][xKey]||[], DATA[r][yKey]||[], maxLag);
    return {x:lags, y:ccf, name:LABELS[r], type:'bar',
      marker:{color:COLORS[r], opacity:0.7}, width:0.6};
  });

  const zeroLine = {x:[-maxLag,maxLag],y:[0,0],type:'scatter',mode:'lines',
    line:{color:'#444',width:1,dash:'dot'},showlegend:false};
  const posLine  = {x:[1,1],y:[-1,1],type:'scatter',mode:'lines',
    line:{color:'rgba(243,156,18,0.4)',width:1.5,dash:'dash'},showlegend:false,
    name:'lag+1'};

  pReact('chart-ccf', [zeroLine, posLine, ...traces], mkLayout({
    barmode:'overlay',
    xaxis:{title:{text:'Lag k  (positive = X leads Y)', font:{size:9}}, dtick:5},
    yaxis:{title:{text:'Correlation', font:{size:9}}, range:[-1,1]},
    margin:{l:52,r:14,t:10,b:52},
  }), CFG);

  // Peak table
  let peaks = '<div class="leg" style="flex-wrap:wrap;gap:.4rem">';
  runs.forEach(r => {
    const {lags, ccf} = computeCCF(DATA[r][xKey]||[], DATA[r][yKey]||[], maxLag);
    const pi = ccf.reduce((bi,v,i)=>Math.abs(v)>Math.abs(ccf[bi])?i:bi, 0);
    peaks += `<div class="ccf-peak"><span style="color:${COLORS[r]}">${LABELS[r]}</span>`+
             ` &nbsp;peak lag <span>${lags[pi]>0?'+':''}${lags[pi]}</span>`+
             ` &nbsp;r = <span>${ccf[pi].toFixed(3)}</span></div>`;
  });
  peaks += '</div>';
  document.getElementById('ccf-peaks').innerHTML = peaks;
}

// ── MIX & MATCH OVERLAY ───────────────────────────────────────────────────────
const OV_COLORS = ['#e2ddd6','#f1c40f','#1abc9c','#e67e22','#9b59b6','#00bcd4','#ff7675','#a29bfe'];

let ovSlots = [
  {col:'equity',     axis:'left',  style:'solid'},
  {col:'bankruptcy', axis:'right', style:'dashed'},
];
let ovLog = {left: false, right: false};

function toggleOvLog(side) {
  ovLog[side] = !ovLog[side];
  const btn = document.getElementById('ov-log-' + side);
  btn.textContent = ovLog[side] ? 'Log' : 'Linear';
  btn.classList.toggle('on', !ovLog[side]);
  renderOverlay();
}

function buildOvRunSelect() { /* no-op — chip-driven now (label updated in renderOverlay) */ }

function addOverlaySlot() {
  if (ovSlots.length >= 8) return;
  ovSlots.push({col:'B', axis:'left', style:'solid'});
  renderOvSlots();
  renderOverlay();
}

function removeOverlaySlot(i) {
  ovSlots.splice(i, 1);
  renderOvSlots();
  renderOverlay();
}

function setOvSlot(i, key, val) {
  ovSlots[i][key] = val;
  renderOvSlots();
  renderOverlay();
}

function renderOvSlots() {
  const allVars = Object.keys(SIG_LABELS).filter(k => k !== 'time');
  const el = document.getElementById('ov-slots');
  el.innerHTML = '';
  ovSlots.forEach((slot, i) => {
    const div = document.createElement('div');
    div.className = 'ov-slot';
    const color = OV_COLORS[i % OV_COLORS.length];
    // variable dropdown
    let varOpts = allVars.map(v =>
      `<option value="${v}"${v===slot.col?' selected':''}>${SIG_LABELS[v]||v}</option>`
    ).join('');
    // axis toggle
    const axCls = slot.axis === 'right' ? 'ov-axis-btn right' : 'ov-axis-btn';
    const axLbl = slot.axis === 'right' ? 'R' : 'L';
    // style selector
    let styleOpts = ['solid','dashed','dotted'].map(s =>
      `<option value="${s}"${s===slot.style?' selected':''}>${s}</option>`
    ).join('');
    div.innerHTML =
      `<span class="ov-color-dot" style="background:${color}"></span>` +
      `<select style="flex:1" onchange="setOvSlot(${i},'col',this.value)">${varOpts}</select>` +
      `<button class="${axCls}" title="Toggle L/R axis" onclick="setOvSlot(${i},'axis',ovSlots[${i}].axis==='left'?'right':'left')">${axLbl}</button>` +
      `<select onchange="setOvSlot(${i},'style',this.value)">${styleOpts}</select>` +
      `<button class="ov-remove" onclick="removeOverlaySlot(${i})" title="Remove">×</button>`;
    el.appendChild(div);
  });
}

function renderOverlay() {
  const run = focusTag();
  const ovSel = document.getElementById('ov-run'); if (ovSel) ovSel.textContent = LABELS[run] || run;
  const w   = parseInt(document.getElementById('ov-smooth').value);
  const t   = DATA[run] ? DATA[run].time : [];
  const traces = [];
  let hasRight = false;

  ovSlots.forEach((slot, i) => {
    const raw = DATA[run] && DATA[run][slot.col] ? DATA[run][slot.col] : [];
    const y   = ma(raw, w);
    const color = OV_COLORS[i % OV_COLORS.length];
    const dash  = slot.style === 'dashed' ? 'dash' : slot.style === 'dotted' ? 'dot' : 'solid';
    if (slot.axis === 'right') hasRight = true;
    traces.push({
      x: t, y, name: SIG_LABELS[slot.col] || slot.col,
      type: 'scatter', mode: 'lines',
      line: {color, width: 1.8, dash},
      yaxis: slot.axis === 'right' ? 'y2' : 'y',
    });
  });

  const extra = {
    hovermode: 'x unified',
    margin: {l:60, r:hasRight?65:20, t:10, b:44},
    xaxis: {title:{text:'Period', font:{size:9}}},
    yaxis: {title:{text:'Left axis', font:{size:9}},
            type: ovLog.left ? 'log' : 'linear'},
  };
  if (hasRight) {
    extra.yaxis2 = {overlaying:'y', side:'right', gridcolor:'transparent',
                    title:{text:'Right axis', font:{size:9}},
                    type: ovLog.right ? 'log' : 'linear'};
  }
  Plotly.react('chart-overlay', traces, mkLayout(extra), CFG);
}

// ── HUB TRACKER ───────────────────────────────────────────────────────────────
function buildHubRunSelect() { /* no-op — chip-driven (label updated in renderHub) */ }

function rollingMode(arr, w) {
  if (w <= 1) return arr;
  return arr.map((_, i) => {
    const s = Math.max(0, i - w + 1);
    const win = arr.slice(s, i + 1).filter(v => v !== null && v >= 0);
    if (!win.length) return null;
    const counts = {};
    win.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
    return parseInt(Object.entries(counts).sort((a,b) => b[1]-a[1])[0][0]);
  });
}

function detectTurnovers(hub_ids) {
  const events = [];
  for (let t = 1; t < hub_ids.length; t++) {
    const prev = hub_ids[t-1], cur = hub_ids[t];
    if (prev !== null && cur !== null && prev !== cur)
      events.push(t);
  }
  return events;
}

function renderHub() {
  const run = focusTag();
  const hubSel = document.getElementById('hub-run'); if (hubSel) hubSel.textContent = LABELS[run] || run;
  const wm  = parseInt(document.getElementById('hub-window').value);
  const ws  = parseInt(document.getElementById('hub-smooth').value);
  const d   = DATA[run] || {};
  const t   = d.time || [];

  const hub_id  = rollingMode(d.best_lender || [], wm);
  const hub_fit = ma(d.best_lender_fitness || [], ws);
  const hub_eq  = ma(d.best_lender_equity  || [], ws);
  const hub_cli = d.best_lender_clients    || [];

  const hasData = hub_id.length > 0 && hub_id.some(v => v !== null && v >= 0);

  // Data note
  document.getElementById('hub-data-note').textContent = hasData
    ? ''
    : 'Hub tracking columns absent — re-run simulations after updating gui_zombie.py.';

  // Turnovers (exact ID changes)
  const turnovers = detectTurnovers(hub_id);

  // Tenure stats
  const tenures = [];
  let prev = 0;
  turnovers.forEach(tv => { tenures.push(tv - prev); prev = tv; });
  tenures.push((t.length || 1000) - prev);
  const avgTen = tenures.length ? Math.round(tenures.reduce((a,b)=>a+b,0)/tenures.length) : 0;
  const maxTen = tenures.length ? Math.max(...tenures) : 0;

  document.getElementById('hub-stats').innerHTML =
    `<div class="mini-card"><div class="lbl">Turnovers</div><div class="val">${turnovers.length}</div></div>` +
    `<div class="mini-card"><div class="lbl">Avg tenure</div><div class="val">${avgTen}</div><div class="sub">periods</div></div>` +
    `<div class="mini-card"><div class="lbl">Max tenure</div><div class="val">${maxTen}</div><div class="sub">periods</div></div>`;

  // Vertical lines for each turnover
  const shapes = turnovers.map(tv => ({
    type:'line', xref:'x', yref:'paper', x0:tv, x1:tv, y0:0, y1:1,
    line:{color:'rgba(231,76,60,0.3)', width:1, dash:'dash'}
  }));

  Plotly.react('chart-hub', [
    {x:t, y:hub_id,  name:'Hub bank ID', type:'scatter', mode:'lines',
     line:{color:'#e2ddd6', width:2, shape:'hv'},
     hovertemplate:'t=%{x}  Hub bank: <b>%{y}</b><extra></extra>'},
    {x:t, y:hub_fit, name:'Hub fitness', type:'scatter', mode:'lines',
     line:{color:'#27ae60', width:1.5, dash:'dot'}, yaxis:'y2'},
  ], mkLayout({
    shapes,
    yaxis:  {title:{text:'Hub bank ID (0–49)', font:{size:9}}, range:[-1,50],
             tickmode:'linear', dtick:5},
    yaxis2: {overlaying:'y', side:'right', title:{text:'Fitness (Φ)', font:{size:9}},
             range:[0,1.05], gridcolor:'transparent'},
    margin: {l:55,r:60,t:10,b:44},
    hovermode:'x unified',
  }), CFG);

  Plotly.react('chart-hub2', [
    {x:t, y:hub_cli, name:'Hub clients', type:'scatter', mode:'lines',
     line:{color:'#e74c3c', width:1.5, dash:'dash'}},
    {x:t, y:hub_eq,  name:'Hub equity',  type:'scatter', mode:'lines',
     line:{color:'#f39c12', width:1.5}, yaxis:'y2'},
  ], mkLayout({
    shapes,
    yaxis:  {title:{text:'Client count', font:{size:9}}},
    yaxis2: {overlaying:'y', side:'right', title:{text:'Hub equity', font:{size:9}},
             gridcolor:'transparent'},
    margin: {l:55,r:60,t:10,b:44},
    hovermode:'x unified',
  }), CFG);

  renderTopology();
}

// Brini Fig. 3 analogue: node-link snapshots at t=0/500/999 for ALL 4 cands
// (rows = candidates, cols = timepoints), at the canonical (regime=st, eta=0.1) cell.
// In-degree computed live from the edge list — bank.numOfBorrowers wasn't
// captured in dump_topology.py so we derive clients = #incoming-edges here,
// Phase 0: Brini-style topology rendering.
// - Single-cand chip-driven view (3 panels horizontally, t=0/500/999)
// - Uniform light-blue nodes scaled by in-degree (hub stands out by size, not color)
// - Directed edges with arrowheads via Plotly annotations (one per edge)
// - Bigger panels (~470px tall) for readability
function _topoTraceForSnap(snap) {
  if (!snap || !snap.nodes || !snap.nodes.length) return null;
  // In-degree
  const inDeg = {};
  snap.nodes.forEach(n => { inDeg[n.id] = 0; });
  snap.edges.forEach(e => { inDeg[e.lender] = (inDeg[e.lender] || 0) + 1; });
  let hubId = -1, hubMax = -1;
  for (const id in inDeg) {
    if (inDeg[id] > hubMax) { hubMax = inDeg[id]; hubId = parseInt(id); }
  }
  const nodePos = {};
  snap.nodes.forEach(n => { nodePos[n.id] = [n.x, n.y]; });
  // Arrows: one annotation per edge, pointing borrower→lender.
  // Shorten the arrow so the head doesn't sit inside the lender's marker.
  const arrows = [];
  snap.edges.forEach(e => {
    const p1 = nodePos[e.borrower], p2 = nodePos[e.lender];
    if (!p1 || !p2) return;
    arrows.push({
      x: p2[0], y: p2[1], ax: p1[0], ay: p1[1],
      xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
      showarrow: true, standoff: 6, startstandoff: 4,
      arrowhead: 2, arrowsize: 1.0, arrowwidth: 0.6,
      arrowcolor: 'rgba(140,140,140,0.55)', text: '',
    });
  });
  // Single light-blue node trace (Brini-style). Hub = biggest by in-degree.
  const nx = snap.nodes.map(n => n.x);
  const ny = snap.nodes.map(n => n.y);
  const nsize = snap.nodes.map(n => {
    const d = inDeg[n.id] || 0;
    return 8 + Math.min(40, d * 4.5);   // bigger range so hub clearly visible
  });
  const ntext = snap.nodes.map(n => {
    const d = inDeg[n.id] || 0;
    const tag = (n.id === hubId && hubMax > 0) ? '<br><b>HUB</b>' : '';
    return `bank ${n.id}<br>E=${(n.equity || 0).toFixed(2)}<br>in-degree=${d}${tag}`;
  });
  const traceNodes = {
    x: nx, y: ny, mode: 'markers', type: 'scatter',
    marker: {
      size: nsize,
      color: '#a9c8e0',                                      // uniform light blue
      line: {color: '#1a3f6b', width: 0.8},
      opacity: 0.92,
    },
    text: ntext, hoverinfo: 'text', showlegend: false,
  };
  return {traces: [traceNodes], arrows, hubId, hubMax};
}

function renderTopology() {
  const container = document.getElementById('chart-topology');
  container.innerHTML = '';
  // Layout: 1 row (current cand) × 3 cols (timepoints) — single-cand chip-driven.
  container.style.display = 'grid';
  container.style.gridTemplateColumns = 'repeat(3, 1fr)';
  container.style.gridAutoRows = '470px';
  container.style.gap = '6px';
  container.style.height = 'auto';
  const timepoints = ['0', '500', '999'];
  const cand = currentFocus.cand;
  if (!TOPOLOGY_DATA[cand]) {
    container.innerHTML = `<div style="grid-column:1/-1;padding:1.5rem;color:#888;
      text-align:center;font-size:.8rem">No topology snapshot available for
      <b style="color:${CAND_COLORS[cand]}">${CAND_LABELS[cand]}</b> —
      run <code>py -3.12 dump_topology_v2.py</code> to generate.</div>`;
    return;
  }
  timepoints.forEach(t => {
    const snap = (TOPOLOGY_DATA[cand] || {})[t];
    const div = document.createElement('div');
    div.style.background = '#161616';
    div.style.position = 'relative';
    div.style.border = `1px solid ${CAND_COLORS[cand]}33`;   // ~20% opacity border
    div.style.borderRadius = '2px';
    const innerId = `chart-topo-${cand}-${t}`;
    div.innerHTML = `<div style="font-size:.7rem;color:#bbb;position:absolute;top:6px;left:10px;z-index:2;letter-spacing:.04em">
                       <span style="color:${CAND_COLORS[cand]};font-weight:600">${CAND_LABELS[cand]}</span> · <span>t = ${t}</span>
                     </div>
                     <div id="${innerId}" style="height:470px"></div>`;
    container.appendChild(div);
    if (!snap) {
      div.querySelector(`#${innerId}`).innerHTML =
        `<div style="color:#666;font-size:.75rem;text-align:center;padding-top:200px">no snapshot at t=${t}</div>`;
      return;
    }
    const built = _topoTraceForSnap(snap);
    if (!built) return;
    const arrows = built.arrows.slice();
    if (built.hubMax > 0) {
      arrows.push({
        x: 1.25, y: -1.22, xref: 'x', yref: 'y',
        text: `hub: bank ${built.hubId} · in-deg = ${built.hubMax}`,
        showarrow: false, font: {size: 10, color: '#e2ddd6'},
        xanchor: 'right', yanchor: 'bottom',
        bgcolor: 'rgba(20,20,20,0.7)', borderpad: 4,
      });
    }
    const layout = {
      paper_bgcolor: '#161616', plot_bgcolor: '#161616',
      margin: {l: 6, r: 6, t: 28, b: 6},
      xaxis: {visible: false, range: [-1.3, 1.3], scaleanchor: 'y', scaleratio: 1},
      yaxis: {visible: false, range: [-1.3, 1.3]},
      hovermode: 'closest',
      annotations: arrows,
    };
    Plotly.newPlot(innerId, built.traces, layout, {responsive: true, displayModeBar: false});
  });
}

// ── BANK DETAIL ───────────────────────────────────────────────────────────────
const BANK_METRICS = [
  {key:'equity',        label:'Equity',              color:'#3498db'},
  {key:'fitness',       label:'Fitness Φ',           color:'#27ae60'},
  {key:'p_j',           label:'Default prob p_j',    color:'#e74c3c'},
  {key:'b_j',           label:'Bailout prob b_j',    color:'#f1c40f'},
  {key:'interest_rate', label:'Interest rate r_ij',  color:'#16a085'},
  {key:'loan',          label:'Loan',                color:'#f39c12'},
  {key:'clients',       label:'Clients',             color:'#9b59b6'},
  {key:'is_hub',        label:'Is hub (lender)',     color:'#1abc9c'},
  {key:'is_top_A',      label:'Is top-A (borrower)', color:'#e67e22'},
];
let activeBankMetrics = new Set(['equity','fitness','b_j','interest_rate','is_hub','is_top_A']);

function buildBankRunSelect() { /* no-op — chip-driven (label updated in renderBank) */ }

function buildBankMetricChips() {
  const el = document.getElementById('bank-metric-chips');
  el.innerHTML = '';   // clear before re-appending; each click rebuilds the chips
  BANK_METRICS.forEach(m => {
    const btn = document.createElement('button');
    btn.className = 'metric-chip' + (activeBankMetrics.has(m.key) ? ' on' : '');
    btn.textContent = m.label;
    btn.style.borderColor = activeBankMetrics.has(m.key) ? m.color : '';
    btn.style.color       = activeBankMetrics.has(m.key) ? m.color : '';
    btn.onclick = () => {
      if (activeBankMetrics.has(m.key)) activeBankMetrics.delete(m.key);
      else activeBankMetrics.add(m.key);
      buildBankMetricChips();
      renderBank();
    };
    el.appendChild(btn);
  });
}

async function buildBankIdSelect() {
  const run = focusTag();
  const bankSel = document.getElementById('bank-run'); if (bankSel) bankSel.textContent = LABELS[run] || run;
  const sel = document.getElementById('bank-id');
  // Lazy-load bank data on demand
  await loadBankDataIfNeeded(run);
  const rows = BANK_DATA[run] || [];
  const ids  = [...new Set(rows.map(r => r.bank_id))].sort((a,b)=>a-b);
  sel.innerHTML = '';
  if (ids.length === 0) {
    const o = document.createElement('option');
    o.text = '— no data —'; sel.appendChild(o); return;
  }
  ids.forEach(id => {
    const o = document.createElement('option');
    o.value = id; o.text = 'Bank ' + id;
    sel.appendChild(o);
  });
}

// ── Sweeps tab ────────────────────────────────────────────────────────────────
// η-view: ω values exposed in the dropdown. Phase1_grid_omega CSVs only exist for
// 0.50 and 0.55; 0.53 and 0.58 fall through to graceful-degrade ("no SWEEP_DATA")
// until those (basis × inertia) ablation grids get re-run. Hardcoded per Phase 0.
const SWEEP_OMEGA_OPTS = [0.50, 0.53, 0.55, 0.58];
const SWEEP_ETA_OPTS   = [0.0, 0.1, 0.3, 0.5, 0.9];   // ω-view: etas in the ω-sweep CSVs
const SWEEP_METRIC_LABELS = {
  total_bk: 'Total bankruptcies',
  contagion: 'Contagion deaths',
  lender_max_tenure: 'Lender max tenure (periods)',
  lender_avg_tenure: 'Lender avg tenure (periods)',
  mortality_frac: 'Mortality fraction of hub changes',
  clamp_fraction: 'C-clamp fraction (bilateral signal)',
  borrower_a_max_tenure: 'Borrower-A max tenure (TBTF identity)',
  borrower_a_avg_tenure: 'Borrower-A avg tenure',
  zombies: 'Fire-sale survivors (zombies)',
  rationing: 'Rationing failures',
};
const SPEC_COLORS = {
  'equity|0':       '#3498db',
  'equity|0.5':     '#2980b9',
  'loan_book|0':    '#27ae60',
  'loan_book|0.5':  '#16a085',
  'bilateral|0':    '#9b59b6',
  'bilateral|0.5':  '#e74c3c',
};
function specKey(basis, inertia) { return `${basis}|${inertia}`; }
function specLabel(basis, inertia) { return `${basis} λ=${inertia}`; }

// Map metric dropdown values → per-cell SUMM keys
const SUMM_KEY_FOR_METRIC = {
  total_bk:               'tot_bkr',
  contagion:              'tot_cntg',
  lender_max_tenure:      'lender_max',
  lender_avg_tenure:      'lender_avg',
  borrower_a_max_tenure:  'borrower_a_max',
  borrower_a_avg_tenure:  'borrower_a_avg',
  zombies:                'tot_zomb',
  rationing:              'tot_rat',
};
const REGIME_DISPLAY_COLORS = {nt:'#3498db', st:'#27ae60', rf:'#e67e22'};
const REGIME_NAME_TO_TAG = {none:'nt', socialized_tax:'st', resolution_fund:'rf'};
const REGIME_TAG_TO_NAME = {nt:'none', st:'socialized_tax', rf:'resolution_fund'};

let currentSweepMode = '';   // lehman | rho | rho01 | omega | cells

function setSweepMode(m) {
  currentSweepMode = m;
  ['lehman','rho','rho01','omega','cells'].forEach(s => {
    const btn = document.getElementById('smode-' + s);
    if (btn) btn.classList.toggle('on', s === m);
  });
  const ctrl = document.getElementById('sweep-controls');
  ctrl.innerHTML = _sweepControlsHtml(m);
  // Wire control onchange
  const sel = ctrl.querySelector('#smetric'); if (sel) sel.onchange = renderSweeps;
  const sav = ctrl.querySelector('#sfixed'); if (sav) sav.onchange = renderSweeps;
  ctrl.querySelectorAll('input[name=saxis]').forEach(r => r.onchange = ()=>{ _refillFixedFor(m); renderSweeps(); });
  ctrl.querySelectorAll('input[name=sview]').forEach(r => r.onchange = renderSweeps);
  renderSweeps();
}

function _sweepControlsHtml(m) {
  const metricOpts = `
    <option value="total_bk">Total bankruptcies</option>
    <option value="contagion">Contagion deaths</option>
    <option value="lender_max_tenure">Lender max tenure</option>
    <option value="lender_avg_tenure">Lender avg tenure</option>
    <option value="borrower_a_max_tenure">Borrower-A max tenure</option>
    <option value="borrower_a_avg_tenure">Borrower-A avg tenure</option>
    <option value="zombies">Zombies (fire-sale survivors)</option>
    <option value="rationing">Rationing failures</option>
    <option value="mortality_frac">Mortality fraction</option>
    <option value="clamp_fraction">C-clamp fraction</option>`;
  if (m === 'omega') {
    return `<span class="ctrl-label">Axis</span>
      <label style="margin-right:.5rem"><input type="radio" name="saxis" value="eta" checked> η-axis (fixed ω)</label>
      <label style="margin-right:.8rem"><input type="radio" name="saxis" value="omega"> ω-axis (fixed η)</label>
      <span class="ctrl-label" id="sfixed-label">ω</span>
      <select id="sfixed"></select>
      <span class="ctrl-label" style="margin-left:.5rem">Metric</span>
      <select id="smetric">${metricOpts}</select>`;
  }
  if (m === 'cells') {
    return `<span class="ctrl-label">View</span>
      <label style="margin-right:.5rem"><input type="radio" name="sview" value="perregime" checked> Per regime</label>
      <label style="margin-right:.8rem"><input type="radio" name="sview" value="regimecmp"> Regime cmp (chip focus)</label>
      <span class="ctrl-label">Metric</span>
      <select id="smetric">${metricOpts}</select>`;
  }
  // lehman / rho / rho01 — no per-mode controls (fixed metric headlines)
  return '';
}

function _refillFixedFor(m) {
  const ctrl = document.getElementById('sweep-controls');
  const sel = ctrl.querySelector('#sfixed');
  const lbl = ctrl.querySelector('#sfixed-label');
  if (!sel || !lbl) return;
  const axisRadio = ctrl.querySelector('input[name=saxis]:checked');
  const axis = axisRadio ? axisRadio.value : 'eta';
  sel.innerHTML = '';
  if (axis === 'eta') {
    lbl.textContent = 'ω =';
    SWEEP_OMEGA_OPTS.forEach(v => {
      const o = document.createElement('option');
      o.value = v; o.text = v.toFixed(2); sel.appendChild(o);
    });
    sel.value = 0.50;
  } else {
    lbl.textContent = 'η =';
    SWEEP_ETA_OPTS.forEach(v => {
      const o = document.createElement('option');
      o.value = v; o.text = v.toFixed(1); sel.appendChild(o);
    });
    sel.value = 0.1;
  }
}

function renderSweeps() {
  // Make sure sfixed is populated for omega mode (first call after setSweepMode)
  if (currentSweepMode === 'omega') {
    const sel = document.getElementById('sfixed');
    if (sel && !sel.options.length) _refillFixedFor('omega');
  }
  const panels = document.getElementById('sweep-panels');
  panels.innerHTML = '';
  if (currentSweepMode === 'lehman') return renderSweepLehman(panels);
  if (currentSweepMode === 'rho')    return renderSweepRho(panels);
  if (currentSweepMode === 'rho01')  return renderSweepRho01(panels);
  if (currentSweepMode === 'omega')  return renderSweepOmega(panels);
  if (currentSweepMode === 'cells')  return renderSweepCells(panels);
}

// Backwards-compat: old showSweepsSubview is now a thin alias to setSweepMode
function showSweepsSubview(v) {
  const map = {perregime:'cells', overlaid:'omega', omega:'omega', regimecmp:'cells'};
  setSweepMode(map[v] || v);
}

function _layoutBase(xaxisTitle, yaxisTitle) {
  return {
    paper_bgcolor: '#0f0f0f', plot_bgcolor: '#0f0f0f',
    font: {family: 'IBM Plex Mono, monospace', size: 10, color: '#e5e5e5'},
    margin: {l: 60, r: 20, t: 10, b: 45},
    xaxis: {title: {text: xaxisTitle, font: {size: 10}}, gridcolor: '#222', zerolinecolor: '#333'},
    yaxis: {title: {text: yaxisTitle, font: {size: 10}}, gridcolor: '#222', zerolinecolor: '#333'},
    legend: {orientation: 'h', y: -0.22, x: 0.5, xanchor: 'center', font: {size: 9}},
    hovermode: 'closest',
  };
}

// ── Mode renderers ─────────────────────────────────────────────────────────
function _chartPanel(parent, title, idSuffix, height) {
  const div = document.createElement('div');
  div.className = 'chart-box';
  div.innerHTML = `<div class="chart-title">${title}</div>
                   <div id="csw-${idSuffix}" style="height:${height||340}px"></div>`;
  parent.appendChild(div);
  return 'csw-' + idSuffix;
}

function _bandTrace(xs, mean, std, color, name, group) {
  const up = mean.map((v,i) => v==null?null:v+std[i]);
  const lo = mean.map((v,i) => v==null?null:v-std[i]);
  return [
    {x: xs.concat(xs.slice().reverse()), y: up.concat(lo.slice().reverse()),
     fill: 'toself', fillcolor: hexA(color, 0.18), line:{color:'transparent'},
     hoverinfo: 'skip', showlegend: false, legendgroup: group},
    {x: xs, y: mean, name, type: 'scatter', mode: 'lines+markers',
     line: {color, width: 2}, marker: {size: 5, color}, legendgroup: group},
  ];
}

// Helper: pick the right thesis dataset for the chip-focused cand,
// fall back to baseline if the focused cand has no data yet.
// Normalizes BA-suffixed chips (bsl_ba3, a_ba5...) to base tag (bsl, a...)
// since thesis sweep dicts (LEHMAN_DATA_BA3 etc.) are keyed by base cand
// within each algorithm family.
function _thesisFor(dataset) {
  const raw = currentFocus.cand;
  const cand = raw.replace(/_ba\d+$/, '');
  if (dataset[cand] && dataset[cand].length) return {cand: raw, rows: dataset[cand]};
  if (dataset['bsl'] && dataset['bsl'].length) return {cand: 'bsl', rows: dataset['bsl']};
  return {cand: null, rows: []};
}

function _candBanner(cand, requestedCand) {
  if (cand === requestedCand) return CAND_LABELS[cand];
  return `<span style="color:#e67e22">${CAND_LABELS[cand]}</span> (${CAND_LABELS[requestedCand]} not yet swept — falling back to baseline)`;
}

// Lehman: η × 3 regimes at the chip-selected cand, ρ=0.4
function renderSweepLehman(panels) {
  const {cand, rows} = _thesisFor(_algoMap(LEHMAN_DATA, LEHMAN_DATA_BA3));
  if (!rows.length) {
    panels.innerHTML = `<div style="padding:1rem;color:#888;font-size:.8rem">
      thesis_lehman*.csv missing — run <code>python run_thesis_repro_sweep.py</code> to generate it.</div>`;
    return;
  }
  // Banner showing which cand is being displayed
  const banner = document.createElement('div');
  banner.className = 'callout';
  banner.style.fontSize = '.7rem';
  banner.innerHTML = `<b>Showing:</b> ${_candBanner(cand, currentFocus.cand)}`;
  panels.appendChild(banner);
  // Group by regime
  const by = {};
  rows.forEach(r => { (by[r.fiscal_regime] = by[r.fiscal_regime] || []).push(r); });
  const regimes = ['none', 'socialized_tax', 'resolution_fund'];

  function buildTraces(metricKey) {
    const traces = [];
    regimes.forEach(rg => {
      if (!by[rg]) return;
      const arr = by[rg].slice().sort((a,b) => a.eta - b.eta);
      const xs = arr.map(r => r.eta);
      const mean = arr.map(r => r[metricKey + '_mean']);
      const std  = arr.map(r => r[metricKey + '_std']);
      const tag = REGIME_NAME_TO_TAG[rg];
      const color = REGIME_DISPLAY_COLORS[tag];
      traces.push(..._bandTrace(xs, mean, std, color, REGIME_LABELS[tag], rg));
    });
    return traces;
  }

  const candP = {bsl: '0.50 equity', a: '0.53 equity', b: '0.52 bilateral'};
  const id1 = _chartPanel(panels, `Total bankruptcies vs η — 3 regimes (ω=${candP[cand]||'?'}, ρ=0.4, 5-seed)`, 'lehman-total', 360);
  const id2 = _chartPanel(panels, 'Contagion deaths vs η — 3 regimes', 'lehman-cntg', 300);
  const lay1 = _layoutBase('η (bailout coverage)', 'Total bankruptcies');
  // Annotations stripped in Phase 0: bsl-specific η*=0.1 / Lehman-zone callouts
  // don't apply to c3 family (η* shifts) or w58 family (fund regime broken).
  pReact(id1, buildTraces('total_bk'), lay1, CFG);
  pReact(id2, buildTraces('contagion'), _layoutBase('η', 'Contagion deaths'), CFG);

  // Data table
  const tbl = document.createElement('div');
  tbl.className = 'callout';
  tbl.style.fontSize = '.65rem';
  let html = '<table style="width:100%;border-collapse:collapse"><thead><tr style="border-bottom:1px solid #444"><th style="text-align:left;padding:4px">η</th>';
  regimes.forEach(rg => { html += `<th style="text-align:right;padding:4px">${REGIME_LABELS[REGIME_NAME_TO_TAG[rg]]}</th>`; });
  html += '</tr></thead><tbody>';
  // Build by-eta map
  const byEta = {};
  rows.forEach(r => { (byEta[r.eta] = byEta[r.eta] || {})[r.fiscal_regime] = r; });
  Object.keys(byEta).map(Number).sort((a,b)=>a-b).forEach(eta => {
    html += `<tr><td style="padding:3px">${eta.toFixed(1)}</td>`;
    regimes.forEach(rg => {
      const r = byEta[eta][rg];
      html += `<td style="text-align:right;padding:3px">${r ? Math.round(r.total_bk_mean).toLocaleString() : '—'}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  tbl.innerHTML = html;
  panels.appendChild(tbl);
}

// ρ-sweep: ρ × no-tax × η=0
function renderSweepRho(panels) {
  const {cand, rows} = _thesisFor(_algoMap(RHO_DATA, RHO_DATA_BA3));
  if (!rows.length) {
    panels.innerHTML = `<div style="padding:1rem;color:#888;font-size:.8rem">
      thesis_rho*.csv missing — run <code>python run_thesis_repro_sweep.py</code>.</div>`;
    return;
  }
  const banner = document.createElement('div');
  banner.className = 'callout';
  banner.style.fontSize = '.7rem';
  banner.innerHTML = `<b>Showing:</b> ${_candBanner(cand, currentFocus.cand)}`;
  panels.appendChild(banner);
  const arr = rows.slice().sort((a,b) => a.rho - b.rho);
  const xs = arr.map(r => r.rho);

  const id1 = _chartPanel(panels, 'Total bankruptcies vs ρ  (η=0, no-tax, 5-seed)', 'rho-total', 340);
  const id2 = _chartPanel(panels, 'Contagion vs ρ  (η=0, no-tax, 5-seed)', 'rho-cntg', 280);
  const id3 = _chartPanel(panels, 'Channel decomposition vs ρ', 'rho-decomp', 320);

  // Total
  const totMean = arr.map(r => r.total_bk_mean);
  const totStd  = arr.map(r => r.total_bk_std);
  const totTraces = _bandTrace(xs, totMean, totStd, '#3498db', 'Total bk', 'total');
  const lay1 = _layoutBase('ρ (fire-sale recovery)', 'Total bankruptcies');
  // 'non-mono peak' annotation stripped in Phase 0 (peak location is cand-dependent).
  pReact(id1, totTraces, lay1, CFG);

  // Contagion
  const cntgMean = arr.map(r => r.contagion_mean);
  const cntgStd  = arr.map(r => r.contagion_std);
  const cntgTraces = _bandTrace(xs, cntgMean, cntgStd, '#e74c3c', 'Contagion', 'cntg');
  const lay2 = _layoutBase('ρ', 'Contagion deaths');
  // 'zombie activation' annotation stripped in Phase 0 (cand-specific).
  pReact(id2, cntgTraces, lay2, CFG);

  // Channel decomp (4 lines)
  const channels = [
    {key: 'shock',     name: 'Shock',     color: '#27ae60'},
    {key: 'rationing', name: 'Rationing', color: '#3498db'},
    {key: 'contagion', name: 'Contagion', color: '#e74c3c'},
    {key: 'repay',     name: 'Repay',     color: '#f39c12'},
  ];
  const chTraces = channels.map(c => ({
    x: xs, y: arr.map(r => r[c.key + '_mean']),
    name: c.name, type: 'scatter', mode: 'lines+markers',
    line: {color: c.color, width: 2}, marker: {size: 5, color: c.color},
  }));
  pReact(id3, chTraces, _layoutBase('ρ', 'Deaths by channel'), CFG);
}

// ρ=0.1 secondary: η × {none, resolution_fund} × ρ=0.1
function renderSweepRho01(panels) {
  const {cand, rows} = _thesisFor(_algoMap(RHO01_DATA, RHO01_DATA_BA3));
  if (!rows.length) {
    panels.innerHTML = `<div style="padding:1rem;color:#888;font-size:.8rem">
      thesis_rho01*.csv missing — run <code>python run_thesis_repro_sweep.py</code>.</div>`;
    return;
  }
  const banner = document.createElement('div');
  banner.className = 'callout';
  banner.style.fontSize = '.7rem';
  banner.innerHTML = `<b>Showing:</b> ${_candBanner(cand, currentFocus.cand)}`;
  panels.appendChild(banner);
  const by = {};
  rows.forEach(r => { (by[r.fiscal_regime] = by[r.fiscal_regime] || []).push(r); });
  const id1 = _chartPanel(panels, 'Total bankruptcies vs η at ρ=0.1 — fund > no-tax everywhere (5-seed)', 'rho01-total', 360);
  const traces = [];
  ['none', 'resolution_fund'].forEach(rg => {
    if (!by[rg]) return;
    const arr = by[rg].slice().sort((a,b) => a.eta - b.eta);
    const xs = arr.map(r => r.eta);
    const mean = arr.map(r => r.total_bk_mean);
    const std  = arr.map(r => r.total_bk_std);
    const tag = REGIME_NAME_TO_TAG[rg];
    const color = REGIME_DISPLAY_COLORS[tag];
    traces.push(..._bandTrace(xs, mean, std, color, REGIME_LABELS[tag], rg));
  });
  const lay = _layoutBase('η', 'Total bankruptcies');
  // 'fund > no-tax / Claim 3 secondary' annotation stripped in Phase 0 (bsl-specific).
  pReact(id1, traces, lay, CFG);
}

// ω-sweep: lift-and-shift of existing renderSweepsOverlaid + renderSweepsOmega
function renderSweepOmega(panels) {
  const ctrl = document.getElementById('sweep-controls');
  const metric = (ctrl.querySelector('#smetric') || {}).value || 'total_bk';
  const fixedSel = ctrl.querySelector('#sfixed');
  const fixed = fixedSel ? parseFloat(fixedSel.value) : 0.50;
  const axis = (ctrl.querySelector('input[name=saxis]:checked') || {}).value || 'eta';
  const meanK = metric + '_mean', stdK = metric + '_std';

  let rows, xKey, xTitle, titleSuffix;
  if (axis === 'eta') {
    rows = SWEEP_DATA.filter(d => Math.abs(d.omega - fixed) < 1e-6);
    xKey = 'eta'; xTitle = 'η (bailout recovery)'; titleSuffix = `ω = ${fixed.toFixed(2)}`;
  } else {
    rows = SWEEP_DATA.filter(d => Math.abs(d.eta - fixed) < 1e-6);
    xKey = 'omega'; xTitle = 'ω (shock dispersion)'; titleSuffix = `η = ${fixed.toFixed(1)}`;
  }
  const byKey = {};
  rows.forEach(r => { const k = specKey(r.basis, r.inertia); (byKey[k] = byKey[k] || []).push(r); });
  const traces = [];
  Object.keys(byKey).sort().forEach(k => {
    const arr = byKey[k].slice().sort((a, b) => a[xKey] - b[xKey]);
    const xs = arr.map(r => r[xKey]);
    const mean = arr.map(r => r[meanK]);
    const std  = arr.map(r => r[stdK]);
    const [basis, inertia] = k.split('|');
    const color = SPEC_COLORS[k] || '#888';
    traces.push(..._bandTrace(xs, mean, std, color, specLabel(basis, inertia), k));
  });
  const id = _chartPanel(panels,
    `${SWEEP_METRIC_LABELS[metric] || metric}  (${titleSuffix}, socialized only)`,
    'omega-main', 420);
  if (!traces.length) {
    document.getElementById(id).innerHTML =
      `<div style="padding:1rem;color:#888;font-size:.8rem">No SWEEP_DATA for ${titleSuffix}, metric=${metric}.</div>`;
    return;
  }
  pReact(id, traces, _layoutBase(xTitle, SWEEP_METRIC_LABELS[metric]||metric), CFG);
}

// 27-cell snapshot: per-regime grid OR regime-cmp bars
function renderSweepCells(panels) {
  const ctrl = document.getElementById('sweep-controls');
  const metric = (ctrl.querySelector('#smetric') || {}).value || 'total_bk';
  const view = (ctrl.querySelector('input[name=sview]:checked') || {}).value || 'perregime';
  const summKey = SUMM_KEY_FOR_METRIC[metric];

  if (view === 'perregime') {
    const wrap = document.createElement('div');
    wrap.style.display = 'grid';
    wrap.style.gridTemplateColumns = 'repeat(3, 1fr)';
    wrap.style.gap = '8px';
    panels.appendChild(wrap);
    if (!summKey) {
      wrap.innerHTML = `<div style="grid-column:1/-1;padding:1rem;color:#888;font-size:.8rem">
        Metric "${metric}" not in 27-cell aggregates (needs bank-level data). Try a different metric.</div>`;
      return;
    }
    REGIME_TAGS.forEach(rt => {
      const div = document.createElement('div');
      div.className = 'chart-box';
      const id = 'csw-cells-' + rt;
      div.innerHTML = `<div class="chart-title">${REGIME_LABELS[rt]}</div>
                       <div id="${id}" style="height:280px"></div>`;
      wrap.appendChild(div);
      const traces = CAND_TAGS.map(ct => {
        const xs = ETA_TAGS.map(et => ({e0:0, e01:0.1, e085:0.85}[et]));
        const ys = ETA_TAGS.map(et => {
          const tag = `${ct}_${rt}_${et}`;
          return (SUMM[tag] && SUMM[tag][summKey] !== undefined) ? SUMM[tag][summKey] : null;
        });
        return {x: xs, y: ys, name: CAND_LABELS[ct],
                type: 'scatter', mode: 'lines+markers',
                line: {color: CAND_COLORS[ct], width: 2},
                marker: {size: 6, color: CAND_COLORS[ct]}};
      });
      pNewPlot(id, traces, _layoutBase('η', SWEEP_METRIC_LABELS[metric]||metric), CFG);
    });
  } else {
    // regime-cmp
    const id = _chartPanel(panels,
      `${SWEEP_METRIC_LABELS[metric]||metric}  (${CAND_LABELS[currentFocus.cand]} · ${ETA_LABELS[currentFocus.eta]} — Lehman reproduction at chip focus)`,
      'cells-cmp', 420);
    if (!summKey) {
      document.getElementById(id).innerHTML =
        `<div style="padding:1rem;color:#888;font-size:.8rem">Metric "${metric}" not in 27-cell aggregates.</div>`;
      return;
    }
    const yVals = REGIME_TAGS.map(rt => {
      const tag = `${currentFocus.cand}_${rt}_${currentFocus.eta}`;
      return (SUMM[tag] && SUMM[tag][summKey] !== undefined) ? SUMM[tag][summKey] : null;
    });
    const trace = {
      type: 'bar', x: REGIME_TAGS.map(rt => REGIME_LABELS[rt]), y: yVals,
      marker: {color: REGIME_TAGS.map(rt => REGIME_DISPLAY_COLORS[rt]), line: {color: '#222', width: 1}},
      text: yVals.map(v => v==null?'n/a':v.toLocaleString()),
      textposition: 'outside', textfont: {size: 11, color: '#e5e5e5'},
    };
    const layout = _layoutBase('fiscal regime', SWEEP_METRIC_LABELS[metric]||metric);
    layout.margin.t = 30;
    Plotly.react(id, [trace], layout, CFG);
  }

  // Multi-seed tenure robustness footnote (settles "max=24 vs 8 — seed artifact?")
  if (Object.keys(ROBUSTNESS_DATA).length > 0) {
    const div = document.createElement('div');
    div.className = 'callout';
    div.style.fontSize = '.65rem';
    div.style.marginTop = '.6rem';
    let html = `<b>Multi-seed lender max-tenure across 30 seeds at η=0.1, ρ=0.4</b> &mdash;
                <span style="color:#888">tests whether the single-seed numbers above are seed-overfit or robust regime properties.</span><br><br>
                <table style="width:100%;border-collapse:collapse;font-family:inherit">
                <thead><tr style="border-bottom:1px solid #444"><th style="text-align:left;padding:4px">cand</th>`;
    REGIME_TAGS.forEach(rt => { html += `<th style="text-align:right;padding:4px;color:${REGIME_DISPLAY_COLORS[rt]}">${REGIME_LABELS[rt]}</th>`; });
    html += '</tr></thead><tbody>';
    CAND_TAGS.forEach(ct => {
      html += `<tr><td style="padding:3px;color:${CAND_COLORS[ct]}">${CAND_LABELS[ct]}</td>`;
      REGIME_TAGS.forEach(rt => {
        const cellName = ({nt: 'none', st: 'socialized_tax', rf: 'resolution_fund'})[rt];
        const r = (ROBUSTNESS_DATA[ct] || {})[cellName];
        if (r) {
          const isFocus = (ct === currentFocus.cand && rt === currentFocus.regime);
          const cell = `${r.max_mean.toFixed(1)} ± ${r.max_std.toFixed(1)} <span style="color:#666">(max ${r.max_max})</span>`;
          html += `<td style="text-align:right;padding:3px;${isFocus?'background:#222;font-weight:bold':''}">${cell}</td>`;
        } else {
          html += `<td style="text-align:right;padding:3px;color:#666">—</td>`;
        }
      });
      html += '</tr>';
    });
    html += '</tbody></table>';
    html += '<div style="margin-top:.4rem;color:#888">Single-seed (26474) numbers in the chart above; multi-seed mean ± std here. Highlighted cell = chip focus.</div>';
    div.innerHTML = html;
    panels.appendChild(div);
  }
}

function hexA(hex, alpha) {
  const h = hex.replace('#','');
  const r = parseInt(h.substr(0,2), 16);
  const g = parseInt(h.substr(2,2), 16);
  const b = parseInt(h.substr(4,2), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

async function renderBank() {
  const run    = focusTag();
  const bankSel = document.getElementById('bank-run'); if (bankSel) bankSel.textContent = LABELS[run] || run;
  const note   = document.getElementById('bank-data-note');
  // Lazy-load bank data on demand (per cand). Show loading state while fetching.
  if (BANK_DATA[run] === undefined) {
    if (note) note.textContent = `Loading bank-detail for ${LABELS[run] || run}…`;
  }
  await loadBankDataIfNeeded(run);
  const bidEl  = document.getElementById('bank-id');
  const bid    = bidEl.value !== '' ? parseInt(bidEl.value) : null;
  const rows   = (BANK_DATA[run] || []).filter(r => r.bank_id === bid);

  const hasData = rows.length > 0;
  document.getElementById('bank-data-note').textContent = hasData
    ? ''
    : 'No bank_detail data — local HTTP server required (run `python -m http.server` in the project dir, browse to localhost:8000/dashboard.html).';
  document.getElementById('bank-chart-title').textContent =
    hasData ? `Bank ${bid} — ${LABELS[run]}` : 'No data';

  if (!hasData) {
    Plotly.react('chart-bank', [], mkLayout({margin:{l:52,r:14,t:10,b:44}}), CFG);
    return;
  }

  const tArr = rows.map(r => r.t);
  // is_top_A is derived from per-period top_A_bank in DATA[run]: 1 if this bank
  // is the current top-A borrower, else 0. Aligned by t value.
  const topA = (DATA[run] && DATA[run].top_A_bank) ? DATA[run].top_A_bank : null;
  const isTopAByT = {};
  if (topA) {
    tArr.forEach((t, i) => {
      const tn = (typeof t === 'number') ? t : Number(t);
      const ta = topA[tn];
      isTopAByT[t] = (ta !== undefined && ta !== null && Number(ta) === bid) ? 1 : 0;
    });
  }
  // Split metrics by axis: small-range on right axis, larger-range on left.
  // interest_rate goes right too — it's typically 0.05-0.5, would be invisible vs equity ~30 / loan ~120.
  const RIGHT_AXIS_KEYS = new Set(['fitness', 'p_j', 'b_j', 'interest_rate', 'is_hub', 'is_top_A']);
  const traces = [];
  BANK_METRICS.forEach((m, mi) => {
    if (!activeBankMetrics.has(m.key)) return;
    let yArr;
    if (m.key === 'is_top_A') {
      if (!topA) return;   // graceful degrade if dash CSV doesn't have the column
      yArr = tArr.map(t => isTopAByT[t]);
    } else {
      yArr = rows.map(r => r[m.key] !== undefined ? r[m.key] : null);
    }
    const onRight = RIGHT_AXIS_KEYS.has(m.key);
    traces.push({
      x: tArr, y: yArr,
      name: m.label + (onRight ? ' (R)' : ''),
      type: 'scatter', mode: 'lines',
      line: {
        color: m.color,
        width: 1.6,
        dash: (m.key === 'is_hub' || m.key === 'is_top_A') ? 'dot' : 'solid',
      },
      yaxis: onRight ? 'y2' : 'y',
    });
  });

  Plotly.react('chart-bank', traces, mkLayout({
    hovermode: 'x unified',
    margin: {l:55, r:60, t:10, b:44},
    xaxis: {title:{text:'Period', font:{size:9}}},
    yaxis: {title:{text:'Equity / Loan / Clients', font:{size:9}}},
    yaxis2:{overlaying:'y', side:'right',
            title:{text:'Φ / p_j / hub  ([0,1])', font:{size:9}},
            range:[-0.05, 1.05], gridcolor:'transparent',
            tickvals:[0, 0.25, 0.5, 0.75, 1]},
  }), CFG);
}

// ── INIT ──────────────────────────────────────────────────────────────────────
updateChipDisplay();
buildVarSelect();
buildDecompTabs();
buildRegCmpVarSelect();
buildCCFSelects();
buildOvRunSelect();
buildHubRunSelect();
buildBankRunSelect();
buildBankIdSelect();
buildBankMetricChips();
renderOverview();
setSweepMode('lehman');   // pre-build the Sweeps tab so first click already shows the C3 figure
// Deferred init: overlay/hub/bank rendered on first tab click
</script>
</body>
</html>
"""

with open(OUT, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'Written: {OUT}  ({len(html):,} chars)')
