"""Shared utilities for thesis figure + table render scripts.

Color palette matches TFG/main.tex (cBlue, cRed, cGreen, cAmber, cGray, etc.).
Provides regime-label normalization, mpl style setup, booktabs LaTeX table emission.
"""
import os
import sys
import csv
import math
import statistics as stat
from collections import defaultdict
from pathlib import Path

# Force UTF-8 stdout to handle Greek characters on Windows cp1252 terminals
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np

# Force a non-interactive matplotlib backend for headless render scripts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Color palette (matches TFG/main.tex) ───────────────────────────────────
COLORS = {
    'cBlue':      '#1A3F6B',
    'cRed':       '#8B1A1A',
    'cGreen':     '#1A5C2A',
    'cAmber':     '#7A4800',
    'cGray':      '#555555',
    'cSteelBlue': '#3A6B8A',
    'cBrown':     '#7A2A00',
    'cPurple':    '#4A1A6B',
}

# Regime → color mapping (for thesis-body figures)
REGIME_COLORS = {
    'no_tax':      COLORS['cBlue'],
    'ex_post':     COLORS['cRed'],
    'ex_ante_t5':  COLORS['cGreen'],     # canonical
    'ex_ante_t4':  COLORS['cAmber'],     # high-levy default
    'ex_ante_t6':  COLORS['cPurple'],    # low-levy
}

REGIME_LABEL_NICE = {
    'no_tax':      'No tax',
    'ex_post':     'Ex-post tax',
    'ex_ante_t5':  r'Ex-ante tax ($\tau{=}10^{-5}$, canonical)',
    'ex_ante_t4':  r'Ex-ante tax ($\tau{=}10^{-4}$)',
    'ex_ante_t6':  r'Ex-ante tax ($\tau{=}10^{-6}$)',
}

# Canonical seed for single-seed visualizations
SEED_VIS = 26474

# Multi-seed pool
SEED_POOL = [26462, 26463, 26464, 26465, 26466]


def setup_mpl():
    """Apply consistent matplotlib defaults for thesis figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.4,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.frameon': False,
        'lines.linewidth': 1.4,
        'figure.dpi': 100,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
    })


def regime_label(row):
    """Map a sweep row to canonical regime label.

    Args:
        row: dict-like with 'fiscal_regime' and (if resolution_fund) 'fund_levy_rate'

    Returns: 'no_tax' | 'ex_post' | 'ex_ante_t4' | 'ex_ante_t5' | 'ex_ante_t6' | unknown
    """
    fr = row.get('fiscal_regime', '')
    if fr == 'none':
        return 'no_tax'
    if fr == 'socialized_tax':
        return 'ex_post'
    if fr == 'resolution_fund':
        try:
            tau = float(row.get('fund_levy_rate', '0.0001'))
        except (ValueError, TypeError):
            tau = 1e-4
        if abs(tau - 1e-4) < 1e-9: return 'ex_ante_t4'
        if abs(tau - 1e-5) < 1e-9: return 'ex_ante_t5'
        if abs(tau - 1e-6) < 1e-9: return 'ex_ante_t6'
        return f'ex_ante_t{tau:.0e}'
    return fr or 'unknown'


def load_sweep(csv_path, fields=None, type_map=None):
    """Load a sweep CSV and optionally cast numeric fields.

    type_map: dict mapping field_name -> type (e.g. {'eta': float, 'total_bk': int}).
    fields: list of fields to keep; if None, keep all.
    """
    rows = []
    with open(csv_path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if fields is not None:
                r = {k: r[k] for k in fields if k in r}
            if type_map:
                for k, t in type_map.items():
                    if k in r:
                        try:
                            r[k] = t(r[k])
                        except (ValueError, TypeError):
                            pass
            rows.append(r)
    return rows


def aggregate_by(rows, group_keys, value_keys):
    """Group rows by (group_keys), compute mean+std for each value_key.

    Returns: dict {(group_key_tuple): {value_key: (mean, std, n)}}
    """
    buckets = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        buckets[key].append(r)
    out = {}
    for key, bucket in buckets.items():
        d = {}
        for vk in value_keys:
            vals = [r[vk] for r in bucket if r.get(vk) is not None]
            if not vals:
                d[vk] = (None, None, 0)
            else:
                m = stat.mean(vals)
                s = stat.stdev(vals) if len(vals) > 1 else 0.0
                d[vk] = (m, s, len(vals))
        out[key] = d
    return out


def ensure_dir(path):
    """Make sure parent dir of `path` exists."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_figure(fig, path):
    """Save matplotlib figure to canonical path; print confirmation."""
    ensure_dir(path)
    fig.savefig(path)
    plt.close(fig)
    print(f'  [saved] {path}', flush=True)


def write_booktabs_table(rows, columns, path, caption=None, label=None,
                         column_format=None, col_headers=None):
    """Write a LaTeX booktabs table.

    rows: list of dicts (or list of lists if columns are positional)
    columns: list of column keys to extract from each row
    col_headers: optional list of header labels (defaults to columns)
    column_format: e.g. 'lrrr'; defaults to 'l' + 'r'*(n-1)
    """
    ensure_dir(path)
    n = len(columns)
    if column_format is None:
        column_format = 'l' + 'r' * (n - 1) if n > 0 else 'l'
    if col_headers is None:
        col_headers = columns

    lines = [
        '% Auto-generated booktabs table',
        '\\begin{table}[H]',
        '\\centering',
    ]
    if caption:
        lines.append(f'\\caption{{{caption}}}')
    if label:
        lines.append(f'\\label{{{label}}}')
    lines.append('\\smallskip')
    lines.append('\\begin{tabular}{' + column_format + '}')
    lines.append('\\toprule')
    lines.append(' & '.join(col_headers) + ' \\\\')
    lines.append('\\midrule')
    for row in rows:
        if isinstance(row, dict):
            cells = [str(row.get(c, '')) for c in columns]
        else:
            cells = [str(c) for c in row]
        lines.append(' & '.join(cells) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  [saved] {path}', flush=True)


def fmt_msd(mean, std, decimals=1):
    """Format mean ± std for table cells."""
    if mean is None: return '--'
    if decimals == 0:
        return f'{mean:.0f} $\\pm$ {std:.0f}' if std is not None and std > 0 else f'{mean:.0f}'
    fmt = f'{{:.{decimals}f}}'
    s_mean = fmt.format(mean)
    if std is None or std == 0:
        return s_mean
    s_std = fmt.format(std)
    return f'{s_mean} $\\pm$ {s_std}'


# ── Path helpers ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'thesis_assets'

def asset_path(chapter_subdir, filename):
    """Build canonical asset path: thesis_assets/{chapter_subdir}/{filename}"""
    return str(ASSETS / chapter_subdir / filename)


if __name__ == '__main__':
    # Simple self-test
    setup_mpl()
    print('thesis_render_utils loaded OK')
    print(f'  ROOT: {ROOT}')
    print(f'  ASSETS: {ASSETS}')
