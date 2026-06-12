"""Phase 0 file migration: move discarded cands + rename c3 -> c3_g10.

User keep-list: bsl, a, w55, c3 (renamed c3_g10), plus 4 to add (c3_g05/c3_g12/w58/w58_g04).
Discarded: b, *_ba1, *_ba3, *_ba5, *_cb, plus legacy 6-cell tags.

Atomic: builds the move/rename list first, then executes. Idempotent — already-moved files are skipped.
"""
import os
import shutil
import glob
from pathlib import Path

# Output / simulation data directory. Override via TBTF_SIM_DIR env var.
# Default: a sibling 'Simulations/' folder next to the repo root.
SIM = str(Path(os.environ.get(
    'TBTF_SIM_DIR',
    str(Path(__file__).resolve().parent.parent / 'Simulations'),
)))

# Project root (the repo directory itself).
PROJ = str(Path(__file__).resolve().parent)
SIM_DISCARD = os.path.join(SIM, 'discarded')
PROJ_DISCARD = os.path.join(PROJ, 'discarded')


def ensure_dirs():
    for d in (SIM_DISCARD, PROJ_DISCARD):
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
            print(f'  mkdir {d}')


def is_discarded(name):
    """Return True iff the file's tag indicates a discarded cand."""
    # legacy 6-cell tags from pre-Phase-1.5
    if any(s in name for s in ('_baseline_', '_bilateral_w52_', '_equity_w53_')):
        return True
    # cash boost
    if '_cb_' in name or name.endswith('_cb.csv') or name.endswith('_cb.json'):
        return True
    # BA m=1, m=3, m=5 — patterns: _ba1_, _ba3_, _ba5_, or _ba1.csv etc
    for ba in ('_ba1', '_ba3', '_ba5'):
        if ba + '_' in name or name.endswith(ba + '.csv') or name.endswith(ba + '.json'):
            return True
    # Cand B — must be careful not to match 'bsl' or 'b_ba*' (already covered)
    # patterns: dash_b_<regime>_, bank_detail_b_<regime>_, thesis_*_b.csv, dashboard_bank_b_
    base = os.path.basename(name)
    for prefix in ('dash_b_', 'bank_detail_b_', 'dashboard_bank_b_', 'topology_b_'):
        if base.startswith(prefix):
            return True
    if base in ('thesis_lehman_b.csv', 'thesis_rho_b.csv', 'thesis_rho01_b.csv'):
        return True
    return False


def move_discarded():
    moved, skipped = 0, 0
    # Simulations folder
    for pattern in ('dash_*.csv', 'bank_detail_*.csv', 'thesis_*.csv', 'topology_*.json'):
        for src in glob.glob(os.path.join(SIM, pattern)):
            base = os.path.basename(src)
            if is_discarded(base):
                dst = os.path.join(SIM_DISCARD, base)
                if os.path.exists(dst):
                    # already moved before — remove src so we converge
                    if os.path.exists(src):
                        os.remove(src)
                        skipped += 1
                else:
                    shutil.move(src, dst)
                    moved += 1
    # Project root (dashboard_bank JSONs)
    for src in glob.glob(os.path.join(PROJ, 'dashboard_bank_*.json')):
        base = os.path.basename(src)
        if is_discarded(base):
            dst = os.path.join(PROJ_DISCARD, base)
            if os.path.exists(dst):
                if os.path.exists(src):
                    os.remove(src)
                    skipped += 1
            else:
                shutil.move(src, dst)
                moved += 1
    print(f'  Moved {moved}, removed {skipped} duplicates')


def rename_c3_to_c3_g10():
    """Rename in-place: c3 -> c3_g10 across all artefact types."""
    renames = []
    # Simulations: dash_c3_<regime>_<eta>.csv x9, bank_detail x9, thesis_lehman_c3.csv x1
    for pattern in ('dash_c3_*.csv', 'bank_detail_c3_*.csv'):
        for src in glob.glob(os.path.join(SIM, pattern)):
            base = os.path.basename(src)
            if base.startswith('dash_c3_g'):  # already a c3_g05/g12/g10 — skip
                continue
            if base.startswith('bank_detail_c3_g'):
                continue
            new_base = base.replace('c3_', 'c3_g10_', 1)
            dst = os.path.join(SIM, new_base)
            renames.append((src, dst))
    # Special: thesis_lehman_c3.csv (no _ between c3 and .csv)
    src = os.path.join(SIM, 'thesis_lehman_c3.csv')
    if os.path.exists(src):
        renames.append((src, os.path.join(SIM, 'thesis_lehman_c3_g10.csv')))
    # Project: dashboard_bank_c3_<regime>_<eta>.json x9
    for src in glob.glob(os.path.join(PROJ, 'dashboard_bank_c3_*.json')):
        base = os.path.basename(src)
        if base.startswith('dashboard_bank_c3_g'):
            continue
        new_base = base.replace('c3_', 'c3_g10_', 1)
        dst = os.path.join(PROJ, new_base)
        renames.append((src, dst))

    done, skipped = 0, 0
    for src, dst in renames:
        if not os.path.exists(src):
            skipped += 1
            continue
        if os.path.exists(dst):
            # already renamed previously — leave the dst and remove the src
            os.remove(src)
            skipped += 1
            continue
        os.rename(src, dst)
        done += 1
    print(f'  Renamed {done}, skipped {skipped}')


def main():
    print('=== Phase 0 migration: discard + rename ===')
    print()
    print('A. Ensure discarded/ folders exist')
    ensure_dirs()
    print()
    print('B. Move discarded files (b, BA1/3/5, CB, legacy)')
    move_discarded()
    print()
    print('C. Rename c3 -> c3_g10')
    rename_c3_to_c3_g10()
    print()
    print('=== Migration complete ===')


if __name__ == '__main__':
    main()
