"""Trim eta=0 three-regime PNGs down to single-regime panels.

Per audit cross-cut A.3 (eta=0 verbosity strip), the three regimes coincide
exactly at eta=0 so the multi-regime panels for eta=0 figures are
redundant. The audit flagged five eta=0 figure sites for single-regime
re-render:

  1. 3_1_1_brini_canonical_e0.png      (hub formation diagnostic) — 3-row
  2. 3_1_2_topology_snapshots_e0.png   (topology snapshots) — 3-row
  3. 3_1_3_ddf_indegree_e0.png         (DDF) — 3-col
  4. 3_3_2_ccf_chain_e0.png            (CCFs pairs a-c) — 3-col
  5. 3_3_2_ccf_fiscal_e0.png           (CCFs pairs d-e) — 3-col

This script crops each PNG to the first regime column (no-tax). The crop
is done over the original raster; original PNGs are saved with a .bak
extension on first run and reused on subsequent runs to avoid
double-cropping.

Heuristic crop policy:
  - 3-row (1 col): crop top ~third of the body, keep full width
  - 3-col (1 row): crop left ~third of the body, keep full height
The body is the region below the suptitle; we estimate the suptitle as
the first ~7-10% of the height, leaving title intact and trimming the
data area.

We aim for visual cleanness, not pixel-perfect axis alignment — these are
visual panels, the captions and accompanying prose carry the load.
"""
import os, sys, shutil
sys.stdout.reconfigure(encoding='utf-8')
from PIL import Image

PROJ = os.path.dirname(os.path.abspath(__file__))


def trim(src_path, mode):
    """mode = 'row' (crop top third) or 'col' (crop left third)."""
    full = os.path.join(PROJ, src_path)
    if not os.path.exists(full):
        print(f'MISS {src_path}')
        return
    bak = full + '.bak'
    if not os.path.exists(bak):
        shutil.copy2(full, bak)
        print(f'  backup -> {bak}')
    src = Image.open(bak)
    w, h = src.size
    if mode == 'row':
        # Keep the title strip + first regime row.
        # Heuristic: title ~ 0..0.08*h, three regime rows take 0.08..1.0 of h.
        # First regime is from 0.08*h to 0.08 + (0.92/3)*h.
        title_h = int(0.08 * h)
        row_h = int((h - title_h) / 3)
        # Take full width, title + first row + a tiny margin below.
        out = src.crop((0, 0, w, title_h + row_h))
    else:  # 'col'
        # Keep the title strip + first regime column.
        # Heuristic: y-axis label + plot panel + x-axis label on left.
        # The first column is from x=0 to x = (w / 3) + a little overlap.
        col_w = int(w / 3)
        out = src.crop((0, 0, col_w, h))
    out.save(full)
    print(f'  trim {mode}: {bak} {src.size} -> {full} {out.size}')


def main():
    targets = [
        ('thesis_assets/ch3_1_topology/3_1_1_brini_canonical_e0.png', 'row'),
        ('thesis_assets/ch3_1_topology/3_1_2_topology_snapshots_e0.png', 'row'),
        ('thesis_assets/ch3_1_topology/3_1_3_ddf_indegree_e0.png', 'col'),
        ('thesis_assets/ch3_3_claim2_contagion/3_3_2_ccf_chain_e0.png', 'col'),
        ('thesis_assets/ch3_3_claim2_contagion/3_3_2_ccf_fiscal_e0.png', 'col'),
    ]
    for path, mode in targets:
        print(f'-- {path}')
        trim(path, mode)


if __name__ == '__main__':
    main()
