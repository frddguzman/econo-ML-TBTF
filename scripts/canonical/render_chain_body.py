"""Chain runner for all body render scripts.

Runs all body render + analysis scripts sequentially. Reports progress.
Skips appendix scripts (deferred per user direction 2026-05-09).
"""
import os, sys, subprocess, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


SCRIPTS = [
    # sec3.1 Network topology
    ('sec3.1.5 hub stability table',         'render_hub_stability_summary.py'),
    ('sec3.1.5 hub turnover causality',      'analyze_hub_turnover_causality.py'),
    ('topology dump (missing regimes)',    'dump_topology_multi_regime.py'),
    ('sec3.1.2 topology snapshots',          'render_topology_snapshots.py'),
    ('sec3.1.3 DDF + power-law fit',         'analyze_ddf_indegree.py'),
    ('sec3.1.4 k-core decomposition',        'analyze_kcore.py'),
    ('sec3.1.7 omega-section',                   'render_omega_section.py'),
    ('sec3.1.8 network metrics regression',  'analyze_network_metrics_regression.py'),
    ('sec3.1.1 Brini canonical',             'render_brini_canonical.py'),

    # sec3.2 Claim 1 (hub stability)
    ('sec3.2.1 avg_ten decoupling',          'render_avg_ten_decoupling.py'),
    ('sec3.2.2 tenure thresholds',           'render_tenure_thresholds.py'),

    # sec3.3 Claim 2 (contagion)
    ('sec3.3.3 contagion vs rho',              'render_contagion_vs_rho.py'),
    ('sec3.3.4 contagion vs eta',              'render_contagion_vs_eta.py'),
    ('sec3.3.5 contagion vs alpha',              'render_contagion_vs_alpha.py'),
    ('sec3.3.6 channel decomp + mortality',  'render_channel_decomp_vs_eta.py'),

    # sec3.4 Claim 3 (rho)
    ('sec3.4.3 rho-sweep multi-regime',        'render_rho_sweep.py'),
    ('sec3.4.4 rho-peak persistence vs alpha',     'render_rho_peak_persistence_alpha.py'),
    ('sec3.4.5 channel decomp by rho',         'render_channel_decomp_by_rho.py'),

    # sec3.5 Claim 4 (eta)
    ('sec3.5.2 total_bk vs eta',               'render_total_bk_vs_eta.py'),
    ('sec3.5.5 fiscal deaths vs eta',          'render_fiscal_deaths_vs_eta.py'),
    ('sec3.5.6 levy calibration',            'render_levy_calibration.py'),
    ('sec3.5.7 categorical regression',      'analyze_categorical_regression.py'),
]


def main():
    py = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv', 'Scripts', 'python.exe')
    if not os.path.exists(py):
        py = sys.executable
    log = []
    n = len(SCRIPTS)
    for i, (label, script) in enumerate(SCRIPTS, 1):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script)
        if not os.path.exists(path):
            print(f'[{i}/{n}] {label} — SKIP (script not found: {script})', flush=True)
            log.append((label, 'MISSING', 0))
            continue
        t0 = time.time()
        print(f'\n[{i}/{n}] {label} -> {script}', flush=True)
        result = subprocess.run([py, script], capture_output=True, text=True,
                                cwd=os.path.dirname(os.path.abspath(__file__)))
        dt = time.time() - t0
        if result.returncode != 0:
            print(f'  ERROR (exit {result.returncode}, {dt:.1f}s)', flush=True)
            print(f'  STDERR (last 500 chars): {result.stderr[-500:]}', flush=True)
            log.append((label, f'ERR({result.returncode})', dt))
        else:
            print(f'  OK ({dt:.1f}s)', flush=True)
            log.append((label, 'OK', dt))
    print('\n' + '='*72)
    print('CHAIN COMPLETE — Summary:')
    print('='*72)
    for label, status, dt in log:
        marker = '+' if status == 'OK' else 'x'
        print(f'  {marker} [{status:>10}] {dt:>6.1f}s  {label}')
    print()


if __name__ == '__main__':
    main()
