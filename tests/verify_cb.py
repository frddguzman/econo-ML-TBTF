"""Quick verification: cash_boost_random tier counts at t=0, 250, 500, 750, 999.
If replacement is working, super+pref counts should stay ~1+14 throughout."""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr
from collections import Counter

cfg = ddr.make_config(basis='equity', omega=0.50, eta=0.1, regime='socialized_tax')
cfg['cash_boost_random'] = True

m = interbank.Model()
m.test = True
m.configure(**cfg)
m.config.lender_change = lc.determine_algorithm("Boltzmann")
m.initialize(seed=26474, generate_plots=False)

snapshots = [0, 250, 500, 750, 999]
print(f't=init  : tier counts = {Counter(b._cash_boost for b in m.banks)}')
print(f'         super C={[round(b.C,1) for b in m.banks if b._cash_boost=="super"]}')
print(f'         pref C (first 3)={[round(b.C,1) for b in m.banks if b._cash_boost=="pref"][:3]}')
print(f'         default C (first 3)={[round(b.C,1) for b in m.banks if b._cash_boost=="default"][:3]}')

for t in range(m.config.T):
    m.forward()
    if t in snapshots:
        tiers = Counter(b._cash_boost for b in m.banks if hasattr(b, '_cash_boost'))
        super_C = [round(b.C,1) for b in m.banks if getattr(b, '_cash_boost', '')=='super']
        print(f't={t:4d}  : tier counts = {dict(tiers)}  super C={super_C}')

m.finish()
