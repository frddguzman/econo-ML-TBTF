"""Standalone A/B test: alternative p_default = E_min/E_j formulation.

Subclasses interbank.Model to override do_interest_rate_common_part with the
new formula. Doesn't modify interbank.py — safe to run alongside the seed
search.

A: baseline (p_default = 1 - E/E_max, the current spec)
B: min-ratio (p_default = E_min/E_j, hyperbolic, bounded above E_min/E_max)
"""
import os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc

SEED = 26474
BASIS = 'equity'
OMEGA = 0.53
ETA = 0.1
REGIME = 'socialized_tax'


class MinRatioModel(interbank.Model):
    """p_default = E_min/E_j → prob_surviving = 1 - E_min/E_j.

    Bounded away from 0 (best bank has p_j ≈ E_min/E_max ≈ 0.3 instead of 0),
    so the C_i clamp doesn't catch infinity edge cases. Hyperbolic in equity:
    small differences near E_min produce large p_j differentials.
    """
    def do_interest_rate_common_part(self):
        if len(self.banks) <= 1:
            return None
        self.maxE = max(self.banks, key=lambda k: k.E).E
        active_E = [b.E for b in self.banks if not b.failed and b.E > 0]
        self.minE = min(active_E) if active_E else 0.01
        for bank in self.banks:
            if bank.E > 0 and self.minE > 0:
                # p_default = E_min/E_j, so prob_surviving = 1 - E_min/E_j
                # Clamped to [0, 1) — the best bank gets prob_surviving = 1 - E_min/E_max
                bank.prob_surviving = max(0.0, min(0.9999, 1.0 - self.minE / bank.E))
            else:
                bank.prob_surviving = 0  # certain-default sentinel
            bank.A = bank.C + bank.L + bank.R
        # Rest of the method (unchanged from original)
        self.max_A_lagged = max((b.A_lagged for b in self.banks if not b.failed), default=1.0)
        basis = self.config.fitness_basis
        if basis == "loan_book":
            self.L_max = max((b.L for b in self.banks if not b.failed), default=1.0)
            if self.L_max <= 0:
                self.L_max = 1.0
        elif basis == "bilateral":
            self.L_max_system = self._compute_L_max_system()


def make_cfg():
    return dict(
        N=50, T=1000,
        mu=0.7, omega=OMEGA,
        eta_bailout=ETA, rho=0.4,
        gamma_capital=0.10, alpha_collateral=0.05,
        beta=5, alfa=0.1,
        fiscal_regime=REGIME,
        fund_levy_rate=0.0001,
        fitness_basis=BASIS, fitness_inertia=0.0,
        equity_heterogeneity=False, equity_cv=0.5,
    )


def _rle_max(keys):
    if not keys: return 0
    runs = []; cur = keys[0]; start = 0
    for i in range(1, len(keys)):
        if keys[i] != cur:
            runs.append(i - start); cur = keys[i]; start = i
    runs.append(len(keys) - start)
    return max(runs)


def _clamp_rate_for(model):
    cnt_total = 0; cnt_clamped = 0
    max_A_lag = max((b.A_lagged for b in model.banks if not b.failed and getattr(b, 'A_lagged', 0) > 0),
                    default=1.0)
    for i, bank_i in enumerate(model.banks):
        if bank_i.failed: continue
        for j, bank_j in enumerate(model.banks):
            if i == j or bank_j.failed: continue
            E_i = bank_i.E
            A_j = getattr(bank_j, 'A', bank_j.C + bank_j.L + bank_j.R)
            p_j = (1 - bank_j.prob_surviving)
            if p_j <= 0 or p_j >= 1: continue
            b_j = (getattr(bank_j, 'A_lagged', A_j) / max_A_lag)
            denom = p_j * (1 - b_j * model.config.eta_bailout)
            if denom <= 0: continue
            num = (model.config.gamma_capital * E_i +
                   p_j * (1 - b_j) * model.config.alpha_collateral * A_j)
            unclamped = num / denom
            cnt_total += 1
            if unclamped > bank_i.C + 1e-12:
                cnt_clamped += 1
    return (cnt_clamped / cnt_total) if cnt_total else 0.0


def run_one(model_cls, label):
    cfg = make_cfg()
    m = model_cls()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm('Boltzmann')
    m.initialize(seed=SEED, generate_plots=False)

    clamp_samples = []
    minE_samples = []
    pj_top_samples = []
    pj_min_samples = []
    for t in range(m.config.T):
        m.forward()
        if t in (100, 250, 500, 750, 999):
            clamp_samples.append(_clamp_rate_for(m))
            active = [b for b in m.banks if not b.failed]
            if active:
                pjs = [1 - b.prob_surviving for b in active]
                minE_samples.append(getattr(m, 'minE', min(b.E for b in active)) /
                                     max(b.E for b in active))
                pj_top_samples.append(max(pjs))
                pj_min_samples.append(min(pjs))
    m.finish()
    T = m.t
    s = m.statistics
    bl = list(s.best_lender[:T]); bg = list(s.best_lender_generation[:T])
    keys = [(b, g) for b, g in zip(bl, bg) if b >= 0]
    deg = list(s.best_lender_clients[:T])
    deg_pos = [d for d in deg if d is not None and d >= 0]
    return {
        'label':      label,
        'total_bk':   int(sum(s.bankruptcy[:T])),
        'max_tenure': _rle_max(keys),
        'mean_deg':   statistics.mean(deg_pos) if deg_pos else 0,
        'max_deg':    max(deg_pos) if deg_pos else 0,
        'p95_deg':    statistics.quantiles(deg_pos, n=20)[18] if len(deg_pos) >= 20 else max(deg_pos, default=0),
        'clamp_mean': statistics.mean(clamp_samples) if clamp_samples else 0,
        'minE_over_maxE': statistics.mean(minE_samples) if minE_samples else 0,
        'pj_max':     statistics.mean(pj_top_samples) if pj_top_samples else 0,
        'pj_min':     statistics.mean(pj_min_samples) if pj_min_samples else 0,
    }


print(f'min-ratio A/B test on Cand A . st . eta=0.1 . seed {SEED}')
print('=' * 96)
print(f'{"label":<30} {"tot_bk":>7} {"max_t":>6} {"meandeg":>8} {"maxdeg":>7} {"clamp%":>7} {"minE/maxE":>10} {"pj_max":>7} {"pj_min":>7}')
print('-' * 96)
results = []
for cls, label in [(interbank.Model, 'baseline (1 - E/E_max)'),
                   (MinRatioModel,    'min-ratio (E_min/E_j)')]:
    r = run_one(cls, label)
    results.append(r)
    print(f'{r["label"]:<30} {r["total_bk"]:>7} {r["max_tenure"]:>6} '
          f'{r["mean_deg"]:>8.2f} {r["max_deg"]:>7} {r["clamp_mean"]*100:>6.1f}% '
          f'{r["minE_over_maxE"]:>10.3f} {r["pj_max"]:>7.3f} {r["pj_min"]:>7.3f}')

print('\n' + '=' * 96)
base = results[0]; new = results[1]
print('Comparison:')
print(f'  total_bk:    {base["total_bk"]:>5} -> {new["total_bk"]:>5} ({(new["total_bk"]/base["total_bk"]-1)*100:+.1f}%)')
print(f'  max_tenure:  {base["max_tenure"]:>5} -> {new["max_tenure"]:>5}')
print(f'  max_deg:     {base["max_deg"]:>5} -> {new["max_deg"]:>5}')
print(f'  clamp_rate:  {base["clamp_mean"]*100:>4.1f}% -> {new["clamp_mean"]*100:>4.1f}%')
print(f'  pj range:    [{base["pj_min"]:.3f}, {base["pj_max"]:.3f}] -> [{new["pj_min"]:.3f}, {new["pj_max"]:.3f}]')
