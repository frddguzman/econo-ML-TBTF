#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Parameter sweep: cumulative bankruptcies vs bailout recovery fraction η."""

import numpy as np
import matplotlib.pyplot as plt
import interbank

eta_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
total_bankruptcies = []
tax_failures = []

print(f"Sweeping eta_bailout over {len(eta_values)} values (N=50, T=1000, seed=26462)...")

model = interbank.Model()
model.test = True

for eta in eta_values:
    model.configure(
        N=50, T=1000,
        eta_bailout=eta,
        gamma_capital=0.10,
        alpha_collateral=0.20,
        beta=5,
        omega=0.5,
        lc='Boltzmann',
    )
    model.initialize(seed=26462, generate_plots=False)
    model.simulate_full()
    model.finish()

    tb = model.statistics.bankruptcy.sum()
    tf = model.statistics.tax_induced_failures.sum()
    total_bankruptcies.append(tb)
    tax_failures.append(tf)
    print(f"  η={eta:<5.2f}  bankruptcies={tb:<6}  tax-induced={tf}")

# Plot
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(eta_values, total_bankruptcies, 'o-', color='black', lw=2, label='Total bankruptcies')
ax.plot(eta_values, tax_failures, 's--', color='red', lw=2, label='Tax-induced failures')

ax.set_xlabel('Bailout recovery fraction (η)', fontsize=12)
ax.set_ylabel('Cumulative failures (T = 1000)', fontsize=12)
ax.set_title('System stability vs. state bailout policy', fontsize=13, fontweight='bold')
ax.set_xticks(eta_values)
ax.legend(fontsize=11)
ax.grid(True, ls=':', alpha=0.6)
fig.tight_layout()

fig.savefig('sweep_eta.png', dpi=300)
print(f"\nSaved sweep_eta.png")
plt.show()
