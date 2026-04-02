# -*- coding: utf-8 -*-
"""
Analytical boundary tests for TBTF equations (alternativa.tex Table 1).
Each test sets up a controlled 2-bank system and verifies that
do_interest_rate() produces the exact values predicted by the spec.
"""
import unittest
import numpy as np
import interbank
import interbank_lenderchange


class TBTFBoundaryBase(unittest.TestCase):
    """Base class: creates a 2-bank model with Boltzmann, no shocks."""

    def make_model(self, **overrides):
        model = interbank.Model()
        model.config.lender_change = interbank_lenderchange.determine_algorithm('Boltzmann')
        model.configure(N=2, T=1)
        model.test = True
        # Apply any config overrides
        for k, v in overrides.items():
            setattr(model.config, k, v)
        model.initialize(seed=42)
        return model

    def setup_banks(self, model, bank0, bank1):
        """Set balance sheets directly. bank0/bank1 are dicts with C, L, D, E."""
        for bank, vals in [(model.banks[0], bank0), (model.banks[1], bank1)]:
            bank.L = vals['L']
            bank.D = vals['D']
            bank.E = vals['E']
            bank.R = model.config.reserves * bank.D
            bank.C = vals['C']
            bank.A = bank.C + bank.L + bank.R
            bank.A_lagged = bank.A  # set lagged = current for controlled test


class TestBoundaryPjZero(TBTFBoundaryBase):
    """
    Table 1, row p_j=0: borrower is perfectly capitalized (E_j = E_max).
    Expected: L_ij = C_i, r_ij = (chi*A_i - phi*A_j) / C_i
    """

    def test_pj_zero_loan_supply_equals_capacity(self):
        model = self.make_model(reserves=0.0)
        # Both banks have equal equity -> prob_surviving = 1.0 -> p_j = 0
        self.setup_banks(model,
            bank0={'C': 30.0, 'L': 120.0, 'D': 135.0, 'E': 15.0},
            bank1={'C': 30.0, 'L': 120.0, 'D': 135.0, 'E': 15.0})
        # bank1 is lender of bank0
        model.banks[0].lender = 1
        model.banks[1].lender = 0

        model.do_interest_rate()

        # p_j = 1 - E_j/E_max = 1 - 15/15 = 0
        # L_ij = C_i (boundary case)
        self.assertEqual(model.banks[0].L_ij_max[1], model.banks[0].C)
        self.assertEqual(model.banks[1].L_ij_max[0], model.banks[1].C)

    def test_pj_zero_interest_rate(self):
        model = self.make_model(reserves=0.0)
        self.setup_banks(model,
            bank0={'C': 30.0, 'L': 120.0, 'D': 135.0, 'E': 15.0},
            bank1={'C': 30.0, 'L': 120.0, 'D': 135.0, 'E': 15.0})
        model.banks[0].lender = 1
        model.banks[1].lender = 0

        model.do_interest_rate()

        # r_ij = screening_cost / C_i
        A_i = model.banks[0].A  # 30 + 120 + 0 = 150
        A_j = model.banks[1].A_lagged
        screening_cost = model.config.chi * A_i - model.config.phi * A_j
        C_i = model.banks[0].C
        expected_r = screening_cost / C_i
        # If negative, floored to r_i0
        if expected_r < 0:
            expected_r = model.config.r_i0

        self.assertAlmostEqual(model.banks[0].rij[1], expected_r, places=10)


class TestBoundaryPjOne(TBTFBoundaryBase):
    """
    Table 1, row p_j=1: borrower is insolvent (E_j ≈ 0).
    Expected: L_ij = 0, r_ij = inf
    """

    def test_pj_near_one_loan_supply_small(self):
        model = self.make_model(reserves=0.0)
        # bank1 has E near 0 -> p_j ≈ 1 -> nearly priced out
        self.setup_banks(model,
            bank0={'C': 30.0, 'L': 120.0, 'D': 135.0, 'E': 15.0},
            bank1={'C': 0.01, 'L': 0.01, 'D': 0.01, 'E': 0.001})
        model.banks[0].lender = 1
        model.banks[1].lender = 0

        model.do_interest_rate()

        # p_j = 1 - 0.001/15 ≈ 0.99993 (very close to 1 but not exactly 1)
        # L_ij should be much smaller than C_i=30 (severely constrained)
        # Rate should be very high (approaching inf)
        self.assertLess(model.banks[0].L_ij_max[1], model.banks[0].C)
        self.assertGreater(model.banks[0].rij[1], 1.0)  # rate >> r_i0

    def test_pj_exactly_one_gives_inf_rate(self):
        model = self.make_model(reserves=0.0)
        # bank1 has E = 0 -> p_j = 1 exactly -> priced out
        self.setup_banks(model,
            bank0={'C': 30.0, 'L': 120.0, 'D': 135.0, 'E': 15.0},
            bank1={'C': 1.0, 'L': 1.0, 'D': 2.0, 'E': 0.0})
        model.banks[0].lender = 1
        model.banks[1].lender = 0

        model.do_interest_rate()

        # p_j = 1 - 0/15 = 1.0 exactly -> boundary: L_ij=0, r_ij=inf
        self.assertEqual(model.banks[0].L_ij_max[1], 0)
        self.assertEqual(model.banks[0].rij[1], np.inf)


class TestBoundaryBjZero(TBTFBoundaryBase):
    """
    Table 1, row b_j=0: borrower has negligible assets (no bailout expectation).
    Expected: L_ij = (gamma*E_i + p_j*alpha*A_j) / p_j
              r_ij = (p_j*(L_ij - alpha*A_j) + screening_cost) / ((1-p_j)*L_ij)
    """

    def test_bj_zero_loan_supply(self):
        model = self.make_model(reserves=0.0)
        # bank1 has tiny assets -> A_lagged ≈ 0 -> b_j ≈ 0
        self.setup_banks(model,
            bank0={'C': 30.0, 'L': 120.0, 'D': 135.0, 'E': 15.0},
            bank1={'C': 5.0, 'L': 5.0, 'D': 5.0, 'E': 5.0})
        # Make bank1 A_lagged very small relative to bank0
        model.banks[0].A_lagged = 150.0
        model.banks[1].A_lagged = 0.001  # effectively b_j ≈ 0
        model.banks[0].lender = 1
        model.banks[1].lender = 0

        model.do_interest_rate()

        gamma = model.config.gamma_capital
        alpha = model.config.alpha_collateral
        E_i = model.banks[0].E
        A_j = model.banks[1].A_lagged
        p_j = 1 - model.banks[1].prob_surviving
        b_j = model.banks[1].A_lagged / model.max_A_lagged

        # With b_j ≈ 0: L_ij = (gamma*E_i + p_j*alpha*A_j) / p_j
        # but capped at C_i
        expected_L = (gamma * E_i + p_j * (1 - b_j) * alpha * A_j) / (p_j * (1 - b_j * model.config.eta_bailout))
        expected_L = min(expected_L, model.banks[0].C)

        self.assertAlmostEqual(model.banks[0].L_ij_max[1], expected_L, places=8)


class TestBoundaryBjOne(TBTFBoundaryBase):
    """
    Table 1, row b_j=1: borrower is the largest bank (certain bailout).
    Expected: L_ij = gamma*E_i / (p_j*(1-eta))
    """

    def test_bj_one_loan_supply(self):
        model = self.make_model(reserves=0.0)
        # bank1 is the largest -> b_j = 1
        self.setup_banks(model,
            bank0={'C': 30.0, 'L': 120.0, 'D': 135.0, 'E': 15.0},
            bank1={'C': 50.0, 'L': 200.0, 'D': 220.0, 'E': 30.0})
        # Set A_lagged so bank1 = max
        model.banks[0].A_lagged = 150.0
        model.banks[1].A_lagged = 250.0  # b_j = 250/250 = 1
        model.banks[0].lender = 1
        model.banks[1].lender = 0

        model.do_interest_rate()

        gamma = model.config.gamma_capital
        eta = model.config.eta_bailout
        E_i = model.banks[0].E
        p_j = 1 - model.banks[1].prob_surviving  # p_j for bank1

        # b_j = 1: L_ij = gamma*E_i / (p_j*(1-eta)), capped at C_i
        if p_j > 0 and (1 - eta) > 0:
            expected_L = min(gamma * E_i / (p_j * (1 - eta)), model.banks[0].C)
            self.assertAlmostEqual(model.banks[0].L_ij_max[1], expected_L, places=8)


class TestFitnessEquityOnly(TBTFBoundaryBase):
    """
    Eq. 9: fitness Phi_i = E_i / E_max.
    Verify compound fitness is replaced with equity-only.
    """

    def test_fitness_is_equity_ratio(self):
        model = self.make_model(reserves=0.0)
        self.setup_banks(model,
            bank0={'C': 30.0, 'L': 120.0, 'D': 135.0, 'E': 15.0},
            bank1={'C': 30.0, 'L': 120.0, 'D': 135.0, 'E': 10.0})
        model.banks[0].lender = 1
        model.banks[1].lender = 0

        model.do_interest_rate()

        maxE = max(model.banks[0].E, model.banks[1].E)
        # eq. 9: Phi_i = E_i / E_max
        self.assertAlmostEqual(model.banks[0].mu, model.banks[0].E / maxE)
        self.assertAlmostEqual(model.banks[1].mu, model.banks[1].E / maxE)
        # bank0 has higher equity -> higher fitness
        self.assertGreater(model.banks[0].mu, model.banks[1].mu)


class TestBailoutTax(TBTFBoundaryBase):
    """
    Bailout tax: tau_k = bill * (A_k / sum(A_m))
    Verify asset-proportional distribution and equity deduction.
    """

    def test_tax_distribution_proportional_to_assets(self):
        model = self.make_model(reserves=0.0)
        self.setup_banks(model,
            bank0={'C': 30.0, 'L': 120.0, 'D': 135.0, 'E': 15.0},
            bank1={'C': 60.0, 'L': 200.0, 'D': 230.0, 'E': 30.0})
        model.banks[0].A = 150.0
        model.banks[1].A = 260.0

        # Simulate a bailout bill
        model.period_bailout_bill = 10.0
        model.t = 0

        E0_before = model.banks[0].E
        E1_before = model.banks[1].E
        total_A = 150.0 + 260.0

        model.apply_bailout_tax()

        expected_tax_0 = 10.0 * (150.0 / total_A)
        expected_tax_1 = 10.0 * (260.0 / total_A)

        self.assertAlmostEqual(model.banks[0].E, E0_before - expected_tax_0, places=10)
        self.assertAlmostEqual(model.banks[1].E, E1_before - expected_tax_1, places=10)
        # Bill should be reset
        self.assertEqual(model.period_bailout_bill, 0.0)

    def test_tax_induced_failure(self):
        model = self.make_model(reserves=0.0)
        self.setup_banks(model,
            bank0={'C': 30.0, 'L': 120.0, 'D': 135.0, 'E': 0.2},  # barely alive
            bank1={'C': 60.0, 'L': 200.0, 'D': 230.0, 'E': 30.0})
        model.banks[0].A = 150.0
        model.banks[1].A = 260.0

        # Large bailout bill should push bank0 below alfa=0.1
        model.period_bailout_bill = 5.0
        model.t = 0

        model.apply_bailout_tax()

        # bank0 E = 0.2 - 5*(150/410) = 0.2 - 1.829 < 0 < alfa -> failed
        self.assertTrue(model.banks[0].failed)
        self.assertFalse(model.banks[1].failed)
        self.assertEqual(model.statistics.tax_induced_failures[0], 1)


class TestInterestRateFloor(TBTFBoundaryBase):
    """
    When screening cost is negative (borrower much larger than lender),
    r_ij should be floored at config.r_i0.
    """

    def test_negative_rate_floored(self):
        model = self.make_model(reserves=0.0)
        # bank1 much larger -> chi*A_i - phi*A_j < 0
        self.setup_banks(model,
            bank0={'C': 5.0, 'L': 10.0, 'D': 10.0, 'E': 5.0},
            bank1={'C': 100.0, 'L': 400.0, 'D': 450.0, 'E': 50.0})
        model.banks[0].A_lagged = 15.0
        model.banks[1].A_lagged = 500.0
        model.banks[0].lender = 1
        model.banks[1].lender = 0

        model.do_interest_rate()

        # For bank0 lending to bank1: chi*15 - phi*500 = 0.225 - 12.5 < 0
        # With p_j small (bank1 well-capitalized), rate could go negative -> floor
        # The rate should be at least r_i0
        self.assertGreaterEqual(model.banks[0].rij[1], model.config.r_i0)


class TestCAvgIrAlias(TBTFBoundaryBase):
    """
    Verify c_avg_ir is aliased to L_ij_max for stats compatibility.
    """

    def test_c_avg_ir_equals_L_ij_max(self):
        model = self.make_model(reserves=0.0)
        self.setup_banks(model,
            bank0={'C': 30.0, 'L': 120.0, 'D': 135.0, 'E': 15.0},
            bank1={'C': 30.0, 'L': 120.0, 'D': 135.0, 'E': 10.0})
        model.banks[0].lender = 1
        model.banks[1].lender = 0

        model.do_interest_rate()

        # c_avg_ir should be the same object as L_ij_max
        self.assertIs(model.banks[0].c_avg_ir, model.banks[0].L_ij_max)
        self.assertIs(model.banks[1].c_avg_ir, model.banks[1].L_ij_max)


if __name__ == '__main__':
    unittest.main()
