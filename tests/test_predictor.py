#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tesztek a predictor funkciókhoz: Poisson PMF, BTTS és Over/Under számításokhoz.

Note: Ez a teszt standalone, nem importál a betting.py-ból, hogy kerülje a függőségeket.
A tesztelt funkciók másolatát tartalmazza.
"""

import unittest
import math
from math import exp, factorial
from typing import List


# ========== Tesztelt funkciók (másolt betting.py-ból) ==========

def poisson_p(k: int, lam: float) -> float:
    """Poisson eloszlás PMF"""
    if lam<=0: return 1.0 if k==0 else 0.0
    return (lam**k * exp(-lam)) / factorial(k)


def btts_probability(lh: float, la: float) -> float:
    """BTTS valószínűség zárt formula alapján"""
    return 1 - math.exp(-lh) - math.exp(-la) + math.exp(-(lh+la))


def over_under_probabilities(lambda_home: float, lambda_away: float, thresholds: List[float] = None) -> dict:
    """
    Calculate Over/Under probabilities for multiple thresholds using Poisson distribution.
    
    Args:
        lambda_home: Expected goals for home team (Poisson parameter)
        lambda_away: Expected goals for away team (Poisson parameter)
        thresholds: List of goal thresholds (default: [0.5, 1.5, 2.5, 3.5])
    
    Returns:
        Dictionary with threshold as key and dict of {over, under, balanced} probabilities
    """
    if thresholds is None:
        thresholds = [0.5, 1.5, 2.5, 3.5]
    
    results = {}
    
    for threshold in thresholds:
        # Calculate probability for each total goals scenario
        # Since we need total_goals > threshold for Over, we sum P(home=h, away=a) for all h+a > threshold
        over_prob = 0.0
        
        # Calculate up to reasonable upper bound (3 sigma ~ mean + 3*sqrt(mean))
        max_goals_home = int(lambda_home + 4 * math.sqrt(lambda_home) + 10)
        max_goals_away = int(lambda_away + 4 * math.sqrt(lambda_away) + 10)
        
        for h in range(max_goals_home + 1):
            for a in range(max_goals_away + 1):
                total = h + a
                if total > threshold:
                    prob_h = poisson_p(h, lambda_home)
                    prob_a = poisson_p(a, lambda_away)
                    over_prob += prob_h * prob_a
        
        under_prob = 1.0 - over_prob
        
        # Balanced: which one has higher probability
        balanced = "over" if over_prob > under_prob else "under"
        
        results[str(threshold)] = {
            "over": over_prob,
            "under": under_prob,
            "balanced": balanced,
            "probabilities": {
                "over": over_prob,
                "under": under_prob
            }
        }
    
    return results


def btts_score_distribution(lambda_home: float, lambda_away: float) -> dict:
    """
    Calculate BTTS probability using score distribution method.
    Sums P(home=h, away=a) for all combinations where both h > 0 and a > 0.
    
    Args:
        lambda_home: Expected goals for home team (Poisson parameter)
        lambda_away: Expected goals for away team (Poisson parameter)
    
    Returns:
        Dictionary with btts_yes and btts_no probabilities
    """
    btts_yes_prob = 0.0
    
    # Calculate up to reasonable upper bound
    max_goals_home = int(lambda_home + 4 * math.sqrt(lambda_home) + 10)
    max_goals_away = int(lambda_away + 4 * math.sqrt(lambda_away) + 10)
    
    for h in range(1, max_goals_home + 1):  # Start from 1 (both teams must score)
        for a in range(1, max_goals_away + 1):
            prob_h = poisson_p(h, lambda_home)
            prob_a = poisson_p(a, lambda_away)
            btts_yes_prob += prob_h * prob_a
    
    btts_no_prob = 1.0 - btts_yes_prob
    
    return {
        "btts_yes": btts_yes_prob,
        "btts_no": btts_no_prob,
        "balanced": "yes" if btts_yes_prob > btts_no_prob else "no"
    }


# ========== Tesztek ==========


class TestPoissonPMF(unittest.TestCase):
    """Poisson eloszlás PMF tesztjei"""
    
    def test_poisson_zero_lambda(self):
        """Teszt: lambda=0 esetén P(0)=1, más esetben 0"""
        self.assertAlmostEqual(poisson_p(0, 0.0), 1.0, places=5)
        self.assertAlmostEqual(poisson_p(1, 0.0), 0.0, places=5)
        self.assertAlmostEqual(poisson_p(5, 0.0), 0.0, places=5)
    
    def test_poisson_basic_values(self):
        """Teszt: alapvető Poisson értékek ellenőrzése"""
        # lambda=1.0, P(k=0) = e^(-1) ≈ 0.3679
        expected_p0 = math.exp(-1.0)
        self.assertAlmostEqual(poisson_p(0, 1.0), expected_p0, places=4)
        
        # lambda=1.0, P(k=1) = 1 * e^(-1) / 1! ≈ 0.3679
        expected_p1 = 1.0 * math.exp(-1.0)
        self.assertAlmostEqual(poisson_p(1, 1.0), expected_p1, places=4)
        
        # lambda=2.0, P(k=2) = 2^2 * e^(-2) / 2! ≈ 0.2707
        expected_p2 = (2.0**2 * math.exp(-2.0)) / 2.0
        self.assertAlmostEqual(poisson_p(2, 2.0), expected_p2, places=4)
    
    def test_poisson_sum_approximates_one(self):
        """Teszt: Poisson eloszlás összege közelíti az 1-et"""
        lambda_val = 1.5
        total = sum(poisson_p(k, lambda_val) for k in range(20))
        self.assertGreater(total, 0.99)  # Should be close to 1


class TestBTTSCalculations(unittest.TestCase):
    """BTTS (Both Teams To Score) számítások tesztjei"""
    
    def test_btts_probability_basic(self):
        """Teszt: BTTS valószínűség alapvető ellenőrzése"""
        # lambda_home=1.5, lambda_away=1.2
        prob = btts_probability(1.5, 1.2)
        self.assertGreater(prob, 0.0)
        self.assertLess(prob, 1.0)
        # Should be around 0.48-0.52
        self.assertGreater(prob, 0.4)
        self.assertLess(prob, 0.6)
    
    def test_btts_probability_zero_lambda(self):
        """Teszt: Ha valamelyik lambda 0, akkor BTTS valószínűség 0"""
        prob1 = btts_probability(0.0, 1.5)
        prob2 = btts_probability(1.5, 0.0)
        self.assertAlmostEqual(prob1, 0.0, places=5)
        self.assertAlmostEqual(prob2, 0.0, places=5)
    
    def test_btts_score_distribution(self):
        """Teszt: BTTS score distribution módszer"""
        result = btts_score_distribution(1.5, 1.2)
        
        # Check structure
        self.assertIn("btts_yes", result)
        self.assertIn("btts_no", result)
        self.assertIn("balanced", result)
        
        # Check probabilities sum to 1
        total = result["btts_yes"] + result["btts_no"]
        self.assertAlmostEqual(total, 1.0, places=5)
        
        # Check values are in valid range
        self.assertGreaterEqual(result["btts_yes"], 0.0)
        self.assertLessEqual(result["btts_yes"], 1.0)
        self.assertGreaterEqual(result["btts_no"], 0.0)
        self.assertLessEqual(result["btts_no"], 1.0)
        
        # Check balanced field
        self.assertIn(result["balanced"], ["yes", "no"])
    
    def test_btts_consistency(self):
        """Teszt: btts_probability és btts_score_distribution konzisztenciája"""
        lambda_h, lambda_a = 1.8, 1.5
        
        prob_formula = btts_probability(lambda_h, lambda_a)
        result_dist = btts_score_distribution(lambda_h, lambda_a)
        
        # Both methods should give similar results (within small tolerance)
        # Score distribution is more accurate but both should be close
        diff = abs(prob_formula - result_dist["btts_yes"])
        self.assertLess(diff, 0.05, 
            f"Methods differ too much: formula={prob_formula:.4f}, dist={result_dist['btts_yes']:.4f}")


class TestOverUnderCalculations(unittest.TestCase):
    """Over/Under számítások tesztjei"""
    
    def test_over_under_basic_structure(self):
        """Teszt: Over/Under kimenet struktúrája"""
        result = over_under_probabilities(1.5, 1.2, thresholds=[2.5])
        
        self.assertIn("2.5", result)
        threshold_data = result["2.5"]
        
        self.assertIn("over", threshold_data)
        self.assertIn("under", threshold_data)
        self.assertIn("balanced", threshold_data)
        self.assertIn("probabilities", threshold_data)
    
    def test_over_under_probabilities_sum_to_one(self):
        """Teszt: Over és Under valószínűségek összege 1"""
        result = over_under_probabilities(1.5, 1.2, thresholds=[0.5, 1.5, 2.5, 3.5])
        
        for threshold_str, data in result.items():
            total = data["over"] + data["under"]
            self.assertAlmostEqual(total, 1.0, places=5,
                msg=f"Threshold {threshold_str}: over+under != 1.0")
    
    def test_over_under_default_thresholds(self):
        """Teszt: alapértelmezett küszöbök (0.5, 1.5, 2.5, 3.5)"""
        result = over_under_probabilities(1.5, 1.2)
        
        self.assertIn("0.5", result)
        self.assertIn("1.5", result)
        self.assertIn("2.5", result)
        self.assertIn("3.5", result)
    
    def test_over_under_logical_ordering(self):
        """Teszt: magasabb küszöb -> alacsonyabb over valószínűség"""
        result = over_under_probabilities(1.5, 1.2, thresholds=[0.5, 1.5, 2.5, 3.5])
        
        # Over probabilities should decrease as threshold increases
        over_05 = result["0.5"]["over"]
        over_15 = result["1.5"]["over"]
        over_25 = result["2.5"]["over"]
        over_35 = result["3.5"]["over"]
        
        self.assertGreater(over_05, over_15, "Over 0.5 should be > Over 1.5")
        self.assertGreater(over_15, over_25, "Over 1.5 should be > Over 2.5")
        self.assertGreater(over_25, over_35, "Over 2.5 should be > Over 3.5")
    
    def test_over_under_balanced_field(self):
        """Teszt: balanced mező helyesen van-e beállítva"""
        result = over_under_probabilities(2.0, 2.0, thresholds=[2.5])
        
        # With lambda_total=4.0, Over 2.5 should be more likely
        data = result["2.5"]
        if data["over"] > data["under"]:
            self.assertEqual(data["balanced"], "over")
        else:
            self.assertEqual(data["balanced"], "under")
    
    def test_over_under_extreme_values(self):
        """Teszt: extrém lambda értékek kezelése"""
        # Very low goals expected (lambda_total = 0.8)
        result_low = over_under_probabilities(0.5, 0.3, thresholds=[1.5])
        self.assertLess(result_low["1.5"]["over"], 0.3)  # Under more likely for 1.5
        
        # Very high goals expected
        result_high = over_under_probabilities(3.0, 3.0, thresholds=[0.5])
        self.assertGreater(result_high["0.5"]["over"], 0.99)  # Over very likely


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
