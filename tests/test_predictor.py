#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tesztek a Predictor osztály Poisson függvényéhez
"""

import sys
import math
from pathlib import Path

# Add parent directory to path to import main
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import Predictor


def test_poisson_probability_basic():
    """Alapvető Poisson valószínűség teszt"""
    predictor = Predictor()
    
    # Lambda=1, k=1 esetén P(X=1) = 1 * e^(-1) / 1! = e^(-1) ≈ 0.368
    prob = predictor.poisson_probability(1.0, 1)
    expected = math.exp(-1.0)
    assert abs(prob - expected) < 0.001, f"Expected {expected}, got {prob}"
    
    print(f"✓ test_poisson_probability_basic passed: P(X=1|λ=1) = {prob:.4f}")


def test_poisson_probability_zero_goals():
    """Nulla gól valószínűség teszt"""
    predictor = Predictor()
    
    # Lambda=1.5, k=0 esetén P(X=0) = e^(-1.5) ≈ 0.223
    prob = predictor.poisson_probability(1.5, 0)
    expected = math.exp(-1.5)
    assert abs(prob - expected) < 0.001, f"Expected {expected}, got {prob}"
    
    print(f"✓ test_poisson_probability_zero_goals passed: P(X=0|λ=1.5) = {prob:.4f}")


def test_poisson_probability_high_goals():
    """Magas gólszám valószínűség teszt"""
    predictor = Predictor()
    
    # Lambda=2.0, k=3 esetén P(X=3) = 2^3 * e^(-2) / 3! = 8 * e^(-2) / 6
    prob = predictor.poisson_probability(2.0, 3)
    expected = (2.0 ** 3) * math.exp(-2.0) / math.factorial(3)
    assert abs(prob - expected) < 0.001, f"Expected {expected}, got {prob}"
    
    print(f"✓ test_poisson_probability_high_goals passed: P(X=3|λ=2.0) = {prob:.4f}")


def test_poisson_probability_negative_lambda():
    """Negatív lambda kezelés teszt"""
    predictor = Predictor()
    
    # Negatív lambda esetén 0.0-t kell visszaadnia
    prob = predictor.poisson_probability(-1.0, 2)
    assert prob == 0.0, f"Expected 0.0 for negative lambda, got {prob}"
    
    print(f"✓ test_poisson_probability_negative_lambda passed: P(X=2|λ=-1) = {prob}")


def test_score_distribution_sum():
    """Score distribution összege ~1.0 kell legyen"""
    predictor = Predictor()
    
    distribution = predictor.calculate_score_distribution(1.5, 1.2, max_goals=5)
    total_prob = sum(distribution.values())
    
    # Az összeg közel 1.0-hoz kell legyen (nem pontosan, mert max_goals korlátozott)
    assert 0.95 < total_prob < 1.0, f"Total probability should be close to 1.0, got {total_prob}"
    
    print(f"✓ test_score_distribution_sum passed: Total probability = {total_prob:.4f}")


def test_1x2_probabilities_sum():
    """1X2 valószínűségek összege 1.0 kell legyen"""
    predictor = Predictor()
    
    distribution = predictor.calculate_score_distribution(1.5, 1.0)
    probs_1x2 = predictor.calculate_1x2_probabilities(distribution)
    
    total = probs_1x2["home_win"] + probs_1x2["draw"] + probs_1x2["away_win"]
    
    # Az összeg nagyon közel 1.0-hoz kell legyen
    assert 0.99 < total < 1.01, f"1X2 probabilities should sum to 1.0, got {total}"
    
    print(f"✓ test_1x2_probabilities_sum passed: Sum = {total:.4f}")
    print(f"  Home win: {probs_1x2['home_win']:.4f}")
    print(f"  Draw: {probs_1x2['draw']:.4f}")
    print(f"  Away win: {probs_1x2['away_win']:.4f}")


def test_top_scorelines_ordering():
    """Top scorelines helyes rendezés teszt"""
    predictor = Predictor()
    
    distribution = predictor.calculate_score_distribution(1.5, 1.0, max_goals=4)
    top_scorelines = predictor.get_top_n_scorelines(distribution, n=5)
    
    # Ellenőrizd, hogy csökkenő sorrendben vannak
    for i in range(len(top_scorelines) - 1):
        assert top_scorelines[i]["probability"] >= top_scorelines[i+1]["probability"], \
            "Scorelines should be ordered by probability (descending)"
    
    print(f"✓ test_top_scorelines_ordering passed")
    print(f"  Top scoreline: {top_scorelines[0]['scoreline']} ({top_scorelines[0]['probability']:.4f})")


def run_all_tests():
    """Összes teszt futtatása"""
    print("\n" + "="*60)
    print("Running Predictor Poisson Tests")
    print("="*60 + "\n")
    
    tests = [
        test_poisson_probability_basic,
        test_poisson_probability_zero_goals,
        test_poisson_probability_high_goals,
        test_poisson_probability_negative_lambda,
        test_score_distribution_sum,
        test_1x2_probabilities_sum,
        test_top_scorelines_ordering
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
