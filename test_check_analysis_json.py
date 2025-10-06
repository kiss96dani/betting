#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for check_analysis_json.py

Run with: python3 test_check_analysis_json.py
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from check_analysis_json import (
    MarketNormalizer,
    extract_bookmaker_odds_from_odds_response,
    calculate_margin,
    NormalizedMarket
)

def test_1x2_normalization():
    """Test 1X2 market normalization"""
    print("TEST: 1X2 Market Normalization...")
    
    market = MarketNormalizer.normalize_market(
        "Match Winner",
        [
            {"value": "Home", "odd": "2.50"},
            {"value": "Draw", "odd": "3.20"},
            {"value": "Away", "odd": "2.80"}
        ],
        "Bet365"
    )
    
    assert market is not None, "Market should be recognized"
    assert market.market_key == "1X2", f"Expected '1X2', got {market.market_key}"
    assert market.market_type == "1X2", f"Expected '1X2', got {market.market_type}"
    assert len(market.selections) == 3, f"Expected 3 selections, got {len(market.selections)}"
    assert "HOME" in market.selections, "HOME selection missing"
    assert "DRAW" in market.selections, "DRAW selection missing"
    assert "AWAY" in market.selections, "AWAY selection missing"
    
    print("  ✓ PASSED")


def test_double_chance_normalization():
    """Test Double Chance market normalization"""
    print("TEST: Double Chance Market Normalization...")
    
    market = MarketNormalizer.normalize_market(
        "Double Chance",
        [
            {"value": "Home/Draw", "odd": "1.40"},
            {"value": "Draw/Away", "odd": "1.50"},
            {"value": "Home/Away", "odd": "1.35"}
        ],
        "Bet365"
    )
    
    assert market is not None, "Market should be recognized"
    assert market.market_key == "DoubleChance", f"Expected 'DoubleChance', got {market.market_key}"
    assert len(market.selections) == 3, f"Expected 3 selections, got {len(market.selections)}"
    assert "1X" in market.selections, "1X selection missing"
    assert "X2" in market.selections, "X2 selection missing"
    assert "12" in market.selections, "12 selection missing"
    
    print("  ✓ PASSED")


def test_over_under_normalization():
    """Test Over/Under market normalization with various thresholds"""
    print("TEST: Over/Under Market Normalization...")
    
    # Test 2.5
    market_25 = MarketNormalizer.normalize_market(
        "Goals Over/Under",
        [
            {"value": "Over 2.5", "odd": "1.85"},
            {"value": "Under 2.5", "odd": "2.00"}
        ],
        "Bet365"
    )
    
    assert market_25 is not None, "Market 2.5 should be recognized"
    assert market_25.market_key == "Goals|OU:2.5", f"Expected 'Goals|OU:2.5', got {market_25.market_key}"
    assert market_25.threshold == 2.5, f"Expected threshold 2.5, got {market_25.threshold}"
    assert "OVER" in market_25.selections, "OVER selection missing"
    assert "UNDER" in market_25.selections, "UNDER selection missing"
    
    # Test 1.5
    market_15 = MarketNormalizer.normalize_market(
        "Goals Over/Under",
        [
            {"value": "Over 1.5", "odd": "1.35"},
            {"value": "Under 1.5", "odd": "3.20"}
        ],
        "Bet365"
    )
    
    assert market_15 is not None, "Market 1.5 should be recognized"
    assert market_15.market_key == "Goals|OU:1.5", f"Expected 'Goals|OU:1.5', got {market_15.market_key}"
    assert market_15.threshold == 1.5, f"Expected threshold 1.5, got {market_15.threshold}"
    
    # Test 3.5
    market_35 = MarketNormalizer.normalize_market(
        "Goals Over/Under",
        [
            {"value": "Over 3.5", "odd": "3.40"},
            {"value": "Under 3.5", "odd": "1.30"}
        ],
        "Bet365"
    )
    
    assert market_35 is not None, "Market 3.5 should be recognized"
    assert market_35.threshold == 3.5, f"Expected threshold 3.5, got {market_35.threshold}"
    
    print("  ✓ PASSED")


def test_btts_normalization():
    """Test BTTS market normalization"""
    print("TEST: BTTS Market Normalization...")
    
    market = MarketNormalizer.normalize_market(
        "Both Teams to Score",
        [
            {"value": "Yes", "odd": "1.75"},
            {"value": "No", "odd": "2.10"}
        ],
        "Bet365"
    )
    
    assert market is not None, "Market should be recognized"
    assert market.market_key == "BTTS", f"Expected 'BTTS', got {market.market_key}"
    assert market.market_type == "BTTS", f"Expected 'BTTS', got {market.market_type}"
    assert len(market.selections) == 2, f"Expected 2 selections, got {len(market.selections)}"
    assert "YES" in market.selections, "YES selection missing"
    assert "NO" in market.selections, "NO selection missing"
    
    print("  ✓ PASSED")


def test_handicap_normalization():
    """Test Asian Handicap market normalization"""
    print("TEST: Asian Handicap Market Normalization...")
    
    market = MarketNormalizer.normalize_market(
        "Asian Handicap",
        [
            {"value": "Home -1.5", "odd": "3.20"},
            {"value": "Away +1.5", "odd": "1.33"}
        ],
        "Bet365"
    )
    
    assert market is not None, "Market should be recognized"
    assert market.market_key == "Handicap:1.5", f"Expected 'Handicap:1.5', got {market.market_key}"
    assert market.market_type == "Handicap", f"Expected 'Handicap', got {market.market_type}"
    assert market.threshold == 1.5, f"Expected threshold 1.5, got {market.threshold}"
    assert len(market.selections) == 2, f"Expected 2 selections, got {len(market.selections)}"
    
    print("  ✓ PASSED")


def test_htft_normalization():
    """Test HT/FT market normalization"""
    print("TEST: HT/FT Market Normalization...")
    
    market = MarketNormalizer.normalize_market(
        "Halftime/Fulltime",
        [
            {"value": "Home/Home", "odd": "3.75"},
            {"value": "Home/Draw", "odd": "15.00"},
            {"value": "Home/Away", "odd": "21.00"},
            {"value": "Draw/Home", "odd": "6.50"},
            {"value": "Draw/Draw", "odd": "8.00"},
            {"value": "Draw/Away", "odd": "10.00"},
            {"value": "Away/Home", "odd": "26.00"},
            {"value": "Away/Draw", "odd": "17.00"},
            {"value": "Away/Away", "odd": "9.00"}
        ],
        "Bet365"
    )
    
    assert market is not None, "Market should be recognized"
    assert market.market_key == "HT/FT", f"Expected 'HT/FT', got {market.market_key}"
    assert market.market_type == "HT/FT", f"Expected 'HT/FT', got {market.market_type}"
    assert len(market.selections) == 9, f"Expected 9 selections, got {len(market.selections)}"
    
    print("  ✓ PASSED")


def test_full_extraction():
    """Test full odds response extraction"""
    print("TEST: Full Odds Response Extraction...")
    
    sample_response = [
        {
            "fixture": {"id": 999999},
            "bookmakers": [
                {
                    "id": 8,
                    "name": "Bet365",
                    "bets": [
                        {
                            "name": "Match Winner",
                            "values": [
                                {"value": "Home", "odd": "2.10"},
                                {"value": "Draw", "odd": "3.40"},
                                {"value": "Away", "odd": "3.60"}
                            ]
                        },
                        {
                            "name": "Goals Over/Under",
                            "values": [
                                {"value": "Over 2.5", "odd": "2.00"},
                                {"value": "Under 2.5", "odd": "1.80"}
                            ]
                        },
                        {
                            "name": "Both Teams to Score",
                            "values": [
                                {"value": "Yes", "odd": "1.72"},
                                {"value": "No", "odd": "2.05"}
                            ]
                        }
                    ]
                }
            ]
        }
    ]
    
    markets = extract_bookmaker_odds_from_odds_response(sample_response)
    
    assert len(markets) == 3, f"Expected 3 markets, got {len(markets)}"
    assert "1X2" in markets, "1X2 market missing"
    assert "Goals|OU:2.5" in markets, "Goals|OU:2.5 market missing"
    assert "BTTS" in markets, "BTTS market missing"
    
    print("  ✓ PASSED")


def test_margin_calculation():
    """Test bookmaker margin calculation"""
    print("TEST: Margin Calculation...")
    
    # Test 1X2 margin
    selections = {"HOME": 2.5, "DRAW": 3.2, "AWAY": 2.8}
    margin = calculate_margin(selections)
    
    # Margin should be sum of inverse odds
    expected = 1/2.5 + 1/3.2 + 1/2.8
    assert abs(margin - expected) < 0.001, f"Expected margin {expected}, got {margin}"
    
    # Margin should be > 1 (bookmaker advantage)
    assert margin > 1.0, f"Margin should be > 1, got {margin}"
    
    # Test 2-way margin
    selections_2way = {"OVER": 1.85, "UNDER": 2.0}
    margin_2way = calculate_margin(selections_2way)
    
    expected_2way = 1/1.85 + 1/2.0
    assert abs(margin_2way - expected_2way) < 0.001, f"Expected margin {expected_2way}, got {margin_2way}"
    
    print("  ✓ PASSED")


def test_selection_normalization():
    """Test selection name normalization"""
    print("TEST: Selection Name Normalization...")
    
    # Home variants
    assert MarketNormalizer.normalize_selection_name("Home") == "HOME"
    assert MarketNormalizer.normalize_selection_name("1") == "HOME"
    assert MarketNormalizer.normalize_selection_name("Hazai") == "HOME"
    
    # Draw variants
    assert MarketNormalizer.normalize_selection_name("Draw") == "DRAW"
    assert MarketNormalizer.normalize_selection_name("X") == "DRAW"
    assert MarketNormalizer.normalize_selection_name("Döntetlen") == "DRAW"
    
    # Away variants
    assert MarketNormalizer.normalize_selection_name("Away") == "AWAY"
    assert MarketNormalizer.normalize_selection_name("2") == "AWAY"
    assert MarketNormalizer.normalize_selection_name("Vendég") == "AWAY"
    
    # Over/Under
    assert MarketNormalizer.normalize_selection_name("Over") == "OVER"
    assert MarketNormalizer.normalize_selection_name("Under") == "UNDER"
    assert MarketNormalizer.normalize_selection_name("Több") == "OVER"
    assert MarketNormalizer.normalize_selection_name("Kevesebb") == "UNDER"
    
    # BTTS
    assert MarketNormalizer.normalize_selection_name("Yes") == "YES"
    assert MarketNormalizer.normalize_selection_name("No") == "NO"
    assert MarketNormalizer.normalize_selection_name("Igen") == "YES"
    assert MarketNormalizer.normalize_selection_name("Nem") == "NO"
    
    print("  ✓ PASSED")


def test_threshold_extraction():
    """Test threshold extraction from market names"""
    print("TEST: Threshold Extraction...")
    
    # Test various formats
    assert MarketNormalizer.extract_threshold_from_name("Over 2.5") == 2.5
    assert MarketNormalizer.extract_threshold_from_name("Under 1.5") == 1.5
    assert MarketNormalizer.extract_threshold_from_name("Goals Over/Under 3.5") == 3.5
    assert MarketNormalizer.extract_threshold_from_name("Over 4.5") == 4.5
    assert MarketNormalizer.extract_threshold_from_name("Home -1.5") == 1.5
    assert MarketNormalizer.extract_threshold_from_name("Away +0.5") == 0.5
    
    # Should return None for no threshold
    assert MarketNormalizer.extract_threshold_from_name("Match Winner") is None
    assert MarketNormalizer.extract_threshold_from_name("BTTS") is None
    
    print("  ✓ PASSED")


def run_all_tests():
    """Run all tests"""
    print()
    print("=" * 80)
    print("RUNNING UNIT TESTS FOR check_analysis_json.py")
    print("=" * 80)
    print()
    
    tests = [
        test_1x2_normalization,
        test_double_chance_normalization,
        test_over_under_normalization,
        test_btts_normalization,
        test_handicap_normalization,
        test_htft_normalization,
        test_full_extraction,
        test_margin_calculation,
        test_selection_normalization,
        test_threshold_extraction
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print()
    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
