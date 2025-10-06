#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of check_analysis_json.py API

This script demonstrates how to use the check_analysis_json module programmatically.
"""

from pathlib import Path
from check_analysis_json import (
    MarketNormalizer,
    extract_bookmaker_odds_from_odds_response,
    load_analysis_files,
    generate_ticket_from_analysis,
    format_ticket_display,
    check_analysis_completeness
)

def example_1_market_normalization():
    """Example 1: Normalize individual markets"""
    print("=" * 80)
    print("EXAMPLE 1: Market Normalization")
    print("=" * 80)
    print()
    
    # Example 1X2 market
    market_1x2 = MarketNormalizer.normalize_market(
        "Match Winner",
        [
            {"value": "Home", "odd": "2.50"},
            {"value": "Draw", "odd": "3.20"},
            {"value": "Away", "odd": "2.80"}
        ],
        "Bet365"
    )
    
    print(f"1X2 Market:")
    print(f"  Key: {market_1x2.market_key}")
    print(f"  Selections: {market_1x2.selections}")
    print()
    
    # Example Over/Under market
    market_ou = MarketNormalizer.normalize_market(
        "Goals Over/Under",
        [
            {"value": "Over 2.5", "odd": "1.85"},
            {"value": "Under 2.5", "odd": "2.00"}
        ],
        "Bet365"
    )
    
    print(f"Over/Under 2.5 Market:")
    print(f"  Key: {market_ou.market_key}")
    print(f"  Threshold: {market_ou.threshold}")
    print(f"  Selections: {market_ou.selections}")
    print()


def example_2_extract_all_markets():
    """Example 2: Extract all markets from API-Football response"""
    print("=" * 80)
    print("EXAMPLE 2: Extract All Markets from Odds Response")
    print("=" * 80)
    print()
    
    # Sample API-Football odds response
    sample_response = [
        {
            "fixture": {"id": 123456},
            "bookmakers": [
                {
                    "id": 8,
                    "name": "Bet365",
                    "bets": [
                        {
                            "name": "Match Winner",
                            "values": [
                                {"value": "Home", "odd": "2.50"},
                                {"value": "Draw", "odd": "3.20"},
                                {"value": "Away", "odd": "2.80"}
                            ]
                        },
                        {
                            "name": "Goals Over/Under",
                            "values": [
                                {"value": "Over 2.5", "odd": "1.85"},
                                {"value": "Under 2.5", "odd": "2.00"}
                            ]
                        },
                        {
                            "name": "Both Teams to Score",
                            "values": [
                                {"value": "Yes", "odd": "1.75"},
                                {"value": "No", "odd": "2.10"}
                            ]
                        }
                    ]
                }
            ]
        }
    ]
    
    markets = extract_bookmaker_odds_from_odds_response(sample_response)
    
    print(f"Extracted {len(markets)} markets:")
    for key, market in markets.items():
        print(f"  {key}: {market.selections}")
    print()


def example_3_load_and_check_analysis():
    """Example 3: Load and check analysis files"""
    print("=" * 80)
    print("EXAMPLE 3: Load and Check Analysis Files")
    print("=" * 80)
    print()
    
    # Load analysis files
    data_root = Path(".")
    analysis_files = load_analysis_files(data_root)
    
    print(f"Loaded {len(analysis_files)} analysis files")
    print()
    
    # Check each file
    for item in analysis_files[:3]:  # Show first 3
        fixture_id = item.get("fixture_id", "unknown")
        result = check_analysis_completeness(item.get("data", {}))
        
        status = "✓" if result["complete"] else "✗"
        print(f"{status} Fixture {fixture_id}: Quality={result['quality_score']:.2f}")
        
        if result["issues"]:
            for issue in result["issues"]:
                print(f"  ERROR: {issue}")
        
        if result["warnings"]:
            for warning in result["warnings"]:
                print(f"  WARNING: {warning}")
        print()


def example_4_generate_ticket():
    """Example 4: Generate betting ticket"""
    print("=" * 80)
    print("EXAMPLE 4: Generate Betting Ticket")
    print("=" * 80)
    print()
    
    # Load analysis files
    data_root = Path(".")
    analysis_files = load_analysis_files(data_root)
    
    if not analysis_files:
        print("No analysis files found")
        return
    
    # Generate ticket
    picks = generate_ticket_from_analysis(
        analysis_files,
        min_edge=0.05,
        max_picks=5,
        max_odds=4.0
    )
    
    if picks:
        print(f"Generated ticket with {len(picks)} picks:")
        print()
        ticket_text = format_ticket_display(picks)
        print(ticket_text)
    else:
        print("No picks meet the criteria")
    print()


def example_5_custom_filtering():
    """Example 5: Custom filtering logic"""
    print("=" * 80)
    print("EXAMPLE 5: Custom Filtering Logic")
    print("=" * 80)
    print()
    
    # Load analysis files with raw odds
    data_root = Path(".")
    analysis_files = load_analysis_files(data_root, include_raw_odds=True)
    
    print(f"Loaded {len(analysis_files)} files")
    print()
    
    # Count markets by type
    market_counts = {}
    for item in analysis_files:
        normalized = item.get("normalized_markets", {})
        for market_key, market in normalized.items():
            market_type = market.market_type
            market_counts[market_type] = market_counts.get(market_type, 0) + 1
    
    print("Market type distribution:")
    for market_type, count in sorted(market_counts.items()):
        print(f"  {market_type:15s}: {count:3d}")
    print()


def main():
    """Run all examples"""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "check_analysis_json.py Usage Examples" + " " * 21 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    try:
        example_1_market_normalization()
        example_2_extract_all_markets()
        example_3_load_and_check_analysis()
        example_4_generate_ticket()
        example_5_custom_filtering()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)
    print("Examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
