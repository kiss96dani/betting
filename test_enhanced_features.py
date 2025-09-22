#!/usr/bin/env python3
"""
Test script for enhanced betting features
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from feature_engineering import AdvancedFeatureEngineer, extract_enhanced_features
import betting

def create_mock_fixture_data():
    """Create mock fixture data for testing"""
    return {
        "fixture": {
            "id": 12345,
            "date": "2024-01-15T15:00:00+00:00",
            "venue": {
                "id": 1,
                "name": "Test Stadium"
            }
        },
        "league": {
            "id": 39,
            "name": "Premier League",
            "season": 2024
        },
        "teams": {
            "home": {
                "id": 33,
                "name": "Manchester United"
            },
            "away": {
                "id": 40,
                "name": "Liverpool"
            }
        }
    }

def test_feature_engineering():
    """Test the feature engineering functionality"""
    print("🧪 Testing Enhanced Feature Engineering...")
    
    # Create temporary data directory
    test_data_root = Path("/tmp/betting_test")
    test_data_root.mkdir(exist_ok=True)
    
    # Create mock fixture data
    fixture_data = create_mock_fixture_data()
    fixture_id = fixture_data["fixture"]["id"]
    
    # Create mock data structure
    fixture_dir = test_data_root / f"out_fixture_{fixture_id}"
    fixture_dir.mkdir(exist_ok=True)
    
    # Save primary fixture data
    with open(fixture_dir / "primary_fixture.json", 'w') as f:
        json.dump(fixture_data, f)
    
    # Test feature extraction
    try:
        engineer = AdvancedFeatureEngineer(test_data_root)
        features = engineer.extract_features(fixture_data)
        
        print(f"✅ Feature extraction successful!")
        print(f"   - Fixture ID: {features.fixture_id}")
        print(f"   - Home Team ID: {features.home_team_id}")
        print(f"   - Away Team ID: {features.away_team_id}")
        print(f"   - League ID: {features.league_id}")
        print(f"   - Season: {features.season}")
        print(f"   - Home ELO: {features.home_elo_rating}")
        print(f"   - Away ELO: {features.away_elo_rating}")
        
        # Test feature matrix creation
        feature_matrix = engineer.create_feature_matrix([features])
        print(f"✅ Feature matrix created: shape {feature_matrix.shape}")
        
        # Test feature scaling
        if feature_matrix.size > 0:
            scaled_features = engineer.apply_feature_scaling(feature_matrix)
            print(f"✅ Feature scaling successful: shape {scaled_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        return False

def test_api_endpoints():
    """Test the enhanced API endpoints"""
    print("\n🌐 Testing Enhanced API Endpoints...")
    
    print(f"✅ Total API endpoints: {len(betting.FIXTURE_ENDPOINTS)}")
    
    # Check for new endpoints
    endpoint_tags = [ep[0] for ep in betting.FIXTURE_ENDPOINTS]
    new_endpoints = [
        "h2h_extended", "form_home_extended", "form_away_extended",
        "team_home_seasons", "team_away_seasons", "coach_home", "coach_away",
        "injuries_home", "injuries_away", "team_home_transfers", "team_away_transfers"
    ]
    
    found_new = [ep for ep in new_endpoints if ep in endpoint_tags]
    print(f"✅ New endpoints found: {len(found_new)}/{len(new_endpoints)}")
    for ep in found_new[:5]:  # Show first 5
        print(f"   - {ep}")
    
    return True

def test_betting_integration():
    """Test the betting.py integration"""
    print("\n⚽ Testing Betting Integration...")
    
    # Check configuration
    print(f"✅ Feature engineering enabled: {betting.ENABLE_FEATURE_ENGINEERING}")
    print(f"✅ Extended stats fetching: {betting.FETCH_EXTENDED_STATS}")
    print(f"✅ Injury details fetching: {betting.FETCH_INJURY_DETAILS}")
    print(f"✅ Feature scaling method: {betting.FEATURE_SCALING_METHOD}")
    print(f"✅ PCA components: {betting.FEATURE_PCA_COMPONENTS}")
    
    # Test enhanced statistics function existence
    try:
        func = getattr(betting, 'generate_enhanced_statistics')
        print(f"✅ Enhanced statistics function available")
    except AttributeError:
        print(f"❌ Enhanced statistics function not found")
        return False
    
    # Test feature importance function
    try:
        func = getattr(betting, 'analyze_feature_importance')
        print(f"✅ Feature importance analysis function available")
    except AttributeError:
        print(f"❌ Feature importance function not found")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Enhanced Betting System - Feature Test Suite\n")
    
    tests = [
        ("Feature Engineering", test_feature_engineering),
        ("API Endpoints", test_api_endpoints), 
        ("Betting Integration", test_betting_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")
        
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced betting system ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())