# Pull Request Summary: Dynamic Markets Feature

## Overview
This PR implements dynamic market extraction and model-based probability/edge computation for Over/Under, corners, cards, and handicap markets.

## Branch Information
- **Source Branch**: `copilot/fix-3d7f3b9a-8741-40e3-9d18-3955fd22f48b`
- **Target Branch**: `main`
- **Commits**: 3
  1. Initial plan (9448f88)
  2. Add market_extensions.py and integrate dynamic markets into betting.py (591e364)
  3. Add .gitignore and remove __pycache__ files (ccab07a)

## Changes Made

### 1. New File: market_extensions.py
Created a comprehensive module at the repository root with:

**Poisson Helper Functions:**
- `poisson_p(k, lam)`: Computes Poisson PMF P(X=k) = (λ^k * e^(-λ)) / k!
- `prob_over_threshold(lam, threshold)`: Computes P(X > threshold) using cumulative Poisson distribution
- `prob_goal_diff_gt(lambda_h, lambda_a, margin)`: Computes probability that home team wins by more than margin goals

**Market Extraction Functions:**
- `extract_dynamic_markets(odds_response_list)`: 
  - Parses raw odds API responses
  - Extracts Over/Under markets (using regex patterns)
  - Extracts corners markets
  - Extracts cards markets  
  - Extracts handicap markets
  - Returns dictionary with market types as keys

- Helper functions for extraction:
  - `extract_over_under_market(name, values, markets)`: Pattern matching for O/U markets
  - `extract_corners_market(name, values, markets)`: Pattern matching for corners
  - `extract_cards_market(name, values, markets)`: Pattern matching for cards
  - `extract_handicap_market(name, values, markets)`: Pattern matching for handicaps

**Probability Computation:**
- `compute_probs_for_dynamic_markets(dynamic_markets, model_context)`:
  - Takes extracted markets and model parameters (lambda_home, lambda_away, lambda_total, corners_total, lambda_cards)
  - Computes model probabilities for each market using Poisson distributions
  - Calculates edges (model_prob * odds - 1)
  - Returns (market_probs, market_edges, details) tuple

**Utility Functions:**
- `flatten_market_dicts(market_probs, market_edges)`: Flattens nested dictionaries (currently pass-through, extensible)

**Features:**
- Comprehensive error handling with try/except blocks
- Type hints throughout
- Safe defaults for missing data
- Handles malformed input gracefully

### 2. Modified File: betting.py

**Added Import (line 15):**
```python
from market_extensions import extract_dynamic_markets, compute_probs_for_dynamic_markets, flatten_market_dicts
```

**Modified analyze_fixture() function (after line 2720):**
1. Added feature_engineering.advanced_features section construction
2. Populated advanced_features with context and extra data (lambda_home, lambda_away, model_prob, model_edge)
3. Added dynamic markets processing block:
   - Loads raw odds data from out_fixture_{id}/raw/odds__*.json
   - Extracts dynamic markets using `extract_dynamic_markets()`
   - Builds model context with lambda values
   - Computes probabilities and edges using `compute_probs_for_dynamic_markets()`
   - Flattens results using `flatten_market_dicts()`
   - Writes model_prob_{key} and model_edge_{key} to advanced_features (with safe key conversion)
   - Stores dynamic_market_details in advanced_features
4. Wrapped entire block in try/except to prevent breaking existing analysis
5. Added feature_engineering section to result dict

### 3. New File: .gitignore
Added standard Python .gitignore to exclude:
- `__pycache__/` directories
- `*.pyc`, `*.pyo` files
- Virtual environments
- IDE files
- Build artifacts

## Output Format

The analysis.json file now includes:
```json
{
  "fixture_id": 12345,
  // ... existing fields ...
  "feature_engineering": {
    "advanced_features": {
      "lambda_home": 1.5,
      "lambda_away": 1.0,
      "model_prob": 0.45,
      "model_edge": 0.05,
      "market_prob_btts_yes": 0.63,
      "market_edge_btts_yes": 0.08,
      // Dynamic market probabilities
      "model_prob_ou_2.5_over": 0.456,
      "model_prob_ou_2.5_under": 0.544,
      "model_edge_ou_2.5_over": -0.156,
      "model_edge_ou_2.5_under": 0.115,
      // Corners (if corners_total available)
      "model_prob_corners_9.5_over": 0.52,
      "model_edge_corners_9.5_over": 0.04,
      // Cards
      "model_prob_cards_4.5_over": 0.38,
      "model_edge_cards_4.5_over": -0.02,
      // Handicap
      "model_prob_handicap_home_-1.5": 0.25,
      "model_edge_handicap_home_-1.5": 0.12,
      // Details
      "dynamic_market_details": {
        "ou_2.5_over": {
          "threshold": 2.5,
          "side": "over",
          "odd": 1.85,
          "model_prob": 0.456,
          "edge": -0.156
        },
        // ... more market details
      }
    }
  }
}
```

## Testing

Tested the market_extensions module with sample data:
- ✅ Poisson functions working correctly
- ✅ Market extraction from sample odds data
- ✅ Probability computation with model context
- ✅ Edge calculation
- ✅ Flattening function
- ✅ Error handling with malformed input

## Files Changed
- `market_extensions.py` (NEW): 456 lines added
- `betting.py`: 77 lines added
- `.gitignore` (NEW): 37 lines added

## Integration Points

The dynamic market feature integrates seamlessly with existing code:
1. Does not modify existing betting.py logic
2. Wrapped in try/except to prevent breaking analysis
3. Uses existing data structures (raw odds files)
4. Extends analysis output without changing existing fields

## Next Steps

To complete the PR:
1. Navigate to https://github.com/kiss96dani/betting/compare/main...copilot/fix-3d7f3b9a-8741-40e3-9d18-3955fd22f48b
2. Click "Create Pull Request"
3. Use title: "Feature: dynamic markets (OU/corners/handicap) — include model probs/edges in analysis.json"
4. Paste this summary in the PR description
5. Review changes and merge

## PR Title
```
Feature: dynamic markets (OU/corners/handicap) — include model probs/edges in analysis.json
```

## PR Description Template
```markdown
## Summary
This PR adds dynamic market extraction and model-based probability/edge computation for Over/Under, corners, cards, and handicap markets.

## New Module: market_extensions.py
- Poisson probability helpers (poisson_p, prob_over_threshold, prob_goal_diff_gt)
- Dynamic market extraction from raw odds responses (regex-based)
- Model probability and edge computation using lambda parameters
- Comprehensive error handling and type hints

## Modified: betting.py
- Added imports from market_extensions
- Extended analyze_fixture() to process dynamic markets
- Added feature_engineering.advanced_features section to analysis.json
- Injects model_prob_* and model_edge_* keys for each extracted market
- Stores dynamic_market_details for debugging

## Integration
- Reads from existing raw odds files
- Non-breaking changes (wrapped in try/except)
- Extends analysis output with new fields
- Maintains backward compatibility

## Testing
- ✅ Module imports successfully
- ✅ Poisson functions validated
- ✅ Market extraction tested with sample data
- ✅ Probability computation working correctly
- ✅ Error handling verified

## Files Changed
- market_extensions.py (NEW)
- betting.py (modified)
- .gitignore (NEW)
```
