# check_analysis_json.py

Comprehensive betting market analysis and ticket generation tool for API-Football and local analysis files.

## Quick Start

```bash
# Check help
python3 check_analysis_json.py --help

# Show summary
python3 check_analysis_json.py

# Validate analysis files
python3 check_analysis_json.py --check

# Generate betting ticket
python3 check_analysis_json.py --ticket
```

## Key Features

✓ **Comprehensive Market Support**
  - Match Winner / 1X2
  - Double Chance
  - Over/Under (0.5, 1.5, 2.5, 3.5, 4.5+)
  - Both Teams To Score (BTTS)
  - Asian Handicap
  - Half Time / Full Time
  - Correct Score

✓ **Multiple Modes**
  - `--check`: Validate analysis files
  - `--report`: Market coverage statistics
  - `--markets`: Show available markets
  - `--ticket`: Generate betting slip

✓ **Smart Normalization**
  - Automatic market recognition
  - Threshold extraction (e.g., 2.5 from "Over 2.5")
  - Selection normalization (Home/1/Hazai → HOME)
  - Best odds selection (lowest margin)

✓ **Robust Ticket Generation**
  - Edge-based filtering
  - Confidence levels (HIGH/MEDIUM/LOW)
  - Customizable parameters
  - JSON export

## Examples

### Generate Conservative Ticket
```bash
python3 check_analysis_json.py --ticket \
  --min-edge 0.10 \
  --max-picks 3 \
  --max-odds 3.0 \
  --output ticket.json
```

### Check Analysis Quality
```bash
python3 check_analysis_json.py --check
```

### View Market Coverage
```bash
python3 check_analysis_json.py --report
```

### Explore Available Markets
```bash
python3 check_analysis_json.py --markets
```

## Documentation

See [CHECK_ANALYSIS_JSON_GUIDE.md](CHECK_ANALYSIS_JSON_GUIDE.md) for complete documentation.

## Market Key Format

| Market Type | Key Format | Example |
|------------|-----------|---------|
| 1X2 | `1X2` | `1X2` |
| Double Chance | `DoubleChance` | `DoubleChance` |
| Over/Under | `Goals\|OU:X.X` | `Goals\|OU:2.5` |
| BTTS | `BTTS` | `BTTS` |
| Handicap | `Handicap:X.X` | `Handicap:1.5` |
| HT/FT | `HT/FT` | `HT/FT` |

## Requirements

- Python 3.8+
- No external dependencies beyond standard library
- Works with betting.py analysis files

## Testing

Run the comprehensive test suite:

```bash
# Test market normalization
python3 /tmp/test_market_normalizer.py

# Test all market types
python3 /tmp/test_comprehensive_markets.py

# Test handicap markets
python3 /tmp/test_handicap.py
```

All tests should pass with ✓ markers.

## Integration

Works seamlessly with `betting.py`:

```bash
# 1. Fetch and analyze fixtures
python3 betting.py --fetch --analyze

# 2. Check analysis quality
python3 check_analysis_json.py --check

# 3. Generate ticket
python3 check_analysis_json.py --ticket --output today_picks.json
```

## API Reference

### MarketNormalizer Class

Main class for market parsing and normalization:

```python
from check_analysis_json import MarketNormalizer

# Normalize a market
market = MarketNormalizer.normalize_market(
    "Match Winner",
    [
        {"value": "Home", "odd": "2.50"},
        {"value": "Draw", "odd": "3.20"},
        {"value": "Away", "odd": "2.80"}
    ],
    "Bet365"
)

print(market.market_key)  # "1X2"
print(market.selections)  # {"HOME": 2.5, "DRAW": 3.2, "AWAY": 2.8}
```

### extract_bookmaker_odds_from_odds_response()

Extract all markets from API-Football response:

```python
from check_analysis_json import extract_bookmaker_odds_from_odds_response

markets = extract_bookmaker_odds_from_odds_response(odds_response)
for key, market in markets.items():
    print(f"{key}: {market.selections}")
```

## Supported Bookmakers

The normalization works with all bookmakers in API-Football:
- Bet365
- 1xBet  
- William Hill
- Unibet
- Betway
- And 100+ others

Market names and selections are automatically normalized.

## Output Formats

### Ticket JSON
```json
{
  "generated_at": "2025-10-06T17:30:00Z",
  "picks": [
    {
      "fixture_id": 123456,
      "home_team": "Manchester United",
      "away_team": "Liverpool",
      "market_type": "1X2",
      "selection": "HOME",
      "odds": 2.50,
      "edge": 0.085,
      "confidence": "MEDIUM"
    }
  ],
  "combined_odds": 4.63
}
```

## Performance

- Default mode: < 1 second for 100 fixtures
- Check mode: < 2 seconds for 100 fixtures
- Markets mode: < 5 seconds for 100 fixtures (loads raw odds)
- Ticket mode: < 1 second for 100 fixtures

## Troubleshooting

**No analysis files found**
→ Check `--data-root` parameter

**No picks available**  
→ Lower `--min-edge` or increase `--max-odds`

**Market not recognized**
→ Check raw odds with `--markets` mode

## License

Part of the betting analysis system. See main repository for license.

## Version

1.0.0 (2025-10-06)
