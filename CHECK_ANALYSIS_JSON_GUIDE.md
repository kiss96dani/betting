# check_analysis_json.py - User Guide

## Overview

`check_analysis_json.py` is a comprehensive betting market analysis and ticket generation tool designed to work with API-Football bookmaker responses and local analysis files.

## Features

### 1. Comprehensive Market Support

The script recognizes and normalizes a wide range of betting markets:

#### Match Markets
- **Match Winner / 1X2**: Home, Draw, Away
- **Double Chance**: 1X (Home or Draw), X2 (Draw or Away), 12 (Home or Away)

#### Goals Markets  
- **Over/Under** with multiple thresholds:
  - Goals|OU:0.5
  - Goals|OU:1.5
  - Goals|OU:2.5
  - Goals|OU:3.5
  - Goals|OU:4.5
  - Any other threshold (automatically detected)

#### Other Markets
- **Both Teams To Score (BTTS)**: Yes/No
- **Asian Handicap**: Various handicap values (e.g., -1.5, +0.5)
- **Half Time / Full Time (HT/FT)**: 9 possible outcomes
- **Correct Score**: Multiple score outcomes

### 2. Multiple Operating Modes

#### Default Mode
Shows summary statistics of analysis files:
```bash
python3 check_analysis_json.py
```

Output:
```
Total fixtures: 5
Complete: 4/5
With odds: 3/5
```

#### Check Mode (`--check`)
Validates analysis files for completeness and quality:
```bash
python3 check_analysis_json.py --check
```

Output:
```
✓ Fixture 123456: Quality=1.00
✗ Fixture 789012: Quality=0.70
  WARNING: No odds data available
  ERROR: Model probabilities don't sum to 1.0
```

#### Report Mode (`--report`)
Generates market coverage statistics:
```bash
python3 check_analysis_json.py --report
```

Output:
```
================================================================================
MARKET COVERAGE REPORT
================================================================================

Total fixtures analyzed: 10

Market Availability:
  1X2                 :   8 /  10 ( 80.0%)
  BTTS                :   7 /  10 ( 70.0%)
  O/U 2.5             :   7 /  10 ( 70.0%)
```

#### Markets Mode (`--markets`)
Shows all available markets from raw odds files:
```bash
python3 check_analysis_json.py --markets
```

Output:
```
Fixture 123456:
  1X2                  [Bet365]
    HOME           : 2.50
    DRAW           : 3.20
    AWAY           : 2.80
  Goals|OU:2.5         [Bet365]
    OVER           : 1.85
    UNDER          : 2.00
```

#### Ticket Mode (`--ticket` or `--slip`)
Generates a betting ticket from best picks:
```bash
python3 check_analysis_json.py --ticket --min-edge 0.05 --max-picks 5 --max-odds 4.0
```

Output:
```
================================================================================
BETTING TICKET / SLIP
================================================================================
Generated: 2025-10-06 17:30:00 UTC
Total picks: 3

[1] Manchester United vs Liverpool
    League: Premier League
    Kickoff: 2025-10-07T14:00:00+00:00
    Market: 1X2 | Selection: HOME
    Odds: 2.50 | Edge: 8.50% | Confidence: MEDIUM
    Model Prob: 45.00% | Market Prob: 40.00%

[2] Barcelona vs Real Madrid
    Market: O/U 2.5 | Selection: OVER
    Odds: 1.85 | Edge: 12.30% | Confidence: HIGH
    ...

--------------------------------------------------------------------------------
Combined odds: 4.63
Expected value: 15.20%
================================================================================
```

Save ticket to JSON file:
```bash
python3 check_analysis_json.py --ticket --output ticket.json
```

## Command Line Options

### General Options
- `--data-root PATH`: Root directory containing fixture data (default: current directory)

### Mode Selection
- `--check`: Check analysis files for completeness and quality
- `--report`: Generate market coverage report
- `--markets`: Show all available markets from raw odds files
- `--ticket` or `--slip`: Generate betting ticket/slip

### Ticket Options
- `--min-edge FLOAT`: Minimum edge for ticket picks (default: 0.05 = 5%)
- `--max-picks INT`: Maximum number of picks for ticket (default: 5)
- `--max-odds FLOAT`: Maximum odds to accept (default: 4.0)
- `--output PATH`: Output file for ticket JSON (optional)

## API-Football Market Normalization

The script follows API-Football's market naming conventions and normalizes them to standardized keys:

### Market Name Mapping

| API-Football Name | Normalized Key | Example Selections |
|------------------|----------------|-------------------|
| Match Winner | 1X2 | HOME, DRAW, AWAY |
| 1X2 | 1X2 | HOME, DRAW, AWAY |
| Double Chance | DoubleChance | 1X, X2, 12 |
| Goals Over/Under | Goals\|OU:X.X | OVER, UNDER |
| Both Teams to Score | BTTS | YES, NO |
| Asian Handicap | Handicap:X.X | Home -1.5, Away +1.5 |
| Halftime/Fulltime | HT/FT | Home/Home, Draw/Away, etc. |

### Selection Name Normalization

The script automatically normalizes selection names:

| Raw Selection | Normalized |
|--------------|------------|
| Home, 1, Hazai | HOME |
| Draw, X, Döntetlen | DRAW |
| Away, 2, Vendég | AWAY |
| Over, Több | OVER |
| Under, Kevesebb, Alatt | UNDER |
| Yes, Igen, Goal | YES |
| No, Nem, No Goal | NO |

## Example Workflows

### 1. Check All Analysis Files
```bash
# First, check if all analysis files are valid
python3 check_analysis_json.py --check

# Generate coverage report
python3 check_analysis_json.py --report
```

### 2. Generate Betting Ticket
```bash
# Generate a conservative ticket (high confidence only)
python3 check_analysis_json.py --ticket --min-edge 0.10 --max-picks 3 --max-odds 3.0

# Generate an aggressive ticket (more picks, accept higher odds)
python3 check_analysis_json.py --ticket --min-edge 0.03 --max-picks 8 --max-odds 6.0

# Save ticket to file
python3 check_analysis_json.py --ticket --output tickets/ticket_$(date +%Y%m%d).json
```

### 3. Explore Available Markets
```bash
# See what markets are available in raw odds files
python3 check_analysis_json.py --markets

# Pipe to file for analysis
python3 check_analysis_json.py --markets > available_markets.txt
```

## Technical Details

### Market Recognition Algorithm

1. **Pattern Matching**: Uses predefined patterns to identify market types
2. **Threshold Extraction**: Automatically extracts numeric thresholds (e.g., 2.5 from "Over 2.5")
3. **Selection Normalization**: Converts bookmaker-specific names to standard format
4. **Margin Calculation**: Calculates bookmaker margin and selects best odds

### Ticket Generation Algorithm

The ticket generator:
1. Loads all analysis files
2. Identifies picks with edge >= `min-edge`
3. Filters by maximum odds (`max-odds`)
4. Assigns confidence levels (HIGH, MEDIUM, LOW) based on edge
5. Sorts by edge (descending)
6. Returns top N picks (`max-picks`)

### Quality Scoring

Analysis quality is scored based on:
- **Issues** (0.3 penalty each): Missing required fields, invalid probabilities
- **Warnings** (0.1 penalty each): Missing odds, incomplete markets
- **Score**: 1.0 - (issues × 0.3 + warnings × 0.1)

## Integration with betting.py

The script is designed to complement `betting.py`:

1. `betting.py` fetches fixtures and generates `analysis.json` files
2. `check_analysis_json.py` validates and analyzes these files
3. Comprehensive market parsing works with API-Football odds format
4. Ticket generation uses the same edge calculations

## Troubleshooting

### "No analysis files found"
- Check that `--data-root` points to the correct directory
- Ensure `out_fixture_*` directories exist with `analysis.json` files

### "No odds data available"
- Check that raw odds files exist in `out_fixture_*/raw/odds*.json`
- Use `--markets` mode to see what's available
- Some fixtures may not have odds data from API-Football

### "No picks available for ticket"
- Try lowering `--min-edge` (e.g., from 0.05 to 0.03)
- Increase `--max-odds` to accept higher-risk bets
- Check that analysis files have odds data

## Performance Considerations

- Loading raw odds files (`--markets` mode) is slower than default mode
- Large number of fixtures may take time to process
- Consider using `--data-root` to limit scope to specific fixtures

## Future Enhancements

Potential improvements:
- Kelly criterion staking for tickets
- Correlation analysis between picks
- Historical performance tracking
- Web interface for ticket generation
- Real-time odds comparison
- Machine learning for optimal pick selection

## Support

For issues or questions:
1. Check this guide first
2. Review `betting.py` documentation
3. Examine sample output with `--markets` or `--check`
4. Open an issue on GitHub with debug output

## Version

Current version: 1.0.0 (2025-10-06)

Last updated: 2025-10-06
