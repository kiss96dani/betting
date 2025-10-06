"# Betting Analysis System

Advanced betting analysis system with comprehensive market support and intelligent ticket generation.

## Quick Start

### Check Analysis Files
```bash
python3 check_analysis_json.py --check
```

### Generate Betting Ticket
```bash
python3 check_analysis_json.py --ticket
```

### View Available Markets
```bash
python3 check_analysis_json.py --markets
```

## New: check_analysis_json.py

Comprehensive betting market analysis tool with support for:
- ✓ Match Winner / 1X2
- ✓ Double Chance  
- ✓ Over/Under (0.5-4.5+)
- ✓ Both Teams To Score
- ✓ Asian Handicap
- ✓ Half Time / Full Time
- ✓ Correct Score

### Features
- 🎯 Automatic market recognition from API-Football
- 🔄 Multi-language selection normalization
- 🎫 Smart ticket generation with edge filtering
- 📊 Market coverage reporting
- ✅ Quality validation for analysis files
- 📁 JSON export capability

### Documentation
- [Quick Start Guide](CHECK_ANALYSIS_JSON_README.md)
- [Complete User Guide](CHECK_ANALYSIS_JSON_GUIDE.md)
- [Implementation Details](IMPLEMENTATION_SUMMARY.md)

### Examples
```bash
# Conservative ticket (high confidence only)
python3 check_analysis_json.py --ticket --min-edge 0.10 --max-picks 3

# Aggressive ticket (more picks)
python3 check_analysis_json.py --ticket --min-edge 0.03 --max-picks 8

# Export to JSON
python3 check_analysis_json.py --ticket --output ticket.json

# Generate coverage report
python3 check_analysis_json.py --report
```

### Testing
```bash
# Run unit tests
python3 test_check_analysis_json.py

# Run examples
python3 example_usage.py
```

All 10 unit tests pass ✓

## Main Components

### betting.py
Main betting analysis script with fixture fetching and analysis.

### check_analysis_json.py
New comprehensive market analysis and ticket generation tool.

## Requirements
- Python 3.8+
- Standard library only (no external dependencies for check_analysis_json.py)

## License
See repository for license details" 
