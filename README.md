"# Betting Analysis System

A comprehensive betting analysis system with integrated API-Football data fetching, advanced modeling, and intelligent ticket generation.

## Components

### Main System: `betting.py`
The core betting analysis system that fetches fixture data, analyzes odds, and generates picks.

### Analysis Checker: `check_analysis_json.py`
CLI tool to check and summarize analysis.json files with ticket generation mode.

#### Usage

**Default Summary Mode:**
```bash
python check_analysis_json.py [--data-root PATH]
```
Lists all analyzed fixtures and shows today's matches.

**Ticket Mode:**
```bash
python check_analysis_json.py --ticket [--ticket-markets MARKETS]
python check_analysis_json.py --slip [--ticket-markets MARKETS]
```

Builds a betting ticket with one top tip per market, ensuring no fixture appears more than once.

**Options:**
- `--data-root PATH`: Root directory containing `out_fixture_*` folders (default: current directory)
- `--ticket` or `--slip`: Enable ticket generation mode
- `--ticket-markets MARKETS`: Comma-separated list of markets (default: `1X2,BTTS,OU:1.5,OU:2.5,OU:3.5,Corners,Cards`)
- `--local-tz TZ`: Local timezone for date filtering (default: `Europe/Budapest`)

**Example:**
```bash
# Generate ticket with default markets
python check_analysis_json.py --ticket

# Generate ticket with specific markets
python check_analysis_json.py --ticket --ticket-markets "1X2,BTTS,OU:2.5"

# Use custom data directory
python check_analysis_json.py --data-root /path/to/data --slip
```

#### Ticket Selection Algorithm

1. Iterates through markets in the specified order
2. For each market, finds all candidates from today's fixtures
3. Selects the candidate with the highest edge that uses a fixture not yet selected
4. If all fixtures are already used, leaves that market empty on the ticket

This ensures each fixture appears at most once on the ticket, maximizing diversification." 
