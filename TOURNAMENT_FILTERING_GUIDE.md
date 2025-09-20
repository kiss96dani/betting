# Tournament.json-based TOP League Filtering - Implementation Guide

## Overview

This implementation replaces the previous tier-based league filtering system with a more flexible tournaments.json-based approach that uses name matching to identify TOP leagues. The system provides comprehensive Hungarian explanations for all betting recommendations.

## Key Features

### 1. tournaments.json-based TOP League Identification
- Loads TOP leagues from `tournaments.json` based on the `is_top_level` field
- Uses multiple name variants: `name`, `template_name`, `translated_name`, `short_translated_name`
- Case-insensitive matching with partial matching support
- Fallback to existing tier-based system for compatibility

### 2. Enhanced Hungarian Explanations
- Detailed Hungarian rationale for every betting tip
- Includes TOP league status with emoji indicators (🌟 for TOP, ⚪ for non-TOP)
- Confidence levels and edge categorization in Hungarian
- Technical details preserved for advanced users

### 3. Improved Logging and Statistics
- Logs number of TOP leagues identified from tournaments.json
- Detailed filtering statistics (total matches, TOP matches, included matches)
- Debug logging for all identified TOP league names
- Enhanced Telegram bot output with structured Hungarian text

## File Structure

### tournaments.json Format
```json
[
  {
    "id": 1,
    "name": "Premier League",
    "template_name": "English Premier League", 
    "translated_name": "Angol Premier League",
    "short_translated_name": "Premier League",
    "is_top_level": true,
    "venue_id": 1
  },
  {
    "id": 7,
    "name": "Championship",
    "template_name": "English Championship",
    "translated_name": "Angol Championship", 
    "short_translated_name": "Championship",
    "is_top_level": false,
    "venue_id": 7
  }
]
```

### Required Fields
- `is_top_level`: Boolean indicating if this is a TOP league
- At least one name field (`name`, `template_name`, `translated_name`, `short_translated_name`)

## Code Changes Made

### 1. New Functions Added
- `load_top_leagues_from_tournaments()`: Loads TOP league names from tournaments.json
- `is_top_league_by_name()`: Case-insensitive name matching for TOP leagues
- Enhanced `build_rationale()`: Generates detailed Hungarian explanations

### 2. Modified Classes
- `LeagueTierManager`: Enhanced with tournaments.json support
  - `_load_top_leagues_from_tournaments()`: Loads TOP league data
  - `is_top_with_reason()`: Returns TOP status with Hungarian explanation
  - `get_league_name()`: Retrieves league name by ID

### 3. Enhanced Data Structures
- `FixtureContext`: Added `league_name` field
- Analysis results now include `league_name` for rationale generation

### 4. Updated Functions
- `fetch_upcoming_fixtures()`: Enhanced logging and statistics
- `analyze_fixture()`: Includes league_name in results
- All `build_rationale()` calls: Include league information
- Telegram bot: Enhanced `/tiers` command with TOP league details

## Usage Examples

### Basic TOP League Check
```python
# Load TOP leagues
top_data = load_top_leagues_from_tournaments()
print(f"Found {top_data['count']} TOP league names")

# Check if a league is TOP
is_top = is_top_league_by_name("Premier League", top_data['names'])
print(f"Premier League is TOP: {is_top}")
```

### Using Enhanced LeagueTierManager
```python
# Check TOP status with explanation
is_top, reason = LEAGUE_MANAGER.is_top_with_reason(39, "Premier League")
print(f"League 39: {is_top} - {reason}")
```

### Hungarian Rationale Generation
The enhanced `build_rationale()` function now generates comprehensive Hungarian explanations:

```
🎯 TIPP: Hazai győzelem (Végeredmény)
📊 Elemzés: A modellünk 60.0% valószínűséget ad erre az eredményre, míg a piac 50.0%-ot ár be.
📈 Értékítélet: Jó value bet (10.0% edge), bizalmi szint: Közepes
⚽ Liga státusz: 🌟 TOP liga (név alapján): Premier League
🔢 Gólvárakozás: Hazai 1.50, Vendég 1.20
💪 Erőviszony: Hazai előny (+0.15)
🧠 Modell: Ensemble algoritmus
```

## Configuration

### Environment Variables
- `TOP_MODE`: Controls filtering behavior (`all`, `top_only`, `hybrid`)
- Existing tier-based variables remain functional as fallback

### Telegram Bot Commands
- `/tiers`: Enhanced to show TOP league statistics from tournaments.json
- Shows emoji indicators and TOP league names
- Displays both tier-based and name-based statistics

## Error Handling

### Missing tournaments.json
- Graceful fallback to tier-based system
- Warning logged but system continues to function
- Empty TOP league list returned

### Invalid tournaments.json
- JSON parsing errors are logged
- System falls back to tier-based approach
- Detailed error information in logs

### Missing League Names
- Functions handle None/empty league names gracefully
- Fallback to league ID display
- No system crashes on missing data

## Testing

The implementation includes comprehensive tests that verify:
- tournaments.json loading and parsing
- Case-insensitive name matching
- Hungarian rationale generation
- Integration with existing LeagueTierManager
- Error handling for edge cases

## Migration Guide

### From Tier-based to tournaments.json
1. Create `tournaments.json` file with your league definitions
2. Set `is_top_level: true` for leagues you want to classify as TOP
3. Include multiple name variants for better matching
4. The system will automatically use name-based matching as primary method
5. Tier-based system remains as fallback for compatibility

### Backward Compatibility
- All existing functionality preserved
- Tier-based system works as fallback
- Existing configuration variables honored
- No breaking changes to API or data structures

## Performance Considerations

- tournaments.json is loaded once at startup
- Name matching uses efficient string operations
- Minimal performance impact on existing functionality
- Enhanced logging may increase log volume

## Troubleshooting

### Common Issues
1. **TOP leagues not recognized**: Check tournaments.json format and `is_top_level` values
2. **Name matching fails**: Ensure league names in tournaments.json match API data
3. **Hungarian text issues**: Check character encoding (UTF-8)
4. **Missing rationale data**: Verify league_name is populated in analysis results

### Debug Commands
```python
# Check loaded TOP leagues
print(LEAGUE_MANAGER.top_league_data)

# Test name matching
result = is_top_league_by_name("test league", top_names)
print(f"Match result: {result}")

# Get detailed TOP status
is_top, reason = LEAGUE_MANAGER.is_top_with_reason(league_id, league_name)
print(f"Status: {is_top}, Reason: {reason}")
```

## Future Enhancements

### Possible Improvements
- Dynamic tournament.json reloading
- Web interface for tournament management  
- Multiple language support for explanations
- Advanced name matching algorithms (fuzzy matching)
- Integration with external league databases

### Extension Points
- Custom rationale templates
- Additional explanation languages
- Enhanced statistical analysis
- Tournament hierarchy support