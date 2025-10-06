#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_analysis_json.py - Comprehensive betting market analysis and ticket generation tool

This script handles comprehensive parsing and normalization of betting markets from:
- API-Football bookmaker responses
- Local analysis.json files

Supports markets including:
- Match Winner / 1X2
- Double Chance (1X, X2, 12)
- Over/Under with various thresholds (0.5, 1.5, 2.5, 3.5, 4.5, etc.)
- Both Teams To Score (BTTS)
- And more common betting markets

Usage:
    python check_analysis_json.py [options]
    python check_analysis_json.py --ticket    # Generate ticket/slip from best picks
    python check_analysis_json.py --check     # Check analysis files for completeness
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict

# ============= LOGGING =================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)
logger = logging.getLogger("check_analysis")

# ============= CONFIGURATION =================
DATA_ROOT = Path(os.getenv("DATA_ROOT", ".")).resolve()


@dataclass
class NormalizedMarket:
    """Normalized market representation"""
    market_key: str  # e.g., "1X2", "Goals|OU:2.5", "DoubleChance|1X"
    market_type: str  # e.g., "1X2", "OverUnder", "DoubleChance", "BTTS"
    selections: Dict[str, float]  # e.g., {"HOME": 2.50, "DRAW": 3.20, "AWAY": 2.80}
    threshold: Optional[float] = None  # For O/U markets
    bookmaker: Optional[str] = None
    raw_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TicketPick:
    """A single betting pick for a ticket"""
    fixture_id: int
    home_team: str
    away_team: str
    kickoff_utc: str
    market_type: str
    selection: str
    odds: float
    edge: float
    model_prob: float
    market_prob: float
    confidence: str
    league_name: Optional[str] = None
    rationale: Optional[str] = None


class MarketNormalizer:
    """
    Comprehensive market name parser and normalizer for API-Football and common bookmakers
    
    Reference: https://www.api-football.com/documentation-v3
    """
    
    # Market name patterns for 1X2 / Match Winner
    MATCH_WINNER_PATTERNS = [
        "match winner",
        "1x2",
        "fulltime result",
        "match result",
        "full time result",
        "ft result",
        "full-time result",
        "3way",
        "three way",
        "match betting"
    ]
    
    # Double Chance patterns
    DOUBLE_CHANCE_PATTERNS = [
        "double chance",
        "dbl chance",
        "dc"
    ]
    
    # Over/Under patterns
    OVER_UNDER_PATTERNS = [
        "goals over/under",
        "over/under",
        "goals",
        "total goals",
        "match goals",
        "total",
        "over under",
        "ou"
    ]
    
    # BTTS patterns
    BTTS_PATTERNS = [
        "both teams to score",
        "btts",
        "both teams score",
        "goal/no goal",
        "gg/ng",
        "both score",
        "teams to score"
    ]
    
    # Handicap patterns
    HANDICAP_PATTERNS = [
        "asian handicap",
        "handicap",
        "asian hcp",
        "ah"
    ]
    
    # Correct Score patterns
    CORRECT_SCORE_PATTERNS = [
        "correct score",
        "exact score",
        "score"
    ]
    
    # Half Time / Full Time patterns
    HT_FT_PATTERNS = [
        "halftime/fulltime",
        "half time/full time",
        "ht/ft",
        "halftime fulltime"
    ]
    
    @staticmethod
    def normalize_selection_name(name: str) -> str:
        """Normalize selection/outcome names to standard format"""
        name_lower = name.strip().lower()
        
        # 1X2 selections
        if name_lower in ("home", "1", "h", "hazai"):
            return "HOME"
        elif name_lower in ("draw", "x", "d", "döntetlen", "tie"):
            return "DRAW"
        elif name_lower in ("away", "2", "a", "vendég", "visitor"):
            return "AWAY"
        
        # Double Chance selections
        elif name_lower in ("1x", "home/draw", "home or draw", "1 or x"):
            return "1X"
        elif name_lower in ("x2", "draw/away", "draw or away", "x or 2"):
            return "X2"
        elif name_lower in ("12", "1/2", "home/away", "home or away", "1 or 2"):
            return "12"
        
        # Over/Under selections
        elif "over" in name_lower or "több" in name_lower:
            return "OVER"
        elif "under" in name_lower or "kevesebb" in name_lower or "alatt" in name_lower:
            return "UNDER"
        
        # BTTS selections
        elif name_lower in ("yes", "y", "goal", "igen", "both"):
            return "YES"
        elif name_lower in ("no", "n", "no goal", "nem", "neither"):
            return "NO"
        
        # Handicap selections
        elif any(x in name_lower for x in ["+", "−", "-"]) and any(c.isdigit() for c in name_lower):
            # Try to extract handicap value
            return name.strip().upper()
        
        # Return normalized version if not matched
        return name.strip().upper()
    
    @staticmethod
    def extract_threshold_from_name(name: str) -> Optional[float]:
        """Extract numeric threshold from market name (e.g., '2.5' from 'Over/Under 2.5')"""
        import re
        # Look for patterns like "2.5", "1.5", "0.5", etc.
        patterns = [
            r'(\d+\.5)',  # Matches 0.5, 1.5, 2.5, etc.
            r'(\d+\.\d+)',  # Matches any decimal
            r'over\s*(\d+)',  # Matches "over 2", "over 3", etc.
            r'under\s*(\d+)',  # Matches "under 2", "under 3", etc.
        ]
        
        for pattern in patterns:
            match = re.search(pattern, name.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None
    
    @classmethod
    def normalize_market(cls, market_name: str, values: List[Dict[str, Any]], 
                        bookmaker: Optional[str] = None) -> Optional[NormalizedMarket]:
        """
        Normalize a market from API-Football format to standardized format
        
        Args:
            market_name: Raw market name from bookmaker
            values: List of {value: str, odd: float} dicts
            bookmaker: Bookmaker name (optional)
        
        Returns:
            NormalizedMarket object or None if market not recognized
        """
        name_lower = market_name.strip().lower()
        
        # Try to match 1X2 / Match Winner
        if any(pattern in name_lower for pattern in cls.MATCH_WINNER_PATTERNS):
            selections = {}
            for val in values:
                sel_name = cls.normalize_selection_name(val.get("value", ""))
                try:
                    odd = float(val.get("odd", 0))
                    if odd > 0 and sel_name in ("HOME", "DRAW", "AWAY"):
                        selections[sel_name] = odd
                except (ValueError, TypeError):
                    continue
            
            if len(selections) == 3:
                return NormalizedMarket(
                    market_key="1X2",
                    market_type="1X2",
                    selections=selections,
                    bookmaker=bookmaker,
                    raw_name=market_name
                )
        
        # Try to match Double Chance
        elif any(pattern in name_lower for pattern in cls.DOUBLE_CHANCE_PATTERNS):
            selections = {}
            for val in values:
                sel_name = cls.normalize_selection_name(val.get("value", ""))
                try:
                    odd = float(val.get("odd", 0))
                    if odd > 0 and sel_name in ("1X", "X2", "12"):
                        selections[sel_name] = odd
                except (ValueError, TypeError):
                    continue
            
            if len(selections) == 3:
                return NormalizedMarket(
                    market_key="DoubleChance",
                    market_type="DoubleChance",
                    selections=selections,
                    bookmaker=bookmaker,
                    raw_name=market_name
                )
        
        # Try to match Over/Under
        elif any(pattern in name_lower for pattern in cls.OVER_UNDER_PATTERNS):
            threshold = cls.extract_threshold_from_name(market_name)
            
            # Also check values for threshold info
            if threshold is None:
                for val in values:
                    val_name = val.get("value", "")
                    extracted = cls.extract_threshold_from_name(val_name)
                    if extracted:
                        threshold = extracted
                        break
            
            selections = {}
            for val in values:
                sel_name = cls.normalize_selection_name(val.get("value", ""))
                try:
                    odd = float(val.get("odd", 0))
                    if odd > 0 and sel_name in ("OVER", "UNDER"):
                        selections[sel_name] = odd
                except (ValueError, TypeError):
                    continue
            
            if len(selections) == 2 and threshold is not None:
                market_key = f"Goals|OU:{threshold}"
                return NormalizedMarket(
                    market_key=market_key,
                    market_type="OverUnder",
                    selections=selections,
                    threshold=threshold,
                    bookmaker=bookmaker,
                    raw_name=market_name
                )
        
        # Try to match BTTS
        elif any(pattern in name_lower for pattern in cls.BTTS_PATTERNS):
            # Exclude halftime/extra time variants
            if any(exclude in name_lower for exclude in 
                   ["1st", "2nd", "first half", "second half", "1h", "2h", "half", "extra", "corners", "penalt"]):
                return None
            
            selections = {}
            for val in values:
                sel_name = cls.normalize_selection_name(val.get("value", ""))
                try:
                    odd = float(val.get("odd", 0))
                    if odd > 0 and sel_name in ("YES", "NO"):
                        selections[sel_name] = odd
                except (ValueError, TypeError):
                    continue
            
            if len(selections) == 2:
                return NormalizedMarket(
                    market_key="BTTS",
                    market_type="BTTS",
                    selections=selections,
                    bookmaker=bookmaker,
                    raw_name=market_name
                )
        
        # Try to match Asian Handicap
        elif any(pattern in name_lower for pattern in cls.HANDICAP_PATTERNS):
            # Extract handicap value from market name first
            threshold = cls.extract_threshold_from_name(market_name)
            
            # If not in name, try to extract from values
            if threshold is None:
                for val in values:
                    val_name = val.get("value", "")
                    extracted = cls.extract_threshold_from_name(val_name)
                    if extracted:
                        threshold = extracted
                        break
            
            selections = {}
            for val in values:
                sel_name = val.get("value", "").strip()
                try:
                    odd = float(val.get("odd", 0))
                    if odd > 0:
                        # Keep raw selection for handicap (e.g., "Home -1.5", "Away +1.5")
                        selections[sel_name] = odd
                except (ValueError, TypeError):
                    continue
            
            if len(selections) == 2 and threshold is not None:
                market_key = f"Handicap:{threshold}"
                return NormalizedMarket(
                    market_key=market_key,
                    market_type="Handicap",
                    selections=selections,
                    threshold=threshold,
                    bookmaker=bookmaker,
                    raw_name=market_name
                )
        
        # Try to match HT/FT (Half Time / Full Time)
        elif any(pattern in name_lower for pattern in cls.HT_FT_PATTERNS):
            selections = {}
            for val in values:
                sel_name = val.get("value", "").strip()
                try:
                    odd = float(val.get("odd", 0))
                    if odd > 0:
                        # Keep raw selection for HT/FT (e.g., "Home/Home", "Draw/Away")
                        selections[sel_name] = odd
                except (ValueError, TypeError):
                    continue
            
            if len(selections) >= 3:  # HT/FT typically has 9 outcomes
                return NormalizedMarket(
                    market_key="HT/FT",
                    market_type="HT/FT",
                    selections=selections,
                    bookmaker=bookmaker,
                    raw_name=market_name
                )
        
        # Try to match Correct Score
        elif any(pattern in name_lower for pattern in cls.CORRECT_SCORE_PATTERNS):
            # Only match if it's the main correct score market, not corners or other variants
            if any(exclude in name_lower for exclude in ["corner", "card", "booking"]):
                return None
            
            selections = {}
            for val in values:
                sel_name = val.get("value", "").strip()
                try:
                    odd = float(val.get("odd", 0))
                    if odd > 0:
                        # Keep raw selection for correct score (e.g., "1-0", "2-1")
                        selections[sel_name] = odd
                except (ValueError, TypeError):
                    continue
            
            if len(selections) >= 3:
                return NormalizedMarket(
                    market_key="CorrectScore",
                    market_type="CorrectScore",
                    selections=selections,
                    bookmaker=bookmaker,
                    raw_name=market_name
                )
        
        return None


def extract_bookmaker_odds_from_odds_response(odds_response: List[Dict]) -> Dict[str, NormalizedMarket]:
    """
    Extract and normalize all recognized markets from API-Football odds response
    
    Args:
        odds_response: The 'response' field from API-Football odds endpoint
    
    Returns:
        Dict mapping market_key to NormalizedMarket objects
    """
    all_markets = {}
    
    for fixture_odds in odds_response:
        bookmakers = fixture_odds.get("bookmakers", [])
        
        for bookmaker_data in bookmakers:
            bookmaker_name = bookmaker_data.get("name", "Unknown")
            bets = bookmaker_data.get("bets", [])
            
            for bet in bets:
                market_name = bet.get("name", "")
                values = bet.get("values", [])
                
                normalized = MarketNormalizer.normalize_market(
                    market_name, values, bookmaker_name
                )
                
                if normalized:
                    # Store best odds per market (lowest margin)
                    market_key = normalized.market_key
                    if market_key not in all_markets:
                        all_markets[market_key] = normalized
                    else:
                        # Compare margins and keep best
                        existing = all_markets[market_key]
                        new_margin = calculate_margin(normalized.selections)
                        existing_margin = calculate_margin(existing.selections)
                        if new_margin < existing_margin:
                            all_markets[market_key] = normalized
    
    return all_markets


def calculate_margin(selections: Dict[str, float]) -> float:
    """Calculate bookmaker margin (overround) from odds"""
    if not selections:
        return 999.0
    
    try:
        total_implied = sum(1.0 / odd for odd in selections.values())
        return total_implied
    except (ZeroDivisionError, ValueError):
        return 999.0


def load_analysis_files(data_root: Path, include_raw_odds: bool = False) -> List[Dict]:
    """
    Load all analysis.json files from fixture directories
    
    Args:
        data_root: Root directory containing fixture folders
        include_raw_odds: If True, also load and parse raw odds files
    
    Returns:
        List of dicts containing analysis data and optionally raw odds
    """
    results = []
    
    for fixture_dir in data_root.glob("out_fixture_*"):
        if not fixture_dir.is_dir():
            continue
        
        analysis_file = fixture_dir / "analysis.json"
        if not analysis_file.exists():
            continue
        
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                item = {
                    "fixture_id": data.get("fixture_id"),
                    "path": analysis_file,
                    "data": data
                }
                
                # Optionally load raw odds
                if include_raw_odds:
                    raw_dir = fixture_dir / "raw"
                    if raw_dir.exists():
                        for odds_file in raw_dir.glob("odds*.json"):
                            try:
                                with open(odds_file, 'r', encoding='utf-8') as of:
                                    odds_data = json.load(of)
                                    odds_response = odds_data.get("response", [])
                                    if odds_response:
                                        normalized_markets = extract_bookmaker_odds_from_odds_response(odds_response)
                                        item["normalized_markets"] = normalized_markets
                                        break
                            except Exception as e:
                                logger.debug(f"Failed to load odds from {odds_file}: {e}")
                
                results.append(item)
        except Exception as e:
            logger.error(f"Failed to load {analysis_file}: {e}")
    
    return results


def check_analysis_completeness(analysis_data: Dict) -> Dict[str, Any]:
    """Check an analysis file for completeness and quality"""
    issues = []
    warnings = []
    
    # Check required fields
    required_fields = [
        "fixture_id", "kickoff_utc", "league_id", "model_probs",
        "teams", "lambda_home", "lambda_away"
    ]
    
    for field in required_fields:
        if field not in analysis_data:
            issues.append(f"Missing required field: {field}")
    
    # Check odds availability
    odds = analysis_data.get("odds")
    if not odds:
        warnings.append("No odds data available")
    else:
        if not odds.get("home") or not odds.get("draw") or not odds.get("away"):
            warnings.append("Incomplete 1X2 odds")
    
    market_odds = analysis_data.get("market_odds", {})
    if not market_odds:
        warnings.append("No market odds (BTTS/O/U) available")
    
    # Check model probabilities
    model_probs = analysis_data.get("model_probs", {})
    if model_probs:
        total_prob = model_probs.get("home", 0) + model_probs.get("draw", 0) + model_probs.get("away", 0)
        if abs(total_prob - 1.0) > 0.01:
            issues.append(f"Model probabilities don't sum to 1.0 (sum={total_prob:.4f})")
    
    # Check edges
    edge = analysis_data.get("edge", {})
    if edge:
        best_edge = max(edge.values()) if edge.values() else 0
        if best_edge > 0.15:
            warnings.append(f"High edge detected ({best_edge:.2%}) - verify odds")
    
    return {
        "issues": issues,
        "warnings": warnings,
        "complete": len(issues) == 0,
        "quality_score": 1.0 - (len(issues) * 0.3 + len(warnings) * 0.1)
    }


def generate_ticket_from_analysis(analysis_files: List[Dict], 
                                  min_edge: float = 0.05,
                                  max_picks: int = 5,
                                  max_odds: float = 4.0) -> List[TicketPick]:
    """
    Generate a betting ticket (slip) from analyzed fixtures
    
    Args:
        analysis_files: List of loaded analysis data
        min_edge: Minimum edge to consider a pick
        max_picks: Maximum number of picks for the ticket
        max_odds: Maximum odds to accept
    
    Returns:
        List of TicketPick objects
    """
    picks = []
    
    for item in analysis_files:
        data = item.get("data", {})
        
        fixture_id = data.get("fixture_id")
        if not fixture_id:
            continue
        
        # Get basic info
        teams = data.get("teams", {})
        home_name = teams.get("home_name", "Unknown")
        away_name = teams.get("away_name", "Unknown")
        kickoff = data.get("kickoff_utc", "")
        league_name = data.get("league_name", "Unknown League")
        
        # Get odds and edges
        odds = data.get("odds")
        edges = data.get("edge", {})
        model_probs = data.get("model_probs", {})
        
        # Check 1X2 market
        if odds:
            for selection in ["home", "draw", "away"]:
                odd_val = odds.get(selection)
                edge_val = edges.get(selection, 0)
                model_prob = model_probs.get(selection, 0)
                
                if (odd_val and edge_val >= min_edge and 
                    1.0 < odd_val <= max_odds and model_prob > 0):
                    
                    # Calculate implied probability
                    market_prob = 1.0 / odd_val if odd_val > 0 else 0
                    
                    # Determine confidence
                    if edge_val >= 0.15:
                        confidence = "HIGH"
                    elif edge_val >= 0.08:
                        confidence = "MEDIUM"
                    else:
                        confidence = "LOW"
                    
                    picks.append(TicketPick(
                        fixture_id=fixture_id,
                        home_team=home_name,
                        away_team=away_name,
                        kickoff_utc=kickoff,
                        market_type="1X2",
                        selection=selection.upper(),
                        odds=odd_val,
                        edge=edge_val,
                        model_prob=model_prob,
                        market_prob=market_prob,
                        confidence=confidence,
                        league_name=league_name
                    ))
        
        # Check 2-way markets (BTTS, O/U)
        market_odds = data.get("market_odds", {})
        market_edge = data.get("market_edge", {})
        market_probs = data.get("market_probs", {})
        
        for market_key in ["btts_yes", "btts_no", "over25", "under25"]:
            odd_val = market_odds.get(market_key)
            edge_val = market_edge.get(market_key, 0)
            model_prob = market_probs.get(market_key, 0)
            
            if (odd_val and edge_val >= min_edge and 
                1.0 < odd_val <= max_odds and model_prob > 0):
                
                market_prob = 1.0 / odd_val if odd_val > 0 else 0
                
                if edge_val >= 0.15:
                    confidence = "HIGH"
                elif edge_val >= 0.08:
                    confidence = "MEDIUM"
                else:
                    confidence = "LOW"
                
                # Determine market type and selection
                if "btts" in market_key:
                    market_type = "BTTS"
                    selection = "YES" if "yes" in market_key else "NO"
                else:
                    market_type = "O/U 2.5"
                    selection = "OVER" if "over" in market_key else "UNDER"
                
                picks.append(TicketPick(
                    fixture_id=fixture_id,
                    home_team=home_name,
                    away_team=away_name,
                    kickoff_utc=kickoff,
                    market_type=market_type,
                    selection=selection,
                    odds=odd_val,
                    edge=edge_val,
                    model_prob=model_prob,
                    market_prob=market_prob,
                    confidence=confidence,
                    league_name=league_name
                ))
    
    # Sort by edge (descending) and take top picks
    picks.sort(key=lambda x: x.edge, reverse=True)
    return picks[:max_picks]


def format_ticket_display(picks: List[TicketPick]) -> str:
    """Format ticket picks for display"""
    if not picks:
        return "No picks available for ticket."
    
    lines = []
    lines.append("=" * 80)
    lines.append("BETTING TICKET / SLIP")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Total picks: {len(picks)}")
    lines.append("")
    
    total_odds = 1.0
    for i, pick in enumerate(picks, 1):
        total_odds *= pick.odds
        
        lines.append(f"[{i}] {pick.home_team} vs {pick.away_team}")
        lines.append(f"    League: {pick.league_name}")
        lines.append(f"    Kickoff: {pick.kickoff_utc}")
        lines.append(f"    Market: {pick.market_type} | Selection: {pick.selection}")
        lines.append(f"    Odds: {pick.odds:.2f} | Edge: {pick.edge:.2%} | Confidence: {pick.confidence}")
        lines.append(f"    Model Prob: {pick.model_prob:.2%} | Market Prob: {pick.market_prob:.2%}")
        lines.append("")
    
    lines.append("-" * 80)
    lines.append(f"Combined odds: {total_odds:.2f}")
    lines.append(f"Expected value: {(total_odds * (sum(p.model_prob for p in picks) / len(picks)) - 1):.2%}")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def generate_market_coverage_report(analysis_files: List[Dict]) -> str:
    """Generate a report showing market coverage across analysis files"""
    lines = []
    lines.append("=" * 80)
    lines.append("MARKET COVERAGE REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    total_fixtures = len(analysis_files)
    market_stats = {
        "1X2": 0,
        "BTTS": 0,
        "O/U 2.5": 0,
        "O/U 1.5": 0,
        "O/U 3.5": 0,
        "Double Chance": 0,
        "Handicap": 0,
        "Other": 0
    }
    
    for item in analysis_files:
        data = item.get("data", {})
        
        # Check 1X2
        if data.get("odds"):
            market_stats["1X2"] += 1
        
        # Check 2-way markets
        market_odds = data.get("market_odds", {})
        if "btts_yes" in market_odds or "btts_no" in market_odds:
            market_stats["BTTS"] += 1
        if "over25" in market_odds or "under25" in market_odds:
            market_stats["O/U 2.5"] += 1
    
    lines.append(f"Total fixtures analyzed: {total_fixtures}")
    lines.append("")
    lines.append("Market Availability:")
    for market, count in market_stats.items():
        if count > 0:
            percentage = (count / total_fixtures * 100) if total_fixtures > 0 else 0
            lines.append(f"  {market:20s}: {count:3d} / {total_fixtures:3d} ({percentage:5.1f}%)")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Check and analyze betting analysis files with comprehensive market support"
    )
    parser.add_argument(
        "--ticket", "--slip",
        action="store_true",
        help="Generate betting ticket/slip from best picks"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check analysis files for completeness and quality"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate market coverage report"
    )
    parser.add_argument(
        "--markets",
        action="store_true",
        help="Show all available markets from raw odds files"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Root directory containing fixture data"
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.05,
        help="Minimum edge for ticket picks (default: 0.05)"
    )
    parser.add_argument(
        "--max-picks",
        type=int,
        default=5,
        help="Maximum picks for ticket (default: 5)"
    )
    parser.add_argument(
        "--max-odds",
        type=float,
        default=4.0,
        help="Maximum odds to accept (default: 4.0)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for ticket JSON (optional)"
    )
    
    args = parser.parse_args()
    
    # Load analysis files
    logger.info(f"Loading analysis files from {args.data_root}")
    include_odds = args.markets  # Load raw odds if markets mode
    analysis_files = load_analysis_files(args.data_root, include_raw_odds=include_odds)
    logger.info(f"Loaded {len(analysis_files)} analysis files")
    
    if not analysis_files:
        logger.warning("No analysis files found")
        return 1
    
    # Mode: Markets
    if args.markets:
        logger.info("Showing available markets from raw odds files...")
        
        for item in analysis_files:
            fixture_id = item.get("fixture_id", "unknown")
            normalized = item.get("normalized_markets", {})
            
            if normalized:
                print(f"\nFixture {fixture_id}:")
                for market_key, market in normalized.items():
                    print(f"  {market_key:20s} [{market.bookmaker or 'N/A'}]")
                    for sel, odd in market.selections.items():
                        print(f"    {sel:15s}: {odd:.2f}")
            else:
                print(f"\nFixture {fixture_id}: No odds data available")
        
        return 0
    
    # Mode: Report
    if args.report:
        logger.info("Generating market coverage report...")
        report = generate_market_coverage_report(analysis_files)
        print(report)
        return 0
    
    # Mode: Check
    if args.check:
        logger.info("Checking analysis files for completeness...")
        
        for item in analysis_files:
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
        
        return 0
    
    # Mode: Ticket/Slip
    if args.ticket:
        logger.info("Generating betting ticket...")
        
        picks = generate_ticket_from_analysis(
            analysis_files,
            min_edge=args.min_edge,
            max_picks=args.max_picks,
            max_odds=args.max_odds
        )
        
        # Display ticket
        ticket_text = format_ticket_display(picks)
        print(ticket_text)
        
        # Save to file if requested
        if args.output:
            output_data = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "picks": [asdict(p) for p in picks],
                "combined_odds": round(
                    sum(p.odds for p in picks) if picks else 0, 2
                )
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Ticket saved to {args.output}")
        
        return 0
    
    # Default: Show stats
    logger.info("Analysis files summary:")
    print(f"Total fixtures: {len(analysis_files)}")
    
    complete_count = sum(
        1 for item in analysis_files 
        if check_analysis_completeness(item.get("data", {}))["complete"]
    )
    print(f"Complete: {complete_count}/{len(analysis_files)}")
    
    # Count fixtures with odds
    with_odds = sum(
        1 for item in analysis_files 
        if item.get("data", {}).get("odds")
    )
    print(f"With odds: {with_odds}/{len(analysis_files)}")
    
    print("\nUse --ticket to generate betting slip")
    print("Use --check to verify analysis completeness")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
