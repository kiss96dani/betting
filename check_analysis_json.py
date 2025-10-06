#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI tool to check and summarize analysis.json files.
Includes --ticket mode to build a betting slip with top tips across markets.
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional

# Default markets for ticket mode
DEFAULT_TICKET_MARKETS = "1X2,BTTS,OU:1.5,OU:2.5,OU:3.5,Corners,Cards"


def load_json(path: Path) -> Optional[dict]:
    """Load JSON file safely."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_fixture_meta(fixture_id: int, data_root: Path) -> dict:
    """
    Fast resolution of fixture metadata from primary_fixture.json.
    Returns team names and venue info.
    """
    pf = data_root / f"out_fixture_{fixture_id}" / "primary_fixture.json"
    if not pf.exists():
        return {}
    try:
        js = json.loads(pf.read_text(encoding="utf-8"))
        fixture = js.get("fixture", {}) or {}
        venue = fixture.get("venue", {}) or {}
        teams = js.get("teams", {}) or {}
        league = js.get("league", {}) or {}
        return {
            "home_name": teams.get("home", {}).get("name"),
            "away_name": teams.get("away", {}).get("name"),
            "league_name": league.get("name"),
            "league_country": league.get("country"),
            "venue_name": venue.get("name"),
            "venue_city": venue.get("city"),
        }
    except Exception:
        return {}


def load_all_analysis(data_root: Path) -> list[dict]:
    """Load all analysis.json files from fixture directories."""
    results = []
    for p in data_root.glob("out_fixture_*"):
        if not p.is_dir():
            continue
        af = p / "analysis.json"
        if af.exists():
            try:
                js = json.loads(af.read_text(encoding="utf-8"))
                results.append(js)
            except Exception:
                pass
    return results


def is_today(iso_utc: str, local_tz: str = "Europe/Budapest") -> bool:
    """Check if the given UTC timestamp is today in local timezone."""
    try:
        tz = ZoneInfo(local_tz)
        dt_utc = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
        today_local = datetime.now(tz=tz).date()
        return dt_utc.astimezone(tz).date() == today_local
    except:
        return False


def extract_markets_for_analysis(analysis: dict) -> Dict[str, List[dict]]:
    """
    Extract market selections and calculate candidate scores for each market.
    Returns a dict mapping market names to list of candidates.
    
    Supported markets:
    - 1X2: home/draw/away from model_probs and odds
    - BTTS: yes/no from market_probs and market_odds
    - OU:1.5, OU:2.5, OU:3.5: over/under from market_probs (if available)
    - Corners, Cards: placeholder for future enhancement
    """
    fixture_id = analysis.get("fixture_id")
    kickoff_utc = analysis.get("kickoff_utc")
    
    candidates = {}
    
    # === 1X2 market ===
    odds_1x2 = analysis.get("odds") or {}
    probs_1x2 = analysis.get("model_probs") or {}
    edges_1x2 = analysis.get("edge") or {}
    
    if odds_1x2 and probs_1x2:
        candidates["1X2"] = []
        for sel in ("home", "draw", "away"):
            edge_val = edges_1x2.get(sel, 0)
            prob = probs_1x2.get(sel, 0)
            odd = odds_1x2.get(sel)
            if odd and edge_val > 0:
                try:
                    odd_f = float(odd)
                    candidates["1X2"].append({
                        "fixture_id": fixture_id,
                        "kickoff_utc": kickoff_utc,
                        "market": "1X2",
                        "selection": sel.upper(),
                        "edge": edge_val,
                        "odds": odd_f,
                        "model_prob": prob,
                    })
                except:
                    pass
    
    # === BTTS market ===
    market_odds = analysis.get("market_odds") or {}
    market_probs = analysis.get("market_probs") or {}
    market_edge = analysis.get("market_edge") or {}
    
    if "btts_yes" in market_probs and "btts_yes" in market_odds:
        candidates["BTTS"] = []
        for sel in ("btts_yes", "btts_no"):
            edge_val = market_edge.get(sel, 0)
            prob = market_probs.get(sel, 0)
            odd = market_odds.get(sel)
            if odd and edge_val > 0:
                try:
                    odd_f = float(odd)
                    label = sel.replace("btts_", "").upper()
                    candidates["BTTS"].append({
                        "fixture_id": fixture_id,
                        "kickoff_utc": kickoff_utc,
                        "market": "BTTS",
                        "selection": label,
                        "edge": edge_val,
                        "odds": odd_f,
                        "model_prob": prob,
                    })
                except:
                    pass
    
    # === Over/Under markets ===
    # OU:2.5 is the main one typically available
    if "over25" in market_probs and "over25" in market_odds:
        candidates["OU:2.5"] = []
        for sel in ("over25", "under25"):
            edge_val = market_edge.get(sel, 0)
            prob = market_probs.get(sel, 0)
            odd = market_odds.get(sel)
            if odd and edge_val > 0:
                try:
                    odd_f = float(odd)
                    label = "OVER" if "over" in sel else "UNDER"
                    candidates["OU:2.5"].append({
                        "fixture_id": fixture_id,
                        "kickoff_utc": kickoff_utc,
                        "market": "OU:2.5",
                        "selection": label,
                        "edge": edge_val,
                        "odds": odd_f,
                        "model_prob": prob,
                    })
                except:
                    pass
    
    # OU:1.5 and OU:3.5 - check if available in market_probs
    # These are less common, but we'll try to extract if present
    for threshold in ["15", "35"]:
        over_key = f"over{threshold}"
        under_key = f"under{threshold}"
        market_name = f"OU:{threshold[0]}.{threshold[1]}"
        
        if over_key in market_probs and over_key in market_odds:
            candidates[market_name] = []
            for sel in (over_key, under_key):
                edge_val = market_edge.get(sel, 0)
                prob = market_probs.get(sel, 0)
                odd = market_odds.get(sel)
                if odd and edge_val > 0:
                    try:
                        odd_f = float(odd)
                        label = "OVER" if "over" in sel else "UNDER"
                        candidates[market_name].append({
                            "fixture_id": fixture_id,
                            "kickoff_utc": kickoff_utc,
                            "market": market_name,
                            "selection": label,
                            "edge": edge_val,
                            "odds": odd_f,
                            "model_prob": prob,
                        })
                    except:
                        pass
    
    # Corners and Cards are placeholders - not typically in analysis.json yet
    # Could be added in future if odds/probs become available
    
    return candidates


def candidate_score(candidate: dict) -> float:
    """
    Calculate score for a candidate tip.
    Simple scoring: use edge as the primary metric.
    """
    return candidate.get("edge", 0)


def build_ticket(
    analyzed_results: List[dict],
    markets: List[str],
    local_tz: str = "Europe/Budapest"
) -> Dict[str, Optional[dict]]:
    """
    Build a betting ticket by selecting one tip per requested market.
    
    Selection algorithm:
    1. Iterate through markets in the provided order
    2. For each market, collect all candidates from all fixtures
    3. Pick the candidate with highest edge that uses a fixture not yet selected
    4. If no candidate available (all fixtures used), leave market empty
    
    Returns: dict mapping market name -> selected tip (or None if empty)
    """
    ticket = {}
    used_fixtures = set()
    
    for market in markets:
        # Collect all candidates for this market across all fixtures
        all_candidates = []
        for analysis in analyzed_results:
            # Only include today's fixtures
            if not is_today(analysis.get("kickoff_utc", ""), local_tz):
                continue
            
            candidates_by_market = extract_markets_for_analysis(analysis)
            market_candidates = candidates_by_market.get(market, [])
            all_candidates.extend(market_candidates)
        
        # Sort by score (edge) descending
        all_candidates.sort(key=candidate_score, reverse=True)
        
        # Pick the best candidate that doesn't use an already-selected fixture
        selected = None
        for cand in all_candidates:
            fid = cand.get("fixture_id")
            if fid not in used_fixtures:
                selected = cand
                used_fixtures.add(fid)
                break
        
        ticket[market] = selected
    
    return ticket


def format_ticket(ticket: Dict[str, Optional[dict]], data_root: Path) -> str:
    """
    Format the ticket for human-readable output.
    
    Output format:
    === BETTING TICKET ===
    Date: 2024-01-15
    
    1X2:      HOME - Arsenal vs Chelsea @2.50 (edge: 12.5%)
    BTTS:     YES - Liverpool vs Man City @1.85 (edge: 8.3%)
    OU:2.5:   OVER - Barcelona vs Real Madrid @1.95 (edge: 6.7%)
    ...
    
    Total tips: 5/7 markets filled
    """
    lines = []
    lines.append("=" * 60)
    lines.append("           BETTING TICKET")
    lines.append("=" * 60)
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    
    filled = 0
    total = len(ticket)
    
    for market, tip in ticket.items():
        if tip is None:
            lines.append(f"{market:12} (no tip available)")
        else:
            fixture_id = tip.get("fixture_id")
            selection = tip.get("selection", "")
            odds = tip.get("odds", 0)
            edge = tip.get("edge", 0)
            
            # Get team names
            meta = load_fixture_meta(fixture_id, data_root)
            home_name = meta.get("home_name") or "Unknown"
            away_name = meta.get("away_name") or "Unknown"
            
            # Format match name
            match_str = f"{home_name} vs {away_name}"
            
            # Format line
            line = f"{market:12} {selection:8} - {match_str} @{odds:.2f} (edge: {edge*100:.1f}%)"
            lines.append(line)
            filled += 1
    
    lines.append("")
    lines.append("-" * 60)
    lines.append(f"Total: {filled}/{total} markets filled")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Check and summarize analysis.json files"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Root directory containing out_fixture_* folders (default: current directory)"
    )
    parser.add_argument(
        "--ticket",
        "--slip",
        action="store_true",
        dest="ticket_mode",
        help="Build a betting ticket with top tips"
    )
    parser.add_argument(
        "--ticket-markets",
        type=str,
        default=DEFAULT_TICKET_MARKETS,
        help=f"Comma-separated list of markets for ticket (default: {DEFAULT_TICKET_MARKETS})"
    )
    parser.add_argument(
        "--local-tz",
        type=str,
        default="Europe/Budapest",
        help="Local timezone for date filtering (default: Europe/Budapest)"
    )
    
    args = parser.parse_args()
    
    # Load all analysis files
    print(f"Loading analysis files from {args.data_root}...")
    analyzed_results = load_all_analysis(args.data_root)
    print(f"Loaded {len(analyzed_results)} analysis files.")
    
    if args.ticket_mode:
        # Parse markets
        markets = [m.strip() for m in args.ticket_markets.split(",") if m.strip()]
        print(f"Building ticket for markets: {markets}")
        print()
        
        # Build ticket
        ticket = build_ticket(analyzed_results, markets, args.local_tz)
        
        # Format and print
        output = format_ticket(ticket, args.data_root)
        print(output)
    else:
        # Default summary mode (not requested in requirements, but useful)
        print("\nSummary of analysis files:")
        print(f"Total fixtures analyzed: {len(analyzed_results)}")
        
        today_count = sum(1 for a in analyzed_results if is_today(a.get("kickoff_utc", ""), args.local_tz))
        print(f"Today's fixtures: {today_count}")
        
        if today_count > 0:
            print("\nToday's fixtures:")
            for analysis in analyzed_results:
                if is_today(analysis.get("kickoff_utc", ""), args.local_tz):
                    fid = analysis.get("fixture_id")
                    meta = load_fixture_meta(fid, args.data_root)
                    home = meta.get("home_name", f"fixture_{fid}")
                    away = meta.get("away_name", "Unknown")
                    print(f"  - {home} vs {away}")


if __name__ == "__main__":
    main()
