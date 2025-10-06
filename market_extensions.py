#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
market_extensions.py
Dynamic market extraction and probability computation for O/U, corners, cards, handicap markets.
"""

from __future__ import annotations
import re
import math
from typing import Dict, List, Tuple, Any, Optional


# ============= POISSON HELPERS =================

def poisson_p(k: int, lam: float) -> float:
    """
    Poisson PMF: P(X=k) = (lam^k * e^(-lam)) / k!
    
    Args:
        k: Number of events
        lam: Expected value (lambda)
    
    Returns:
        Probability of exactly k events
    """
    if lam <= 0:
        return 0.0
    try:
        return (lam ** k) * math.exp(-lam) / math.factorial(k)
    except (OverflowError, ValueError):
        return 0.0


def prob_over_threshold(lam: float, threshold: float) -> float:
    """
    P(X > threshold) using Poisson distribution.
    
    Args:
        lam: Expected value (lambda)
        threshold: Threshold value
    
    Returns:
        Probability of exceeding threshold
    """
    if lam <= 0:
        return 0.0
    # P(X > threshold) = 1 - P(X <= threshold)
    # For integer threshold, P(X <= n) = sum(poisson_p(k, lam) for k in 0..n)
    n = int(threshold)
    cumulative = sum(poisson_p(k, lam) for k in range(n + 1))
    return max(0.0, min(1.0, 1.0 - cumulative))


def prob_goal_diff_gt(lambda_h: float, lambda_a: float, margin: int) -> float:
    """
    P(goals_home - goals_away > margin) using Poisson distributions.
    
    Args:
        lambda_h: Home team expected goals
        lambda_a: Away team expected goals
        margin: Goal difference margin
    
    Returns:
        Probability that home team wins by more than margin goals
    """
    if lambda_h <= 0 or lambda_a <= 0:
        return 0.0
    
    prob = 0.0
    max_goals = 12  # Reasonable upper limit for computation
    
    try:
        for gh in range(max_goals + 1):
            p_gh = poisson_p(gh, lambda_h)
            if p_gh < 1e-10:
                continue
            for ga in range(max_goals + 1):
                if gh - ga > margin:
                    p_ga = poisson_p(ga, lambda_a)
                    prob += p_gh * p_ga
    except (OverflowError, ValueError):
        return 0.0
    
    return max(0.0, min(1.0, prob))


# ============= DYNAMIC MARKET EXTRACTION =================

def extract_dynamic_markets(odds_response_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract dynamic markets (O/U, corners, cards, handicap) from raw odds responses.
    
    Args:
        odds_response_list: List of odds response items from API
    
    Returns:
        Dictionary with market types as keys and lists of market data as values
    """
    markets = {
        "over_under": [],
        "corners": [],
        "cards": [],
        "handicap": []
    }
    
    if not odds_response_list or not isinstance(odds_response_list, list):
        return markets
    
    for item in odds_response_list:
        try:
            bookmakers = item.get("bookmakers", [])
            if not isinstance(bookmakers, list):
                continue
            
            for bookmaker in bookmakers:
                bets = bookmaker.get("bets", [])
                if not isinstance(bets, list):
                    continue
                
                for bet in bets:
                    name = (bet.get("name") or "").strip().lower()
                    values = bet.get("values", [])
                    if not isinstance(values, list):
                        continue
                    
                    # Extract Over/Under markets
                    if extract_over_under_market(name, values, markets):
                        continue
                    
                    # Extract Corners markets
                    if extract_corners_market(name, values, markets):
                        continue
                    
                    # Extract Cards markets
                    if extract_cards_market(name, values, markets):
                        continue
                    
                    # Extract Handicap markets
                    if extract_handicap_market(name, values, markets):
                        continue
        
        except Exception:
            continue
    
    return markets


def extract_over_under_market(name: str, values: List[Dict], markets: Dict) -> bool:
    """Extract Over/Under markets from bet data."""
    # Match patterns like "goals over/under", "over/under 2.5", etc.
    if "over" not in name or "under" not in name:
        return False
    
    # Skip corners and cards
    if "corner" in name or "card" in name:
        return False
    
    for value_item in values:
        try:
            val_str = (value_item.get("value") or "").strip().lower()
            odd = float(value_item.get("odd", 0))
            
            # Extract threshold from value like "Over 2.5" or "Under 1.5"
            match = re.search(r'(over|under)\s*(\d+\.?\d*)', val_str)
            if match:
                side = match.group(1)
                threshold = float(match.group(2))
                
                markets["over_under"].append({
                    "threshold": threshold,
                    "side": side,
                    "odd": odd,
                    "market_name": name
                })
        except (ValueError, AttributeError):
            continue
    
    return True


def extract_corners_market(name: str, values: List[Dict], markets: Dict) -> bool:
    """Extract Corners markets from bet data."""
    if "corner" not in name:
        return False
    
    for value_item in values:
        try:
            val_str = (value_item.get("value") or "").strip().lower()
            odd = float(value_item.get("odd", 0))
            
            # Extract threshold from corners O/U
            match = re.search(r'(over|under)\s*(\d+\.?\d*)', val_str)
            if match:
                side = match.group(1)
                threshold = float(match.group(2))
                
                markets["corners"].append({
                    "threshold": threshold,
                    "side": side,
                    "odd": odd,
                    "market_name": name
                })
        except (ValueError, AttributeError):
            continue
    
    return True


def extract_cards_market(name: str, values: List[Dict], markets: Dict) -> bool:
    """Extract Cards markets from bet data."""
    if "card" not in name:
        return False
    
    for value_item in values:
        try:
            val_str = (value_item.get("value") or "").strip().lower()
            odd = float(value_item.get("odd", 0))
            
            # Extract threshold from cards O/U
            match = re.search(r'(over|under)\s*(\d+\.?\d*)', val_str)
            if match:
                side = match.group(1)
                threshold = float(match.group(2))
                
                markets["cards"].append({
                    "threshold": threshold,
                    "side": side,
                    "odd": odd,
                    "market_name": name
                })
        except (ValueError, AttributeError):
            continue
    
    return True


def extract_handicap_market(name: str, values: List[Dict], markets: Dict) -> bool:
    """Extract Handicap markets from bet data."""
    # Match patterns for handicap markets
    if "handicap" not in name and "asian" not in name:
        return False
    
    for value_item in values:
        try:
            val_str = (value_item.get("value") or "").strip().lower()
            odd = float(value_item.get("odd", 0))
            
            # Extract handicap value like "+1.5", "-2.0", etc.
            match = re.search(r'([+-]?\d+\.?\d*)', val_str)
            if match:
                handicap = float(match.group(1))
                
                # Determine team (home/away)
                team = "home" if "home" in val_str or "1" in val_str else "away"
                
                markets["handicap"].append({
                    "handicap": handicap,
                    "team": team,
                    "odd": odd,
                    "market_name": name
                })
        except (ValueError, AttributeError):
            continue
    
    return True


# ============= PROBABILITY COMPUTATION =================

def compute_probs_for_dynamic_markets(
    dynamic_markets: Dict[str, List[Dict[str, Any]]],
    model_context: Dict[str, Any]
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """
    Compute model probabilities and edges for extracted dynamic markets.
    
    Args:
        dynamic_markets: Output from extract_dynamic_markets
        model_context: Dict with keys:
            - lambda_home: Home team expected goals
            - lambda_away: Away team expected goals
            - lambda_total: Total expected goals
            - corners_total: Expected total corners (optional)
            - lambda_cards: Expected cards (optional)
    
    Returns:
        Tuple of (market_probs, market_edges, details)
        - market_probs: Dict mapping market keys to probabilities
        - market_edges: Dict mapping market keys to edge values
        - details: Dict with additional market information
    """
    market_probs: Dict[str, float] = {}
    market_edges: Dict[str, float] = {}
    details: Dict[str, Any] = {}
    
    # Extract model parameters with safe defaults
    lambda_home = float(model_context.get("lambda_home", 0.05))
    lambda_away = float(model_context.get("lambda_away", 0.05))
    lambda_total = float(model_context.get("lambda_total", lambda_home + lambda_away))
    corners_total = model_context.get("corners_total")
    lambda_cards = float(model_context.get("lambda_cards", 0.9))
    
    # Ensure positive lambda values
    lambda_home = max(0.05, lambda_home)
    lambda_away = max(0.05, lambda_away)
    lambda_total = max(0.1, lambda_total)
    lambda_cards = max(0.05, lambda_cards)
    
    try:
        # Process Over/Under markets
        for ou_item in dynamic_markets.get("over_under", []):
            try:
                threshold = ou_item["threshold"]
                side = ou_item["side"]
                odd = ou_item["odd"]
                
                if side == "over":
                    prob = prob_over_threshold(lambda_total, threshold)
                else:  # under
                    prob = 1.0 - prob_over_threshold(lambda_total, threshold)
                
                key = f"ou_{threshold}_{side}"
                market_probs[key] = prob
                market_edges[key] = prob * odd - 1.0
                
                details[key] = {
                    "threshold": threshold,
                    "side": side,
                    "odd": odd,
                    "model_prob": prob,
                    "edge": prob * odd - 1.0
                }
            except (KeyError, ValueError, TypeError):
                continue
        
        # Process Corners markets
        if corners_total is not None:
            corners_lam = max(0.1, float(corners_total))
            for corner_item in dynamic_markets.get("corners", []):
                try:
                    threshold = corner_item["threshold"]
                    side = corner_item["side"]
                    odd = corner_item["odd"]
                    
                    if side == "over":
                        prob = prob_over_threshold(corners_lam, threshold)
                    else:  # under
                        prob = 1.0 - prob_over_threshold(corners_lam, threshold)
                    
                    key = f"corners_{threshold}_{side}"
                    market_probs[key] = prob
                    market_edges[key] = prob * odd - 1.0
                    
                    details[key] = {
                        "threshold": threshold,
                        "side": side,
                        "odd": odd,
                        "model_prob": prob,
                        "edge": prob * odd - 1.0
                    }
                except (KeyError, ValueError, TypeError):
                    continue
        
        # Process Cards markets
        for card_item in dynamic_markets.get("cards", []):
            try:
                threshold = card_item["threshold"]
                side = card_item["side"]
                odd = card_item["odd"]
                
                if side == "over":
                    prob = prob_over_threshold(lambda_cards, threshold)
                else:  # under
                    prob = 1.0 - prob_over_threshold(lambda_cards, threshold)
                
                key = f"cards_{threshold}_{side}"
                market_probs[key] = prob
                market_edges[key] = prob * odd - 1.0
                
                details[key] = {
                    "threshold": threshold,
                    "side": side,
                    "odd": odd,
                    "model_prob": prob,
                    "edge": prob * odd - 1.0
                }
            except (KeyError, ValueError, TypeError):
                continue
        
        # Process Handicap markets
        for hcap_item in dynamic_markets.get("handicap", []):
            try:
                handicap = hcap_item["handicap"]
                team = hcap_item["team"]
                odd = hcap_item["odd"]
                
                # For handicap, compute probability using goal difference
                # Positive handicap gives advantage to the team
                if team == "home":
                    # Home team with handicap: need to win by more than abs(handicap)
                    if handicap >= 0:
                        # Favorable handicap for home
                        margin = int(abs(handicap)) - 1
                    else:
                        # Unfavorable handicap for home
                        margin = int(abs(handicap))
                    prob = prob_goal_diff_gt(lambda_home, lambda_away, margin)
                else:
                    # Away team with handicap
                    if handicap >= 0:
                        margin = int(abs(handicap)) - 1
                    else:
                        margin = int(abs(handicap))
                    prob = prob_goal_diff_gt(lambda_away, lambda_home, margin)
                
                key = f"handicap_{team}_{handicap}"
                market_probs[key] = prob
                market_edges[key] = prob * odd - 1.0
                
                details[key] = {
                    "handicap": handicap,
                    "team": team,
                    "odd": odd,
                    "model_prob": prob,
                    "edge": prob * odd - 1.0
                }
            except (KeyError, ValueError, TypeError):
                continue
    
    except Exception:
        # Return empty results on any error
        pass
    
    return market_probs, market_edges, details


def flatten_market_dicts(
    market_probs: Dict[str, float],
    market_edges: Dict[str, float]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Flatten market probability and edge dictionaries.
    
    This is a pass-through function that can be extended for additional processing.
    
    Args:
        market_probs: Dictionary of market probabilities
        market_edges: Dictionary of market edges
    
    Returns:
        Tuple of (flattened_probs, flattened_edges)
    """
    # For now, just return as-is since they're already flat dictionaries
    # Can be extended later for nested structure handling
    return dict(market_probs), dict(market_edges)
