#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced Feature Engineering Layer for Betting Analysis
Integrates with API-Football v3 to extract rich statistical features
"""

from __future__ import annotations
import json, math, logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

logger = logging.getLogger("feature_engineering")

@dataclass
class AdvancedFeatures:
    """Enhanced feature set for betting analysis"""
    # Basic features
    fixture_id: int
    home_team_id: int
    away_team_id: int
    league_id: int
    season: int
    
    # Form and momentum features
    home_form_points: float = 0.0
    away_form_points: float = 0.0
    home_momentum: float = 0.0
    away_momentum: float = 0.0
    home_goal_momentum: float = 0.0
    away_goal_momentum: float = 0.0
    
    # Head-to-head features
    h2h_home_wins: int = 0
    h2h_away_wins: int = 0
    h2h_draws: int = 0
    h2h_avg_goals: float = 0.0
    h2h_btts_rate: float = 0.0
    
    # Attack/Defense strength features
    home_attack_strength: float = 0.0
    away_attack_strength: float = 0.0
    home_defense_strength: float = 0.0
    away_defense_strength: float = 0.0
    home_xg_per_game: float = 0.0
    away_xg_per_game: float = 0.0
    
    # Tactical features
    home_possession_avg: float = 0.0
    away_possession_avg: float = 0.0
    home_pass_accuracy: float = 0.0
    away_pass_accuracy: float = 0.0
    home_shots_per_game: float = 0.0
    away_shots_per_game: float = 0.0
    
    # Injury and lineup features
    home_key_players_missing: int = 0
    away_key_players_missing: int = 0
    home_lineup_stability: float = 0.0
    away_lineup_stability: float = 0.0
    
    # Market and odds features
    market_consensus_home: float = 0.0
    market_consensus_draw: float = 0.0
    market_consensus_away: float = 0.0
    odds_efficiency: float = 0.0
    market_volume_indicator: float = 0.0
    
    # Advanced statistical features
    home_elo_rating: float = 1500.0
    away_elo_rating: float = 1500.0
    rating_differential: float = 0.0
    home_recent_performance: float = 0.0
    away_recent_performance: float = 0.0
    
    # Time and context features
    days_since_last_match_home: int = 0
    days_since_last_match_away: int = 0
    match_importance_factor: float = 1.0
    weather_factor: float = 1.0
    crowd_factor: float = 1.0

class AdvancedFeatureEngineer:
    """Advanced feature engineering for betting analysis"""
    
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.scalers = {}
        self.team_stats_cache = {}
        self.league_averages_cache = {}
        
    def extract_features(self, fixture_data: Dict[str, Any]) -> AdvancedFeatures:
        """Extract comprehensive features from fixture data"""
        fixture = fixture_data.get("fixture", {})
        league = fixture_data.get("league", {})
        teams = fixture_data.get("teams", {})
        
        features = AdvancedFeatures(
            fixture_id=fixture.get("id", 0),
            home_team_id=teams.get("home", {}).get("id", 0),
            away_team_id=teams.get("away", {}).get("id", 0),
            league_id=league.get("id", 0),
            season=league.get("season", 0)
        )
        
        # Extract all available features
        self._extract_form_features(features, fixture_data)
        self._extract_h2h_features(features, fixture_data)
        self._extract_strength_features(features, fixture_data)
        self._extract_tactical_features(features, fixture_data)
        self._extract_lineup_features(features, fixture_data)
        self._extract_market_features(features, fixture_data)
        self._extract_advanced_stats(features, fixture_data)
        self._extract_context_features(features, fixture_data)
        
        return features
    
    def _extract_form_features(self, features: AdvancedFeatures, data: Dict[str, Any]):
        """Extract form and momentum features"""
        try:
            # Get recent matches data
            home_recent = self._load_raw_data(features.fixture_id, "form_home_last")
            away_recent = self._load_raw_data(features.fixture_id, "form_away_last")
            
            if home_recent:
                features.home_form_points = self._calculate_form_points(home_recent, features.home_team_id)
                features.home_momentum = self._calculate_momentum(home_recent, features.home_team_id)
                features.home_goal_momentum = self._calculate_goal_momentum(home_recent, features.home_team_id)
                
            if away_recent:
                features.away_form_points = self._calculate_form_points(away_recent, features.away_team_id)
                features.away_momentum = self._calculate_momentum(away_recent, features.away_team_id)
                features.away_goal_momentum = self._calculate_goal_momentum(away_recent, features.away_team_id)
                
        except Exception as e:
            logger.warning(f"Error extracting form features: {e}")
    
    def _extract_h2h_features(self, features: AdvancedFeatures, data: Dict[str, Any]):
        """Extract head-to-head features"""
        try:
            h2h_data = self._load_raw_data(features.fixture_id, "h2h")
            if not h2h_data:
                return
                
            matches = h2h_data.get("response", [])
            if not matches:
                return
                
            home_wins = away_wins = draws = 0
            total_goals = 0
            btts_count = 0
            
            for match in matches:
                home_goals = match.get("goals", {}).get("home", 0) or 0
                away_goals = match.get("goals", {}).get("away", 0) or 0
                total_goals += home_goals + away_goals
                
                if home_goals > 0 and away_goals > 0:
                    btts_count += 1
                    
                home_id = match.get("teams", {}).get("home", {}).get("id")
                if home_id == features.home_team_id:
                    if home_goals > away_goals:
                        home_wins += 1
                    elif away_goals > home_goals:
                        away_wins += 1
                    else:
                        draws += 1
                else:
                    if away_goals > home_goals:
                        home_wins += 1
                    elif home_goals > away_goals:
                        away_wins += 1
                    else:
                        draws += 1
            
            total_matches = len(matches)
            if total_matches > 0:
                features.h2h_home_wins = home_wins
                features.h2h_away_wins = away_wins
                features.h2h_draws = draws
                features.h2h_avg_goals = total_goals / total_matches
                features.h2h_btts_rate = btts_count / total_matches
                
        except Exception as e:
            logger.warning(f"Error extracting H2H features: {e}")
    
    def _extract_strength_features(self, features: AdvancedFeatures, data: Dict[str, Any]):
        """Extract attack/defense strength features"""
        try:
            home_stats = self._load_raw_data(features.fixture_id, "team_stats_home")
            away_stats = self._load_raw_data(features.fixture_id, "team_stats_away")
            
            league_avg = self._get_league_averages(features.league_id, features.season)
            
            if home_stats:
                features.home_attack_strength = self._calculate_attack_strength(home_stats, league_avg)
                features.home_defense_strength = self._calculate_defense_strength(home_stats, league_avg)
                
            if away_stats:
                features.away_attack_strength = self._calculate_attack_strength(away_stats, league_avg)
                features.away_defense_strength = self._calculate_defense_strength(away_stats, league_avg)
                
        except Exception as e:
            logger.warning(f"Error extracting strength features: {e}")
    
    def _extract_tactical_features(self, features: AdvancedFeatures, data: Dict[str, Any]):
        """Extract tactical and style features"""
        try:
            home_stats = self._load_raw_data(features.fixture_id, "team_stats_home")
            away_stats = self._load_raw_data(features.fixture_id, "team_stats_away")
            
            if home_stats:
                stats = home_stats.get("response", {})
                if stats:
                    features.home_possession_avg = self._extract_stat_value(stats, "ball_possession_for")
                    features.home_pass_accuracy = self._extract_stat_value(stats, "passes_accuracy")
                    features.home_shots_per_game = self._extract_stat_value(stats, "goals_for_average_total")
                    
            if away_stats:
                stats = away_stats.get("response", {})
                if stats:
                    features.away_possession_avg = self._extract_stat_value(stats, "ball_possession_for")
                    features.away_pass_accuracy = self._extract_stat_value(stats, "passes_accuracy")
                    features.away_shots_per_game = self._extract_stat_value(stats, "goals_for_average_total")
                    
        except Exception as e:
            logger.warning(f"Error extracting tactical features: {e}")
    
    def _extract_lineup_features(self, features: AdvancedFeatures, data: Dict[str, Any]):
        """Extract lineup and player availability features"""
        try:
            home_squad = self._load_raw_data(features.fixture_id, "squad_home")
            away_squad = self._load_raw_data(features.fixture_id, "squad_away")
            
            if home_squad:
                features.home_key_players_missing = self._count_missing_key_players(home_squad)
                features.home_lineup_stability = self._calculate_lineup_stability(home_squad)
                
            if away_squad:
                features.away_key_players_missing = self._count_missing_key_players(away_squad)
                features.away_lineup_stability = self._calculate_lineup_stability(away_squad)
                
        except Exception as e:
            logger.warning(f"Error extracting lineup features: {e}")
    
    def _extract_market_features(self, features: AdvancedFeatures, data: Dict[str, Any]):
        """Extract market and odds-based features"""
        try:
            odds_data = self._load_raw_data(features.fixture_id, "odds")
            if not odds_data:
                return
                
            # Extract market consensus from multiple bookmakers
            bookmaker_odds = []
            for bookmaker in odds_data.get("response", []):
                for bet in bookmaker.get("bets", []):
                    if bet.get("name") == "Match Winner":
                        values = bet.get("values", [])
                        if len(values) >= 3:
                            home_odds = float(values[0].get("odd", 0))
                            draw_odds = float(values[1].get("odd", 0))
                            away_odds = float(values[2].get("odd", 0))
                            
                            if all(odd > 0 for odd in [home_odds, draw_odds, away_odds]):
                                bookmaker_odds.append((home_odds, draw_odds, away_odds))
            
            if bookmaker_odds:
                # Calculate market consensus
                avg_home = np.mean([1/odds[0] for odds in bookmaker_odds])
                avg_draw = np.mean([1/odds[1] for odds in bookmaker_odds])
                avg_away = np.mean([1/odds[2] for odds in bookmaker_odds])
                
                total = avg_home + avg_draw + avg_away
                features.market_consensus_home = avg_home / total
                features.market_consensus_draw = avg_draw / total
                features.market_consensus_away = avg_away / total
                
                # Calculate odds efficiency (inverse of market margin)
                features.odds_efficiency = 1.0 / total if total > 0 else 0.0
                
                # Market volume indicator (number of bookmakers)
                features.market_volume_indicator = len(bookmaker_odds) / 10.0  # Normalized
                
        except Exception as e:
            logger.warning(f"Error extracting market features: {e}")
    
    def _extract_advanced_stats(self, features: AdvancedFeatures, data: Dict[str, Any]):
        """Extract advanced statistical features"""
        try:
            # Calculate ELO ratings based on recent performance
            features.home_elo_rating = self._calculate_elo_rating(features.home_team_id, features.league_id)
            features.away_elo_rating = self._calculate_elo_rating(features.away_team_id, features.league_id)
            features.rating_differential = features.home_elo_rating - features.away_elo_rating
            
            # Recent performance indicators
            features.home_recent_performance = self._calculate_recent_performance(features.home_team_id)
            features.away_recent_performance = self._calculate_recent_performance(features.away_team_id)
            
        except Exception as e:
            logger.warning(f"Error extracting advanced stats: {e}")
    
    def _extract_context_features(self, features: AdvancedFeatures, data: Dict[str, Any]):
        """Extract contextual features"""
        try:
            # Days since last match
            features.days_since_last_match_home = self._days_since_last_match(features.home_team_id)
            features.days_since_last_match_away = self._days_since_last_match(features.away_team_id)
            
            # Match importance (based on league position, cup stage, etc.)
            features.match_importance_factor = self._calculate_match_importance(features.league_id, features.season)
            
            # Weather and venue factors would require additional data sources
            features.weather_factor = 1.0  # Placeholder
            features.crowd_factor = 1.0   # Placeholder
            
        except Exception as e:
            logger.warning(f"Error extracting context features: {e}")
    
    def create_feature_matrix(self, features_list: List[AdvancedFeatures]) -> np.ndarray:
        """Create a feature matrix for machine learning models"""
        if not features_list:
            return np.array([])
            
        # Convert features to dictionaries and then to DataFrame
        features_dicts = [asdict(f) for f in features_list]
        df = pd.DataFrame(features_dicts)
        
        # Select numeric features only
        numeric_features = df.select_dtypes(include=[np.number]).columns
        feature_matrix = df[numeric_features].fillna(0).values
        
        return feature_matrix
    
    def apply_feature_scaling(self, X: np.ndarray, scaler_type: str = "standard") -> np.ndarray:
        """Apply feature scaling to the feature matrix"""
        if scaler_type not in self.scalers:
            if scaler_type == "standard":
                self.scalers[scaler_type] = StandardScaler()
            elif scaler_type == "minmax":
                self.scalers[scaler_type] = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")
                
        return self.scalers[scaler_type].fit_transform(X)
    
    def apply_dimensionality_reduction(self, X: np.ndarray, n_components: int = 10) -> np.ndarray:
        """Apply PCA for dimensionality reduction"""
        pca = PCA(n_components=n_components)
        return pca.fit_transform(X)
    
    def create_team_clusters(self, features_list: List[AdvancedFeatures], n_clusters: int = 5) -> Dict[int, int]:
        """Create team clusters based on playing style"""
        if len(features_list) < n_clusters:
            return {}
            
        # Create team-level aggregated features
        team_features = {}
        for features in features_list:
            for team_id in [features.home_team_id, features.away_team_id]:
                if team_id not in team_features:
                    team_features[team_id] = []
                    
                # Add relevant features for this team
                if team_id == features.home_team_id:
                    team_features[team_id].append([
                        features.home_attack_strength,
                        features.home_defense_strength,
                        features.home_possession_avg,
                        features.home_pass_accuracy,
                        features.home_shots_per_game
                    ])
                else:
                    team_features[team_id].append([
                        features.away_attack_strength,
                        features.away_defense_strength,
                        features.away_possession_avg,
                        features.away_pass_accuracy,
                        features.away_shots_per_game
                    ])
        
        # Average features per team
        team_avg_features = {}
        for team_id, feature_list in team_features.items():
            if feature_list:
                team_avg_features[team_id] = np.mean(feature_list, axis=0)
        
        if len(team_avg_features) < n_clusters:
            return {}
            
        # Apply clustering
        X = np.array(list(team_avg_features.values()))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Return team -> cluster mapping
        return {team_id: int(cluster_labels[i]) 
                for i, team_id in enumerate(team_avg_features.keys())}
    
    # Helper methods
    def _load_raw_data(self, fixture_id: int, data_type: str) -> Optional[Dict]:
        """Load raw data for a specific fixture and data type"""
        try:
            data_dir = self.data_root / f"out_fixture_{fixture_id}" / "raw"
            files = list(data_dir.glob(f"{data_type}__*.json"))
            if files:
                with open(files[0], 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return None
    
    def _calculate_form_points(self, matches_data: Dict, team_id: int) -> float:
        """Calculate weighted form points"""
        matches = matches_data.get("response", [])
        if not matches:
            return 0.0
            
        points = 0
        total_weight = 0
        
        for i, match in enumerate(matches[:10]):  # Last 10 matches
            weight = 1.0 / (i + 1)  # More recent matches have higher weight
            
            home_id = match.get("teams", {}).get("home", {}).get("id")
            home_goals = match.get("goals", {}).get("home", 0) or 0
            away_goals = match.get("goals", {}).get("away", 0) or 0
            
            if home_id == team_id:
                if home_goals > away_goals:
                    points += 3 * weight
                elif home_goals == away_goals:
                    points += 1 * weight
            else:
                if away_goals > home_goals:
                    points += 3 * weight
                elif away_goals == home_goals:
                    points += 1 * weight
                    
            total_weight += weight
            
        return points / total_weight if total_weight > 0 else 0.0
    
    def _calculate_momentum(self, matches_data: Dict, team_id: int) -> float:
        """Calculate team momentum based on recent results trend"""
        matches = matches_data.get("response", [])
        if len(matches) < 3:
            return 0.0
            
        recent_points = []
        for match in matches[:5]:  # Last 5 matches
            home_id = match.get("teams", {}).get("home", {}).get("id")
            home_goals = match.get("goals", {}).get("home", 0) or 0
            away_goals = match.get("goals", {}).get("away", 0) or 0
            
            if home_id == team_id:
                if home_goals > away_goals:
                    recent_points.append(3)
                elif home_goals == away_goals:
                    recent_points.append(1)
                else:
                    recent_points.append(0)
            else:
                if away_goals > home_goals:
                    recent_points.append(3)
                elif away_goals == home_goals:
                    recent_points.append(1)
                else:
                    recent_points.append(0)
        
        if len(recent_points) < 3:
            return 0.0
            
        # Calculate trend (positive = improving, negative = declining)
        weights = np.arange(1, len(recent_points) + 1)
        weighted_avg = np.average(recent_points, weights=weights)
        overall_avg = np.mean(recent_points)
        
        return (weighted_avg - overall_avg) / 3.0  # Normalize to [-1, 1]
    
    def _calculate_goal_momentum(self, matches_data: Dict, team_id: int) -> float:
        """Calculate goal scoring momentum"""
        matches = matches_data.get("response", [])
        if len(matches) < 3:
            return 0.0
            
        recent_goals = []
        for match in matches[:5]:
            home_id = match.get("teams", {}).get("home", {}).get("id")
            home_goals = match.get("goals", {}).get("home", 0) or 0
            away_goals = match.get("goals", {}).get("away", 0) or 0
            
            goals_scored = home_goals if home_id == team_id else away_goals
            recent_goals.append(goals_scored)
            
        if len(recent_goals) < 3:
            return 0.0
            
        # Calculate goal scoring trend
        weights = np.arange(1, len(recent_goals) + 1)
        weighted_avg = np.average(recent_goals, weights=weights)
        overall_avg = np.mean(recent_goals)
        
        return weighted_avg - overall_avg
    
    def _get_league_averages(self, league_id: int, season: int) -> Dict[str, float]:
        """Get or calculate league averages for normalization"""
        cache_key = f"{league_id}_{season}"
        if cache_key not in self.league_averages_cache:
            # This would typically be calculated from historical data
            # For now, return reasonable defaults
            self.league_averages_cache[cache_key] = {
                "goals_per_game": 2.5,
                "shots_per_game": 12.0,
                "possession": 50.0,
                "pass_accuracy": 80.0
            }
        return self.league_averages_cache[cache_key]
    
    def _calculate_attack_strength(self, stats_data: Dict, league_avg: Dict) -> float:
        """Calculate relative attack strength"""
        try:
            stats = stats_data.get("response", {})
            if not stats:
                return 1.0
                
            goals_for = self._extract_stat_value(stats, "goals_for_total_home") + \
                       self._extract_stat_value(stats, "goals_for_total_away")
            matches_played = self._extract_stat_value(stats, "fixtures_played_total")
            
            if matches_played > 0:
                goals_per_game = goals_for / matches_played
                return goals_per_game / league_avg.get("goals_per_game", 2.5)
        except Exception:
            pass
        return 1.0
    
    def _calculate_defense_strength(self, stats_data: Dict, league_avg: Dict) -> float:
        """Calculate relative defense strength (inverse of goals conceded)"""
        try:
            stats = stats_data.get("response", {})
            if not stats:
                return 1.0
                
            goals_against = self._extract_stat_value(stats, "goals_against_total_home") + \
                           self._extract_stat_value(stats, "goals_against_total_away")
            matches_played = self._extract_stat_value(stats, "fixtures_played_total")
            
            if matches_played > 0:
                goals_against_per_game = goals_against / matches_played
                # Lower goals against = higher defense strength
                return league_avg.get("goals_per_game", 2.5) / max(goals_against_per_game, 0.1)
        except Exception:
            pass
        return 1.0
    
    def _extract_stat_value(self, stats: Dict, key: str) -> float:
        """Extract numeric value from nested stats structure"""
        try:
            value = stats
            for part in key.split('_'):
                if isinstance(value, dict):
                    value = value.get(part, 0)
                else:
                    return 0.0
            
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                return float(value.rstrip('%'))
        except (ValueError, TypeError, AttributeError):
            pass
        return 0.0
    
    def _count_missing_key_players(self, squad_data: Dict) -> int:
        """Count missing key players (simplified implementation)"""
        # This would require more sophisticated logic to identify key players
        # For now, return a random number for demonstration
        return 0
    
    def _calculate_lineup_stability(self, squad_data: Dict) -> float:
        """Calculate lineup stability factor"""
        # This would analyze lineup consistency over recent matches
        # For now, return a default value
        return 0.8
    
    def _calculate_elo_rating(self, team_id: int, league_id: int) -> float:
        """Calculate ELO rating for a team"""
        # This would maintain ELO ratings based on match results
        # For now, return a base rating
        return 1500.0
    
    def _calculate_recent_performance(self, team_id: int) -> float:
        """Calculate recent performance indicator"""
        # This would analyze recent match performance metrics
        # For now, return a neutral value
        return 0.0
    
    def _days_since_last_match(self, team_id: int) -> int:
        """Calculate days since team's last match"""
        # This would require tracking team schedules
        # For now, return a default value
        return 3
    
    def _calculate_match_importance(self, league_id: int, season: int) -> float:
        """Calculate match importance factor"""
        # This would consider league standings, cup stages, etc.
        # For now, return a neutral value
        return 1.0

def extract_enhanced_features(data_root: Path, fixture_data: Dict[str, Any]) -> AdvancedFeatures:
    """Convenience function to extract enhanced features"""
    engineer = AdvancedFeatureEngineer(data_root)
    return engineer.extract_features(fixture_data)

def create_feature_pipeline(data_root: Path, fixtures_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[AdvancedFeatures]]:
    """Create a complete feature engineering pipeline"""
    engineer = AdvancedFeatureEngineer(data_root)
    
    # Extract features for all fixtures
    features_list = []
    for fixture_data in fixtures_data:
        features = engineer.extract_features(fixture_data)
        features_list.append(features)
    
    # Create feature matrix
    X = engineer.create_feature_matrix(features_list)
    
    # Apply scaling
    if X.size > 0:
        X_scaled = engineer.apply_feature_scaling(X, "standard")
        return X_scaled, features_list
    
    return np.array([]), features_list