#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Napi futball előrejelző rendszer - Single-file implementation
Minden logika egyetlen fájlban: config, data collection, analysis, prediction, DB, FastAPI
"""

from __future__ import annotations
import os
import sys
import math
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path
import asyncio

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)
logger = logging.getLogger("betting_predictor")


# ============= CONFIG =================
@dataclass
class Config:
    """Környezeti változók és konfiguráció"""
    api_football_key: str
    database_url: str
    timeout: float
    max_retries: int
    base_url: str
    
    @classmethod
    def from_env(cls) -> "Config":
        """Betöltés környezeti változókból"""
        return cls(
            api_football_key=os.getenv("API_FOOTBALL_KEY", ""),
            database_url=os.getenv("DATABASE_URL", "sqlite:///predictions.db"),
            timeout=float(os.getenv("REQUEST_TIMEOUT", "15")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            base_url=os.getenv("API_BASE_URL", "https://v3.football.api-sports.io")
        )


# ============= DATA COLLECTOR =================
class DataCollector:
    """API-Football v3 adatgyűjtő httpx-szel, retry/backoff kezelés"""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = None
        
    async def __aenter__(self):
        import httpx
        self.session = httpx.AsyncClient(
            timeout=self.config.timeout,
            headers={
                "x-apisports-key": self.config.api_football_key
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def _request_with_retry(self, endpoint: str, params: dict = None) -> dict:
        """Retry logika exponential backoff-fal"""
        import httpx
        url = f"{self.config.base_url}/{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self.session.get(url, params=params or {})
                response.raise_for_status()
                data = response.json()
                
                if data.get("errors"):
                    logger.warning(f"API error: {data['errors']}")
                    
                return data
                
            except httpx.HTTPStatusError as e:
                wait_time = 2 ** attempt
                logger.warning(f"HTTP {e.response.status_code} on attempt {attempt+1}, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                
            except (httpx.RequestError, httpx.TimeoutException) as e:
                wait_time = 2 ** attempt
                logger.warning(f"Request error on attempt {attempt+1}: {e}, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        
        logger.error(f"Failed to fetch {endpoint} after {self.config.max_retries} attempts")
        return {"response": [], "errors": ["max_retries_exceeded"]}
    
    async def get_fixtures(self, date: str = None) -> List[dict]:
        """Mérkőzések lekérése adott napra"""
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            
        logger.info(f"Fetching fixtures for date: {date}")
        data = await self._request_with_retry("fixtures", {"date": date})
        fixtures = data.get("response", [])
        logger.info(f"Found {len(fixtures)} fixtures")
        return fixtures
    
    async def get_team_statistics(self, team_id: int, season: int, league_id: int) -> dict:
        """Csapat statisztikák lekérése"""
        logger.info(f"Fetching team statistics: team={team_id}, season={season}, league={league_id}")
        data = await self._request_with_retry(
            "teams/statistics",
            {"team": team_id, "season": season, "league": league_id}
        )
        return data.get("response", {})


# ============= ANALYZER =================
class Analyzer:
    """Egyszerű feature engineering és elemzés"""
    
    def __init__(self, collector: DataCollector):
        self.collector = collector
    
    def calculate_last_5_form(self, fixtures: List[dict], team_id: int) -> dict:
        """Utolsó 5 meccs alapján forma számítás"""
        team_fixtures = [
            f for f in fixtures 
            if f.get("teams", {}).get("home", {}).get("id") == team_id 
            or f.get("teams", {}).get("away", {}).get("id") == team_id
        ]
        
        # Rendezés dátum szerint (legutolsó először)
        team_fixtures.sort(
            key=lambda x: x.get("fixture", {}).get("date", ""), 
            reverse=True
        )
        
        last_5 = team_fixtures[:5]
        
        goals_scored = 0
        goals_conceded = 0
        points = 0
        form_string = ""
        
        for fixture in last_5:
            if fixture.get("fixture", {}).get("status", {}).get("short") != "FT":
                continue
                
            home_team = fixture.get("teams", {}).get("home", {})
            away_team = fixture.get("teams", {}).get("away", {})
            goals = fixture.get("goals", {})
            
            is_home = home_team.get("id") == team_id
            
            if is_home:
                team_goals = goals.get("home", 0) or 0
                opponent_goals = goals.get("away", 0) or 0
            else:
                team_goals = goals.get("away", 0) or 0
                opponent_goals = goals.get("home", 0) or 0
            
            goals_scored += team_goals
            goals_conceded += opponent_goals
            
            if team_goals > opponent_goals:
                points += 3
                form_string += "W"
            elif team_goals == opponent_goals:
                points += 1
                form_string += "D"
            else:
                form_string += "L"
        
        num_matches = len([f for f in last_5 if f.get("fixture", {}).get("status", {}).get("short") == "FT"])
        
        return {
            "matches_played": num_matches,
            "goals_scored": goals_scored,
            "goals_conceded": goals_conceded,
            "goals_per_match": goals_scored / num_matches if num_matches > 0 else 0.0,
            "conceded_per_match": goals_conceded / num_matches if num_matches > 0 else 0.0,
            "points": points,
            "form_string": form_string,
            "avg_points": points / num_matches if num_matches > 0 else 0.0
        }
    
    async def analyze_fixture(self, fixture: dict, all_fixtures: List[dict]) -> dict:
        """Egy mérkőzés elemzése"""
        home_team = fixture.get("teams", {}).get("home", {})
        away_team = fixture.get("teams", {}).get("away", {})
        
        home_id = home_team.get("id")
        away_id = away_team.get("id")
        
        home_form = self.calculate_last_5_form(all_fixtures, home_id)
        away_form = self.calculate_last_5_form(all_fixtures, away_id)
        
        return {
            "fixture_id": fixture.get("fixture", {}).get("id"),
            "home_team": home_team.get("name"),
            "away_team": away_team.get("name"),
            "home_form": home_form,
            "away_form": away_form,
            "date": fixture.get("fixture", {}).get("date"),
            "league": fixture.get("league", {}).get("name"),
            "league_id": fixture.get("league", {}).get("id")
        }


# ============= PREDICTOR =================
class Predictor:
    """Poisson-alapú becslés és predikciók"""
    
    @staticmethod
    def poisson_probability(lambda_param: float, k: int) -> float:
        """Poisson eloszlás valószínűség számítás"""
        if lambda_param < 0:
            return 0.0
        try:
            return (lambda_param ** k) * math.exp(-lambda_param) / math.factorial(k)
        except (ValueError, OverflowError):
            return 0.0
    
    def calculate_score_distribution(self, home_lambda: float, away_lambda: float, 
                                     max_goals: int = 6) -> Dict[tuple, float]:
        """Score distribution számítás"""
        distribution = {}
        
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                prob = (
                    self.poisson_probability(home_lambda, home_goals) * 
                    self.poisson_probability(away_lambda, away_goals)
                )
                distribution[(home_goals, away_goals)] = prob
        
        return distribution
    
    def get_top_n_scorelines(self, distribution: Dict[tuple, float], n: int = 5) -> List[dict]:
        """Top N legvalószínűbb végeredmény"""
        sorted_scores = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                "scoreline": f"{score[0]}-{score[1]}",
                "home_goals": score[0],
                "away_goals": score[1],
                "probability": prob
            }
            for score, prob in sorted_scores[:n]
        ]
    
    def calculate_1x2_probabilities(self, distribution: Dict[tuple, float]) -> dict:
        """1X2 valószínűségek aggregálása"""
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        
        for (home_goals, away_goals), prob in distribution.items():
            if home_goals > away_goals:
                home_win += prob
            elif home_goals == away_goals:
                draw += prob
            else:
                away_win += prob
        
        return {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win
        }
    
    def predict_fixture(self, analysis: dict) -> dict:
        """Mérkőzés predikció"""
        home_form = analysis.get("home_form", {})
        away_form = analysis.get("away_form", {})
        
        # Lambda paraméterek becslése forma alapján
        home_lambda = max(0.5, home_form.get("goals_per_match", 1.2))
        away_lambda = max(0.5, away_form.get("goals_per_match", 1.0))
        
        # Score distribution
        distribution = self.calculate_score_distribution(home_lambda, away_lambda)
        
        # Top scorelines
        top_scorelines = self.get_top_n_scorelines(distribution, n=5)
        
        # 1X2 probabilities
        probabilities_1x2 = self.calculate_1x2_probabilities(distribution)
        
        # Confidence calculation
        max_prob = max(probabilities_1x2.values())
        confidence = "Magas" if max_prob > 0.5 else "Közepes" if max_prob > 0.4 else "Alacsony"
        
        # Explanation
        explanation = self._generate_explanation(analysis, home_lambda, away_lambda, probabilities_1x2)
        
        return {
            "fixture_id": analysis.get("fixture_id"),
            "home_team": analysis.get("home_team"),
            "away_team": analysis.get("away_team"),
            "date": analysis.get("date"),
            "league": analysis.get("league"),
            "prediction": {
                "home_lambda": home_lambda,
                "away_lambda": away_lambda,
                "probabilities_1x2": probabilities_1x2,
                "top_scorelines": top_scorelines,
                "confidence": confidence,
                "explanation": explanation
            }
        }
    
    def _generate_explanation(self, analysis: dict, home_lambda: float, 
                            away_lambda: float, probs: dict) -> str:
        """Magyar nyelvű magyarázat generálás"""
        home_team = analysis.get("home_team", "Hazai")
        away_team = analysis.get("away_team", "Vendég")
        
        home_form_str = analysis.get("home_form", {}).get("form_string", "N/A")
        away_form_str = analysis.get("away_form", {}).get("form_string", "N/A")
        
        max_outcome = max(probs, key=probs.get)
        max_prob = probs[max_outcome]
        
        outcome_map = {
            "home_win": f"{home_team} győzelem",
            "draw": "Döntetlen",
            "away_win": f"{away_team} győzelem"
        }
        
        explanation = (
            f"A modell {outcome_map[max_outcome]} kimenetelt valószínűsíti ({max_prob:.1%} esély). "
            f"Várható gólok: {home_team} {home_lambda:.2f}, {away_team} {away_lambda:.2f}. "
            f"Forma: {home_team} ({home_form_str}), {away_team} ({away_form_str})."
        )
        
        return explanation


# ============= DATABASE =================
def get_db_engine(database_url: str):
    """SQLAlchemy engine létrehozás"""
    from sqlalchemy import create_engine
    return create_engine(database_url, echo=False)


def get_db_session(engine):
    """SQLAlchemy session létrehozás"""
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    return Session()


class PredictionModel:
    """SQLAlchemy model a predikciók tárolásához"""
    
    @staticmethod
    def create_tables(engine):
        """Táblák létrehozása"""
        from sqlalchemy import Column, Integer, String, Float, DateTime, Text, MetaData, Table
        
        metadata = MetaData()
        
        predictions_table = Table(
            'predictions',
            metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('fixture_id', Integer, nullable=False),
            Column('home_team', String(200), nullable=False),
            Column('away_team', String(200), nullable=False),
            Column('league', String(200)),
            Column('match_date', DateTime),
            Column('home_win_prob', Float),
            Column('draw_prob', Float),
            Column('away_win_prob', Float),
            Column('home_lambda', Float),
            Column('away_lambda', Float),
            Column('confidence', String(50)),
            Column('explanation', Text),
            Column('top_scorelines', Text),
            Column('created_at', DateTime, default=datetime.utcnow),
        )
        
        metadata.create_all(engine)
        logger.info("Database tables created successfully")
        return predictions_table
    
    @staticmethod
    def save_prediction(engine, prediction: dict):
        """Predikció mentése adatbázisba"""
        from sqlalchemy import text
        
        pred_data = prediction.get("prediction", {})
        probs = pred_data.get("probabilities_1x2", {})
        
        query = text("""
            INSERT INTO predictions 
            (fixture_id, home_team, away_team, league, match_date, 
             home_win_prob, draw_prob, away_win_prob, home_lambda, away_lambda,
             confidence, explanation, top_scorelines)
            VALUES 
            (:fixture_id, :home_team, :away_team, :league, :match_date,
             :home_win_prob, :draw_prob, :away_win_prob, :home_lambda, :away_lambda,
             :confidence, :explanation, :top_scorelines)
        """)
        
        with engine.connect() as conn:
            conn.execute(query, {
                "fixture_id": prediction.get("fixture_id"),
                "home_team": prediction.get("home_team"),
                "away_team": prediction.get("away_team"),
                "league": prediction.get("league"),
                "match_date": prediction.get("date"),
                "home_win_prob": probs.get("home_win"),
                "draw_prob": probs.get("draw"),
                "away_win_prob": probs.get("away_win"),
                "home_lambda": pred_data.get("home_lambda"),
                "away_lambda": pred_data.get("away_lambda"),
                "confidence": pred_data.get("confidence"),
                "explanation": pred_data.get("explanation"),
                "top_scorelines": json.dumps(pred_data.get("top_scorelines"))
            })
            conn.commit()


# ============= FASTAPI APP =================
def create_app(config: Config):
    """FastAPI alkalmazás létrehozása"""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    
    app = FastAPI(
        title="Betting Predictor API",
        description="Napi futball mérkőzés előrejelzések",
        version="1.0.0"
    )
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Betting Predictor API",
            "endpoints": ["/predictions", "/health"]
        }
    
    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}
    
    @app.get("/predictions")
    async def get_predictions():
        """Mai mérkőzések predikcióinak lekérése"""
        try:
            # Config és komponensek inicializálása
            async with DataCollector(config) as collector:
                # Mai fixtures lekérése
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                fixtures = await collector.get_fixtures(today)
                
                if not fixtures:
                    return JSONResponse(
                        content={"message": "Nincs mérkőzés ma", "predictions": []},
                        status_code=200
                    )
                
                # Analyzer és Predictor
                analyzer = Analyzer(collector)
                predictor = Predictor()
                
                predictions = []
                
                # Első néhány mérkőzés elemzése (demo célra)
                for fixture in fixtures[:10]:  # Korlátozás: első 10 mérkőzés
                    try:
                        analysis = await analyzer.analyze_fixture(fixture, fixtures)
                        prediction = predictor.predict_fixture(analysis)
                        predictions.append(prediction)
                    except Exception as e:
                        logger.error(f"Error analyzing fixture {fixture.get('fixture', {}).get('id')}: {e}")
                        continue
                
                return JSONResponse(
                    content={
                        "date": today,
                        "total_predictions": len(predictions),
                        "predictions": predictions
                    },
                    status_code=200
                )
                
        except Exception as e:
            logger.exception("Error generating predictions")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# ============= CLI ENTRY POINT =================
def main():
    """Parancssori belépési pont"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Betting Predictor System")
    parser.add_argument(
        "--mode",
        choices=["server", "predict", "init-db"],
        default="server",
        help="Működési mód: server (FastAPI), predict (egyszeri predikció), init-db (adatbázis inicializálás)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="FastAPI host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port (default: 8000)")
    parser.add_argument("--date", help="Dátum mérkőzésekhez (YYYY-MM-DD formátum)")
    
    args = parser.parse_args()
    
    # Config betöltése
    config = Config.from_env()
    
    if not config.api_football_key:
        logger.error("API_FOOTBALL_KEY környezeti változó nincs beállítva!")
        logger.error("Állítsd be a .env fájlban vagy exportáld: export API_FOOTBALL_KEY=your_key")
        sys.exit(1)
    
    if args.mode == "init-db":
        # Adatbázis inicializálás
        logger.info("Initializing database...")
        engine = get_db_engine(config.database_url)
        PredictionModel.create_tables(engine)
        logger.info("Database initialized successfully")
        
    elif args.mode == "predict":
        # Egyszeri predikció futtatás
        logger.info("Running prediction...")
        
        async def run_prediction():
            async with DataCollector(config) as collector:
                date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
                fixtures = await collector.get_fixtures(date)
                
                logger.info(f"Found {len(fixtures)} fixtures for {date}")
                
                if not fixtures:
                    logger.info("No fixtures found")
                    return
                
                analyzer = Analyzer(collector)
                predictor = Predictor()
                engine = get_db_engine(config.database_url)
                
                for fixture in fixtures[:10]:  # Első 10 mérkőzés
                    try:
                        analysis = await analyzer.analyze_fixture(fixture, fixtures)
                        prediction = predictor.predict_fixture(analysis)
                        
                        logger.info(f"\n{prediction['home_team']} vs {prediction['away_team']}")
                        logger.info(f"  Prediction: {prediction['prediction']['explanation']}")
                        logger.info(f"  Confidence: {prediction['prediction']['confidence']}")
                        
                        # Mentés adatbázisba
                        PredictionModel.save_prediction(engine, prediction)
                        
                    except Exception as e:
                        logger.error(f"Error processing fixture: {e}")
                        continue
        
        asyncio.run(run_prediction())
        
    else:  # server mode
        # FastAPI szerver indítás
        logger.info(f"Starting FastAPI server on {args.host}:{args.port}")
        logger.info("Use: uvicorn main:app --host 0.0.0.0 --port 8000")
        
        app = create_app(config)
        
        # Export app for uvicorn
        globals()['app'] = app
        
        # Uvicorn indítás
        try:
            import uvicorn
            uvicorn.run(app, host=args.host, port=args.port)
        except ImportError:
            logger.error("uvicorn not installed. Install with: pip install uvicorn")
            logger.info("Or run manually: uvicorn main:app --host 0.0.0.0 --port 8000")
            sys.exit(1)


if __name__ == "__main__":
    main()
