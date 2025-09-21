#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrált Sportfogadási Bot – SUPerset (API-Football + TippmixPro WAMP + Telegram + Kalibráció + Bayes + MC)

FRISSÍTÉS (C, D, E PONTOK BEÉPÍTVE) – 2025-09-10

ÚJ FUNKCIÓK / MÓDOSÍTÁSOK (a kérés szerinti C, D, E csomag):
C) Odds és edge bővítés:
   - Overround / margin számítás 1X2 és 2-way (BTTS, O/U 2.5) piacokra.
   - Normalizált (margin levont) implied valószínűségek mentése.
   - Több edge metrika: raw_edge (p*odds-1), margin_adj_edge (p_model - p_implied_norm), z_edge
     (Bernoulli variancia alapján standardizált különbség), rel_value (raw_edge / odds).
   - analysis.json kiegészítve: market_prob_details. Picks / ticket rationale bővítve margin/diff értékekkel.

D) Kalibráció dataset gondozás:
   - Kalibrációs history trimming (CALIBRATION_HISTORY_MAX, default 15000 minta kategóriánként).
   - Ensemble valószínűségek (ha vannak) eltárolása külön szekcióban (hist["ensemble"][cat]).
   - Reliability diagram generálás JSON formátumban (bin szerinti empirikus hit rate) ha GENERATE_RELIABILITY=1.
     (Paraméterezhető CALIBRATION_BIN_SIZE – default 0.05)
   - update_calibration_history_from_results most az ensemble_probs-et is gyűjti (ha van).
   - retrain_calibrators jelenleg továbbra is a base raw-ot használja (ensemble kalibráció opcionálisan később).

E) Bayes optimalizáció:
   - Minimum mintaszám feltételek: BAYES_MIN_MATCHES, BAYES_MIN_TEAM_MATCHES.
   - Lambda posterior átlagok clipping: BAYES_LAMBDA_MIN <= λ <= BAYES_LAMBDA_MAX.
   - Időlimit a Bayes futásra (BAYES_MAX_SECONDS) – külön threadben futtatva, timeout-ra skip.
   - Ha feltételek nem teljesülnek vagy timeout / hiba történik → Bayes disabled fallback.

Megjegyzés:
  A monolit fájl (betting.py) szerkezetét megőriztem, de a releváns részeknél jelöltem a
  # === (C) / (D) / (E) === kommenteket.

FIGYELEM:
  - A korábbi kódhoz képest minden eddigi funkció megmaradt.
  - A tokenek és fallback kulcsok továbbra is a felhasználó által megadott logika szerint működnek;
    éles környezetben ajánlott a fallback API kulcs végleges eltávolítása.

"""

from __future__ import annotations
import os, math, json, asyncio, aiohttp, logging, re, argparse, shutil, random, hashlib, unicodedata, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
from datetime import date, datetime, timezone, timedelta
from math import exp, factorial
from zoneinfo import ZoneInfo
from aiohttp import ClientTimeout
import concurrent.futures

# ================= LOGGING =================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)
logger = logging.getLogger("betting")

# ============= API-FOOTBALL CONF =================
API_HOST = "v3.football.api-sports.io"
API_BASE = os.getenv("API_BASE_URL", f"https://{API_HOST}")
_API_FALLBACK = "0467e77465788bd14dcd3524f9bd99df"
API_KEY = os.getenv("API_FOOTBALL_KEY") or _API_FALLBACK
if os.getenv("API_FOOTBALL_KEY") is None:
    logger.warning("API_FOOTBALL_KEY nincs ENV-ben – fallback kulcs (csak dev).")

# ============= ODDS-API CONF =================
ODDS_API_BASE = os.getenv("ODDS_API_BASE", "https://api.the-odds-api.com/v4")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
if not ODDS_API_KEY:
    logger.warning("ODDS_API_KEY nincs beállítva – OddsAPI funkciók korlátottak.")

# ============= TELEGRAM CONF =================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or "7854789524:AAGKoURk4w6ZMFY5HIUd4bb70dnJm4Gepto"
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID") or "8042762229"
RATIONALE_MAXLEN = int(os.getenv("RATIONALE_MAXLEN", "420"))

# ============= GENERAL ENV =================
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "15"))
PARALLEL_CONNECTIONS = int(os.getenv("PARALLEL_CONNECTIONS", "6"))
SAVE_RAW = os.getenv("SAVE_RAW", "1") == "1"
DATA_ROOT = Path(os.getenv("DATA_ROOT", ".")).resolve()

LEAGUE_WHITELIST = {int(x) for x in os.getenv("LEAGUE_WHITELIST","").split(",") if x.strip().isdigit()}
LEAGUE_BLACKLIST = {int(x) for x in os.getenv("LEAGUE_BLACKLIST","").split(",") if x.strip().isdigit()}

MAJOR_LEAGUE_IDS = {int(x) for x in os.getenv("MAJOR_LEAGUE_IDS","").split(",") if x.strip().isdigit()}
MAJOR_LEAGUE_RATING_BOOST = float(os.getenv("MAJOR_LEAGUE_RATING_BOOST", "0.05"))
MAJOR_LEAGUE_EDGE_BONUS = float(os.getenv("MAJOR_LEAGUE_EDGE_BONUS", "0.01"))

HOME_ADV = float(os.getenv("HOME_ADV", "0.20"))
DRAW_BASE = float(os.getenv("DRAW_BASE", "0.25"))
KELLY_FRACTION_LIMIT = float(os.getenv("KELLY_FRACTION_LIMIT", "0.25"))
MISSING_TOP_SCORER_PCT = float(os.getenv("MISSING_TOP_SCORER_PCT", "0.10"))

FORM_WEIGHT = float(os.getenv("FORM_WEIGHT", "0.30"))
ATK_WEIGHT = float(os.getenv("ATK_WEIGHT", "0.35"))
DEF_WEIGHT = float(os.getenv("DEF_WEIGHT", "0.20"))

FETCH_DAYS_AHEAD = int(os.getenv("FETCH_DAYS_AHEAD", "3"))
BANKROLL_DAILY = float(os.getenv("BANKROLL_DAILY", "1000"))
MAX_SINGLE_STAKE_FRACTION = float(os.getenv("MAX_SINGLE_STAKE_FRACTION", "0.05"))
KELLY_STAKE_MULTIPLIER = float(os.getenv("KELLY_STAKE_MULTIPLIER", "0.5"))

LOCAL_TZ = os.getenv("LOCAL_TZ", "Europe/Budapest")

TICKET_ONLY_TODAY = os.getenv("TICKET_ONLY_TODAY", "1") == "1"

EDGE_CAP_1X2 = float(os.getenv("EDGE_CAP_1X2", "1.0"))
EDGE_CAP_BTTs_OU = float(os.getenv("EDGE_CAP_BTTs_OU", "1.0"))
TICKET_MAX_ODDS_1X2 = float(os.getenv("TICKET_MAX_ODDS_1X2", "4.0"))
TICKET_MAX_ODDS_2WAY = float(os.getenv("TICKET_MAX_ODDS_2WAY", "4.0"))
TICKET_DIFF_TOL_1X2 = float(os.getenv("TICKET_DIFF_TOL_1X2", "0.15"))
TICKET_DIFF_TOL_2WAY = float(os.getenv("TICKET_DIFF_TOL_2WAY", "0.22"))
TICKET_FALLBACK_ENABLE = os.getenv("TICKET_FALLBACK_ENABLE", "1") == "1"
TICKET_FALLBACK_DIFF_TOL_2WAY = float(os.getenv("TICKET_FALLBACK_DIFF_TOL_2WAY", "0.35"))
TICKET_BTTs_LAST_RESORT = os.getenv("TICKET_BTTS_LAST_RESORT", "1") == "1"

MAX_ODDS_1X2_PICKS = float(os.getenv("MAX_ODDS_1X2_PICKS", "6.0"))
PICK_EDGE_CAP_1X2 = float(os.getenv("PICK_EDGE_CAP_1X2", "1.0"))

ENABLE_ENHANCED_MODELING = os.getenv("ENABLE_ENHANCED_MODELING", "1") == "1"
ENABLE_CALIBRATION = os.getenv("ENABLE_CALIBRATION", "1") == "1" and ENABLE_ENHANCED_MODELING
ENABLE_BAYES = os.getenv("ENABLE_BAYES", "1") == "1" and ENABLE_ENHANCED_MODELING
ENABLE_MC = os.getenv("ENABLE_MC", "1") == "1" and ENABLE_ENHANCED_MODELING
ENSEMBLE_WEIGHTS_RAW = os.getenv("ENSEMBLE_WEIGHTS", '{"base":0.55,"cal":0.20,"bayes":0.10,"mc":0.15}')
MC_SIMS = int(os.getenv("MC_SIMS", "12000"))
CALIBRATION_HISTORY_FILE = os.getenv("CALIBRATION_HISTORY_FILE", "calibration_history.json")
BAYES_HISTORY_DAYS = int(os.getenv("BAYES_HISTORY_DAYS", "60"))
DAILY_REPORT_DIR = os.getenv("DAILY_REPORT_DIR", "daily_reports")

ENABLE_DYNAMIC_LEAGUES = os.getenv("ENABLE_DYNAMIC_LEAGUES", "1") == "1"
TOP_MODE = os.getenv("TOP_MODE", "all").lower()
TIER_CONFIG_PATH = Path(os.getenv("TIER_CONFIG_PATH", "config/leagues_tiers.yaml"))
LEAGUE_CLASSIFY_CACHE = Path(os.getenv("LEAGUE_CLASSIFY_CACHE", "data/leagues_classified.json"))
FORCE_RECLASSIFY = os.getenv("FORCE_RECLASSIFY", "0") == "1"

EXTRA_TIER1_IDS = {int(x) for x in os.getenv("EXTRA_TIER1_IDS","").split(",") if x.strip().isdigit()}
EXTRA_TIER1B_IDS = {int(x) for x in os.getenv("EXTRA_TIER1B_IDS","").split(",") if x.strip().isdigit()}
EXTRA_CUPS_ELITE_IDS = {int(x) for x in os.getenv("EXTRA_CUPS_ELITE_IDS","").split(",") if x.strip().isdigit()}
EXTRA_NT_MAJOR_IDS = {int(x) for x in os.getenv("EXTRA_NT_MAJOR_IDS","").split(",") if x.strip().isdigit()}

PUBLISH_MIN_EDGE_TOP = float(os.getenv("PUBLISH_MIN_EDGE_TOP", "0.05"))
PUBLISH_MIN_EDGE_OTHER = float(os.getenv("PUBLISH_MIN_EDGE_OTHER", "0.08"))

# ============ Új betting funkciók environment változók ============
# Value betting minimum edge threshold
MIN_EDGE_THRESHOLD = float(os.getenv("MIN_EDGE_THRESHOLD", "0.03"))  # 3%

# Fix stake amount
FIX_STAKE_AMOUNT = float(os.getenv("FIX_STAKE_AMOUNT", "10000"))  # 10,000 Ft

# Margin filtering thresholds
MAX_MARGIN_1X2 = float(os.getenv("MAX_MARGIN_1X2", "1.10"))
MAX_MARGIN_2WAY = float(os.getenv("MAX_MARGIN_2WAY", "1.10"))

# Liga szűrés - csak TIER1 és TIER1B ligák
ENABLE_TIER_FILTERING = os.getenv("ENABLE_TIER_FILTERING", "1") == "1"

# Döntetlen kizárása az 1X2 piacokról
EXCLUDE_DRAW_1X2 = os.getenv("EXCLUDE_DRAW_1X2", "1") == "1"

# ============ TippmixPro integráció flag-ek ============
USE_TIPPMIX = os.getenv("USE_TIPPMIX", "1") == "1"
TIPPMIX_DAYS_AHEAD = int(os.getenv("TIPPMIX_DAYS_AHEAD", str(FETCH_DAYS_AHEAD)))
TIPPMIX_MARKET_GROUP = os.getenv("TIPPMIX_MARKET_GROUP", "NEPSZERU")
TIPPMIX_SIMILARITY_THRESHOLD = float(os.getenv("TIPPMIX_SIMILARITY_THRESHOLD", "0.78"))
TIPPMIX_TIME_TOLERANCE_MIN = int(os.getenv("TIPPMIX_TIME_TOLERANCE_MIN", "15"))

USE_ODDS_WATCH = os.getenv("USE_ODDS_WATCH", "1") == "1"
WATCH_INTERVAL_SEC = int(os.getenv("ODDS_WATCH_INTERVAL", "120"))

# === (D) Kalibrációs trimming paraméterek ===
CALIBRATION_HISTORY_MAX = int(os.getenv("CALIBRATION_HISTORY_MAX", "15000"))
CALIBRATION_BIN_SIZE = float(os.getenv("CALIBRATION_BIN_SIZE", "0.05"))
GENERATE_RELIABILITY = os.getenv("GENERATE_RELIABILITY", "1") == "1"

# === (E) Bayes optimalizáció paraméterek ===
BAYES_MIN_MATCHES = int(os.getenv("BAYES_MIN_MATCHES", "200"))
BAYES_MIN_TEAM_MATCHES = int(os.getenv("BAYES_MIN_TEAM_MATCHES", "15"))
BAYES_LAMBDA_MIN = float(os.getenv("BAYES_LAMBDA_MIN", "0.2"))
BAYES_LAMBDA_MAX = float(os.getenv("BAYES_LAMBDA_MAX", "3.5"))
BAYES_MAX_SECONDS = int(os.getenv("BAYES_MAX_SECONDS", "25"))  # sampling time limit

# ============ Opcionális csomagok ============
try:
    import numpy as np
except ImportError:
    np = None
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False
try:
    import pymc as pm
    HAVE_PYMC = True
except ImportError:
    HAVE_PYMC = False
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

def parse_ensemble_weights() -> dict:
    try:
        return json.loads(ENSEMBLE_WEIGHTS_RAW)
    except Exception:
        return {"base":0.55,"cal":0.20,"bayes":0.10,"mc":0.15}

ENSEMBLE_WEIGHTS = parse_ensemble_weights()

# ============ Állapot ============
STATE_FILE = DATA_ROOT / "risk_state.json"
RUNTIME_STATE = {
    "date": date.today().isoformat(),
    "bankroll_start": BANKROLL_DAILY,
    "bankroll_current": BANKROLL_DAILY,
    "picks": [],
    "last_run": None
}
try:
    GLOBAL_RUNTIME
except NameError:
    GLOBAL_RUNTIME = {
        "fixture_limit": 0,
        "fetch_days_ahead": FETCH_DAYS_AHEAD,
        "last_summary": None,
        "last_picks_file": None,
        "tippmix_mapping": {},
        "tippmix_odds_cache": {}
    }

# =========================================================
# TippmixPro WAMP kliens és odds extractor
# =========================================================
WAMP_WELCOME=2
WAMP_CALL=48
WAMP_RESULT=50
WAMP_ERROR=8
HELLO_FRAME=[1,"www.tippmixpro.hu",{
 "agent":"Wampy.js v6.2.2",
 "roles":{
  "publisher":{"features":{"subscriber_blackwhite_listing":True,"publisher_exclusion":True,"publisher_identification":True}},
  "subscriber":{"features":{"pattern_based_subscription":True,"publication_trustlevels":True}},
  "caller":{"features":{"caller_identification":True,"progressive_call_results":True,"call_canceling":True,"call_timeout":True}},
  "callee":{"features":{"caller_identification":True,"call_trustlevels":True,"pattern_based_registration":True,"shared_registration":True}}
 },
 "authmethods":["wampcra"],
 "authid":"webapi-wampy"
}]

@dataclass
class StandardOdds:
    one_x_two: Optional[Dict[str,float]]
    btts: Optional[Dict[str,float]]
    ou25: Optional[Dict[str,float]]
    meta: Dict[str,Any]

class TippmixProWampClient:
    def __init__(self, cluster=2901, lang="hu", sport_id="1", verbose=False):
        self.cluster=cluster
        self.lang=lang
        self.sport_id=str(sport_id)
        self.verbose=verbose
        self._ws=None
        self._rid=1000
    async def __aenter__(self):
        import websockets
        self._ws=await websockets.connect("wss://sportsapi.tippmixpro.hu/v2", subprotocols=["wamp.2.json"])
        await self._ws.send(json.dumps(HELLO_FRAME))
        while True:
            raw=await self._ws.recv()
            try: msg=json.loads(raw)
            except: continue
            if isinstance(msg,list) and msg and msg[0]==WAMP_WELCOME:
                if self.verbose: logger.info("[TIPP] WELCOME")
                break
        return self
    async def __aexit__(self, *exc):
        if self._ws:
            await self._ws.close()
    async def _call(self, proc: str, kwargs: dict, expect_initial_dump=False, timeout=8) -> dict:
        rid=self._rid; self._rid+=1
        frame=[WAMP_CALL, rid, {}, proc, [], kwargs]
        await self._ws.send(json.dumps(frame))
        t0=time.time()
        while time.time()-t0<timeout:
            try:
                raw=await asyncio.wait_for(self._ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                break
            try: msg=json.loads(raw)
            except: continue
            if not isinstance(msg,list): continue
            if msg[0]==WAMP_ERROR and len(msg)>2 and msg[2]==rid:
                return {"error": True, "frame": msg}
            if msg[0]==WAMP_RESULT and msg[1]==rid:
                payload=msg[4] if len(msg)>4 else {}
                if expect_initial_dump:
                    if payload.get("messageType")=="INITIAL_DUMP":
                        return payload
                else:
                    return payload
        return {"timeout": True}
    async def get_matches_for_venue_tournament(self, venue_id: str, tournament_id: str) -> List[dict]:
        payload=await self._call("/sports#matches", {
            "lang": self.lang,
            "sportId": self.sport_id,
            "venueId": venue_id,
            "tournamentId": tournament_id
        })
        return payload.get("records",[])
    async def initial_dump_topic(self, topic: str) -> List[dict]:
        payload=await self._call("/sports#initialDump", {"topic": topic}, expect_initial_dump=True)
        return payload.get("records",[])
    async def fetch_match_markets_group(self, match_id: str, group_key="NEPSZERU") -> List[dict]:
        topics=[
            f"/sports/{self.cluster}/{self.lang}/match/{match_id}",
            f"/sports/{self.cluster}/{self.lang}/event/{match_id}/market-groups",
            f"/sports/{self.cluster}/{self.lang}/{match_id}/match-odds/market-group/{group_key}"
        ]
        out=[]
        for tp in topics:
            recs=await self.initial_dump_topic(tp)
            out.extend([r for r in recs if isinstance(r,dict)])
        return out

class TippmixOddsExtractor:
    ONE_X_TWO_BTIDS = {1, 1001, 5001}
    BTTS_KEYWORDS = ("Mindkét", "BTTS", "Mindkét csapat szerez gólt", "Goal/No Goal")
    def extract(self, records: List[dict]) -> StandardOdds:
        markets={}; outcomes={}; offers_by_outcome={}
        for r in records:
            tp=r.get("_type")
            if tp=="MARKET": markets[r.get("id")]=r
            elif tp=="OUTCOME": outcomes[r.get("id")]=r
            elif tp=="BETTING_OFFER":
                oid=r.get("outcomeId")
                if oid: offers_by_outcome.setdefault(oid,[]).append(r)
        def best_offer(oid):
            off=offers_by_outcome.get(oid,[])
            return max(off, key=lambda x: x.get("lastChangedTime",0)) if off else None
        one_x_two=None; btts=None; ou25=None
        for mk_id,mk in markets.items():
            btid=mk.get("bettingTypeId")
            name=(mk.get("name") or "").lower()
            ocs=[oc for oc in outcomes.values() if oc.get("marketId")==mk_id]
            if len(ocs)==3 and (btid in self.ONE_X_TWO_BTIDS or "1x2" in name or "eredmény" in name):
                tmp={}
                for oc in ocs:
                    on=(oc.get("translatedName") or oc.get("name") or "").strip().lower()
                    code=None
                    if on in ("1","hazai","home"): code="HOME"
                    elif on in ("x","döntetlen","draw"): code="DRAW"
                    elif on in ("2","vendég","away"): code="AWAY"
                    else:
                        if "vend" in on: code="AWAY"
                        elif "hazai" in on: code="HOME"
                        elif "öntetlen" in on: code="DRAW"
                    if code:
                        bo=best_offer(oc.get("id"))
                        if bo and bo.get("odds"):
                            try: tmp[code]=float(bo["odds"])
                            except: pass
                if len(tmp)==3:
                    one_x_two=tmp; break
        for mk_id,mk in markets.items():
            mk_name=(mk.get("name") or "")
            if not any(kw.lower() in mk_name.lower() for kw in self.BTTS_KEYWORDS):
                continue
            ocs=[oc for oc in outcomes.values() if oc.get("marketId")==mk_id]
            if len(ocs)!=2: continue
            tmp={}
            for oc in ocs:
                on=(oc.get("translatedName") or oc.get("name") or "").lower()
                code=None
                if "igen" in on or on in ("yes","goal"): code="YES"
                elif "nem" in on or on in ("no","no goal"): code="NO"
                if code:
                    bo=best_offer(oc.get("id"))
                    if bo and bo.get("odds"):
                        try: tmp[code]=float(bo["odds"])
                        except: pass
            if len(tmp)==2:
                btts=tmp; break
        for mk_id,mk in markets.items():
            mk_name=(mk.get("name") or "")
            param=mk.get("paramFloat1")
            pass_flag=False
            if param is not None:
                try:
                    if abs(float(param)-2.5)<1e-6: pass_flag=True
                except: pass
            else:
                pass_flag="2.5" in mk_name
            if not pass_flag: continue
            ocs=[oc for oc in outcomes.values() if oc.get("marketId")==mk_id]
            if len(ocs)!=2: continue
            tmp={}
            for oc in ocs:
                on=(oc.get("translatedName") or oc.get("name") or "").lower()
                code=None
                if "over" in on or "több" in on: code="OVER"
                elif "under" in on or "kevesebb" in on: code="UNDER"
                if code:
                    bo=best_offer(oc.get("id"))
                    if bo and bo.get("odds"):
                        try: tmp[code]=float(bo["odds"])
                        except: pass
            if len(tmp)==2:
                ou25=tmp; break
        return StandardOdds(one_x_two=one_x_two, btts=btts, ou25=ou25,
                            meta={"market_count": len(markets), "outcome_count": len(outcomes)})

# Odds watcher
async def watch_tippmix_odds(stop_event: asyncio.Event):
    if not USE_TIPPMIX:
        logger.info("[WATCH] Tippmix inaktív – watcher nem indul.")
        return
    extractor = TippmixOddsExtractor()
    logger.info("[WATCH] Odds watcher indul (interval=%ds)...", WATCH_INTERVAL_SEC)
    while not stop_event.is_set():
        mapping = GLOBAL_RUNTIME.get("tippmix_mapping") or {}
        if not mapping:
            logger.info("[WATCH] Nincs tippmix_mapping – várakozás...")
        else:
            try:
                async with TippmixProWampClient(verbose=False) as cli:
                    new_cache={}
                    for api_fid, tip_mid in mapping.items():
                        recs = await cli.fetch_match_markets_group(tip_mid, group_key=TIPPMIX_MARKET_GROUP)
                        std = extractor.extract(recs)
                        new_cache[api_fid] = std
                        old = GLOBAL_RUNTIME.get("tippmix_odds_cache", {}).get(api_fid)
                        if old and std:
                            if old.one_x_two and std.one_x_two:
                                for k in ("HOME","DRAW","AWAY"):
                                    ov = old.one_x_two.get(k); nv = std.one_x_two.get(k)
                                    if ov and nv and ov != nv:
                                        logger.info("[WATCH][1X2] FI %s %s odds: %.2f -> %.2f", api_fid, k, ov, nv)
                            if old.btts and std.btts:
                                for k in ("YES","NO"):
                                    ov = old.btts.get(k); nv = std.btts.get(k)
                                    if ov and nv and ov != nv:
                                        logger.info("[WATCH][BTTS] FI %s %s odds: %.2f -> %.2f", api_fid, k, ov, nv)
                            if old.ou25 and std.ou25:
                                for k in ("OVER","UNDER"):
                                    ov = old.ou25.get(k); nv = std.ou25.get(k)
                                    if ov and nv and ov != nv:
                                        logger.info("[WATCH][OU2.5] FI %s %s odds: %.2f -> %.2f", api_fid, k, ov, nv)
                        await asyncio.sleep(0.05)
                GLOBAL_RUNTIME["tippmix_odds_cache"] = new_cache
            except Exception:
                logger.exception("[WATCH] Hiba odds frissítés közben.")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=WATCH_INTERVAL_SEC)
        except asyncio.TimeoutError:
            continue
    logger.info("[WATCH] Odds watcher leállt.")

_ws_re = re.compile(r"\s+")
def normalize_team_name(name: str) -> str:
    if not name: return ""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower()
    n = re.sub(r"\b(fc|sc|afc|cf)\b","", n)
    n = re.sub(r"\b(u19|u20|u21|u23)\b","", n)
    n = re.sub(r"[^a-z0-9]+"," ", n)
    n = _ws_re.sub(" ", n).strip()
    return n

def similarity(a: str, b: str) -> float:
    try:
        from rapidfuzz import fuzz
        return fuzz.token_set_ratio(a,b) / 100.0
    except ImportError:
        from difflib import SequenceMatcher
        return SequenceMatcher(None,a,b).ratio()

# =========================================================
# Data model
# =========================================================
@dataclass
class TeamRatingInput:
    team_id: int
    season: int
    form_string: str
    goals_for_avg: float
    goals_against_avg: float
    top_scorer_ids: List[int]
    squad_player_ids: set[int]
    league_id: int

@dataclass
class TeamComputedRating:
    team_id: int
    raw_attack: float
    raw_defense: float
    form_score: float
    missing_top_scorers: int
    adjusted_attack: float
    combined_rating: float

@dataclass
class FixtureContext:
    fixture_id: int
    league_id: int
    league_name: str
    season: int
    home_team_id: int
    away_team_id: int
    timestamp: int
    kickoff_utc: datetime
    ratings_home: TeamComputedRating
    ratings_away: TeamComputedRating
    probs: dict
    odds: dict | None
    edge: dict
    kelly: dict
    fair_odds: dict

# =========================================================
# TOP Liga Management (tournaments.json alapú)
# =========================================================
def load_top_leagues_from_tournaments() -> Dict[str, List[str]]:
    """
    Betölti a TOP ligákat a tournaments.json fájlból az 'is_top_level' mező alapján.
    Visszaadja az összes név variációt (name, template_name, translated_name, short_translated_name).
    """
    tournaments_path = Path("tournaments.json")
    top_league_names = set()
    
    if not tournaments_path.exists():
        logger.warning("tournaments.json hiányzik – TOP liga azonosítás korlátozott.")
        return {"names": [], "count": 0}
    
    try:
        tdata = json.loads(tournaments_path.read_text(encoding="utf-8"))
        if isinstance(tdata, list):
            tournaments = tdata
        else:
            tournaments = tdata.get("tournaments", [])
        
        for tournament in tournaments:
            if tournament.get("is_top_level", False):
                # Összes név variáció hozzáadása
                name_fields = ["name", "template_name", "translated_name", "short_translated_name"]
                for field in name_fields:
                    name = tournament.get(field)
                    if name and isinstance(name, str) and name.strip():
                        top_league_names.add(name.strip())
        
        result = {
            "names": sorted(list(top_league_names)),
            "count": len(top_league_names)
        }
        
        logger.info("TOP ligák betöltve tournaments.json-ból: %d név variáció azonosítva", len(top_league_names))
        for name in sorted(top_league_names):
            logger.debug("TOP liga név: %s", name)
            
        return result
        
    except Exception as e:
        logger.exception("tournaments.json feldolgozási hiba: %s", e)
        return {"names": [], "count": 0}

def is_top_league_by_name(league_name: str, top_league_names: List[str]) -> bool:
    """
    Ellenőrzi, hogy a liga neve megegyezik-e valamelyik TOP liga névvel (kis-nagybetű függetlenül).
    """
    if not league_name or not top_league_names:
        return False
    
    league_name_lower = league_name.lower().strip()
    
    for top_name in top_league_names:
        if top_name.lower().strip() == league_name_lower:
            return True
        # Részleges egyezés is elfogadható (pl. "Champions League" vs "UEFA Champions League")
        if len(top_name) > 5 and top_name.lower() in league_name_lower:
            return True
        if len(league_name) > 5 and league_name_lower in top_name.lower():
            return True
    
    return False

# =========================================================
# LeagueTierManager (tournaments.json-nal kiterjesztve)
# =========================================================
class LeagueTierManager:
    DEFAULT_TIER_CFG = {
        "generated": True,
        "tiers": {
            "TIER1":  [39, 140, 135, 78, 61, 2, 3, 848],
            "TIER1B": [88, 94, 203, 179, 253, 71, 128],
            "CUPS_ELITE": [45, 81, 137, 66, 556],
            "NT_MAJOR": [1, 4, 762]
        }
    }
    EXCLUDE_KW = [
        r"\bWomen\b", r"\bU19\b", r"\bU20\b", r"\bU21\b",
        r"Friend", r"Club Friend", r"Reserve", r"Primavera",
        r"Youth", r"Test"
    ]
    def __init__(self, tier_config_path: Path, classify_cache: Path,
                 enable_dynamic: bool, force_reclassify: bool):
        self.tier_config_path=tier_config_path
        self.classify_cache=classify_cache
        self.enable_dynamic=enable_dynamic
        self.force_reclassify=force_reclassify
        self.tier_cfg={}
        self.classified={}
        self._top_ids_set=set()
        # ÚJ: tournaments.json alapú TOP liga nevek
        self.top_league_data = {"names": [], "count": 0}
        self._load_or_init_config()
        self._load_or_empty_classified()
        self._load_top_leagues_from_tournaments()
        self._compute_top_ids()
    def _load_or_init_config(self):
        if self.tier_config_path.exists():
            try:
                self.tier_cfg=json.loads(self.tier_config_path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Hibás tier config – fallback.")
                self.tier_cfg=self.DEFAULT_TIER_CFG
        else:
            self.tier_cfg=self.DEFAULT_TIER_CFG
            self._save_tier_cfg()
        if EXTRA_TIER1_IDS:
            self.tier_cfg["tiers"].setdefault("TIER1",[]).extend(list(EXTRA_TIER1_IDS))
        if EXTRA_TIER1B_IDS:
            self.tier_cfg["tiers"].setdefault("TIER1B",[]).extend(list(EXTRA_TIER1B_IDS))
        if EXTRA_CUPS_ELITE_IDS:
            self.tier_cfg["tiers"].setdefault("CUPS_ELITE",[]).extend(list(EXTRA_CUPS_ELITE_IDS))
        if EXTRA_NT_MAJOR_IDS:
            self.tier_cfg["tiers"].setdefault("NT_MAJOR",[]).extend(list(EXTRA_NT_MAJOR_IDS))
        for k,v in self.tier_cfg["tiers"].items():
            self.tier_cfg["tiers"][k]=sorted(set(v))
    def _save_tier_cfg(self):
        self.tier_config_path.parent.mkdir(parents=True, exist_ok=True)
        self.tier_config_path.write_text(json.dumps(self.tier_cfg, indent=2), encoding="utf-8")
    def _load_or_empty_classified(self):
        if self.classify_cache.exists() and not self.force_reclassify:
            try:
                self.classified=json.loads(self.classify_cache.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Liga classify cache hiba – üres.")
                self.classified={}
        else:
            self.classified={}
    def _save_classified(self):
        self.classify_cache.parent.mkdir(parents=True, exist_ok=True)
        self.classify_cache.write_text(json.dumps(self.classified, indent=2), encoding="utf-8")
    def _load_top_leagues_from_tournaments(self):
        """Betölti a TOP ligákat a tournaments.json fájlból."""
        self.top_league_data = load_top_leagues_from_tournaments()
        
    def _compute_top_ids(self):
        # Megtartjuk a régi tier-alapú logikát fallback-ként
        tiers=self.tier_cfg.get("tiers",{})
        top=set()
        for grp in ("TIER1","TIER1B","CUPS_ELITE","NT_MAJOR"):
            for lid in tiers.get(grp, []):
                top.add(lid)
        self._top_ids_set=top
        
    def is_top(self, league_id: int)->bool:
        """
        Meghatározza, hogy egy liga TOP-e.
        Elsősorban tournaments.json név alapú, másodsorban ID alapú tier logika.
        """
        # Először próbálkozás név alapon
        league_name = self.get_league_name(league_id)
        if league_name and self.top_league_data["names"]:
            is_top_by_name = is_top_league_by_name(league_name, self.top_league_data["names"])
            if is_top_by_name:
                return True
        
        # Fallback: régi ID-alapú logika
        return league_id in self._top_ids_set
    
    def get_league_name(self, league_id: int) -> str:
        """Lekéri a liga nevét az ID alapján a classified adatokból."""
        meta = self.classified.get(str(league_id))
        if meta:
            return meta.get("name", "")
        return ""
    
    def is_top_with_reason(self, league_id: int, league_name: str = None) -> Tuple[bool, str]:
        """
        Meghatározza, hogy egy liga TOP-e, és megindokolja a döntést.
        Visszaadja: (is_top: bool, reason: str)
        """
        if not league_name:
            league_name = self.get_league_name(league_id)
        
        # Először név alapú ellenőrzés
        if league_name and self.top_league_data["names"]:
            is_top_by_name = is_top_league_by_name(league_name, self.top_league_data["names"])
            if is_top_by_name:
                return True, f"TOP liga (név alapján): {league_name}"
        
        # Fallback: ID-alapú tier logika
        if league_id in self._top_ids_set:
            tier = self.tier_of(league_id)
            return True, f"TOP liga (tier alapján): {league_name} (tier: {tier})"
        
        return False, f"Nem TOP liga: {league_name or f'ID={league_id}'}"
    def tier_of(self, league_id: int)->str:
        for k,ids in self.tier_cfg.get("tiers",{}).items():
            if league_id in ids: return k
        meta=self.classified.get(str(league_id))
        if meta: return meta.get("tier","OTHER")
        return "OTHER"
    def _is_excluded_name(self, name: str)->bool:
        for pat in self.EXCLUDE_KW:
            if re.search(pat, name, re.IGNORECASE): return True
        return False
    async def fetch_and_classify(self):
        if not self.enable_dynamic:
            logger.info("Dinamikus ligák tiltva.")
            return
        logger.info("Ligák lekérése...")
        try:
            # API-Football adatok lekérése
            async with aiohttp.ClientSession(timeout=ClientTimeout(total=REQUEST_TIMEOUT)) as s:
                headers={"x-apisports-key": API_KEY,"Accept":"application/json"}
                url=API_BASE.rstrip("/")+"/leagues"
                async with s.get(url, headers=headers) as resp:
                    js=await resp.json(content_type=None)
            resp=js.get("response") or []
            
            # OddsAPI adatok lekérése (új funkció)
            odds_api_leagues = await fetch_leagues_from_odds_api()
            
            tiers_cfg_ids={k:set(v) for k,v in self.tier_cfg.get("tiers",{}).items()}
            classified={}
            for entry in resp:
                league=entry.get("league",{}) or {}
                country=entry.get("country",{}) or {}
                lid=league.get("id"); name=league.get("name","")
                if not lid: continue
                tier=None
                
                # Először próbálja az OddsAPI adatokból
                for odds_league_name, odds_data in odds_api_leagues.items():
                    if name.lower() in odds_league_name.lower() or odds_league_name.lower() in name.lower():
                        tier = odds_data["tier"]
                        break
                
                # Ha nem találja, használja a konfigurációt
                if tier is None:
                    for grp,ids in tiers_cfg_ids.items():
                        if lid in ids: tier=grp; break
                
                if tier is None and self._is_excluded_name(name): tier="EXCLUDE"
                if tier is None: tier="OTHER"
                seasons=entry.get("seasons") or []
                latest_year=None
                if seasons:
                    latest_year=sorted(seasons,key=lambda s:s.get("year",0))[-1].get("year")
                classified[str(lid)]={"league_id":lid,"name":name,"country":country.get("name"),
                                      "tier":tier,"latest_season":latest_year}
            self.classified=classified
            self._save_classified()
            self._compute_top_ids()
            logger.info("Liga klasszifikáció kész: %d, OddsAPI mapping: %d", len(classified), len(odds_api_leagues))
        except Exception:
            logger.exception("Liga fetch/classify hiba.")
    def summarize_tiers(self)->Dict[str,int]:
        counts={}
        for meta in self.classified.values():
            t=meta.get("tier","OTHER")
            counts[t]=counts.get(t,0)+1
        return counts
    def search_leagues(self, pattern: str, limit=25)->List[dict]:
        pat=pattern.lower()
        out=[]
        for meta in self.classified.values():
            if pat in (meta.get("name","") or "").lower() or pat in (meta.get("country","") or "").lower():
                out.append(meta)
            if len(out)>=limit: break
        return out

LEAGUE_MANAGER = LeagueTierManager(
    tier_config_path=TIER_CONFIG_PATH,
    classify_cache=LEAGUE_CLASSIFY_CACHE,
    enable_dynamic=ENABLE_DYNAMIC_LEAGUES,
    force_reclassify=FORCE_RECLASSIFY
)

# =========================================================
# Model segédek
# =========================================================
def extract_form_score(form_string: str) -> float:
    if not form_string: return 0.5
    pts=0; tot=0
    for ch in form_string:
        if ch in ("W","L","D"):
            tot+=1
            if ch=="W": pts+=3
            elif ch=="D": pts+=1
    if tot==0: return 0.5
    return pts/(3*tot)

def parse_goals_avg(team_stats_json: dict) -> tuple[float,float,str]:
    try: gf=float(team_stats_json["goals"]["for"]["average"]["total"])
    except: gf=1.0
    try: ga=float(team_stats_json["goals"]["against"]["average"]["total"])
    except: ga=1.0
    form=team_stats_json.get("form") or ""
    return gf, ga, form

def collect_squad_player_ids(squad_json: dict) -> set[int]:
    if not squad_json: return set()
    resp=squad_json.get("response") or []
    if not resp: return set()
    players=resp[0].get("players", [])
    return {p.get("id") for p in players if p.get("id")}

def collect_top_scorers_ids(topscorers_json: dict, team_id: int, limit: int = 5) -> List[int]:
    if not topscorers_json: return []
    resp=topscorers_json.get("response") or []
    ids=[]
    for entry in resp:
        player=entry.get("player")
        stats=entry.get("statistics") or []
        if not player or not stats: continue
        st_team=stats[0].get("team", {}).get("id")
        if st_team==team_id:
            ids.append(player.get("id"))
        if len(ids)>=limit: break
    return ids

def compute_team_rating(inp: TeamRatingInput) -> TeamComputedRating:
    form_score=extract_form_score(inp.form_string)
    attack_raw=inp.goals_for_avg
    defense_raw=1.0/max(inp.goals_against_avg,0.05)
    missing=sum(1 for pid in inp.top_scorer_ids if pid not in inp.squad_player_ids)
    attack_adj=attack_raw*(1 - MISSING_TOP_SCORER_PCT*missing)
    base=(attack_adj*ATK_WEIGHT)+(defense_raw*DEF_WEIGHT)+(form_score*FORM_WEIGHT)
    if inp.league_id in MAJOR_LEAGUE_IDS:
        base*=(1 + MAJOR_LEAGUE_RATING_BOOST)
    return TeamComputedRating(
        team_id=inp.team_id,
        raw_attack=attack_raw,
        raw_defense=defense_raw,
        form_score=form_score,
        missing_top_scorers=missing,
        adjusted_attack=attack_adj,
        combined_rating=base
    )

def logistic_probabilities(r_home: float, r_away: float) -> tuple[float,float,float]:
    diff=(r_home - r_away) + HOME_ADV
    p_home_raw=1/(1+math.exp(-diff))
    p_away_raw=1 - p_home_raw
    p_draw=DRAW_BASE
    scale=p_home_raw + p_away_raw
    p_home=p_home_raw*(1 - p_draw)/scale
    p_away=p_away_raw*(1 - p_draw)/scale
    return p_home, p_draw, p_away

def kelly_fraction(p: float, odds: float) -> float:
    b=odds-1
    if b<=0: return 0.0
    q=1-p
    numer=b*p - q
    if numer<=0: return 0.0
    raw=numer/b
    return max(0.0, min(KELLY_FRACTION_LIMIT, raw))

def fair_odds_from_prob(probs: dict) -> dict:
    return {k:(1/v if v>0 else None) for k,v in probs.items()}

def poisson_p(k: int, lam: float) -> float:
    if lam<=0: return 1.0 if k==0 else 0.0
    return (lam**k * exp(-lam)) / factorial(k)

def btts_probability(lh: float, la: float) -> float:
    return 1 - math.exp(-lh) - math.exp(-la) + math.exp(-(lh+la))

def over25_probability(lambda_total: float) -> float:
    return 1 - (poisson_p(0, lambda_total) + poisson_p(1, lambda_total) + poisson_p(2, lambda_total))

def safe_edge(prob: float, odds: float) -> float:
    if odds<=0 or prob<=0: return -1.0
    return prob*odds - 1

# ========== Enhanced Modeling ==========
class OneVsRestCalibrator:
    def __init__(self, name: str):
        self.name=name
        self._platt=None
        self._iso=None
        self.fitted=False
    def fit(self, raw_probs: List[float], outcomes: List[int]):
        if not HAVE_SKLEARN or len(raw_probs)<40: return
        import numpy as _np
        X=_np.array(raw_probs).reshape(-1,1)
        y=_np.array(outcomes)
        try:
            self._platt=LogisticRegression(max_iter=400)
            self._platt.fit(X,y)
            self._iso=IsotonicRegression(out_of_bounds="clip")
            self._iso.fit(raw_probs,y)
            self.fitted=True
        except Exception:
            logging.exception("Kalibráció fit hiba (%s)", self.name)
    def transform(self, p: float) -> dict:
        if not self.fitted or not HAVE_SKLEARN:
            return {"raw":p,"platt":p,"iso":p,"cal":p,"used":"raw"}
        try:
            p_platt=float(self._platt.predict_proba([[p]])[0,1])
            p_iso=float(self._iso.predict([p])[0])
            p_avg=0.5*(p_platt+p_iso)
            return {"raw":p,"platt":p_platt,"iso":p_iso,"cal":p_avg,"used":"avg"}
        except Exception:
            return {"raw":p,"platt":p,"iso":p,"cal":p,"used":"raw"}

def load_calibration_history(path: Path)->dict:
    if not path.exists(): return {}
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logging.exception("Kalibráció history hiba"); return {}

def save_calibration_history(path: Path, data: dict):
    try: path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception: logging.exception("Kalibráció history mentés hiba")

def run_mc_1x2(lambda_home: float, lambda_away: float, sims: int)->dict:
    rng=random.Random(202901 + int(lambda_home*1000) + int(lambda_away*2000))
    home=draw=away=btts_yes=over25=0
    for _ in range(sims):
        gh=_sample_poisson(lambda_home, rng)
        ga=_sample_poisson(lambda_away, rng)
        if gh>ga: home+=1
        elif gh==ga: draw+=1
        else: away+=1
        if gh>=1 and ga>=1: btts_yes+=1
        if gh+ga>=3: over25+=1
    tot=float(sims)
    return {
        "home": home/tot,
        "draw": draw/tot,
        "away": away/tot,
        "btts_yes": btts_yes/tot,
        "btts_no": 1-btts_yes/tot,
        "over25": over25/tot,
        "under25": 1-over25/tot
    }

def _sample_poisson(lmbd: float, rng: random.Random)->int:
    L=math.exp(-lmbd); k=0; p=1.0
    while p>L:
        k+=1
        p*=rng.random()
    return k-1

# === (E) Bayes optimalizált osztály ===
class BayesianAttackDefense:
    def __init__(self):
        self.trace=None
        self.team_index={}
        self.enabled=ENABLE_BAYES and HAVE_PYMC
        self._teams=[]
    def fit(self, matches: List[dict]):
        if not self.enabled:
            logger.info("Bayes disabled vagy dependency hiányzik.")
            return
        if len(matches) < BAYES_MIN_MATCHES:
            logger.info("Bayes skip – összes meccs kevés (%d < %d)", len(matches), BAYES_MIN_MATCHES)
            self.enabled=False
            return
        team_counts={}
        for m in matches:
            team_counts[m["home_id"]]=team_counts.get(m["home_id"],0)+1
            team_counts[m["away_id"]]=team_counts.get(m["away_id"],0)+1
        insufficient=[t for t,c in team_counts.items() if c < BAYES_MIN_TEAM_MATCHES]
        if insufficient:
            logger.info("Bayes skip – csapatoknál nincs elég minta (pl. %s)", insufficient[:5])
            self.enabled=False
            return
        teams=sorted(team_counts.keys())
        self.team_index={t:i for i,t in enumerate(teams)}
        self._teams=teams
        home_idx=[self.team_index[m["home_id"]] for m in matches]
        away_idx=[self.team_index[m["away_id"]] for m in matches]
        g_home=[m["goals_home"] for m in matches]
        g_away=[m["goals_away"] for m in matches]

        if not HAVE_PYMC:
            self.enabled=False
            return

        def _sample():
            with pm.Model() as mdl:
                mu_att=pm.Normal("mu_att",0,1)
                mu_def=pm.Normal("mu_def",0,1)
                sigma_att=pm.HalfNormal("sigma_att",1)
                sigma_def=pm.HalfNormal("sigma_def",1)
                att=pm.Normal("att",mu_att,sigma_att,shape=len(teams))
                dfn=pm.Normal("def",mu_def,sigma_def,shape=len(teams))
                home_adv=pm.Normal("home_adv",0.2,0.2)
                lam_h=pm.math.exp(att[home_idx]-dfn[away_idx]+home_adv)
                lam_a=pm.math.exp(att[away_idx]-dfn[home_idx])
                pm.Poisson("gh",lam_h,observed=g_home)
                pm.Poisson("ga",lam_a,observed=g_away)
                tr=pm.sample(draws=300,tune=300,chains=2,cores=1,
                             target_accept=0.9,progressbar=False)
            return tr
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut=ex.submit(_sample)
                self.trace=fut.result(timeout=BAYES_MAX_SECONDS)
            logging.info("Bayes modell kész (teams=%d matches=%d)", len(teams), len(matches))
        except concurrent.futures.TimeoutError:
            logger.warning("Bayes sampling timeout (%ds) – skip.", BAYES_MAX_SECONDS)
            self.trace=None
            self.enabled=False
        except Exception:
            logger.exception("Bayes modell hiba")
            self.trace=None
            self.enabled=False
    def posterior_lambda_means(self, home_id:int, away_id:int)->Optional[Tuple[float,float]]:
        if not self.enabled or self.trace is None or home_id not in self.team_index or away_id not in self.team_index or np is None:
            return None
        try:
            att=self.trace.posterior["att"].stack(draws=("chain","draw")).values
            dfn=self.trace.posterior["def"].stack(draws=("chain","draw")).values
            home_adv=self.trace.posterior["home_adv"].stack(draws=("chain","draw")).values
            hi=self.team_index[home_id]; ai=self.team_index[away_id]
            lamh=np.exp(att[hi]-dfn[ai]+home_adv)
            lama=np.exp(att[ai]-dfn[hi])
            lmh=float(np.mean(lamh)); lma=float(np.mean(lama))
            # clipping
            lmh=max(BAYES_LAMBDA_MIN, min(BAYES_LAMBDA_MAX, lmh))
            lma=max(BAYES_LAMBDA_MIN, min(BAYES_LAMBDA_MAX, lma))
            return lmh, lma
        except Exception:
            return None

def ensemble_probs(base: Dict[str,float],
                   cal: Optional[Dict[str,float]],
                   bayes: Optional[Dict[str,float]],
                   mc: Optional[Dict[str,float]])->Dict[str,float]:
    weights=ENSEMBLE_WEIGHTS
    agg={"home":0.0,"draw":0.0,"away":0.0}; totw=0.0
    def add(p,w):
        nonlocal totw
        if not p or w<=0: return
        for k in agg: agg[k]+=p.get(k,0)*w
        totw+=w
    add(base, weights.get("base",0))
    add(cal, weights.get("cal",0))
    add(bayes, weights.get("bayes",0))
    add(mc, weights.get("mc",0))
    if totw<=0: return base
    for k in agg: agg[k]/=totw
    s=sum(agg.values())
    if s>0:
        for k in agg: agg[k]/=s
    return agg

def build_rationale(market: str,
                    selection: str,
                    model_prob: float,
                    market_prob: float,
                    odds: float,
                    edge_val: float,
                    lambda_home: float,
                    lambda_away: float,
                    rating_diff: float,
                    strict_flag: Optional[bool],
                    diff_limit: float | None,
                    ensemble_used: bool,
                    notes: str = "",
                    margin_adj_diff: float | None = None,
                    raw_edge: float | None = None,
                    z_edge: float | None = None,
                    league_id: int = None,
                    league_name: str = None) -> str:
    """
    Részletes magyar nyelvű indoklás generálása minden tipphez.
    """
    diff = model_prob - market_prob
    
    # Magyar piac név fordítás
    market_hu = {
        "1X2": "Végeredmény",
        "BTTS": "Mindkét csapat gólokat szerez",
        "O/U 2.5": "Gólok száma (2.5 felett/alatt)"
    }.get(market, market)
    
    # Szelekció magyar fordítása
    selection_hu = {
        "HOME": "Hazai győzelem",
        "AWAY": "Vendég győzelem", 
        "DRAW": "Döntetlen",
        "YES": "Igen",
        "NO": "Nem",
        "OVER": "Felett",
        "UNDER": "Alatt"
    }.get(selection.upper(), selection)
    
    # Liga információ és TOP státusz
    top_status = ""
    if league_id and LEAGUE_MANAGER:
        is_top, reason = LEAGUE_MANAGER.is_top_with_reason(league_id, league_name)
        if is_top:
            top_status = f" 🌟 {reason}"
        else:
            top_status = f" ⚪ {reason}"
    
    # Statisztikai magyarázat
    confidence_level = "Alacsony"
    if abs(diff) >= 0.15:
        confidence_level = "Magas"
    elif abs(diff) >= 0.08:
        confidence_level = "Közepes"
    
    # Edge kategorizálás
    edge_desc = "Gyenge"
    if edge_val >= 0.15:
        edge_desc = "Kiváló"
    elif edge_val >= 0.08:
        edge_desc = "Jó"
    elif edge_val >= 0.04:
        edge_desc = "Elfogadható"
    
    # Magyar nyelvű fő indoklás
    explanation = f"""
🎯 TIPP: {selection_hu} ({market_hu})
📊 Elemzés: A modellünk {model_prob:.1%} valószínűséget ad erre az eredményre, míg a piac {market_prob:.1%}-ot ár be. 
📈 Értékítélet: {edge_desc} value bet ({edge_val:.1%} edge), bizalmi szint: {confidence_level}
⚽ Liga státusz:{top_status}
🔢 Gólvárakozás: Hazai {lambda_home:.2f}, Vendég {lambda_away:.2f}
💪 Erőviszony: {'Hazai előny' if rating_diff > 0.1 else 'Vendég előny' if rating_diff < -0.1 else 'Kiegyenlített'} ({rating_diff:+.2f})
🧠 Modell: {'Ensemble' if ensemble_used else 'Alap'} algoritmus
""".strip()
    
    # Technikai részletek (eredeti formátumban)
    tech_parts = [
        f"Piac={market}",
        f"Pick={selection}",
        f"Model={model_prob:.4f}",
        f"Piac={market_prob:.4f}",
        f"Diff={diff:+.4f}",
        f"Odds={odds:.2f}",
        f"Edge={edge_val*100:.1f}%"
    ]
    
    if raw_edge is not None:
        tech_parts.append(f"RawEdge={raw_edge:+.4f}")
    if margin_adj_diff is not None:
        tech_parts.append(f"AdjDiff={margin_adj_diff:+.4f}")
    if z_edge is not None:
        tech_parts.append(f"Z={z_edge:+.2f}")
    
    tech_parts.extend([
        f"λH={lambda_home:.2f}",
        f"λA={lambda_away:.2f}",
        f"RatingDiff={rating_diff:+.3f}",
        f"Ens={'Y' if ensemble_used else 'N'}"
    ])
    
    if strict_flag is not None:
        tech_parts.append("Qual=" + ("STRICT" if strict_flag else "FALLBACK"))
    if diff_limit is not None:
        tech_parts.append(f"DiffTol={diff_limit:.2f}")
    if notes:
        tech_parts.append("Notes=" + notes)
    
    technical_details = " | ".join(tech_parts)
    
    return f"{explanation}\n\n📋 Technikai adatok: {technical_details}"

# =========================================================
# Állapot mentés / betöltés
# =========================================================
def load_state():
    if STATE_FILE.exists():
        try:
            st=json.loads(STATE_FILE.read_text(encoding="utf-8"))
            tz=ZoneInfo(LOCAL_TZ)
            today_local=datetime.now(tz=tz).date().isoformat()
            if st.get("date")!=today_local:
                logger.info("Új nap – bankroll reset")
                st["date"]=today_local
                st["bankroll_start"]=BANKROLL_DAILY
                st["bankroll_current"]=BANKROLL_DAILY
                st["picks"]=[]  # napi pick reset
            RUNTIME_STATE.update(st)
        except Exception:
            logger.warning("Állapot betöltési hiba – reset")
            save_state()
    else:
        save_state()

def save_state():
    try:
        STATE_FILE.write_text(json.dumps(RUNTIME_STATE, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("State mentési hiba")

def load_json(path: Path)->Any:
    if not path.exists(): return None
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("JSON betöltési hiba: %s", path)
        return None

def safe_write_json(path: Path, data: Any, indent=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

def hash_params(params: dict)->str:
    raw=json.dumps(params, sort_keys=True, separators=(",",":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

# =========================================================
# API-Football HTTP kliens
# =========================================================
class APIFootballClient:
    def __init__(self, api_key: str, base: str):
        self.api_key=api_key
        self.base=base.rstrip("/")
        self._session: aiohttp.ClientSession|None=None
        self._sem=asyncio.Semaphore(PARALLEL_CONNECTIONS)
        self.last_rate_headers={}
    async def __aenter__(self):
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        self._session=aiohttp.ClientSession(timeout=timeout, raise_for_status=False)
        return self
    async def __aexit__(self, *exc):
        if self._session: await self._session.close()
    async def get(self, endpoint: str, params: dict)->dict:
        url=self.base + endpoint
        headers={"x-apisports-key": self.api_key, "Accept":"application/json"}
        async with self._sem:
            if self.last_rate_headers:
                try:
                    remain=int(self.last_rate_headers.get("x-ratelimit-requests-remaining","5"))
                    if remain<3: await asyncio.sleep(1.0)
                except: pass
            for attempt in range(3):
                try:
                    async with self._session.get(url, headers=headers, params=params) as resp:
                        txt=await resp.text()
                        try: js=json.loads(txt)
                        except json.JSONDecodeError:
                            js={"raw":txt,"parse_error":True}
                        for k,v in resp.headers.items():
                            lk=k.lower()
                            if lk.startswith("x-ratelimit"):
                                self.last_rate_headers[lk]=v
                        if resp.status>=500:
                            await asyncio.sleep(1+attempt); continue
                        return js
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    await asyncio.sleep(1+attempt)
            return {"errors":["network_fail"],"response":[]}

# =========================================================
# OddsAPI kliens - Liga adatok lekérésére
# =========================================================
class OddsAPIClient:
    def __init__(self, api_key: str, base: str):
        self.api_key = api_key
        self.base = base.rstrip("/")
        self._session: aiohttp.ClientSession|None = None
        self._sem = asyncio.Semaphore(PARALLEL_CONNECTIONS)
        self.last_rate_headers = {}
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        self._session = aiohttp.ClientSession(timeout=timeout, raise_for_status=False)
        return self
    
    async def __aexit__(self, *exc):
        if self._session: 
            await self._session.close()
    
    async def get(self, endpoint: str, params: dict = None) -> dict:
        if not self.api_key:
            logger.warning("OddsAPI kulcs hiányzik")
            return {"data": [], "error": "No API key"}
        
        url = self.base + endpoint
        params = params or {}
        params["apiKey"] = self.api_key
        
        async with self._sem:
            # Rate limiting
            if self.last_rate_headers:
                try:
                    remain = int(self.last_rate_headers.get("x-requests-remaining", "10"))
                    if remain < 3: 
                        await asyncio.sleep(1.0)
                except: 
                    pass
            
            for attempt in range(3):
                try:
                    async with self._session.get(url, params=params) as resp:
                        txt = await resp.text()
                        try: 
                            js = json.loads(txt)
                        except json.JSONDecodeError:
                            js = {"raw": txt, "parse_error": True}
                        
                        # Store rate limit headers
                        for k, v in resp.headers.items():
                            lk = k.lower()
                            if lk.startswith("x-requests"):
                                self.last_rate_headers[lk] = v
                        
                        if resp.status >= 500:
                            await asyncio.sleep(1 + attempt)
                            continue
                        return js
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    await asyncio.sleep(1 + attempt)
            return {"data": [], "error": "network_fail"}
    
    async def get_sports(self) -> dict:
        """Lekéri az elérhető sportágakat"""
        return await self.get("/sports")
    
    async def get_leagues(self, sport_key: str = "soccer") -> dict:
        """Lekéri az adott sportág ligáit"""
        return await self.get(f"/sports/{sport_key}/events")

# =========================================================
# Liga mapping és tier osztályozás OddsAPI-val
# =========================================================
async def fetch_leagues_from_odds_api() -> Dict[str, dict]:
    """OddsAPI-ból lekéri a ligákat és tier besorolást készít"""
    if not ODDS_API_KEY:
        logger.warning("OddsAPI kulcs nincs beállítva - alapértelmezett tier mapping")
        return {}
    
    leagues_data = {}
    try:
        async with OddsAPIClient(ODDS_API_KEY, ODDS_API_BASE) as client:
            # Lekérjük a futball ligákat
            result = await client.get_leagues("soccer")
            
            if "data" in result:
                events = result.get("data", [])
            else:
                # Ha a válasz közvetlenül a ligák listája
                events = result if isinstance(result, list) else []
            
            # TIER1 és TIER1B ligák meghatározása név alapján
            tier1_keywords = [
                "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
                "Champions League", "UEFA", "World Cup", "Euro", "Copa America"
            ]
            
            tier1b_keywords = [
                "Championship", "Liga 2", "Serie B", "2. Bundesliga", "Ligue 2",
                "Eredivisie", "Primeira Liga", "Pro League", "Super League"
            ]
            
            for event in events:
                league_name = event.get("home_team", {}).get("league", "") or event.get("sport_title", "")
                if not league_name:
                    continue
                
                # Tier meghatározása
                tier = "OTHER"
                for keyword in tier1_keywords:
                    if keyword.lower() in league_name.lower():
                        tier = "TIER1"
                        break
                
                if tier == "OTHER":
                    for keyword in tier1b_keywords:
                        if keyword.lower() in league_name.lower():
                            tier = "TIER1B"
                            break
                
                leagues_data[league_name] = {
                    "name": league_name,
                    "tier": tier,
                    "source": "odds_api"
                }
                
        logger.info(f"OddsAPI liga adatok feldolgozva: {len(leagues_data)} liga")
        
    except Exception as e:
        logger.exception(f"OddsAPI liga lekérés hiba: {e}")
    
    return leagues_data

# =========================================================
# Fixtures lekérése
# =========================================================
async def fetch_upcoming_fixtures(client: APIFootballClient, days_ahead: int)->list[dict]:
    fixtures=[]
    today=date.today()
    total_fixtures_found = 0
    top_fixtures_count = 0
    
    # Statisztikák gyűjtése
    league_stats = {}
    
    for delta in range(days_ahead+1):
        d=today + timedelta(days=delta)
        js=await client.get("/fixtures", {"date": d.isoformat()})
        resp=js.get("response") or []
        for fx in resp:
            total_fixtures_found += 1
            league_id=fx.get("league",{}).get("id")
            league_name=fx.get("league",{}).get("name","")
            
            if league_id:
                # Statisztika gyűjtés
                if league_name not in league_stats:
                    league_stats[league_name] = {"total": 0, "included": 0, "is_top": False}
                league_stats[league_name]["total"] += 1
                
                if LEAGUE_WHITELIST and league_id not in LEAGUE_WHITELIST: continue
                if league_id in LEAGUE_BLACKLIST: continue
                
                # Új tier-alapú szűrés: csak TIER1 és TIER1B ligák
                if ENABLE_TIER_FILTERING:
                    tier = LEAGUE_MANAGER.tier_of(league_id)
                    if tier not in ["TIER1", "TIER1B"]:
                        continue
                
                # TOP_MODE alapú szűrés
                should_include = False
                is_top_league = False
                
                if TOP_MODE=="top_only":
                    is_top_league, reason = LEAGUE_MANAGER.is_top_with_reason(league_id, league_name)
                    should_include = is_top_league
                    if should_include:
                        top_fixtures_count += 1
                        logger.debug("TOP liga mérkőzés: %s - %s", league_name, reason)
                elif TOP_MODE=="hybrid":
                    if LEAGUE_MANAGER.tier_of(league_id)=="EXCLUDE": 
                        should_include = False
                    else:
                        should_include = True
                        is_top_league, _ = LEAGUE_MANAGER.is_top_with_reason(league_id, league_name)
                        if is_top_league:
                            top_fixtures_count += 1
                else:  # "all"
                    if LEAGUE_MANAGER.tier_of(league_id)=="EXCLUDE": 
                        should_include = False
                    else:
                        should_include = True
                        is_top_league, _ = LEAGUE_MANAGER.is_top_with_reason(league_id, league_name)
                        if is_top_league:
                            top_fixtures_count += 1
                
                if should_include:
                    fixtures.append(fx)
                    league_stats[league_name]["included"] += 1
                    league_stats[league_name]["is_top"] = is_top_league
    
    # Statisztikák logolása
    logger.info("Mérkőzés szűrési statisztika:")
    logger.info("  Összes mérkőzés: %d", total_fixtures_found)
    logger.info("  TOP liga mérkőzések: %d", top_fixtures_count)
    logger.info("  Beválogatott mérkőzések: %d", len(fixtures))
    logger.info("  TOP_MODE: %s", TOP_MODE)
    logger.info("  Azonosított TOP ligák száma: %d", LEAGUE_MANAGER.top_league_data["count"])
    
    # TOP ligák részletes listázása
    if LEAGUE_MANAGER.top_league_data["names"]:
        logger.info("TOP liga nevek:")
        for name in LEAGUE_MANAGER.top_league_data["names"]:
            logger.info("  - %s", name)
    
    uniq={}
    for f in fixtures:
        fid=f.get("fixture",{}).get("id")
        if fid: uniq[fid]=f
    return list(uniq.values())

# =========================================================
# Tippmix MATCH gyűjtés
# =========================================================
async def tippmix_fetch_all_leagues_robust() -> list[dict]:
    """
    Robustly fetch all available leagues from TippmixPro, 
    including special championships like World Cup qualifiers
    """
    all_tournaments = []
    
    async with TippmixProWampClient(verbose=False) as cli:
        try:
            # Fetch initial dump of available tournaments/venues
            topics_to_check = [
                "sport.1",  # Football
                "sport.1.venue",  # Football venues
                "tournament",  # Tournament data
            ]
            
            for topic in topics_to_check:
                try:
                    dump_data = await cli.initial_dump_topic(topic)
                    if dump_data:
                        all_tournaments.extend(dump_data)
                        logger.info(f"Fetched {len(dump_data)} items from topic: {topic}")
                except Exception as e:
                    logger.warning(f"Failed to fetch topic {topic}: {e}")
                    continue
                
                await asyncio.sleep(0.1)  # Rate limiting
        
        except Exception as e:
            logger.error(f"Error in robust league fetching: {e}")
    
    # Process and deduplicate tournaments
    unique_tournaments = {}
    special_keywords = [
        "világbajnokság", "world cup", "vb", "wc",
        "selejtező", "qualifier", "qualifying",
        "európa bajnokság", "european championship", "euro",
        "nemzetek ligája", "nations league",
        "copa", "continental", "confederations"
    ]
    
    for item in all_tournaments:
        if item.get("_type") in ("TOURNAMENT", "VENUE"):
            tid = item.get("id")
            name = str(item.get("name", "")).lower()
            
            # Check if it's a special championship
            is_special = any(keyword in name for keyword in special_keywords)
            
            if tid and (is_special or "league" in name or "liga" in name or "championship" in name):
                unique_tournaments[tid] = {
                    "id": tid,
                    "name": item.get("name"),
                    "type": item.get("_type"),
                    "is_special": is_special,
                    "venue_id": item.get("venue_id") or item.get("id")
                }
    
    logger.info(f"Discovered {len(unique_tournaments)} unique tournaments, including {sum(1 for t in unique_tournaments.values() if t['is_special'])} special championships")
    return list(unique_tournaments.values())

async def tippmix_fetch_and_map(days_ahead: int) -> dict:
    # First try to get tournaments from enhanced robust fetching
    try:
        robust_tournaments = await tippmix_fetch_all_leagues_robust()
        if robust_tournaments:
            tours = robust_tournaments
            logger.info(f"Using {len(tours)} tournaments from robust fetching")
        else:
            raise ValueError("No tournaments from robust fetching")
    except Exception as e:
        logger.warning(f"Robust fetching failed: {e}, falling back to tournaments.json")
        # Fallback to tournaments.json
        tournaments_path=Path("tournaments.json")
        if not tournaments_path.exists():
            logger.warning("tournaments.json hiányzik – Tippmix integráció korlátozott.")
            return {}
        try:
            tdata=json.loads(tournaments_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("tournaments.json parse hiba.")
            return {}
        if isinstance(tdata,list): tours=tdata
        else: tours=tdata.get("tournaments", [])
    
    # Limit to prevent excessive API calls
    tours=tours[:500]  # Increased from 300 to handle more leagues
    out=[]
    
    async with TippmixProWampClient(verbose=False) as cli:
        for t in tours:
            try:
                tid=str(t.get("id"))
                venue=str(t.get("venue_id") or t.get("id") or "")
                if not venue: continue
                recs=await cli.get_matches_for_venue_tournament(venue, tid)
                for r in recs:
                    if r.get("_type")=="MATCH":
                        # Add tournament info to match for better tracking
                        r["tournament_info"] = {
                            "name": t.get("name"),
                            "is_special": t.get("is_special", False)
                        }
                        out.append(r)
                await asyncio.sleep(0.05)
            except Exception as e:
                logger.warning(f"Failed to fetch matches for tournament {t.get('name', tid)}: {e}")
                continue
    
    logger.info(f"Fetched {len(out)} matches from {len(tours)} tournaments")
    return {str(m.get("id")):m for m in out}

# =========================================================
# Odds fájl helper
# =========================================================
def find_raw_file(base_dir: Path, prefix: str)->list[Path]:
    rd=base_dir/"raw"
    if not rd.exists(): return []
    return list(rd.glob(f"{prefix}__*.json"))

def overround_1x2(odds: dict)->float:
    try: return 1/float(odds["home"]) + 1/float(odds["draw"]) + 1/float(odds["away"])
    except Exception: return 999.0

def overround_2way(o1: float, o2: float)->float:
    try: return 1/float(o1) + 1/float(o2)
    except Exception: return 999.0

# =========================================================
# Szigorú margin szűrés - Új funkciók
# =========================================================
def passes_margin_filter_1x2(odds: dict) -> bool:
    """Ellenőrzi, hogy az 1X2 odds margin ≤ MAX_MARGIN_1X2"""
    if not odds or not all(k in odds for k in ["home", "draw", "away"]):
        return False
    margin = overround_1x2(odds)
    return margin <= MAX_MARGIN_1X2

def passes_margin_filter_2way(o1: float, o2: float) -> bool:
    """Ellenőrzi, hogy a 2-way odds margin ≤ MAX_MARGIN_2WAY"""
    if not o1 or not o2:
        return False
    margin = overround_2way(o1, o2)
    return margin <= MAX_MARGIN_2WAY

def filter_odds_by_margin(odds_data: dict) -> dict:
    """Odds szűrése margin alapján"""
    filtered = {}
    
    # 1X2 szűrés
    if "one_x_two" in odds_data and passes_margin_filter_1x2(odds_data["one_x_two"]):
        filtered["one_x_two"] = odds_data["one_x_two"]
    
    # BTTS szűrés
    if "btts" in odds_data:
        btts = odds_data["btts"]
        if btts and "yes" in btts and "no" in btts:
            try:
                yes_odds = float(btts["yes"])
                no_odds = float(btts["no"])
                if passes_margin_filter_2way(yes_odds, no_odds):
                    filtered["btts"] = btts
            except:
                pass
    
    # Over/Under 2.5 szűrés
    if "ou25" in odds_data:
        ou25 = odds_data["ou25"]
        if ou25 and "over25" in ou25 and "under25" in ou25:
            try:
                over_odds = float(ou25["over25"])
                under_odds = float(ou25["under25"])
                if passes_margin_filter_2way(over_odds, under_odds):
                    filtered["ou25"] = ou25
            except:
                pass
    
    return filtered

def valid_1x2_parsing(odds: dict)->bool:
    if not odds or any(k not in odds for k in ("home","draw","away")): return False
    try:
        _=float(odds["home"]); _=float(odds["draw"]); _=float(odds["away"])
    except Exception: return False
    orr=overround_1x2(odds)
    return 0.90 <= orr <= 1.20

def valid_2way_parsing(o1: float|None, o2: float|None, max_odds: float=6.0)->bool:
    if not o1 or not o2: return False
    try: o1=float(o1); o2=float(o2)
    except Exception: return False
    if o1<=1.01 or o2<=1.01: return False
    if o1>max_odds or o2>max_odds: return False
    orr=overround_2way(o1,o2)
    return 0.90 <= orr <= 1.20

# === (C) Odds / margin számítás segédek ===
def implied_probs_from_odds_1x2(odds: dict)->Optional[dict]:
    try:
        inv={k:1/float(odds[k]) for k in ("home","draw","away")}
        s=sum(inv.values())
        if s<=0: return None
        return {k:inv[k]/s for k in inv}
    except Exception:
        return None

def implied_probs_from_odds_2way(o1: float, o2: float)->Optional[Tuple[float,float,float]]:
    # returns (pA_raw, pB_raw, overround)
    try:
        ia=1/float(o1); ib=1/float(o2); s=ia+ib
        if s<=0: return None
        return ia/s, ib/s, s
    except Exception:
        return None

def z_score_diff(p_model: float, p_market: float)->float:
    # Bernoulli variance approximate
    var = p_market*(1-p_market)
    if var <= 0: return 0.0
    return (p_model - p_market)/math.sqrt(var)

def format_local_time(iso_utc: str) -> str:
    """Convert UTC ISO string to local time string"""
    try:
        dt_utc = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
        tz = ZoneInfo(LOCAL_TZ)
        return dt_utc.astimezone(tz).strftime("%Y-%m-%d %H:%M")
    except:
        return iso_utc

def calculate_market_strength(odds: dict, market_type: str) -> float:
    """
    Calculate market strength based on overround and betting volume proxy
    Returns a percentage indicating market efficiency/strength
    """
    try:
        if market_type == "1X2":
            if not all(k in odds for k in ("home", "draw", "away")):
                return 0.0
            overround = 1/float(odds["home"]) + 1/float(odds["draw"]) + 1/float(odds["away"])
            # Lower overround = stronger market (more efficient)
            # Typical range: 1.02-1.15, convert to 0-100% strength
            strength = max(0, min(100, (1.15 - overround) / 0.13 * 100))
            return strength
        elif market_type in ("BTTS", "O/U"):
            # For 2-way markets
            if len(odds) != 2:
                return 0.0
            o1, o2 = list(odds.values())
            overround = 1/float(o1) + 1/float(o2)
            # Typical range: 1.02-1.10 for 2-way markets
            strength = max(0, min(100, (1.10 - overround) / 0.08 * 100))
            return strength
    except Exception:
        return 0.0
    return 0.0

def relative_value(raw_edge: float, odds: float)->float:
    if odds<=0: return 0.0
    return raw_edge/odds

# =========================================================
# build_fixture_context – Tippmix odds override + fallback logika + (C) bővítések
# =========================================================
def build_fixture_context(root: Path, fixture_id: int):
    summary_path=root / f"out_fixture_{fixture_id}" / "summary.json"
    summary=load_json(summary_path)
    if not summary:
        logger.error("Nincs summary: %s", fixture_id); return None
    primary=load_json(summary_path.parent / "primary_fixture.json")
    fx_data=primary
    if not fx_data:
        for f in find_raw_file(summary_path.parent,"fixture"):
            js=load_json(f)
            if js and js.get("response"):
                fx_data=js["response"][0]; break
    if not fx_data:
        logger.error("Nincs primary_fixture: %s", fixture_id); return None
    fixture=fx_data.get("fixture",{}) or {}
    league=fx_data.get("league",{}) or {}
    teams=fx_data.get("teams",{}) or {}
    fid=fixture.get("id")
    if not fid: return None
    league_id=league.get("id")
    season=league.get("season")
    home_id=teams.get("home",{}).get("id")
    away_id=teams.get("away",{}).get("id")
    kickoff_ts=fixture.get("timestamp",0)
    kickoff_dt=datetime.fromtimestamp(kickoff_ts, tz=timezone.utc) if kickoff_ts else datetime.now(timezone.utc)
    ts_home_files=find_raw_file(summary_path.parent,"team_stats_home")
    ts_away_files=find_raw_file(summary_path.parent,"team_stats_away")
    team_stats_home=team_stats_away=None
    if ts_home_files:
        js=load_json(ts_home_files[0]); team_stats_home=js.get("response") if js else None
    if ts_away_files:
        js=load_json(ts_away_files[0]); team_stats_away=js.get("response") if js else None
    gf_home,ga_home,form_home=parse_goals_avg(team_stats_home) if team_stats_home else (1.1,1.2,"")
    gf_away,ga_away,form_away=parse_goals_avg(team_stats_away) if team_stats_away else (1.0,1.1,"")
    topscorers_files=find_raw_file(summary_path.parent,"topscorers_primary")
    topscorers_json=load_json(topscorers_files[0]) if topscorers_files else None
    top_home_ids=collect_top_scorers_ids(topscorers_json, home_id)
    top_away_ids=collect_top_scorers_ids(topscorers_json, away_id)
    squad_home_files=find_raw_file(summary_path.parent,"squad_home")
    squad_away_files=find_raw_file(summary_path.parent,"squad_away")
    squad_home_json=load_json(squad_home_files[0]) if squad_home_files else None
    squad_away_json=load_json(squad_away_files[0]) if squad_away_files else None
    squad_home_ids=collect_squad_player_ids(squad_home_json)
    squad_away_ids=collect_squad_player_ids(squad_away_json)
    inp_home=TeamRatingInput(home_id, season, form_home, gf_home, ga_home, top_home_ids, squad_home_ids, league_id)
    inp_away=TeamRatingInput(away_id, season, form_away, gf_away, ga_away, top_away_ids, squad_away_ids, league_id)
    rating_home=compute_team_rating(inp_home)
    rating_away=compute_team_rating(inp_away)
    p_home,p_draw,p_away=logistic_probabilities(rating_home.combined_rating, rating_away.combined_rating)
    model_probs={"home":p_home,"draw":p_draw,"away":p_away}
    lambda_home=max(0.05,(gf_home + ga_away)/2)
    lambda_away=max(0.05,(gf_away + ga_home)/2)
    lambda_total=lambda_home + lambda_away
    odds_1x2=None
    odds_btts=None
    odds_overunder25=None
    if USE_TIPPMIX:
        tip_map=GLOBAL_RUNTIME.get("tippmix_mapping", {})
        tip_odds_cache=GLOBAL_RUNTIME.get("tippmix_odds_cache", {})
        if fixture_id in tip_map and fixture_id in tip_odds_cache:
            std=tip_odds_cache[fixture_id]
            if std.one_x_two:
                odds_1x2={"home":std.one_x_two.get("HOME"),"draw":std.one_x_two.get("DRAW"),"away":std.one_x_two.get("AWAY")}
            if std.btts:
                odds_btts={"yes":std.btts.get("YES"),"no":std.btts.get("NO")}
            if std.ou25:
                odds_overunder25={"over25":std.ou25.get("OVER"),"under25":std.ou25.get("UNDER")}
    if (not USE_TIPPMIX) or odds_1x2 is None:
        odds_files=find_raw_file(summary_path.parent,"odds")
        if odds_files:
            js=load_json(odds_files[0]); resp=js.get("response") or []
            best_1x2=None; best_1x2_margin=999
            best_btts=None; best_btts_margin=999
            best_ou=None; best_ou_margin=999
            for it in resp:
                for bm in (it.get("bookmakers") or []):
                    bets=bm.get("bets") or []
                    cand_1x2=None; cand_btts_pair=(None,None); cand_ou_pair=(None,None)
                    for b in bets:
                        name_l=(b.get("name") or "").strip().lower()
                        values=b.get("values") or []
                        if name_l in ("match winner","1x2","fulltime result"):
                            tmp={}
                            for v in values:
                                val=(v.get("value") or "").strip().lower()
                                try: odd=float(v.get("odd"))
                                except: continue
                                if val.startswith("home"): tmp["home"]=odd
                                elif val.startswith("draw"): tmp["draw"]=odd
                                elif val.startswith("away"): tmp["away"]=odd
                            if valid_1x2_parsing(tmp): cand_1x2=tmp
                        elif (("both" in name_l and "team" in name_l and "score" in name_l) or name_l in ("btts","goal/no goal","goals - both teams to score")) \
                             and not any(k in name_l for k in ("1st","2nd","first half","second half","1h","2h","half","extra","corners","penalt")):
                            yes_odd=no_odd=None
                            for v in values:
                                val=(v.get("value") or "").strip().lower()
                                try: odd=float(v.get("odd"))
                                except: continue
                                if val in ("yes","goal"): yes_odd=odd
                                elif val in ("no","no goal"): no_odd=odd
                            if valid_2way_parsing(yes_odd,no_odd,6.0):
                                cand_btts_pair=(yes_odd,no_odd)
                        elif name_l in ("goals over/under","over/under"):
                            over25=None; under25=None
                            for v in values:
                                val=(v.get("value") or "").strip().lower()
                                try: odd=float(v.get("odd"))
                                except: continue
                                if val=="over 2.5": over25=odd
                                elif val=="under 2.5": under25=odd
                            if valid_2way_parsing(over25,under25,6.0):
                                cand_ou_pair=(over25,under25)
                    if cand_1x2:
                        m=overround_1x2(cand_1x2)
                        if m<best_1x2_margin:
                            best_1x2=cand_1x2; best_1x2_margin=m
                    yes,no=cand_btts_pair
                    if yes and no:
                        m=overround_2way(yes,no)
                        if m<best_btts_margin:
                            best_btts={"yes":yes,"no":no}; best_btts_margin=m
                    ov,un=cand_ou_pair
                    if ov and un:
                        m=overround_2way(ov,un)
                        if m<best_ou_margin:
                            best_ou={"over25":ov,"under25":un}; best_ou_margin=m
            if odds_1x2 is None: odds_1x2=best_1x2
            if odds_btts is None: odds_btts=best_btts
            if odds_overunder25 is None: odds_overunder25=best_ou

    fair_from_model=fair_odds_from_prob(model_probs)

    # === (C) 1X2 market metrics ===
    edge={}; kelly={}; prob_details_1x2={}
    implied_1x2=None; overround_val=None
    if odds_1x2:
        implied_1x2_raw=implied_probs_from_odds_1x2(odds_1x2)
        if implied_1x2_raw:
            # raw (already normalized to sum=1) -> overround = sum(1/odds)
            ov=overround_1x2(odds_1x2)
            overround_val=ov
            # margin = ov -1
            margin=ov - 1 if ov!=999.0 else None
            prob_details_1x2={
                "implied_norm": implied_1x2_raw,
                "overround": ov,
                "margin": margin
            }
            implied_1x2=implied_1x2_raw
        for k in ("home","draw","away"):
            odd=odds_1x2.get(k)
            if not odd:
                edge[k]=0.0; kelly[k]=0.0; continue
            try:
                odd_f=float(odd)
            except:
                odd_f=0.0
            raw_edge = model_probs[k]*odd_f - 1
            if league_id in MAJOR_LEAGUE_IDS:
                raw_edge += MAJOR_LEAGUE_EDGE_BONUS
            margin_adj_diff = None
            z_e=None
            if implied_1x2 and k in implied_1x2:
                margin_adj_diff = model_probs[k] - implied_1x2[k]
                z_e = z_score_diff(model_probs[k], implied_1x2[k])
            edge[k]=raw_edge
            kelly[k]=kelly_fraction(model_probs[k], odd_f)
            # store advanced metrics per selection
            prob_details_1x2.setdefault("selections", {})[k]={
                "model_prob": model_probs[k],
                "odd": odd_f,
                "raw_edge": raw_edge,
                "margin_adj_diff": margin_adj_diff,
                "z_edge": z_e,
                "rel_value": relative_value(raw_edge, odd_f)
            }
    else:
        for k in ("home","draw","away"):
            edge[k]=0.0; kelly[k]=0.0

    # === (C) 2-way markets (BTTS / O/U) metrics ===
    market_probs={}; market_odds={}; market_edge={}
    market_prob_details={}

    p_btts_yes=btts_probability(lambda_home, lambda_away)
    p_btts_no=1-p_btts_yes
    market_probs["btts_yes"]=p_btts_yes
    market_probs["btts_no"]=p_btts_no
    if odds_btts and "yes" in odds_btts and "no" in odds_btts:
        o_yes=float(odds_btts["yes"]); o_no=float(odds_btts["no"])
        ip=implied_probs_from_odds_2way(o_yes, o_no)
        if ip:
            p_yes_mkt, p_no_mkt, ov2=ip
            market_prob_details["BTTS"]={
                "overround": ov2,
                "margin": ov2-1,
                "selections": {
                    "btts_yes":{
                        "model_prob": p_btts_yes,
                        "implied_prob": p_yes_mkt,
                        "raw_edge": p_btts_yes*o_yes -1,
                        "margin_adj_diff": p_btts_yes - p_yes_mkt,
                        "z_edge": z_score_diff(p_btts_yes, p_yes_mkt),
                        "rel_value": relative_value(p_btts_yes*o_yes -1, o_yes),
                        "odd": o_yes
                    },
                    "btts_no":{
                        "model_prob": p_btts_no,
                        "implied_prob": p_no_mkt,
                        "raw_edge": p_btts_no*o_no -1,
                        "margin_adj_diff": p_btts_no - p_no_mkt,
                        "z_edge": z_score_diff(p_btts_no, p_no_mkt),
                        "rel_value": relative_value(p_btts_no*o_no -1, o_no),
                        "odd": o_no
                    }
                }
            }
        market_odds["btts_yes"]=o_yes
        market_odds["btts_no"]=o_no
        market_edge["btts_yes"]=safe_edge(p_btts_yes, o_yes)
        market_edge["btts_no"]=safe_edge(p_btts_no, o_no)

    p_over25=over25_probability(lambda_total)
    p_under25=1-p_over25
    market_probs["over25"]=p_over25
    market_probs["under25"]=p_under25
    if odds_overunder25 and "over25" in odds_overunder25 and "under25" in odds_overunder25:
        o_ov=float(odds_overunder25["over25"]); o_un=float(odds_overunder25["under25"])
        ip=implied_probs_from_odds_2way(o_ov, o_un)
        if ip:
            p_ov_mkt, p_un_mkt, ov2=ip
            market_prob_details["OU25"]={
                "overround": ov2,
                "margin": ov2-1,
                "selections":{
                    "over25":{
                        "model_prob": p_over25,
                        "implied_prob": p_ov_mkt,
                        "raw_edge": p_over25*o_ov -1,
                        "margin_adj_diff": p_over25 - p_ov_mkt,
                        "z_edge": z_score_diff(p_over25, p_ov_mkt),
                        "rel_value": relative_value(p_over25*o_ov -1, o_ov),
                        "odd": o_ov
                    },
                    "under25":{
                        "model_prob": p_under25,
                        "implied_prob": p_un_mkt,
                        "raw_edge": p_under25*o_un -1,
                        "margin_adj_diff": p_under25 - p_un_mkt,
                        "z_edge": z_score_diff(p_under25, p_un_mkt),
                        "rel_value": relative_value(p_under25*o_un -1, o_un),
                        "odd": o_un
                    }
                }
            }
        market_odds["over25"]=o_ov
        market_odds["under25"]=o_un
        market_edge["over25"]=safe_edge(p_over25, o_ov)
        market_edge["under25"]=safe_edge(p_under25, o_un)

    injuries_files=find_raw_file(summary_path.parent,"injuries_league")
    injuries_affected=[]
    if injuries_files:
        js=load_json(injuries_files[0])
        resp=js.get("response") or []
        for inj in resp:
            try:
                if inj.get("player",{}).get("id") in (set(top_home_ids)|set(top_away_ids)):
                    injuries_affected.append(inj)
            except: pass

    ctx=FixtureContext(
        fixture_id=fixture_id,
        league_id=league_id,
        league_name=league.get("name", ""),
        season=season,
        home_team_id=home_id,
        away_team_id=away_id,
        timestamp=kickoff_ts,
        kickoff_utc=kickoff_dt,
        ratings_home=rating_home,
        ratings_away=rating_away,
        probs=model_probs,
        odds=odds_1x2,
        edge=edge,
        kelly=kelly,
        fair_odds=fair_from_model
    )
    extra={
        "lambda_home":lambda_home,
        "lambda_away":lambda_away,
        "market_probs":market_probs,
        "market_odds":market_odds,
        "market_edge":market_edge,
        "injuries_hit_top":injuries_affected,
        "market_prob_details": {  # (C) összegyűjtve
            "one_x_two": prob_details_1x2,
            "other_markets": market_prob_details
        }
    }
    return ctx, extra

# =========================================================
# Calibrátor init, Bayes history
# =========================================================
def init_calibrators(cal_history_path: Path)->dict:
    c_map={}
    if not ENABLE_CALIBRATION: return c_map
    hist=load_calibration_history(cal_history_path)
    for k in ("home","draw","away"):
        cal=OneVsRestCalibrator(k)
        data=hist.get(k,{})
        raw=data.get("raw",[]); outcome=data.get("outcome",[])
        if raw and outcome: cal.fit(raw,outcome)
        c_map[k]=cal
    return c_map

def gather_bayes_history(root: Path, days: int)->List[dict]:
    cutoff=datetime.now(timezone.utc)-timedelta(days=days)
    dataset=[]
    for p in root.glob("out_fixture_*"):
        primary=p/"primary_fixture.json"
        if not primary.exists(): continue
        try:
            js=json.loads(primary.read_text(encoding="utf-8"))
            fixture=js.get("fixture",{}) or {}
            status=fixture.get("status",{}).get("short")
            if status!="FT": continue
            ts=fixture.get("timestamp")
            if not ts: continue
            dt=datetime.fromtimestamp(ts, tz=timezone.utc)
            if dt<cutoff: continue
            score=fixture.get("score",{}) or {}
            full=score.get("fulltime",{}) or {}
            gh=full.get("home"); ga=full.get("away")
            if gh is None or ga is None: continue
            league=js.get("league",{}) or {}
            season=league.get("season")
            teams=js.get("teams",{}) or {}
            home_id=teams.get("home",{}).get("id")
            away_id=teams.get("away",{}).get("id")
            if None in (home_id,away_id,gh,ga): continue
            dataset.append({"home_id":home_id,"away_id":away_id,"goals_home":gh,"goals_away":ga,"season":season})
        except: continue
    return dataset

# =========================================================
# Elemzés
# =========================================================
def analyze_fixture(root: Path, fixture_id: int, enhanced_tools: dict|None=None)->dict:
    tup=build_fixture_context(root, fixture_id)
    if not tup: return {}
    ctx,extra=tup
    enhanced_block={}
    if ENABLE_ENHANCED_MODELING and enhanced_tools:
        base_probs=ctx.probs
        cald={}
        ensemble_source={"base": base_probs}
        if ENABLE_CALIBRATION and enhanced_tools.get("calibrators"):
            for k in ("home","draw","away"):
                co=enhanced_tools["calibrators"].get(k)
                if co: cald[k]=co.transform(base_probs[k])
                else: cald[k]={"raw":base_probs[k],"cal":base_probs[k],"used":"raw"}
            calibrated={k: cald[k]["cal"] for k in cald}
            enhanced_block["calibration"]=cald
            ensemble_source["cal"]=calibrated
        else:
            calibrated=None
        bayes_probs=None
        if ENABLE_BAYES and enhanced_tools.get("bayes_model"):
            bm=enhanced_tools["bayes_model"]
            blams=bm.posterior_lambda_means(ctx.home_team_id, ctx.away_team_id)
            if blams:
                lbh,lba=blams
                mc_bayes=run_mc_1x2(lbh,lba, sims=int(MC_SIMS/2))
                bayes_probs={k: mc_bayes[k] for k in ("home","draw","away")}
                enhanced_block["bayes_lambdas"]={"home":lbh,"away":lba}
                enhanced_block["bayes_mc_probs"]=bayes_probs
                ensemble_source["bayes"]=bayes_probs
        mc_probs=None
        if ENABLE_MC:
            mc_all=run_mc_1x2(extra["lambda_home"], extra["lambda_away"], sims=MC_SIMS)
            mc_probs={k: mc_all[k] for k in ("home","draw","away")}
            enhanced_block["mc_full"]=mc_all
            ensemble_source["mc"]=mc_probs
        final_probs=ensemble_probs(
            base=ensemble_source.get("base"),
            cal=ensemble_source.get("cal"),
            bayes=ensemble_source.get("bayes"),
            mc=ensemble_source.get("mc"))
        enhanced_block["ensemble_probs"]=final_probs
        enhanced_block["weights_used"]=ENSEMBLE_WEIGHTS
    out_dir=root / f"out_fixture_{fixture_id}"
    tier=LEAGUE_MANAGER.tier_of(ctx.league_id) if ctx.league_id else None
    result={
        "fixture_id": ctx.fixture_id,
        "kickoff_utc": ctx.kickoff_utc.isoformat(),
        "league_id": ctx.league_id,
        "league_name": ctx.league_name,
        "league_tier": tier,
        "season": ctx.season,
        "teams": {"home_id": ctx.home_team_id, "away_id": ctx.away_team_id},
        "model_probs": ctx.probs,
        "odds": ctx.odds,
        "fair_odds_model": ctx.fair_odds,
        "edge": ctx.edge,
        "kelly": ctx.kelly,
        "home_rating": asdict(ctx.ratings_home),
        "away_rating": asdict(ctx.ratings_away),
        "lambda_home": extra["lambda_home"],
        "lambda_away": extra["lambda_away"],
        "market_probs": extra["market_probs"],
        "market_odds": extra["market_odds"],
        "market_edge": extra["market_edge"],
        "injuries_hit_top": extra["injuries_hit_top"],
        "market_prob_details": extra["market_prob_details"],  # (C)
        "generated_at": datetime.now(timezone.utc).isoformat()
    }
    if enhanced_block:
        result["enhanced_model"]=enhanced_block
    safe_write_json(out_dir/"analysis.json", result)
    return result

# =========================================================
# PICK / STAKE – (C) margin_adj felhasználás kiegészítő adata
# =========================================================
def allocate_stakes(analysis_results: list[dict])->list[dict]:
    bankroll=RUNTIME_STATE["bankroll_current"]
    max_single=bankroll * MAX_SINGLE_STAKE_FRACTION
    picks=[]
    for r in analysis_results:
        odds=r.get("odds")
        if not odds: continue
        
        # Margin szűrés alkalmazása
        filtered_odds = filter_odds_by_margin(odds)
        if not filtered_odds.get("one_x_two"):
            continue  # Ha az 1X2 odds nem felel meg a margin kritériumnak
        
        odds = filtered_odds  # Használjuk a szűrt odds-okat
        
        league_id=r.get("league_id")
        def passes_publish_threshold(edge_val: float)->bool:
            # Value betting logika: minimum edge threshold ellenőrzése
            if edge_val < MIN_EDGE_THRESHOLD:
                return False
                
            if TOP_MODE=="all": return edge_val>0
            if TOP_MODE=="top_only":
                return LEAGUE_MANAGER.is_top(league_id) and edge_val>=PUBLISH_MIN_EDGE_TOP
            if TOP_MODE=="hybrid":
                if LEAGUE_MANAGER.is_top(league_id): return edge_val>=PUBLISH_MIN_EDGE_TOP
                return edge_val>=PUBLISH_MIN_EDGE_OTHER
            return edge_val>0
        edge_d=r.get("edge",{})
        kelly_d=r.get("kelly",{})
        best_sel=None; best_edge=0.0
        
        # Döntetlen kizárása 1X2 piacokról
        selections_to_check = ["home", "away"]
        if not EXCLUDE_DRAW_1X2:
            selections_to_check.append("draw")
        
        for sel in selections_to_check:
            e=edge_d.get(sel,0.0)
            if e>0 and e>best_edge:
                best_edge=e; best_sel=sel
        if not best_sel: continue
        sel_odds=odds.get(best_sel)
        if not sel_odds: continue
        try: sel_odds=float(sel_odds)
        except: continue
        if sel_odds>MAX_ODDS_1X2_PICKS: continue
        if best_edge>PICK_EDGE_CAP_1X2: continue
        if not passes_publish_threshold(best_edge): continue
        
        # Fix tét rendszer - FIX_STAKE_AMOUNT használata Kelly helyett
        stake = FIX_STAKE_AMOUNT
        
        # Bankroll ellenőrzés
        if stake > max_single:
            stake = max_single
        if stake < 1: 
            continue
            
        projected=stake*sel_odds
        model_prob=r.get("model_probs",{}).get(best_sel,0.0)
        implied_block=r.get("market_prob_details",{}).get("one_x_two",{})
        margin_adj_diff=None; z_edge=None; raw_edge_val=best_edge
        if implied_block and "selections" in implied_block:
            sel_info=implied_block["selections"].get(best_sel)
            if sel_info:
                margin_adj_diff=sel_info.get("margin_adj_diff")
                z_edge=sel_info.get("z_edge")
        implied=None
        try:
            inv_sum=sum(1/float(odds[k]) for k in ("home","draw","away"))
            implied=(1/sel_odds)/inv_sum if inv_sum>0 else 0
        except: implied=0
        diff=model_prob-(implied or 0)
        enhanced_used="enhanced_model" in r
        rating_diff=(r.get("home_rating",{}).get("combined_rating",0) -
                     r.get("away_rating",{}).get("combined_rating",0))
        rationale=build_rationale(
            market="1X2",
            selection=best_sel.upper(),
            model_prob=model_prob,
            market_prob=implied or 0,
            odds=sel_odds,
            edge_val=best_edge,
            lambda_home=r.get("lambda_home",0),
            lambda_away=r.get("lambda_away",0),
            rating_diff=rating_diff,
            strict_flag=None,
            diff_limit=None,
            ensemble_used=enhanced_used,
            notes=f"Pick stake kelly={k_frac:.4f} diff={diff:+.4f}",
            margin_adj_diff=margin_adj_diff,
            raw_edge=raw_edge_val,
            z_edge=z_edge,
            league_id=league_id,
            league_name=r.get("league_name")
        )
        picks.append({
            "fixture_id": r["fixture_id"],
            "selection": best_sel,
            "edge": round(best_edge,4),
            "kelly_fraction": round(kelly_d.get(best_sel,0.0),4),
            "stake": round(stake,2),
            "odds": sel_odds,
            "expected_brutto": round(projected,2),
            "kickoff_utc": r["kickoff_utc"],
            "rationale": rationale,
            "league_id": league_id,
            "league_tier": r.get("league_tier")
        })
    return picks

def register_picks(picks: list[dict]):
    if picks:
        RUNTIME_STATE["picks"].extend(picks)
        save_state()

# =========================================================
# Ticket / meta segédek (változatlan + margin info a rationale-ben)
# =========================================================
def load_fixture_meta(fixture_id: int)->dict:
    pf=DATA_ROOT / f"out_fixture_{fixture_id}" / "primary_fixture.json"
    if not pf.exists(): return {}
    try:
        js=json.loads(pf.read_text(encoding="utf-8"))
        fixture=js.get("fixture",{}) or {}
        venue=fixture.get("venue",{}) or {}
        teams=js.get("teams",{}) or {}
        league=js.get("league",{}) or {}
        return {
            "home_name": teams.get("home",{}).get("name"),
            "away_name": teams.get("away",{}).get("name"),
            "league_name": league.get("name"),
            "league_country": league.get("country"),
            "venue_name": venue.get("name"),
            "venue_city": venue.get("city"),
        }
    except Exception:
        logger.exception("Meta betöltési hiba fixture=%s", fixture_id)
        return {}

def to_local_time(iso_utc: str)->str:
    try:
        dt=datetime.fromisoformat(iso_utc.replace("Z","+00:00"))
        tz=ZoneInfo(LOCAL_TZ)
        return dt.astimezone(tz).strftime("%Y-%m-%d %H:%M")
    except:
        return iso_utc

def select_best_tickets_enhanced(analyzed_results: list[dict], only_today: bool=True, max_tips_per_market: int=2) -> dict:
    """
    Enhanced ticket selection that returns multiple top tips per market
    """
    tz=ZoneInfo(LOCAL_TZ)
    today_local=datetime.now(tz=tz).date()
    
    def same_local_day(iso_utc: str)->bool:
        if not only_today: return True
        try:
            dt_utc=datetime.fromisoformat(iso_utc.replace("Z","+00:00"))
            return dt_utc.astimezone(tz).date()==today_local
        except: return False
        
    def implied_probs_1x2(odds: dict)->dict|None:
        try:
            inv={k:1/float(v) for k,v in odds.items()}
            s=sum(inv.values())
            if s<=0: return None
            return {k:inv[k]/s for k in ("home","draw","away")}
        except: return None
        
    def allow_ticket_for_public(r: dict, market: str)->bool:
        league_id=r.get("league_id")
        if TOP_MODE=="all": return True
        if TOP_MODE=="top_only": return LEAGUE_MANAGER.is_top(league_id)
        if TOP_MODE=="hybrid": return LEAGUE_MANAGER.is_top(league_id)
        return True

    # Collect all candidates for each market
    candidates_1x2 = []
    candidates_btts = []
    candidates_ou = []

    # Process 1X2 market
    for r in analyzed_results:
        if not allow_ticket_for_public(r,"1X2"): continue
        ko=r.get("kickoff_utc")
        if not ko: continue
        if only_today and not same_local_day(ko): continue
        odds=r.get("odds") or {}; edges=r.get("edge") or {}; probs=r.get("model_probs") or {}
        if not odds or not edges or not probs: continue
        ip=implied_probs_1x2(odds)
        if not ip: continue
        
        # Calculate market strength
        market_strength = calculate_market_strength(odds, "1X2")
        
        for sel in ("home","draw","away"):
            e=edges.get(sel); p=probs.get(sel); pi=ip.get(sel); o=odds.get(sel)
            if None in (e,p,pi,o): continue
            try: o=float(o)
            except: continue
            if e<=0 or e>EDGE_CAP_1X2: continue
            if o>TICKET_MAX_ODDS_1X2: continue
            if abs(p-pi)>TICKET_DIFF_TOL_1X2: continue
            
            # Enhanced value calculation considering market strength
            value_score = e * (1 + market_strength / 100 * 0.1)  # 10% bonus for strong markets
            
            candidates_1x2.append({
                "fixture_id": r["fixture_id"],
                "league_id": r.get("league_id"),
                "league_name": r.get("league_name"),
                "league_tier": r.get("league_tier"),
                "home_name": r.get("home_name"),
                "away_name": r.get("away_name"),
                "market": "1X2",
                "selection": sel.upper(),
                "edge": e,
                "value_score": value_score,
                "odds": o,
                "model_prob": p,
                "market_prob": pi,
                "market_strength": market_strength,
                "kickoff_utc": r["kickoff_utc"],
                "kickoff_local": format_local_time(r["kickoff_utc"])
            })

    # Process BTTS market
    for r in analyzed_results:
        if not allow_ticket_for_public(r,"BTTS"): continue
        ko=r.get("kickoff_utc")
        if not ko: continue
        if only_today and not same_local_day(ko): continue
        me=r.get("market_edge") or {}; mo=r.get("market_odds") or {}; mp=r.get("market_probs") or {}
        if not mo or not mp: continue
        
        y_odds=mo.get("yes"); n_odds=mo.get("no")
        if y_odds is None or n_odds is None: continue
        try: y_odds=float(y_odds); n_odds=float(n_odds)
        except: continue
        
        # Calculate market strength for BTTS
        btts_odds = {"yes": y_odds, "no": n_odds}
        market_strength = calculate_market_strength(btts_odds, "BTTS")
        
        try:
            ia=1/y_odds; ib=1/n_odds; s=ia+ib
            pair=(ia/s,ib/s) if s>0 else None
        except:
            pair=None
        if not pair: continue
        p_yes_mkt,p_no_mkt=pair
        
        for sel_raw in ("yes","no"):
            e=me.get(sel_raw); p=mp.get(sel_raw)
            if e is None or p is None: continue
            if e<=0 or e>EDGE_CAP_BTTs_OU: continue
            o=y_odds if sel_raw=="yes" else n_odds
            p_mkt=p_yes_mkt if sel_raw=="yes" else p_no_mkt
            if o>TICKET_MAX_ODDS_2WAY: continue
            if abs(p-p_mkt)>TICKET_DIFF_TOL_2WAY: continue
            
            # Enhanced value calculation
            value_score = e * (1 + market_strength / 100 * 0.1)
            
            candidates_btts.append({
                "fixture_id": r["fixture_id"],
                "league_id": r.get("league_id"),
                "league_name": r.get("league_name"),
                "league_tier": r.get("league_tier"),
                "home_name": r.get("home_name"),
                "away_name": r.get("away_name"),
                "market": "BTTS",
                "selection": sel_raw.upper(),
                "edge": e,
                "value_score": value_score,
                "odds": o,
                "model_prob": p,
                "market_prob": p_mkt,
                "market_strength": market_strength,
                "kickoff_utc": r["kickoff_utc"],
                "kickoff_local": format_local_time(r["kickoff_utc"])
            })

    # Process O/U 2.5 market
    for r in analyzed_results:
        if not allow_ticket_for_public(r,"O/U 2.5"): continue
        ko=r.get("kickoff_utc")
        if not ko: continue
        if only_today and not same_local_day(ko): continue
        me=r.get("market_edge") or {}; mo=r.get("market_odds") or {}; mp=r.get("market_probs") or {}
        if not mo or not mp: continue
        
        ov=mo.get("over25"); un=mo.get("under25")
        if ov is None or un is None: continue
        try: ov=float(ov); un=float(un)
        except: continue
        
        # Calculate market strength for O/U
        ou_odds = {"over": ov, "under": un}
        market_strength = calculate_market_strength(ou_odds, "O/U")
        
        try:
            ia=1/ov; ib=1/un; s=ia+ib
            pair=(ia/s,ib/s) if s>0 else None
        except:
            pair=None
        if not pair: continue
        p_ov_mkt,p_un_mkt=pair
        
        for sel_raw in ("over25","under25"):
            e=me.get(sel_raw); p=mp.get(sel_raw)
            if e is None or p is None: continue
            if e<=0 or e>EDGE_CAP_BTTs_OU: continue
            o=ov if sel_raw=="over25" else un
            p_mkt=p_ov_mkt if sel_raw=="over25" else p_un_mkt
            if o>TICKET_MAX_ODDS_2WAY: continue
            if abs(p-p_mkt)>TICKET_DIFF_TOL_2WAY: continue
            
            # Enhanced value calculation
            value_score = e * (1 + market_strength / 100 * 0.1)
            
            label="OVER 2.5" if sel_raw=="over25" else "UNDER 2.5"
            candidates_ou.append({
                "fixture_id": r["fixture_id"],
                "league_id": r.get("league_id"),
                "league_name": r.get("league_name"),
                "league_tier": r.get("league_tier"),
                "home_name": r.get("home_name"),
                "away_name": r.get("away_name"),
                "market": "O/U 2.5",
                "selection": label,
                "edge": e,
                "value_score": value_score,
                "odds": o,
                "model_prob": p,
                "market_prob": p_mkt,
                "market_strength": market_strength,
                "kickoff_utc": r["kickoff_utc"],
                "kickoff_local": format_local_time(r["kickoff_utc"])
            })

    # Sort and select top candidates for each market
    candidates_1x2.sort(key=lambda x: x["value_score"], reverse=True)
    candidates_btts.sort(key=lambda x: x["value_score"], reverse=True)
    candidates_ou.sort(key=lambda x: x["value_score"], reverse=True)

    return {
        "x1x2": candidates_1x2[:max_tips_per_market] if candidates_1x2 else [],
        "btts": candidates_btts[:max_tips_per_market] if candidates_btts else [],
        "overunder": candidates_ou[:max_tips_per_market] if candidates_ou else []
    }

def select_best_tickets(analyzed_results: list[dict], only_today: bool=True) -> dict:
    tz=ZoneInfo(LOCAL_TZ)
    today_local=datetime.now(tz=tz).date()
    def same_local_day(iso_utc: str)->bool:
        if not only_today: return True
        try:
            dt_utc=datetime.fromisoformat(iso_utc.replace("Z","+00:00"))
            return dt_utc.astimezone(tz).date()==today_local
        except: return False
    def implied_probs_1x2(odds: dict)->dict|None:
        try:
            inv={k:1/float(v) for k,v in odds.items()}
            s=sum(inv.values())
            if s<=0: return None
            return {k:inv[k]/s for k in ("home","draw","away")}
        except: return None
    def allow_ticket_for_public(r: dict, market: str)->bool:
        league_id=r.get("league_id")
        if TOP_MODE=="all": return True
        if TOP_MODE=="top_only": return LEAGUE_MANAGER.is_top(league_id)
        if TOP_MODE=="hybrid": return LEAGUE_MANAGER.is_top(league_id)
        return True
    best_1x2=None; best_edge_1x2=-1
    for r in analyzed_results:
        if not allow_ticket_for_public(r,"1X2"): continue
        ko=r.get("kickoff_utc")
        if not ko: continue
        if only_today and not same_local_day(ko): continue
        odds=r.get("odds") or {}; edges=r.get("edge") or {}; probs=r.get("model_probs") or {}
        if not odds or not edges or not probs: continue
        ip=implied_probs_1x2(odds)
        if not ip: continue
        implied_block=r.get("market_prob_details",{}).get("one_x_two",{})
        for sel in ("home","draw","away"):
            e=edges.get(sel); p=probs.get(sel); pi=ip.get(sel); o=odds.get(sel)
            if None in (e,p,pi,o): continue
            try: o=float(o)
            except: continue
            if e<=0 or e>EDGE_CAP_1X2: continue
            if o>TICKET_MAX_ODDS_1X2: continue
            if abs(p-pi)>TICKET_DIFF_TOL_1X2: continue
            if e>best_edge_1x2:
                rating_diff=(r.get("home_rating",{}).get("combined_rating",0) -
                             r.get("away_rating",{}).get("combined_rating",0))
                margin_adj=None; z_e=None; raw_edge=e
                if implied_block and "selections" in implied_block:
                    sel_info=implied_block["selections"].get(sel)
                    if sel_info:
                        margin_adj=sel_info.get("margin_adj_diff"); z_e=sel_info.get("z_edge")
                rationale=build_rationale("1X2", sel.upper(), p, pi, o, e,
                                          r.get("lambda_home",0), r.get("lambda_away",0),
                                          rating_diff, True, TICKET_DIFF_TOL_1X2,
                                          "enhanced_model" in r, "Ticket strict",
                                          margin_adj_diff=margin_adj,
                                          raw_edge=raw_edge,
                                          z_edge=z_e,
                                          league_id=r.get("league_id"),
                                          league_name=r.get("league_name"))
                best_edge_1x2=e
                best_1x2={
                    "fixture_id": r["fixture_id"],
                    "league_id": r.get("league_id"),
                    "league_tier": r.get("league_tier"),
                    "market":"1X2",
                    "selection": sel.upper(),
                    "edge": round(e,4),
                    "odds": o,
                    "model_prob": round(p,4),
                    "kickoff_utc": r["kickoff_utc"],
                    "rationale": rationale
                }
    best_btts=None; best_edge_btts=-1
    candidate_btts_fallback=[]
    for r in analyzed_results:
        if not allow_ticket_for_public(r,"BTTS"): continue
        ko=r.get("kickoff_utc")
        if not ko: continue
        if only_today and not same_local_day(ko): continue
        me=r.get("market_edge") or {}
        mo=r.get("market_odds") or {}
        mp=r.get("market_probs") or {}
        if not mo or not mp: continue
        yes_o=mo.get("btts_yes"); no_o=mo.get("btts_no")
        if yes_o is None or no_o is None: continue
        try: yes_o=float(yes_o); no_o=float(no_o)
        except: continue
        if yes_o>TICKET_MAX_ODDS_2WAY or no_o>TICKET_MAX_ODDS_2WAY: continue
        try:
            ia=1/yes_o; ib=1/no_o; s=ia+ib
            pair=(ia/s,ib/s) if s>0 else None
        except:
            pair=None
        if not pair: continue
        p_yes_mkt,p_no_mkt=pair
        prob_detail=r.get("market_prob_details",{}).get("other_markets",{}).get("BTTS",{})
        for sel_raw in ("btts_yes","btts_no"):
            e=me.get(sel_raw); p=mp.get(sel_raw)
            if e is None or p is None: continue
            if e<=0 or e>EDGE_CAP_BTTs_OU: continue
            p_mkt=p_yes_mkt if sel_raw=="btts_yes" else p_no_mkt
            rating_diff=(r.get("home_rating",{}).get("combined_rating",0) -
                         r.get("away_rating",{}).get("combined_rating",0))
            sel_label=sel_raw.replace("btts_","").upper()
            margin_adj=None; z_e=None; raw_edge_val=None
            if prob_detail and "selections" in prob_detail:
                d=prob_detail["selections"].get(sel_raw)
                if d:
                    margin_adj=d.get("margin_adj_diff"); z_e=d.get("z_edge"); raw_edge_val=d.get("raw_edge")
            if abs(p-p_mkt)<=TICKET_DIFF_TOL_2WAY:
                if e>best_edge_btts:
                    rationale=build_rationale(
                        market="BTTS",
                        selection=sel_label,
                        model_prob=p,
                        market_prob=p_mkt,
                        odds=yes_o if sel_raw=="btts_yes" else no_o,
                        edge_val=e,
                        lambda_home=r.get("lambda_home",0),
                        lambda_away=r.get("lambda_away",0),
                        rating_diff=rating_diff,
                        strict_flag=True,
                        diff_limit=TICKET_DIFF_TOL_2WAY,
                        ensemble_used=("enhanced_model" in r),
                        notes="Ticket strict",
                        margin_adj_diff=margin_adj,
                        raw_edge=raw_edge_val,
                        z_edge=z_e,
                        league_id=r.get("league_id"),
                        league_name=r.get("league_name")
                    )
                    best_edge_btts=e
                    best_btts={
                        "fixture_id": r["fixture_id"],
                        "league_id": r.get("league_id"),
                        "league_tier": r.get("league_tier"),
                        "market":"BTTS",
                        "selection": sel_label,
                        "edge": round(e,4),
                        "odds": yes_o if sel_raw=="btts_yes" else no_o,
                        "model_prob": round(p,4),
                        "kickoff_utc": r["kickoff_utc"],
                        "rationale": rationale
                    }
            else:
                if TICKET_FALLBACK_ENABLE and abs(p-p_mkt)<=TICKET_FALLBACK_DIFF_TOL_2WAY and e>0.05:
                    rationale=build_rationale(
                        market="BTTS",
                        selection=sel_label,
                        model_prob=p,
                        market_prob=p_mkt,
                        odds=yes_o if sel_raw=="btts_yes" else no_o,
                        edge_val=e,
                        lambda_home=r.get("lambda_home",0),
                        lambda_away=r.get("lambda_away",0),
                        rating_diff=rating_diff,
                        strict_flag=False,
                        diff_limit=TICKET_FALLBACK_DIFF_TOL_2WAY,
                        ensemble_used=("enhanced_model" in r),
                        notes="Fallback jelölt",
                        margin_adj_diff=margin_adj,
                        raw_edge=raw_edge_val,
                        z_edge=z_e,
                        league_id=r.get("league_id"),
                        league_name=r.get("league_name")
                    )
                    candidate_btts_fallback.append({
                        "fixture_id": r["fixture_id"],
                        "league_id": r.get("league_id"),
                        "league_tier": r.get("league_tier"),
                        "market":"BTTS",
                        "selection": sel_label,
                        "edge": round(e,4),
                        "odds": yes_o if sel_raw=="btts_yes" else no_o,
                        "model_prob": round(p,4),
                        "kickoff_utc": r["kickoff_utc"],
                        "rationale": rationale,
                        "_fallback": True
                    })
    if not best_btts and candidate_btts_fallback:
        candidate_btts_fallback.sort(key=lambda x: x["edge"], reverse=True)
        cand=candidate_btts_fallback[0]
        cand["rationale"] += " | Fallback kiválasztva"
        best_btts=cand
    best_ou=None; best_edge_ou=-1
    candidate_ou_fallback=[]
    for r in analyzed_results:
        if not allow_ticket_for_public(r,"O/U 2.5"): continue
        ko=r.get("kickoff_utc")
        if not ko: continue
        if only_today and not same_local_day(ko): continue
        me=r.get("market_edge") or {}
        mo=r.get("market_odds") or {}
        mp=r.get("market_probs") or {}
        if not mo or not mp: continue
        ov=mo.get("over25"); un=mo.get("under25")
        if ov is None or un is None: continue
        try: ov=float(ov); un=float(un)
        except: continue
        if ov>TICKET_MAX_ODDS_2WAY or un>TICKET_MAX_ODDS_2WAY: continue
        try:
            ia=1/ov; ib=1/un; s=ia+ib
            pair=(ia/s,ib/s) if s>0 else None
        except:
            pair=None
        if not pair: continue
        p_ov_mkt,p_un_mkt=pair
        prob_detail=r.get("market_prob_details",{}).get("other_markets",{}).get("OU25",{})
        for sel_raw in ("over25","under25"):
            e=me.get(sel_raw); p=mp.get(sel_raw)
            if e is None or p is None: continue
            if e<=0 or e>EDGE_CAP_BTTs_OU: continue
            o=ov if sel_raw=="over25" else un
            p_mkt=p_ov_mkt if sel_raw=="over25" else p_un_mkt
            rating_diff=(r.get("home_rating",{}).get("combined_rating",0) -
                         r.get("away_rating",{}).get("combined_rating",0))
            label="OVER 2.5" if sel_raw=="over25" else "UNDER 2.5"
            margin_adj=None; z_e=None; raw_edge_val=None
            if prob_detail and "selections" in prob_detail:
                d=prob_detail["selections"].get(sel_raw)
                if d:
                    margin_adj=d.get("margin_adj_diff"); z_e=d.get("z_edge"); raw_edge_val=d.get("raw_edge")
            if abs(p-p_mkt)<=TICKET_DIFF_TOL_2WAY:
                if e>best_edge_ou:
                    rationale=build_rationale(
                        market="O/U 2.5",
                        selection=label,
                        model_prob=p,
                        market_prob=p_mkt,
                        odds=o,
                        edge_val=e,
                        lambda_home=r.get("lambda_home",0),
                        lambda_away=r.get("lambda_away",0),
                        rating_diff=rating_diff,
                        strict_flag=True,
                        diff_limit=TICKET_DIFF_TOL_2WAY,
                        ensemble_used=("enhanced_model" in r),
                        notes="Ticket strict",
                        margin_adj_diff=margin_adj,
                        raw_edge=raw_edge_val,
                        z_edge=z_e,
                        league_id=r.get("league_id"),
                        league_name=r.get("league_name")
                    )
                    best_edge_ou=e
                    best_ou={
                        "fixture_id": r["fixture_id"],
                        "league_id": r.get("league_id"),
                        "league_tier": r.get("league_tier"),
                        "market":"O/U 2.5",
                        "selection": label,
                        "edge": round(e,4),
                        "odds": o,
                        "model_prob": round(p,4),
                        "kickoff_utc": r["kickoff_utc"],
                        "rationale": rationale
                    }
            else:
                if TICKET_FALLBACK_ENABLE and abs(p-p_mkt)<=TICKET_FALLBACK_DIFF_TOL_2WAY and e>0.05:
                    rationale=build_rationale(
                        market="O/U 2.5",
                        selection=label,
                        model_prob=p,
                        market_prob=p_mkt,
                        odds=o,
                        edge_val=e,
                        lambda_home=r.get("lambda_home",0),
                        lambda_away=r.get("lambda_away",0),
                        rating_diff=rating_diff,
                        strict_flag=False,
                        diff_limit=TICKET_FALLBACK_DIFF_TOL_2WAY,
                        ensemble_used=("enhanced_model" in r),
                        notes="Fallback jelölt",
                        margin_adj_diff=margin_adj,
                        raw_edge=raw_edge_val,
                        z_edge=z_e,
                        league_id=r.get("league_id"),
                        league_name=r.get("league_name")
                    )
                    candidate_ou_fallback.append({
                        "fixture_id": r["fixture_id"],
                        "league_id": r.get("league_id"),
                        "league_tier": r.get("league_tier"),
                        "market":"O/U 2.5",
                        "selection": label,
                        "edge": round(e,4),
                        "odds": o,
                        "model_prob": round(p,4),
                        "kickoff_utc": r["kickoff_utc"],
                        "rationale": rationale,
                        "_fallback": True
                    })
    if not best_ou and candidate_ou_fallback:
        candidate_ou_fallback.sort(key=lambda x: x["edge"], reverse=True)
        cand=candidate_ou_fallback[0]
        cand["rationale"] += " | Fallback kiválasztva"
        best_ou=cand

    def enrich(entry,title):
        if not entry: return entry
        meta=load_fixture_meta(entry["fixture_id"])
        venue_name=(meta.get("venue_name") or "").strip()
        venue_city=(meta.get("venue_city") or "").strip()
        entry.update(meta)
        entry["venue_name"]=venue_name if venue_name else None
        entry["venue_city"]=venue_city if venue_city else None
        entry["kickoff_local"]=to_local_time(entry["kickoff_utc"])
        if entry["market"]=="1X2":
            mapping={"HOME":"Hazai győzelem","AWAY":"Vendég győzelem","DRAW":"Döntetlen"}
            entry["selection_label"]=mapping.get(entry["selection"], entry["selection"])
        elif entry["market"]=="BTTS":
            entry["selection_label"]="Mindkét csapat szerez gólt" if entry["selection"]=="BTTS_YES" else "Nem szerez mindkét csapat gólt"
        elif entry["market"]=="O/U 2.5":
            entry["selection_label"]="Több mint 2.5 gól" if "OVER" in entry["selection"] else "Kevesebb mint 2.5 gól"
        return entry
    return {
        "x1x2": enrich(best_1x2,"1X2"),
        "btts": enrich(best_btts,"BTTS"),
        "overunder": enrich(best_ou,"O/U 2.5")
    }

def load_all_analysis(root: Path)->list[dict]:
    results=[]
    for p in root.glob("out_fixture_*"):
        if not p.is_dir(): continue
        af=p/"analysis.json"
        if af.exists():
            try:
                js=json.loads(af.read_text(encoding="utf-8"))
                results.append(js)
            except:
                logger.exception("Analysis beolvasási hiba: %s", af)
    return results

def build_offline_tickets(root: Path)->dict:
    analyzed=load_all_analysis(root)
    if not analyzed:
        return {"x1x2": None,"btts": None,"overunder": None}
    return select_best_tickets(analyzed, only_today=TICKET_ONLY_TODAY)

def save_ticket_full_analysis(tickets: dict, root: Path)->Optional[Path]:
    if not tickets: return None
    fixture_ids=[]
    for k in ("x1x2","btts","overunder"):
        e=tickets.get(k)
        if e and e.get("fixture_id"):
            fixture_ids.append(e["fixture_id"])
    fixture_ids=sorted(set(fixture_ids))
    if not fixture_ids: return None
    items=[]
    for fid in fixture_ids:
        analysis=load_json(root / f"out_fixture_{fid}" / "analysis.json") or {}
        primary=load_json(root / f"out_fixture_{fid}" / "primary_fixture.json") or {}
        items.append({"fixture_id":fid,"analysis":analysis,"primary_fixture":primary})
    out={
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "local_tz": LOCAL_TZ,
        "top_mode": TOP_MODE,
        "tickets": tickets,
        "fixtures": items
    }
    ts=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path=root / f"ticket_full_analysis_{ts}.json"
    safe_write_json(path, out)
    logger.info("Ticket teljes elemzés mentve: %s", path.name)
    return path

def find_stale_fixture_dirs(root: Path)->list[Path]:
    stale=[]
    for p in root.glob("out_fixture_*"):
        if p.is_dir() and not (p / "summary.json").exists():
            stale.append(p)
    return stale

async def refetch_single_fixture(client: APIFootballClient, fixture_id: int, root: Path):
    js=await client.get("/fixtures", {"id": fixture_id})
    resp=js.get("response") or []
    if not resp:
        logger.warning("Refetch sikertelen: %s", fixture_id)
        return False
    await fetch_fixture_bundle(client, resp[0], root)
    return True

# =========================================================
# FETCH fixture bundle
# =========================================================
FIXTURE_ENDPOINTS = [
    ("fixture", "/fixtures", {"id": "<FIXTURE_ID>"}),
    ("predictions", "/predictions", {"fixture": "<FIXTURE_ID>"}),
    ("odds", "/odds", {"fixture": "<FIXTURE_ID>"}),
    ("h2h", "/fixtures/headtohead", {"h2h": "<HOME_ID>-<AWAY_ID>", "last":"10"}),
    ("form_home_last", "/fixtures", {"team": "<HOME_ID>", "last": 10}),
    ("form_away_last", "/fixtures", {"team": "<AWAY_ID>", "last": 10}),
    ("squad_home", "/players/squads", {"team": "<HOME_ID>"}),
    ("squad_away", "/players/squads", {"team": "<AWAY_ID>"}),
    ("team_home_info", "/teams", {"id": "<HOME_ID>"}),
    ("team_away_info", "/teams", {"id": "<AWAY_ID>"}),
    ("topscorers_primary", "/players/topscorers", {"league": "<LEAGUE_ID>", "season": "<SEASON>"}),
    ("standings_primary", "/standings", {"league": "<LEAGUE_ID>", "season": "<SEASON>"}),
    ("team_stats_home", "/teams/statistics", {"league": "<LEAGUE_ID>", "season": "<SEASON>", "team": "<HOME_ID>"}),
    ("team_stats_away", "/teams/statistics", {"league": "<LEAGUE_ID>", "season": "<SEASON>", "team": "<AWAY_ID>"}),
    ("injuries_league", "/injuries", {"league":"<LEAGUE_ID>", "season":"<SEASON>"}),
    ("fixture_events", "/fixtures/events", {"fixture":"<FIXTURE_ID>"}),
    ("fixture_lineups", "/fixtures/lineups", {"fixture":"<FIXTURE_ID>"}),
    ("fixture_statistics", "/fixtures/statistics", {"fixture":"<FIXTURE_ID>"}),
    ("fixture_players_stats", "/fixtures/players", {"fixture":"<FIXTURE_ID>"}),
]

async def fetch_fixture_bundle(client: APIFootballClient, fx_obj: dict, root: Path):
    fixture=fx_obj.get("fixture", {})
    league=fx_obj.get("league", {})
    teams=fx_obj.get("teams", {})
    fid=fixture.get("id")
    if not fid: return
    league_id=league.get("id")
    season=league.get("season")
    home_id=teams.get("home", {}).get("id")
    away_id=teams.get("away", {}).get("id")
    out_dir=root / f"out_fixture_{fid}"
    out_dir_raw=out_dir/"raw"
    out_dir_raw.mkdir(parents=True, exist_ok=True)
    summary_meta={
        "fixture_id": fid,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "endpoints": [],
        "league_tier": LEAGUE_MANAGER.tier_of(league_id) if league_id else None
    }
    async def fetch_one(tag, endpoint, param_tpl):
        params={}
        for k,v in param_tpl.items():
            if isinstance(v,str):
                v2=(v.replace("<FIXTURE_ID>", str(fid))
                      .replace("<LEAGUE_ID>", str(league_id))
                      .replace("<SEASON>", str(season))
                      .replace("<HOME_ID>", str(home_id))
                      .replace("<AWAY_ID>", str(away_id)))
                params[k]=v2
            else:
                params[k]=v
        result=await client.get(endpoint, params)
        if SAVE_RAW:
            h=hash_params(params)
            safe_write_json(out_dir_raw / f"{tag}__{h}.json", result)
        summary_meta["endpoints"].append({
            "tag": tag,
            "endpoint": endpoint,
            "params": params,
            "results": result.get("results"),
            "errors": result.get("errors", []),
        })
    tasks=[asyncio.create_task(fetch_one(tag, ep, tpl)) for tag,ep,tpl in FIXTURE_ENDPOINTS]
    await asyncio.gather(*tasks, return_exceptions=True)
    safe_write_json(out_dir/"summary.json", summary_meta)
    safe_write_json(out_dir/"primary_fixture.json", fx_obj)

# =========================================================
# (D) Kalibráció history update + trimming + reliability
# =========================================================
def trim_calibration_history(hist: dict):
    # hist: { "home": {"raw":[], "outcome":[]}, ... , "ensemble": {"home":{"raw":[],...}} }
    def trim_list(v: list):
        if len(v) > CALIBRATION_HISTORY_MAX:
            # keep most recent
            return v[-CALIBRATION_HISTORY_MAX:]
        return v
    for main_key in list(hist.keys()):
        block=hist.get(main_key)
        if not isinstance(block, dict): continue
        # regular categories
        if main_key in ("home","draw","away"):
            for k2 in ("raw","outcome"):
                if isinstance(block.get(k2), list):
                    block[k2]=trim_list(block[k2])
        elif main_key=="ensemble":
            for cat in ("home","draw","away"):
                if cat in block:
                    for k2 in ("raw","outcome"):
                        if isinstance(block[cat].get(k2), list):
                            block[cat][k2]=trim_list(block[cat][k2])

def generate_reliability_diagram(hist: dict, out_path: Path):
    if not GENERATE_RELIABILITY:
        return
    bin_size=CALIBRATION_BIN_SIZE
    bins=[round(i*bin_size,5) for i in range(int(1/bin_size)+1)]
    def bucket(p):
        for i in range(len(bins)-1):
            if bins[i] <= p < bins[i+1]:
                return f"{bins[i]:.2f}-{bins[i+1]:.2f}"
        return f"{bins[-2]:.2f}-{bins[-1]:.2f}"
    output={"generated_at": datetime.utcnow().isoformat(), "bin_size": bin_size, "categories": {}}
    
    # Adatok gyűjtése minden kategóriához
    plot_data = {}
    for cat in ("home","draw","away"):
        raw=hist.get(cat,{}).get("raw",[])
        outc=hist.get(cat,{}).get("outcome",[])
        if not raw or not outc or len(raw)!=len(outc): continue
        agg={}
        for p,o in zip(raw,outc):
            b=bucket(p)
            agg.setdefault(b, {"count":0,"sum":0.0})
            agg[b]["count"]+=1
            agg[b]["sum"]+=o
        diag=[]
        mid_probs = []
        empiricals = []
        for b,info in sorted(agg.items(), key=lambda x:x[0]):
            avg_obs=info["sum"]/info["count"] if info["count"]>0 else 0
            mid_prob = sum(float(x) for x in b.split("-"))/2
            diag.append({
                "bin": b,
                "mid_prob": mid_prob,
                "count": info["count"],
                "empirical": avg_obs
            })
            mid_probs.append(mid_prob)
            empiricals.append(avg_obs)
        output["categories"][cat]=diag
        plot_data[cat] = (mid_probs, empiricals)
    
    # JSON mentése
    safe_write_json(out_path, output)
    
    # PNG generálása matplotlib-tel
    if HAVE_MATPLOTLIB and plot_data:
        try:
            plt.figure(figsize=(10, 8))
            
            # Ideális kalibrációs vonal (y=x)
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Calibration')
            
            # Minden kategória külön színnel
            colors = {'home': 'blue', 'draw': 'green', 'away': 'red'}
            
            for cat, (mid_probs, empiricals) in plot_data.items():
                if mid_probs and empiricals:
                    plt.scatter(mid_probs, empiricals, 
                              label=f'{cat.capitalize()} predictions', 
                              color=colors.get(cat, 'gray'), 
                              alpha=0.7, s=50)
                    plt.plot(mid_probs, empiricals, 
                           color=colors.get(cat, 'gray'), 
                           alpha=0.5, linewidth=1)
            
            plt.xlabel('Predicted Probability')
            plt.ylabel('Empirical Frequency')
            plt.title('Reliability Diagram - Model Calibration')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            
            # PNG mentése
            png_path = out_path.with_suffix('.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Reliability diagram mentve: {png_path}")
            
        except Exception as e:
            logger.exception(f"Reliability diagram PNG generálás hiba: {e}")
    
    logger.info(f"Reliability diagram JSON mentve: {out_path}")

def update_calibration_history_from_results(cal_path: Path, picks: List[dict]):
    hist=load_calibration_history(cal_path)
    for cat in ("home","draw","away"):
        hist.setdefault(cat, {"raw":[],"outcome":[]})
    # ensemble block
    hist.setdefault("ensemble", {})
    for cat in ("home","draw","away"):
        hist["ensemble"].setdefault(cat, {"raw":[],"outcome":[]})
    updated=False
    for pick in picks:
        fid=pick.get("fixture_id")
        af=DATA_ROOT / f"out_fixture_{fid}" / "analysis.json"
        primary=DATA_ROOT / f"out_fixture_{fid}" / "primary_fixture.json"
        if not (af.exists() and primary.exists()): continue
        try:
            analysis=json.loads(af.read_text(encoding="utf-8"))
            primary_js=json.loads(primary.read_text(encoding="utf-8"))
            fixture=primary_js.get("fixture",{}) or {}
            status=fixture.get("status",{}).get("short")
            if status!="FT": continue
            score=fixture.get("score",{}) or {}
            full=score.get("fulltime",{}) or {}
            gh=full.get("home"); ga=full.get("away")
            if gh is None or ga is None: continue
            base_probs=analysis.get("model_probs", {})
            ens_probs=None
            if "enhanced_model" in analysis and "ensemble_probs" in analysis["enhanced_model"]:
                ens_probs=analysis["enhanced_model"]["ensemble_probs"]
            is_home=1 if gh>ga else 0
            is_draw=1 if gh==ga else 0
            is_away=1 if ga>gh else 0
            if all(cat in base_probs for cat in ("home","draw","away")):
                hist["home"]["raw"].append(base_probs["home"])
                hist["home"]["outcome"].append(is_home)
                hist["draw"]["raw"].append(base_probs["draw"])
                hist["draw"]["outcome"].append(is_draw)
                hist["away"]["raw"].append(base_probs["away"])
                hist["away"]["outcome"].append(is_away)
                if ens_probs:
                    hist["ensemble"]["home"]["raw"].append(ens_probs["home"])
                    hist["ensemble"]["home"]["outcome"].append(is_home)
                    hist["ensemble"]["draw"]["raw"].append(ens_probs["draw"])
                    hist["ensemble"]["draw"]["outcome"].append(is_draw)
                    hist["ensemble"]["away"]["raw"].append(ens_probs["away"])
                    hist["ensemble"]["away"]["outcome"].append(is_away)
                updated=True
        except:
            continue
    if updated:
        trim_calibration_history(hist)
        save_calibration_history(cal_path, hist)
        if GENERATE_RELIABILITY:
            reliability_path=Path("reliability_diagram.json")
            generate_reliability_diagram(hist, reliability_path)
        logger.info("Kalibráció history frissítve + trimming alkalmazva.")

# =========================================================
# Daily riport (változatlan)
# =========================================================
def generate_comprehensive_stats(analyzed_results: list[dict], output_path: Path):
    """
    Generate comprehensive statistics file with detailed match data, odds, 
    model-market percentages, value, confidence level, market strength, league classification
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_matches": len(analyzed_results),
        "summary": {
            "leagues_covered": len(set(r.get("league_id") for r in analyzed_results if r.get("league_id"))),
            "top_leagues": 0,
            "other_leagues": 0,
            "markets_analyzed": ["1X2", "BTTS", "O/U 2.5"]
        },
        "detailed_matches": []
    }
    
    for r in analyzed_results:
        # Basic match info
        match_data = {
            "fixture_id": r.get("fixture_id"),
            "league_id": r.get("league_id"),
            "league_name": r.get("league_name"),
            "league_tier": r.get("league_tier"),
            "is_top_league": LEAGUE_MANAGER.is_top(r.get("league_id")) if r.get("league_id") else False,
            "home_team": r.get("home_name"),
            "away_team": r.get("away_name"),
            "kickoff_utc": r.get("kickoff_utc"),
            "kickoff_local": format_local_time(r.get("kickoff_utc", "")),
            "markets": {}
        }
        
        # Update league counts
        if match_data["is_top_league"]:
            stats["summary"]["top_leagues"] += 1
        else:
            stats["summary"]["other_leagues"] += 1
        
        # 1X2 Market Analysis
        odds_1x2 = r.get("odds", {})
        edges_1x2 = r.get("edge", {})
        probs_1x2 = r.get("model_probs", {})
        
        if odds_1x2 and edges_1x2 and probs_1x2:
            market_strength_1x2 = calculate_market_strength(odds_1x2, "1X2")
            implied_probs = None
            try:
                inv = {k: 1/float(v) for k, v in odds_1x2.items() if k in ("home", "draw", "away")}
                s = sum(inv.values())
                if s > 0:
                    implied_probs = {k: inv[k]/s for k in inv}
            except:
                pass
            
            match_data["markets"]["1X2"] = {
                "market_strength": market_strength_1x2,
                "overround": sum(1/float(odds_1x2.get(k, 1)) for k in ("home", "draw", "away") if k in odds_1x2),
                "selections": {}
            }
            
            for sel in ("home", "draw", "away"):
                if sel in odds_1x2 and sel in edges_1x2 and sel in probs_1x2:
                    edge_val = edges_1x2[sel]
                    model_prob = probs_1x2[sel]
                    market_prob = implied_probs.get(sel, 0) if implied_probs else 0
                    odds_val = float(odds_1x2[sel])
                    
                    # Value calculation with market strength
                    value_score = edge_val * (1 + market_strength_1x2 / 100 * 0.1)
                    
                    # Confidence level
                    confidence = "Alacsony"
                    if edge_val >= 0.15:
                        confidence = "Magas"
                    elif edge_val >= 0.08:
                        confidence = "Közepes"
                    
                    match_data["markets"]["1X2"]["selections"][sel] = {
                        "odds": odds_val,
                        "model_probability": model_prob,
                        "market_probability": market_prob,
                        "edge": edge_val,
                        "value_score": value_score,
                        "confidence_level": confidence,
                        "raw_value": model_prob * odds_val - 1,
                        "probability_difference": model_prob - market_prob,
                        "meets_publish_threshold": edge_val >= PUBLISH_MIN_EDGE_TOP if match_data["is_top_league"] else edge_val >= PUBLISH_MIN_EDGE_OTHER
                    }
        
        # BTTS Market Analysis
        market_odds_btts = r.get("market_odds", {})
        market_edges_btts = r.get("market_edge", {})
        market_probs_btts = r.get("market_probs", {})
        
        if market_odds_btts and market_edges_btts and market_probs_btts:
            y_odds = market_odds_btts.get("yes")
            n_odds = market_odds_btts.get("no")
            
            if y_odds is not None and n_odds is not None:
                try:
                    y_odds, n_odds = float(y_odds), float(n_odds)
                    btts_odds = {"yes": y_odds, "no": n_odds}
                    market_strength_btts = calculate_market_strength(btts_odds, "BTTS")
                    
                    # Implied probabilities
                    ia, ib = 1/y_odds, 1/n_odds
                    s = ia + ib
                    implied_probs_btts = {"yes": ia/s, "no": ib/s} if s > 0 else {}
                    
                    match_data["markets"]["BTTS"] = {
                        "market_strength": market_strength_btts,
                        "overround": s,
                        "selections": {}
                    }
                    
                    for sel in ("yes", "no"):
                        if sel in market_edges_btts and sel in market_probs_btts:
                            edge_val = market_edges_btts[sel]
                            model_prob = market_probs_btts[sel]
                            market_prob = implied_probs_btts.get(sel, 0)
                            odds_val = y_odds if sel == "yes" else n_odds
                            
                            value_score = edge_val * (1 + market_strength_btts / 100 * 0.1)
                            
                            confidence = "Alacsony"
                            if edge_val >= 0.15:
                                confidence = "Magas"
                            elif edge_val >= 0.08:
                                confidence = "Közepes"
                            
                            match_data["markets"]["BTTS"]["selections"][sel] = {
                                "odds": odds_val,
                                "model_probability": model_prob,
                                "market_probability": market_prob,
                                "edge": edge_val,
                                "value_score": value_score,
                                "confidence_level": confidence,
                                "raw_value": model_prob * odds_val - 1,
                                "probability_difference": model_prob - market_prob,
                                "meets_publish_threshold": edge_val >= PUBLISH_MIN_EDGE_TOP if match_data["is_top_league"] else edge_val >= PUBLISH_MIN_EDGE_OTHER
                            }
                except:
                    pass
        
        # O/U 2.5 Market Analysis
        if market_odds_btts and market_edges_btts and market_probs_btts:
            ov_odds = market_odds_btts.get("over25")
            un_odds = market_odds_btts.get("under25")
            
            if ov_odds is not None and un_odds is not None:
                try:
                    ov_odds, un_odds = float(ov_odds), float(un_odds)
                    ou_odds = {"over": ov_odds, "under": un_odds}
                    market_strength_ou = calculate_market_strength(ou_odds, "O/U")
                    
                    # Implied probabilities
                    ia, ib = 1/ov_odds, 1/un_odds
                    s = ia + ib
                    implied_probs_ou = {"over25": ia/s, "under25": ib/s} if s > 0 else {}
                    
                    match_data["markets"]["O/U 2.5"] = {
                        "market_strength": market_strength_ou,
                        "overround": s,
                        "selections": {}
                    }
                    
                    for sel_raw in ("over25", "under25"):
                        if sel_raw in market_edges_btts and sel_raw in market_probs_btts:
                            edge_val = market_edges_btts[sel_raw]
                            model_prob = market_probs_btts[sel_raw]
                            market_prob = implied_probs_ou.get(sel_raw, 0)
                            odds_val = ov_odds if sel_raw == "over25" else un_odds
                            
                            value_score = edge_val * (1 + market_strength_ou / 100 * 0.1)
                            
                            confidence = "Alacsony"
                            if edge_val >= 0.15:
                                confidence = "Magas"
                            elif edge_val >= 0.08:
                                confidence = "Közepes"
                            
                            selection_name = "over" if sel_raw == "over25" else "under"
                            
                            match_data["markets"]["O/U 2.5"]["selections"][selection_name] = {
                                "odds": odds_val,
                                "model_probability": model_prob,
                                "market_probability": market_prob,
                                "edge": edge_val,
                                "value_score": value_score,
                                "confidence_level": confidence,
                                "raw_value": model_prob * odds_val - 1,
                                "probability_difference": model_prob - market_prob,
                                "meets_publish_threshold": edge_val >= PUBLISH_MIN_EDGE_TOP if match_data["is_top_league"] else edge_val >= PUBLISH_MIN_EDGE_OTHER
                            }
                except:
                    pass
        
        stats["detailed_matches"].append(match_data)
    
    # Sort matches by highest value scores across all markets
    for match in stats["detailed_matches"]:
        max_value = 0
        for market_data in match["markets"].values():
            for sel_data in market_data.get("selections", {}).values():
                max_value = max(max_value, sel_data.get("value_score", 0))
        match["max_value_score"] = max_value
    
    stats["detailed_matches"].sort(key=lambda x: x.get("max_value_score", 0), reverse=True)
    
    # Write to JSON file
    safe_write_json(output_path, stats)
    logger.info("Comprehensive statistics generated: %s", output_path)
    return output_path

def generate_daily_report(picks_file: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        data=json.loads(picks_file.read_text(encoding="utf-8"))
    except:
        logger.error("Daily report: picks fájl olvasási hiba: %s", picks_file)
        return
    picks=data.get("picks", [])
    stats={
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "total_picks": len(picks),
        "edge_buckets": {},
        "overall":{"wins":0,"losses":0,"roi":0.0,"stake_sum":0.0,"return_sum":0.0}
    }
    def bucket(edge):
        pct=edge*100
        if pct<5: return "0-5%"
        if pct<10: return "5-10%"
        if pct<20: return "10-20%"
        return "20%+"
    for p in picks:
        fid=p["fixture_id"]
        af=DATA_ROOT / f"out_fixture_{fid}" / "primary_fixture.json"
        if not af.exists(): continue
        try:
            js=json.loads(af.read_text(encoding="utf-8"))
            fixture=js.get("fixture",{}) or {}
            status=fixture.get("status",{}).get("short")
            if status!="FT": continue
            score=fixture.get("score",{}) or {}
            full=score.get("fulltime",{}) or {}
            gh=full.get("home"); ga=full.get("away")
            sel=p.get("selection")
            if gh is None or ga is None or not sel: continue
            stake=p.get("stake",1.0)
            odds=p.get("odds",0.0)
            stats["overall"]["stake_sum"]+=stake
            bucket_key=bucket(p.get("edge",0))
            stats["edge_buckets"].setdefault(bucket_key, {"picks":0,"wins":0,"losses":0,"stake":0.0,"ret":0.0,"roi":0.0})
            bd=stats["edge_buckets"][bucket_key]
            bd["picks"]+=1; bd["stake"]+=stake
            home_win= gh>ga; draw= gh==ga; away_win= ga>gh
            win=False
            if sel=="home" and home_win: win=True
            elif sel=="draw" and draw: win=True
            elif sel=="away" and away_win: win=True
            if win:
                stats["overall"]["wins"]+=1
                ret=stake*odds
                stats["overall"]["return_sum"]+=ret
                bd["wins"]+=1; bd["ret"]+=ret
            else:
                stats["overall"]["losses"]+=1
                bd["losses"]+=1
        except:
            continue
    def roi(ret,stk): return (ret-stk)/stk if stk>0 else 0
    for bk,bd in stats["edge_buckets"].items():
        bd["roi"]=roi(bd["ret"], bd["stake"])
    stats["overall"]["roi"]=roi(stats["overall"]["return_sum"], stats["overall"]["stake_sum"])
    json_path=out_dir / f"report_{stats['date']}.json"
    safe_write_json(json_path, stats)
    logger.info("Napi riport kész: %s", json_path)

# =========================================================
# Pipeline
# =========================================================
async def run_pipeline(fetch: bool, analyze: bool,
                       fixture_ids: list[int]|None=None,
                       limit: int=0,
                       cleanup_stale: bool=False,
                       refetch_missing: bool=False,
                       days_ahead_override: int|None=None,
                       reload_leagues: bool=False,
                       list_tiers: bool=False)->dict:
    load_state()
    new_fetched=[]; analyzed_res=[]; picks=[]
    if reload_leagues:
        await LEAGUE_MANAGER.fetch_and_classify()
    if list_tiers:
        logger.info("Tier összegzés: %s", LEAGUE_MANAGER.summarize_tiers())
    days_ahead = days_ahead_override if days_ahead_override is not None else GLOBAL_RUNTIME["fetch_days_ahead"]
    enhanced_tools={}
    if analyze and ENABLE_ENHANCED_MODELING:
        logger.info("Enhanced ON (Cal=%s Bayes=%s MC=%s)", ENABLE_CALIBRATION, ENABLE_BAYES, ENABLE_MC)
        if ENABLE_CALIBRATION:
            enhanced_tools["calibrators"]=init_calibrators(Path(CALIBRATION_HISTORY_FILE))
        if ENABLE_BAYES:
            bayes_model=BayesianAttackDefense()
            hist_matches=gather_bayes_history(DATA_ROOT, BAYES_HISTORY_DAYS)
            bayes_model.fit(hist_matches)
            enhanced_tools["bayes_model"]=bayes_model
    elif analyze:
        logger.info("Enhanced modeling OFF")
    tippmix_mapping={}
    tippmix_odds_cache={}
    if fetch:
        async with APIFootballClient(API_KEY, API_BASE) as client:
            if fixture_ids:
                fixture_objs=[]
                for fid in fixture_ids:
                    js=await client.get("/fixtures", {"id": fid})
                    resp=js.get("response") or []
                    if resp: fixture_objs.append(resp[0])
            else:
                fixture_objs=await fetch_upcoming_fixtures(client, days_ahead)
            logger.info("API-Football fixture jelöltek: %d (TOP_MODE=%s)", len(fixture_objs), TOP_MODE)
            if USE_TIPPMIX:
                logger.info("TippmixPro integráció – meccsek párosítása...")
                tipp_matches=await tippmix_fetch_and_map(TIPPMIX_DAYS_AHEAD)
                logger.info("TippmixPro MATCH rekordok: %d", len(tipp_matches))
                tip_index=[]
                for tm in tipp_matches.values():
                    home=tm.get("homeParticipantName") or ""
                    away=tm.get("awayParticipantName") or ""
                    start_ms=tm.get("startTime")
                    tip_index.append({
                        "match_id": tm.get("id"),
                        "home_n": normalize_team_name(home),
                        "away_n": normalize_team_name(away),
                        "start_ms": start_ms,
                        "raw": tm
                    })
                matched=[]; mapping_api_to_tip={}
                for fx in fixture_objs:
                    fixture=fx.get("fixture",{}) or {}
                    teams=fx.get("teams",{}) or {}
                    fid=fixture.get("id"); ts=fixture.get("timestamp")
                    if not (fid and ts and teams.get("home") and teams.get("away")): continue
                    home_name=teams["home"].get("name","")
                    away_name=teams["away"].get("name","")
                    hn=normalize_team_name(home_name)
                    an=normalize_team_name(away_name)
                    ts_ms=ts*1000
                    best=None; best_score=0
                    for tm in tip_index:
                        if tm["start_ms"] is None: continue
                        if abs(tm["start_ms"] - ts_ms) > TIPPMIX_TIME_TOLERANCE_MIN*60*1000: continue
                        sim_home=similarity(hn, tm["home_n"])
                        sim_away=similarity(an, tm["away_n"])
                        sim=(sim_home+sim_away)/2
                        if sim>best_score:
                            best_score=sim; best=tm
                    if best and best_score>=TIPPMIX_SIMILARITY_THRESHOLD:
                        mapping_api_to_tip[fid]=best["match_id"]
                        matched.append(fx)
                logger.info("Párosított fixturek: %d / %d (threshold=%.2f)", len(matched), len(fixture_objs), TIPPMIX_SIMILARITY_THRESHOLD)
                fixture_objs=matched
                tippmix_mapping=mapping_api_to_tip
            else:
                logger.info("TippmixPro integráció kikapcsolva (USE_TIPPMIX=0).")
            filtered=[]
            for fx in fixture_objs:
                status=fx.get("fixture",{}).get("status",{}).get("short")
                if status in ("NS","TBD","PST","CANC","SUSP"):
                    filtered.append(fx)
            if limit>0 and len(filtered)>limit:
                filtered=filtered[:limit]
            logger.info("Szűrt upcoming fixturek: %d", len(filtered))
            tasks=[asyncio.create_task(fetch_fixture_bundle(client, fx, DATA_ROOT)) for fx in filtered]
            if tasks: await asyncio.gather(*tasks)
            new_fetched=[fx.get("fixture",{}).get("id") for fx in filtered if fx.get("fixture",{}).get("id")]
            if USE_TIPPMIX and tippmix_mapping:
                extractor=TippmixOddsExtractor()
                async with TippmixProWampClient(verbose=False) as cli:
                    for api_fid, tip_mid in tippmix_mapping.items():
                        recs=await cli.fetch_match_markets_group(tip_mid, TIPPMIX_MARKET_GROUP)
                        std=extractor.extract(recs)
                        tippmix_odds_cache[api_fid]=std
                        await asyncio.sleep(0.05)
                GLOBAL_RUNTIME["tippmix_mapping"]=tippmix_mapping
                GLOBAL_RUNTIME["tippmix_odds_cache"]=tippmix_odds_cache
            else:
                GLOBAL_RUNTIME["tippmix_mapping"]={}
                GLOBAL_RUNTIME["tippmix_odds_cache"]={}
    tickets=None
    if analyze:
        if cleanup_stale:
            stale=find_stale_fixture_dirs(DATA_ROOT)
            for d in stale: shutil.rmtree(d, ignore_errors=True)
        if fixture_ids:
            targets=fixture_ids
        else:
            targets=[int(p.name.split("_")[-1]) for p in DATA_ROOT.glob("out_fixture_*") if (p/"summary.json").exists()]
        if refetch_missing and fetch:
            missing=[fid for fid in targets if not (DATA_ROOT/f"out_fixture_{fid}"/"summary.json").exists()]
            if missing:
                async with APIFootballClient(API_KEY, API_BASE) as client:
                    for fid in missing:
                        await refetch_single_fixture(client, fid, DATA_ROOT)
        valid=[fid for fid in targets if (DATA_ROOT/f"out_fixture_{fid}"/"summary.json").exists()]
        logger.info("Elemzendő fixture: %d", len(valid))
        for idx,fid in enumerate(valid,1):
            res=analyze_fixture(DATA_ROOT, fid, enhanced_tools)
            if res: analyzed_res.append(res)
            if idx%100==0:
                logger.info("Elemzés haladás: %d / %d", idx, len(valid))
        picks=allocate_stakes(analyzed_res)
        register_picks(picks)
        tickets=select_best_tickets(analyzed_res, only_today=TICKET_ONLY_TODAY)
        if tickets: save_ticket_full_analysis(tickets, DATA_ROOT)
        
        # Generate comprehensive statistics with all match details
        if analyzed_res:
            stats_path = DATA_ROOT / f"comprehensive_stats_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
            generate_comprehensive_stats(analyzed_res, stats_path)
            logger.info("Comprehensive statistics generated: %s", stats_path.name)
    RUNTIME_STATE["last_run"]=datetime.now(timezone.utc).isoformat()
    save_state()
    summary={
        "fetched": new_fetched,
        "analyzed_count": len(analyzed_res),
        "analyzed_results": analyzed_res,  # Add analyzed results for enhanced telegram
        "picks_count": len(picks),
        "picks": picks,
        "tickets": tickets,
        "top_mode": TOP_MODE,
        "use_tippmix": USE_TIPPMIX
    }
    picks_file=DATA_ROOT / f"picks_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    safe_write_json(picks_file, summary)
    GLOBAL_RUNTIME["last_summary"]=summary
    GLOBAL_RUNTIME["last_picks_file"]=picks_file.name
    logger.info("Futás kész. fetched=%d analyzed=%d picks=%d Tippmix=%s -> %s",
                len(new_fetched), len(analyzed_res), len(picks), USE_TIPPMIX, picks_file.name)
    return summary

# =========================================================
# Kalibrátor retrain + Bayes dataset export
# =========================================================
def retrain_calibrators(cal_path: Path):
    if not ENABLE_CALIBRATION:
        logger.info("Kalibráció nincs engedélyezve.")
        return
    hist=load_calibration_history(cal_path)
    out={}
    for key in ("home","draw","away"):
        cal=OneVsRestCalibrator(key)
        data=hist.get(key,{})
        raw=data.get("raw",[]); outcome=data.get("outcome",[])
        cal.fit(raw, outcome)
        out[key]={"samples": len(raw), "fitted": cal.fitted}
    logger.info("Kalibrátorok újratanítva: %s", out)

def export_bayes_dataset(path: Path, days: int):
    ds=gather_bayes_history(DATA_ROOT, days)
    out={"generated_at": datetime.utcnow().isoformat(), "sample_count": len(ds), "matches": ds}
    safe_write_json(path, out)
    logger.info("Bayes dataset exportálva: %s (count=%d)", path.name, len(ds))

# =========================================================
# Telegram Bot
# =========================================================
class TelegramBot:
    def __init__(self, token: str, chat_id: str | None, runtime: dict):
        self.token = token
        self.base = f"https://api.telegram.org/bot{token}"
        self.offset = 0
        self.default_chat_id = chat_id
        self._session: aiohttp.ClientSession | None = None
        self.runtime = runtime
    async def send(self, text: str, chat_id: str | None = None):
        if not chat_id:
            chat_id = self.default_chat_id
        if not chat_id:
            logger.warning("Nincs chat_id a Telegram üzenethez.")
            return
        data = {"chat_id": chat_id, "text": text[:4096]}
        try:
            if self._session:
                await self._session.post(self.base + "/sendMessage", data=data)
            else:
                async with aiohttp.ClientSession(timeout=ClientTimeout(total=30)) as s:
                    await s.post(self.base + "/sendMessage", data=data)
        except Exception:
            logger.exception("Telegram küldési hiba")
    async def long_poll(self):
        logger.info("Telegram long-poll indul.")
        backoff = 5
        max_backoff = 300
        timeout_s = 60
        async with aiohttp.ClientSession(timeout=ClientTimeout(total=timeout_s + 20)) as s:
            self._session = s
            while True:
                try:
                    params = {"timeout": timeout_s, "offset": self.offset + 1}
                    async with s.get(self.base + "/getUpdates", params=params) as resp:
                        js = await resp.json(content_type=None)
                    if not js.get("ok"):
                        await asyncio.sleep(3)
                        continue
                    backoff = 5
                    for upd in js.get("result", []):
                        self.offset = upd["update_id"]
                        msg = upd.get("message") or upd.get("edited_message")
                        if not msg:
                            continue
                        chat_id = str(msg["chat"]["id"])
                        if not self.default_chat_id:
                            self.default_chat_id = chat_id
                        text = (msg.get("text") or "").strip()
                        if text.startswith("/") and not text.startswith("/ "):
                            await self.handle_command(text, chat_id)
                except asyncio.CancelledError:
                    logger.info("Bot leállítva (cancel).")
                    break
                except Exception:
                    jitter = random.uniform(0, 1.0)
                    wait_s = min(backoff, max_backoff) + jitter
                    logger.exception("Polling hiba – %.1fs múlva újra.", wait_s)
                    await asyncio.sleep(wait_s)
                    backoff = min(backoff * 2, max_backoff)
    def format_ticket_message(self, enhanced_tickets: dict) -> str:
        """
        Unified ticket formatting function for consistent display in both /ticket and /run commands
        """
        def fmt_tip(entry, title):
            if not entry: return f"{title}:\n  Nincs ajánlás."
            
            # New enhanced telegram format
            market_emoji = {"1X2": "⚽", "BTTS": "🥅", "O/U 2.5": "📊"}
            market_hu = {"1X2": "1X2", "BTTS": "BTTS", "O/U 2.5": "O/U 2.5"}
            
            # Parse selection for Hungarian display
            selection = entry.get('selection', '')
            selection_hu = selection
            if 'home' in selection.lower():
                selection_hu = "Hazai győzelem"
            elif 'away' in selection.lower():
                selection_hu = "Vendég győzelem"
            elif 'draw' in selection.lower():
                selection_hu = "Döntetlen"
            elif selection.upper() == "YES":
                selection_hu = "Igen"
            elif selection.upper() == "NO":
                selection_hu = "Nem"
            elif "OVER" in selection.upper():
                selection_hu = "Felett 2.5"
            elif "UNDER" in selection.upper():
                selection_hu = "Alatt 2.5"
            
            # Confidence level based on edge value
            edge_val = entry.get('edge', 0)
            confidence = "Alacsony"
            if edge_val >= 0.15:
                confidence = "Magas"
            elif edge_val >= 0.08:
                confidence = "Közepes"
            
            # Market strength calculation (if available)
            market_strength = entry.get('market_strength', None)
            market_strength_str = ""
            if market_strength is not None:
                market_strength_str = f"\n💪 Piac-erő: {market_strength:.1f}%"
            
            # Model and market probabilities
            model_prob = entry.get('model_prob', 0) * 100
            market_prob = entry.get('market_prob', 0) * 100
            
            # Format kickoff time
            kickoff = entry.get('kickoff_local', entry.get('kickoff_utc', '?'))
            
            return (
                f"{market_emoji.get(title, '⚽')} {market_hu.get(title, title)} – {entry.get('league_name','?')}\n"
                f"{entry.get('home_name','?')} vs {entry.get('away_name','?')}\n"
                f"🕒 {kickoff}\n"
                f"🎯 Tipp: {selection_hu} @ {entry['odds']}\n"
                f"📊 Modell: {model_prob:.1f}% | Piac: {market_prob:.1f}%\n"
                f"📈 Érték: +{edge_val*100:.1f}%\n"
                f"🔒 Bizalom: {confidence}{market_strength_str}"
            )
        
        def fmt_enhanced_tips(entries, title):
            if not entries: 
                return f"🚫 {title}: Nincs ajánlás"
            
            results = []
            for i, entry in enumerate(entries[:2]):  # Max 2 tips per market
                result = fmt_tip(entry, title)
                if i > 0:  # Add separator for multiple tips
                    result = "─" * 25 + "\n" + result
                results.append(result)
            return "\n".join(results)
        
        msg_parts = [
            fmt_enhanced_tips(enhanced_tickets.get("x1x2", []), "1X2"),
            fmt_enhanced_tips(enhanced_tickets.get("btts", []), "BTTS"), 
            fmt_enhanced_tips(enhanced_tickets.get("overunder", []), "O/U 2.5")
        ]
        
        msg = "\n\n".join([part for part in msg_parts if "Nincs ajánlás" not in part])
        if not msg:
            msg = "🚫 Nincs tipp ma"
        
        return msg

    async def handle_command(self, text: str, chat_id: str):
        parts = text.split()
        cmd = parts[0].lower()
        args = parts[1:]
        if cmd in ("/help","/start"):
            await self.send(
                "Parancsok:\n"
                "/run | /run 1 | /run ids <idk>\n"
                "/ticket (/szelveny)\n"
                "/status\n"
                "/picks\n"
                "/limit <n>\n"
                "/setdays <n>\n"
                "/cleanup\n"
                "/refresh_tickets\n"
                "/dailyreport\n"
                "/mode <top_only|hybrid|all>\n"
                "/tiers | /leagues <minta> | /reloadtiers\n"
                "/updatecal | /retraincal | /exportbayes\n"
                "/tippmixstats\n"
                "/stop", chat_id)
        elif cmd == "/mode":
            if args and args[0] in ("top_only","hybrid","all"):
                global TOP_MODE
                TOP_MODE = args[0]
                await self.send(f"TOP_MODE beállítva: {TOP_MODE}", chat_id)
            else:
                await self.send(f"Jelenlegi TOP_MODE={TOP_MODE}. Használat: /mode top_only|hybrid|all", chat_id)
        elif cmd == "/tiers":
            counts = LEAGUE_MANAGER.summarize_tiers()
            lines = [f"🏆 Liga Statisztika (összes: {sum(counts.values())})"]
            lines.append("")
            
            # Tier alapú statisztika
            lines.append("📊 Tier alapú besorolás:")
            for k in sorted(counts.keys()):
                emoji = "🥇" if k == "TIER1" else "🥈" if k == "TIER1B" else "🏆" if "ELITE" in k else "⚽"
                lines.append(f"  {emoji} {k}: {counts[k]} liga")
            
            lines.append("")
            lines.append(f"🎯 TOP_MODE: {TOP_MODE}")
            
            # TOP liga információk tournaments.json-ból
            top_data = LEAGUE_MANAGER.top_league_data
            lines.append(f"🌟 TOP ligák (tournaments.json): {top_data['count']} név")
            
            if top_data['names']:
                lines.append("")
                lines.append("🏅 Azonosított TOP liga nevek:")
                for i, name in enumerate(top_data['names'][:10], 1):  # Max 10 név
                    lines.append(f"  {i}. {name}")
                if len(top_data['names']) > 10:
                    lines.append(f"  ... és még {len(top_data['names'])-10} további")
            
            await self.send("\n".join(lines), chat_id)
        elif cmd == "/leagues":
            if not args:
                await self.send("Használat: /leagues <részlet>", chat_id)
            else:
                pattern=" ".join(args)
                res=LEAGUE_MANAGER.search_leagues(pattern, limit=25)
                if not res:
                    await self.send("Nincs találat.", chat_id)
                else:
                    out=[]
                    for r in res:
                        out.append(f"{r['league_id']}: {r['name']} ({r['country']}) tier={r['tier']}")
                    await self.send("\n".join(out), chat_id)
        elif cmd == "/reloadtiers":
            await self.send("Ligák frissítése indul...", chat_id)
            try:
                await LEAGUE_MANAGER.fetch_and_classify()
                await self.send("Ligák újratöltve.", chat_id)
            except Exception as e:
                await self.send(f"Hiba a reload során: {e}", chat_id)
        elif cmd == "/limit" and args:
            try:
                val = int(args[0])
                self.runtime["fixture_limit"] = val
                await self.send(f"Fixture limit beállítva: {val}", chat_id)
            except:
                await self.send("Hibás szám.", chat_id)
        elif cmd == "/setdays" and args:
            try:
                val = int(args[0])
                self.runtime["fetch_days_ahead"] = val
                await self.send(f"FETCH_DAYS_AHEAD beállítva: {val}", chat_id)
            except:
                await self.send("Hibás szám.", chat_id)
        elif cmd == "/cleanup":
            stale = find_stale_fixture_dirs(DATA_ROOT)
            for d in stale:
                shutil.rmtree(d, ignore_errors=True)
            await self.send(f"Stale törölve: {len(stale)}", chat_id)
        elif cmd == "/status":
            summ = self.runtime.get("last_summary")
            if not summ:
                await self.send("Még nincs futás.", chat_id)
            else:
                tm_info=""
                if USE_TIPPMIX:
                    mp=self.runtime.get("tippmix_mapping") or {}
                    tm_info=f" | Tippmix matched={len(mp)}"
                await self.send(
                    f"Utolsó futás: fetched={len(summ['fetched'])} "
                    f"analyzed={summ['analyzed_count']} picks={summ['picks_count']} "
                    f"file={self.runtime.get('last_picks_file')} TOP_MODE={TOP_MODE}{tm_info}", chat_id)
        elif cmd == "/picks":
            summ = self.runtime.get("last_summary")
            if not summ:
                await self.send("Nincs pick adat.", chat_id)
            else:
                picks = summ.get("picks", [])
                if not picks:
                    await self.send("Nincsenek pickek.", chat_id)
                else:
                    lines=[]
                    for i,p in enumerate(picks[:30], start=1):
                        rat=p.get("rationale","")
                        if len(rat)>120: rat=rat[:120]+"..."
                        lines.append(f"{i}. FI#{p['fixture_id']} {p['selection']} @ {p['odds']} "
                                     f"stake={p['stake']} edge={p['edge']} tier={p.get('league_tier')} | {rat}")
                    if len(picks)>30: lines.append(f"... összesen {len(picks)}")
                    await self.send("\n".join(lines), chat_id)
        elif cmd in ("/ticket","/szelveny"):
            # Use enhanced ticket selection for best value/edge tips (consistent with /run command)
            summ = self.runtime.get("last_summary")
            if summ and summ.get("analyzed_results"):
                enhanced_tickets = select_best_tickets_enhanced(summ.get("analyzed_results"), only_today=True, max_tips_per_market=1)
            else:
                enhanced_tickets = {"x1x2": [], "btts": [], "overunder": []}
            
            # Use unified formatting function
            msg = self.format_ticket_message(enhanced_tickets)
            await self.send(msg, chat_id)
        elif cmd == "/refresh_tickets":
            tickets = build_offline_tickets(DATA_ROOT)
            if tickets:
                save_ticket_full_analysis(tickets, DATA_ROOT)
            def lab(entry,t):
                if not entry: return f"{t}: nincs."
                return f"{t}: FI#{entry['fixture_id']} {entry.get('home_name','?')} vs {entry.get('away_name','?')} {entry.get('selection_label', entry['selection'])} @ {entry['odds']} edge={entry['edge']}"
            await self.send("\n".join([
                lab(tickets.get("x1x2"),"1X2"),
                lab(tickets.get("btts"),"BTTS"),
                lab(tickets.get("overunder"),"O/U 2.5")
            ]), chat_id)
        elif cmd == "/dailyreport":
            picks_files=sorted([p for p in DATA_ROOT.glob("picks_*.json")])
            if not picks_files:
                await self.send("Nincs picks fájl.", chat_id)
            else:
                latest=picks_files[-1]
                out_dir=Path(DAILY_REPORT_DIR)
                generate_daily_report(latest, out_dir)
                await self.send(f"Napi riport generálva: {latest.name}", chat_id)
        elif cmd == "/updatecal":
            picks_files=sorted([p for p in DATA_ROOT.glob("picks_*.json")])
            if not picks_files:
                await self.send("Nincs picks fájl kalibráció frissítéshez.", chat_id)
            else:
                latest=picks_files[-1]
                try:
                    summ=json.loads(latest.read_text(encoding="utf-8"))
                    update_calibration_history_from_results(Path(CALIBRATION_HISTORY_FILE), summ.get("picks",[]))
                    await self.send("Kalibráció history frissítve.", chat_id)
                except Exception as e:
                    await self.send(f"Hiba: {e}", chat_id)
        elif cmd == "/retraincal":
            retrain_calibrators(Path(CALIBRATION_HISTORY_FILE))
            await self.send("Kalibrátorok retrain lefutott.", chat_id)
        elif cmd == "/exportbayes":
            export_bayes_dataset(Path("bayes_dataset.json"), BAYES_HISTORY_DAYS)
            await self.send("Bayes dataset exportálva (bayes_dataset.json).", chat_id)
        elif cmd == "/tippmixstats":
            if not USE_TIPPMIX:
                await self.send("USE_TIPPMIX=0 – Tippmix integráció inaktív.", chat_id)
            else:
                mp=self.runtime.get("tippmix_mapping") or {}
                oc=self.runtime.get("tippmix_odds_cache") or {}
                await self.send(
                    f"Tippmix integráció:\n"
                    f"  Párosított fixturek: {len(mp)}\n"
                    f"  Odds cache entries: {len(oc)}\n"
                    f"  THRESHOLD={TIPPMIX_SIMILARITY_THRESHOLD} TIME_TOL={TIPPMIX_TIME_TOLERANCE_MIN}min\n"
                    f"  MarketGroup={TIPPMIX_MARKET_GROUP}", chat_id)
        elif cmd == "/run":
            fixture_ids=None
            days_override=None
            if args:
                if args[0]=="ids" and len(args)>1:
                    fids=[]
                    for a in args[1:]:
                        if a.isdigit(): fids.append(int(a))
                    fixture_ids=fids
                elif args[0].isdigit():
                    days_override=int(args[0])
            await self.send(f"Futás indult (fetch+analyze) TOP_MODE={TOP_MODE} USE_TIPPMIX={USE_TIPPMIX}...", chat_id)
            try:
                summary = await run_pipeline(
                    fetch=True,
                    analyze=True,
                    fixture_ids=fixture_ids,
                    limit=self.runtime["fixture_limit"],
                    cleanup_stale=False,
                    refetch_missing=False,
                    days_ahead_override=days_override
                )
                self.runtime["last_summary"] = summary
                
                # Send pipeline status summary first
                await self.send(
                    f"Kész: fetched={len(summary['fetched'])} analyzed={summary['analyzed_count']} "
                    f"picks={summary['picks_count']} TOP_MODE={TOP_MODE}", chat_id)
                
                # Automatically generate and send daily ticket recommendations
                if summary.get("analyzed_results"):
                    enhanced_tickets = select_best_tickets_enhanced(
                        summary.get("analyzed_results"), 
                        only_today=True, 
                        max_tips_per_market=1  # Use max 1 tip per market for best value
                    )
                    
                    # Check if we have any tickets to send
                    has_tickets = any(enhanced_tickets.get(market, []) for market in ["x1x2", "btts", "overunder"])
                    
                    if has_tickets:
                        # Send a header message for the daily recommendations
                        await self.send("📊 NAPI SZELVÉNY AJÁNLÁSOK (Legjobb value/edge tippek)", chat_id)
                        
                        # Send the formatted ticket recommendations
                        ticket_msg = self.format_ticket_message(enhanced_tickets)
                        await self.send(ticket_msg, chat_id)
                    else:
                        await self.send("📭 Nincs napi szelvény ajánlás - nem található megfelelő value tipp", chat_id)
                else:
                    await self.send("⚠️ Nincs elemzési eredmény - szelvény nem generálható", chat_id)
                    
            except Exception as e:
                logger.exception("Run hiba (telegram)")
                await self.send(f"Hiba: {e}", chat_id)

        elif cmd == "/stop":
            await self.send("Leállítás kérve – viszlát!", chat_id)
            raise KeyboardInterrupt()

        else:
            await self.send("Ismeretlen parancs. /help", chat_id)


# =========================================================
# CLI parse
# =========================================================
def parse_args():
    ap=argparse.ArgumentParser(description="Integrált Betting Bot (API-Football + TippmixPro + Telegram)")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--fixture-ids", nargs="*", type=int)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cleanup-stale", action="store_true")
    ap.add_argument("--refetch-missing", action="store_true")
    ap.add_argument("--days-ahead", type=int)
    ap.add_argument("--reload-leagues", action="store_true")
    ap.add_argument("--list-tiers", action="store_true")
    ap.add_argument("--daily-report", action="store_true")
    ap.add_argument("--update-calibration", action="store_true")
    ap.add_argument("--retrain-calibration", action="store_true")
    ap.add_argument("--export-bayes-dataset", action="store_true")
    ap.add_argument("--telegram-bot", action="store_true")
    ap.add_argument("--watch-odds", action="store_true", help="Odds watcher indítása a Telegram bot mellett")
    return ap.parse_args()


# =========================================================
# MAIN
# =========================================================
def main():
    import sys
    logger.info(">>> MAIN START | USE_TIPPMIX=%s | TOP_MODE=%s | argv=%s", USE_TIPPMIX, TOP_MODE, sys.argv)
    args=parse_args()

    if args.reload_leagues:
        asyncio.run(LEAGUE_MANAGER.fetch_and_classify())

    if args.list_tiers:
        logger.info("Tier összegzés: %s", LEAGUE_MANAGER.summarize_tiers())

    if args.daily_report:
        picks_files=sorted([p for p in DATA_ROOT.glob("picks_*.json")])
        if picks_files:
            generate_daily_report(picks_files[-1], Path(DAILY_REPORT_DIR))
        else:
            logger.info("Nincs picks fájl napi riporthoz.")

    if args.update_calibration:
        picks_files=sorted([p for p in DATA_ROOT.glob("picks_*.json")])
        if picks_files:
            latest=picks_files[-1]
            try:
                summ=json.loads(latest.read_text(encoding="utf-8"))
                update_calibration_history_from_results(Path(CALIBRATION_HISTORY_FILE), summ.get("picks",[]))
            except Exception:
                logger.exception("Calibration update hiba.")
        else:
            logger.info("Nincs picks fájl kalibrációhoz.")

    if args.retrain_calibration:
        retrain_calibrators(Path(CALIBRATION_HISTORY_FILE))

    if args.export_bayes_dataset:
        export_bayes_dataset(Path("bayes_dataset.json"), BAYES_HISTORY_DAYS)

    # Telegram Bot + (opcionális) Watcher mód
    if args.telegram_bot:
        if TELEGRAM_BOT_TOKEN.startswith("REPLACE_ME"):
            logger.error("TELEGRAM_BOT_TOKEN nincs beállítva – bot nem indítható.")
            return

        async def bot_main():
            load_state()
            bot = TelegramBot(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, GLOBAL_RUNTIME)
            stop_event = asyncio.Event()

            # Odds watcher indítás – ha kérve vagy ENV szerint
            watch_flag = getattr(args, "watch_odds", False) or USE_ODDS_WATCH
            watcher_task = None
            if watch_flag and USE_TIPPMIX:
                logger.info("Odds watcher indul (watch_flag=%s, interval=%ds)", watch_flag, WATCH_INTERVAL_SEC)
                watcher_task = asyncio.create_task(watch_tippmix_odds(stop_event))
            else:
                logger.info("Odds watcher inaktív (flag/env alapján).")

            try:
                await bot.long_poll()
            finally:
                stop_event.set()
                if watcher_task:
                    try:
                        await watcher_task
                    except Exception:
                        logger.exception("Watcher leállítás hiba.")

        try:
            asyncio.run(bot_main())
        except KeyboardInterrupt:
            logger.info("Bot leállítva (KeyboardInterrupt).")
        return

    # Ha nincs fetch/analyze és nem bot
    if not (args.fetch or args.analyze):
        logger.info("Adj meg legalább egyet: --fetch vagy --analyze (vagy --telegram-bot).")
        return

    if args.limit:
        GLOBAL_RUNTIME["fixture_limit"]=args.limit
    if args.days_ahead is not None:
        GLOBAL_RUNTIME["fetch_days_ahead"]=args.days_ahead

    try:
        asyncio.run(run_pipeline(
            fetch=args.fetch,
            analyze=args.analyze,
            fixture_ids=args.fixture_ids,
            limit=GLOBAL_RUNTIME["fixture_limit"],
            cleanup_stale=args.cleanup_stale,
            refetch_missing=args.refetch_missing,
            days_ahead_override=args.days_ahead,
            reload_leagues=args.reload_leagues,
            list_tiers=args.list_tiers
        ))
    except KeyboardInterrupt:
        logger.info("Megszakítva felhasználó által.")


# ====== VÉGPONT =======
if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Főprogram hiba – kilépés")
        raise