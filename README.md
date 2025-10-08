"# Betting Analysis System

Automatizált fogadási elemző rendszer amely valós idejű statisztikák és predikciók alapján értékel mérkőzéseket az API-Football és további adatforrások felhasználásával.

## Főbb Jellemzők

- **Valós forma számítás**: A csapatok múltbéli mérkőzéseit lekéri az API-ból, nem a szezonstatisztikákra támaszkodik
- **Lambda alapú predikció**: Poisson-eloszlás alapú gólszám becslés a csapatok tényleges teljesítménye alapján
- **1X2, BTTS és Over/Under piacok**: Többféle fogadási piac elemzése
- **Edge számítás**: A modell és piaci odds közötti különbség azonosítása
- **Enhanced modeling**: Kalibrált, Bayes-i és Monte Carlo alapú ensemble előrejelzés
- **TippmixPro integráció**: Magyar Tippmix odds használata ha elérhető

## Telepítés

### Követelmények

- Python 3.10+
- API-Football kulcs (https://api-sports.io/)
- Opcionális: Odds API kulcs további odds adatokhoz

### Függőségek telepítése

```bash
pip install aiohttp pillow
# Opcionális enhanced modeling-hoz:
pip install numpy scikit-learn pymc matplotlib
```

### Környezeti változók

Hozz létre egy `.env` fájlt vagy állítsd be ezeket a változókat:

```bash
# Kötelező
API_FOOTBALL_KEY=your_api_key_here

# Opcionális
ODDS_API_KEY=your_odds_api_key
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Konfiguráció
FETCH_DAYS_AHEAD=3              # Hány napra előre töltse le a mérkőzéseket
HOME_ADV=0.20                   # Hazai pálya előny lambda számításnál
BANKROLL_DAILY=1000             # Napi bankroll
LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR
```

## Használat

### Alapvető használat

```bash
# Mérkőzések letöltése és elemzése
python betting.py --fetch --analyze

# Csak elemzés meglévő adatokon
python betting.py --analyze

# Specifikus fixture-ök elemzése
python betting.py --analyze --fixture-ids 12345,67890

# Liga besorolások újratöltése
python betting.py --reload-leagues
```

### Fejlett opciók

```bash
# Elavult fixture mappák törlése
python betting.py --cleanup-stale

# Hiányzó fixture-ök újra-letöltése
python betting.py --fetch --refetch-missing

# Napok száma felülírása
python betting.py --fetch --days-ahead 7

# Korlátozott számú mérkőzés elemzése
python betting.py --fetch --analyze --limit 50
```

## Adatbázis Inicializálás

A rendszer automatikusan létrehozza a szükséges adatstruktúrákat:

1. **data/**: Fixture adatok és elemzési eredmények
2. **config/**: Konfiguráció fájlok (leagues_tiers.yaml)
3. **daily_reports/**: Napi riportok
4. **calibration_history.json**: Kalibráció történet
5. **runtime_state.json**: Futási állapot

Első futtatás:

```bash
# Ligák osztályozásának letöltése és mérkőzések lekérése
python betting.py --reload-leagues --fetch --analyze
```

## Kimenet Magyarázata

### Elemzési JSON struktúra

```json
{
  "fixture_id": 12345,
  "kickoff_utc": "2024-01-15T18:00:00+00:00",
  "league_id": 39,
  "league_name": "Premier League",
  "model_probs": {
    "home": 0.45,
    "draw": 0.30,
    "away": 0.25
  },
  "lambda_home": 1.65,
  "lambda_away": 1.20,
  "market_probs": {
    "btts_yes": 0.55,
    "btts_no": 0.45,
    "over25": 0.60,
    "under25": 0.40
  },
  "home_recent": {
    "form_string": "WWDLW",
    "goals_per_match": 1.8,
    "goals_against_per_match": 1.2,
    "matches_count": 5
  },
  "away_recent": {
    "form_string": "LDWWL",
    "goals_per_match": 1.4,
    "goals_against_per_match": 1.6,
    "matches_count": 5
  }
}
```

### Mezők jelentése

#### Model Probs (1X2)
- **home**: Hazai győzelem valószínűsége a modell szerint
- **draw**: Döntetlen valószínűsége
- **away**: Vendég győzelem valószínűsége

#### Lambda értékek
- **lambda_home**: Hazai csapat várható gólszáma (Poisson λ paraméter)
- **lambda_away**: Vendég csapat várható gólszáma
- Magasabb lambda = több várható gól

#### Market Probs (BTTS / Over Under)
- **btts_yes**: Both Teams To Score - mindkét csapat szerez gólt
- **btts_no**: Legalább egy csapat nem szerez gólt
- **over25**: Több mint 2.5 gól a mérkőzésen
- **under25**: 2.5 vagy kevesebb gól

#### Recent Stats (Forma)
- **form_string**: W (győzelem), D (döntetlen), L (vereség) az utolsó 5 meccsből
- **goals_per_match**: Átlagos szerzett gólok
- **goals_against_per_match**: Átlagos kapott gólok
- **matches_count**: Elemzett mérkőzések száma

#### Edge és Kelly
- **edge**: Várható profit % (model_prob × odds - 1)
  - Pozitív edge = értékfogadás lehetőség
- **kelly**: Kelly kritérium szerinti ajánlott tét százalék

## Működési Elv

### 1. Adatgyűjtés (DataCollector)

Az `APIFootballClient` lekéri:
- Mérkőzés alapadatok (csapatok, időpont, liga)
- Csapat statisztikák (gólátlag, forma)
- **Új**: Legutóbbi 10 lezárt mérkőzés csapatonként
- Odds adatok (1X2, BTTS, O/U 2.5)
- Játékos adatok (topscorer, sérülések)

A `get_team_recent_fixtures()` metódus cache-eli a lekérdezéseket a rate limit miatt.

### 2. Forma számítás (Analyzer)

A `calculate_last_5_form()` függvény:
1. Veszi a csapat utolsó 10 lezárt meccsét
2. Külön számol home/away helyszínre
3. Meghatározza a formát (W/D/L string)
4. Kiszámolja a gólátlagokat (goals_per_match, goals_against_per_match)

**Előny az előző verzióhoz képest**: Most valós múltbéli meccseket használ, nem csak az aznapi fixture listát.

### 3. Lambda számítás (Predictor)

```python
lambda_home = goals_per_match_home * (1 + HOME_ADV)
lambda_away = goals_per_match_away
```

- A hazai pálya előny (HOME_ADV, alapértelmezetten 0.20) növeli a hazai lambda értékét
- Ezek a lambdák kerülnek be a Poisson-eloszlásba

### 4. Valószínűség számítás

**1X2 piac**: Logisztikus regresszió a csapat rating-ek alapján

**BTTS**: Poisson alapú számítás
```python
P(BTTS=Yes) = P(Home > 0) × P(Away > 0)
```

**Over/Under 2.5**: Összesített lambda Poisson eloszlás
```python
P(Over 2.5) = P(Total Goals > 2)
```

### 5. Enhanced Modeling (Opcionális)

Ha engedélyezve:
- **Calibration**: Platt és Isotonic kalibrálás korábbi eredmények alapján
- **Bayesian**: Hierarchikus Poisson modell csapat attack/defense paraméterekkel
- **Monte Carlo**: Több ezer szimuláció Poisson eloszlásból

Az ensemble ezeket súlyozva kombinálja (config: ENSEMBLE_WEIGHTS).

## Konfigurációs Lehetőségek

### Alapvető Paraméterek

| Változó | Alapértelmezett | Leírás |
|---------|----------------|--------|
| `HOME_ADV` | 0.20 | Hazai pálya előny százalék |
| `FORM_WEIGHT` | 0.30 | Forma súlya a rating számításban |
| `ATK_WEIGHT` | 0.35 | Támadás súlya |
| `DEF_WEIGHT` | 0.20 | Védelem súlya |

### Edge és Margin Szűrők

| Változó | Alapértelmezett | Leírás |
|---------|----------------|--------|
| `MIN_EDGE_THRESHOLD` | 0.03 | Minimum edge (3%) a publikáláshoz |
| `MAX_MARGIN_1X2` | 1.10 | Maximum margin 1X2 piacon |
| `MAX_MARGIN_2WAY` | 1.10 | Maximum margin 2-way piacokon |

### Liga Szűrés

| Változó | Alapértelmezett | Leírás |
|---------|----------------|--------|
| `ENABLE_TIER_FILTERING` | 1 | Liga tier alapú szűrés |
| `TOP_MODE` | "all" | TOP ligák módja |

## Hibaelhárítás

### "No recent fixtures found"
- Ellenőrizd az API kulcsod érvényességét
- Lehet, hogy a csapatnak nincs múltbéli mérkőzése
- A rendszer fallback értékeket használ ilyenkor

### "Rate limit exceeded"
- A cache csökkenti az API hívásokat
- Növeld a `REQUEST_TIMEOUT` értékét
- Csökkentsd a `PARALLEL_CONNECTIONS` számát

### "Form string is empty (N/A)"
- Most már nem kellene előfordulnia az új implementációval
- Ha mégis, a csapatnak lehet, hogy nincsenek lezárt mérkőzései

## Fejlesztés és Tesztelés

```bash
# Szintaxis ellenőrzés
python3 -m py_compile betting.py

# Debug mód
LOG_LEVEL=DEBUG python betting.py --analyze --fixture-ids 12345

# Dry-run specifikus fixture-ökön
python betting.py --analyze --fixture-ids 12345,67890 --limit 2
```

## Jogi Nyilatkozat

Ez a szoftver csak oktatási és elemzési célokra készült. A fogadás kockázattal jár. Mindig felelősségteljesen fogadj és soha ne fogadj többet, mint amennyit megengedhetsz magadnak elveszíteni.

## Licenc

MIT License - lásd a LICENSE fájlt a részletekért.

## Közreműködés

Pull request-ek és issue-k várva! Kérjük tartsd be a meglévő kódstílust és add hozzá a megfelelő teszteket." 
