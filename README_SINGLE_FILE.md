# Betting Predictor - Napi Futball Előrejelző Rendszer

Egyszerű, single-file implementáció futball mérkőzések napi előrejelzéséhez Poisson-alapú modellel.

## Funkciók

- **Adatgyűjtés**: API-Football v3 integráció mai mérkőzések lekéréséhez
- **Elemzés**: Feature engineering - utolsó 5 meccs forma és gólátlagok
- **Predikció**: Poisson-alapú valószínűség számítás, score distribution, 1X2 kimenetel
- **Adatbázis**: SQLite + SQLAlchemy a predikciók tárolásához
- **API**: FastAPI endpoint a predikciók JSON formátumban való lekéréséhez

## Telepítés

### 1. Repository klónozása

```bash
git clone https://github.com/kiss96dani/betting.git
cd betting
```

### 2. Python környezet létrehozása

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# vagy
venv\Scripts\activate  # Windows
```

### 3. Függőségek telepítése

```bash
pip install -r requirements.txt
```

### 4. API kulcs beállítása

Szerezz egy ingyenes API kulcsot: https://www.api-football.com/

Másold le a `.env.example` fájlt `.env` néven, és add meg az API kulcsot:

```bash
cp .env.example .env
```

Szerkeszd a `.env` fájlt:

```
API_FOOTBALL_KEY=ide_jon_a_te_kulcsod
DATABASE_URL=sqlite:///predictions.db
REQUEST_TIMEOUT=15
MAX_RETRIES=3
```

**Fontos**: Ne commitold a `.env` fájlt a git repository-ba!

## Használat

### Adatbázis inicializálás

Első futtatás előtt inicializáld az adatbázist:

```bash
python main.py --mode init-db
```

### FastAPI szerver indítása

```bash
python main.py --mode server --host 0.0.0.0 --port 8000
```

Vagy használd közvetlenül az uvicorn-t:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

A szerver elérhető lesz: http://localhost:8000

### API Endpointok

- `GET /` - Root endpoint, API információk
- `GET /health` - Health check
- `GET /predictions` - Mai mérkőzések predikcióinak lekérése

Példa válasz:

```json
{
  "date": "2024-01-15",
  "total_predictions": 5,
  "predictions": [
    {
      "fixture_id": 123456,
      "home_team": "Manchester United",
      "away_team": "Liverpool",
      "date": "2024-01-15T20:00:00+00:00",
      "league": "Premier League",
      "prediction": {
        "home_lambda": 1.5,
        "away_lambda": 1.2,
        "probabilities_1x2": {
          "home_win": 0.45,
          "draw": 0.30,
          "away_win": 0.25
        },
        "top_scorelines": [
          {"scoreline": "1-1", "probability": 0.22},
          {"scoreline": "2-1", "probability": 0.18}
        ],
        "confidence": "Közepes",
        "explanation": "A modell Manchester United győzelem kimenetelt valószínűsíti..."
      }
    }
  ]
}
```

### Egyszeri predikció futtatás

Mai mérkőzésekre:

```bash
python main.py --mode predict
```

Konkrét dátumra:

```bash
python main.py --mode predict --date 2024-01-15
```

## Automatizálás - Napi futás

### Linux/Mac - Cron beállítás

Szerkeszd a crontab-ot:

```bash
crontab -e
```

Add hozzá a következő sort (minden nap reggel 6 órakor fut):

```
0 6 * * * cd /path/to/betting && /path/to/venv/bin/python main.py --mode predict >> /tmp/betting_predictor.log 2>&1
```

### Windows - Task Scheduler

1. Nyisd meg a Task Scheduler-t
2. Create Basic Task
3. Trigger: Daily, 06:00
4. Action: Start a program
   - Program: `C:\path\to\venv\Scripts\python.exe`
   - Arguments: `main.py --mode predict`
   - Start in: `C:\path\to\betting`

## Fejlesztés

### Tesztek futtatása

```bash
pytest tests/
```

### Kód struktúra

A `main.py` tartalmazza az összes komponenst:

- **Config**: Környezeti változók kezelése
- **DataCollector**: API-Football v3 kliens, retry/backoff logika
- **Analyzer**: Feature engineering, forma számítás
- **Predictor**: Poisson-modell, score distribution, 1X2 aggregáció
- **PredictionModel**: SQLAlchemy adatbázis model
- **FastAPI app**: REST API endpoint

## Hibaelhárítás

### "API_FOOTBALL_KEY nincs beállítva"

Ellenőrizd, hogy a `.env` fájl létezik és tartalmazza az API kulcsot.

### "ModuleNotFoundError"

Telepítsd újra a függőségeket:

```bash
pip install -r requirements.txt
```

### API rate limit

Az ingyenes API-Football kulcs korlátozott (100 kérés/nap). A DataCollector automatikusan kezeli a retry logikát.

## Licenc

MIT

## Hozzájárulás

Pull request-ek és issue-k welcome!
