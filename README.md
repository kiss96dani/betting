"# Betting Predictor

Futball mérkőzés prediktor rendszer támogatással BTTS (Both Teams To Score) és Over/Under piacokhoz.

## Funkciók

- **1X2 predikciók**: Hazai győzelem / Döntetlen / Vendég győzelem valószínűségek
- **BTTS predikciók**: Both Teams To Score (mindkét csapat gólt szerez) valószínűségek
- **Over/Under predikciók**: Több küszöbhöz (0.5, 1.5, 2.5, 3.5 gól)
- **Enhanced modeling**: Bayesi statisztika, Monte Carlo szimuláció, kalibráció
- **Automatikus adatgyűjtés**: API-Football integráció
- **Telegram bot**: Napi tippek és riportok

## Telepítés

```bash
# Klónozás
git clone https://github.com/kiss96dani/betting.git
cd betting

# Függőségek telepítése (példa)
pip install aiohttp Pillow pyyaml
# Optional: pymc arviz scikit-learn numpy pandas

# API kulcs beállítása
export API_FOOTBALL_KEY="your_api_key_here"
```

## Használat

### Alapvető predikció futtatása

```bash
# Fetch upcoming fixtures and analyze them
python3 betting.py --fetch --analyze

# Analyze specific fixture
python3 betting.py --analyze --fixture-ids 1234567

# Run with custom days ahead
python3 betting.py --fetch --analyze --days-ahead 5
```

### Tesztek futtatása

```bash
# Unit tesztek (Poisson, BTTS, Over/Under)
python3 tests/test_predictor.py -v

# Vagy pytest-tel (ha telepítve van)
pytest tests/ -v
```

## API Output Példák

### BTTS Prediction

```json
{
  "btts": {
    "btts_yes": 0.5428,
    "btts_no": 0.4572,
    "balanced": "yes"
  }
}
```

### Over/Under Predictions

```json
{
  "over_under": {
    "0.5": {
      "over": 0.9502,
      "under": 0.0498,
      "balanced": "over",
      "probabilities": {
        "over": 0.9502,
        "under": 0.0498
      }
    },
    "1.5": {
      "over": 0.7408,
      "under": 0.2592,
      "balanced": "over",
      "probabilities": {
        "over": 0.7408,
        "under": 0.2592
      }
    },
    "2.5": {
      "over": 0.4562,
      "under": 0.5438,
      "balanced": "under",
      "probabilities": {
        "over": 0.4562,
        "under": 0.5438
      }
    },
    "3.5": {
      "over": 0.2378,
      "under": 0.7622,
      "balanced": "under",
      "probabilities": {
        "over": 0.2378,
        "under": 0.7622
      }
    }
  }
}
```

### Teljes Prediction Output

```json
{
  "fixture_id": 1234567,
  "kickoff_utc": "2025-01-15T19:00:00+00:00",
  "league_id": 39,
  "league_name": "Premier League",
  "model_probs": {
    "home": 0.4523,
    "draw": 0.2845,
    "away": 0.2632
  },
  "lambda_home": 1.5,
  "lambda_away": 1.2,
  "btts": {
    "btts_yes": 0.5428,
    "btts_no": 0.4572,
    "balanced": "yes"
  },
  "over_under": {
    "0.5": {
      "over": 0.9502,
      "under": 0.0498,
      "balanced": "over"
    },
    "1.5": {
      "over": 0.7408,
      "under": 0.2592,
      "balanced": "over"
    },
    "2.5": {
      "over": 0.4562,
      "under": 0.5438,
      "balanced": "under"
    },
    "3.5": {
      "over": 0.2378,
      "under": 0.7622,
      "balanced": "under"
    }
  }
}
```

## Környezeti Változók

```bash
# API kulcsok
export API_FOOTBALL_KEY="your_api_key"
export TELEGRAM_BOT_TOKEN="your_telegram_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# Predikció beállítások
export HOME_ADV=0.20              # Hazai pálya előny
export KELLY_FRACTION_LIMIT=0.25  # Kelly kritérium limit
export FORM_WEIGHT=0.30           # Forma súlya
export ATK_WEIGHT=0.35            # Támadóerő súlya
export DEF_WEIGHT=0.20            # Védelem súlya

# Enhanced modeling
export ENABLE_ENHANCED_MODELING=1 # Enhanced modellek engedélyezése
export ENABLE_CALIBRATION=1       # Kalibráció engedélyezése
export ENABLE_BAYES=1             # Bayesi modell engedélyezése
export ENABLE_MC=1                # Monte Carlo szimuláció engedélyezése
export MC_SIMS=12000              # Monte Carlo szimulációk száma
```

## Fejlesztési Backlog

### Prioritás A: Lambda becslés javítása
- [ ] Utolsó 5-10 meccs gólátlaga külön home/away
- [ ] Liga-normalizáció implementálása
- [ ] Home-advantage faktor finomítása
- [ ] Csapat formák súlyozása idővel

### Prioritás B: Attack/Defense Poisson modell
- [ ] Strukturált attack/defense paraméterek
- [ ] Regressziós becslés implementálása
- [ ] Historikus adatok használata tanításhoz

### Prioritás C: Confidence-kalibráció
- [ ] Entrópia számítás predikciókhoz
- [ ] Top1-top2 arány metrikák
- [ ] Kalibrációs metrikák valós eredményekkel
- [ ] Confidence szintek visszajelzése

### Prioritás D: Rate-limit és cache
- [ ] Csapatstatisztikák cache-elése
- [ ] Per-endpoint throttling
- [ ] AsyncClient concurrency limit beállítása
- [ ] Redis integráció cache-hez (opcionális)

### Prioritás E: Tesztek és CI
- [ ] Integrációs tesztek mock-olt API-kkal
- [ ] Coverage report
- [ ] GitHub Actions workflow
- [ ] Pre-commit hooks
- [ ] Linting (pylint, flake8, black)

## Architektúra

```
betting.py              # Főmodul (predictor, API integráció)
tests/
  test_predictor.py     # Unit tesztek
tournaments.json        # Liga konfiguráció
config/
  leagues_tiers.yaml    # Liga tier beállítások
```

##算法

### BTTS Számítás

BTTS (Both Teams To Score) valószínűséget két módszerrel számoljuk:

1. **Zárt formula**: `P(BTTS) = 1 - e^(-λh) - e^(-λa) + e^(-(λh+λa))`
2. **Score distribution**: Összeadja `P(h, a)` minden `(h > 0, a > 0)` kombinációra

### Over/Under Számítás

Over/Under valószínűségeket Poisson eloszlással számoljuk:

- **Over X.5**: `P(total > X.5) = Σ P(h, a)` ahol `h + a > X.5`
- **Under X.5**: `P(total ≤ X.5) = 1 - P(Over X.5)`

Poisson paraméterek:
- `λ_home = (goals_for_home + goals_against_away) / 2`
- `λ_away = (goals_for_away + goals_against_home) / 2`

## Licenc

MIT

## Közreműködés

Pull requestek és issue-k mindig szívesen látottak!

```bash
# Fork, clone, create branch
git checkout -b feature/my-new-feature

# Make changes, test
python3 tests/test_predictor.py

# Commit and push
git add .
git commit -m "Add new feature"
git push origin feature/my-new-feature

# Create PR on GitHub
```" 
