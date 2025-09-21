# Automatikus Value Bet Kiválasztás - Használati Útmutató

## Áttekintés

Az automatikus value bet funkció minden támogatott szelvénytípusra (1X2, BTTS, Over/Under) automatikusan kiválasztja a legjobb value bet mérkőzést a legmagasabb edge, confidence és value score alapján. Minden típusra külön szelvényt generál részletes indoklással és külön Telegram üzenetben küldi el.

## Főbb Funkciók

### 1. Automatikus Kiválasztás
- **1X2 Market**: Hazai/Döntetlen/Vendég fogadások
- **BTTS Market**: Mindkét csapat gólt szerez (Igen/Nem)  
- **Over/Under Market**: Gólok száma 2.5 felett/alatt

### 2. Értékelési Kritériumok
- **Value Score**: Edge × (1 + market_strength/100 × 0.1)
- **Edge**: Modell valószínűség vs piaci valószínűség különbsége
- **Confidence**: Alacsony/Közepes/Magas (edge alapján)
- **Market Strength**: Piac hatékonyságának mérése

### 3. Telegram Integráció
- Külön üzenet minden szelvénytípusra
- Magyar nyelvű részletes magyarázat
- Automatikus hibakezelés és logolás

## Telegram Parancsok

### `/autobets` vagy `/autovalue`
Automatikus value bet kiválasztás és küldés:

```
/autobets
```

**Válasz példa:**
```
🎯 Automatikus value bet kiválasztás indul...

[3 különálló üzenet érkezik minden piactípusra]

✅ Automatikus value bet kiválasztás kész! 3 ajánlás elküldve.
```

## Üzenet Formátum

Minden szelvény tartalmazza:

```
🎯 AUTOMATIKUS VALUE BET

⚽ Végeredmény
🌟 Premier League

⚽ Mérkőzés:
Arsenal vs Chelsea  
🕒 2025-09-21 18:00

💰 Ajánlás:
🎯 Hazai győzelem @ 2.10

📊 Elemzés:
📈 Modell valószínűség: 55.0%
🏪 Piac valószínűség: 45.0%
⚡ Edge (előny): +12.0%
🔥 Value Score: 0.126

💪 Bizalmi szint:
⚡ Közepes bizalom

🏛️ Piac információ:
📊 Piac erő: 85.2%
🆔 Fixture ID: #12345

💡 Indoklás:
A modellünk 55.0% valószínűséget ad erre az eredményre, 
míg a piac csak 45.0%-ot ár be. Ez 12.0% előnyt jelent 
számunkra, ami közepes bizalmi szintű value betting lehetőség.
```

## Bizalmi Szintek

| Edge Érték | Bizalmi Szint | Emoji |
|------------|---------------|--------|
| ≥ 15%      | Magas         | 🔥     |
| 8-15%      | Közepes       | ⚡     |
| < 8%       | Alacsony      | 🔸     |

## Liga Besorolás

| Liga Tier | Emoji | Leírás |
|-----------|-------|--------|
| TIER1, TIER1B | 🌟 | TOP ligák |
| Egyéb | ⚪ | Standard ligák |

## Hibakezelés

A rendszer automatikusan kezeli a következő helyzeteket:

- **Nincs elemzési adat**: "❌ Nincs elérhető elemzés. Futtasd a /run parancsot először!"
- **Nincs value bet**: "🚫 Nincs megfelelő value bet ma. Próbáld újra később!"
- **Üzenet küldési hiba**: Részletes hibaüzenet logolással
- **Rendszer hiba**: "❌ Hiba az automatikus value bet kiválasztáskor: [részletek]"

## Technikai Részletek

### Függvények
- `select_auto_value_bets()`: Legjobb value bet kiválasztás
- `generate_detailed_bet_message()`: Magyar üzenet generálás
- `select_best_tickets_enhanced()`: Alapul szolgáló ticket selection

### Szűrési Kritériumok
- Edge > 0
- Odds < max límit (1X2: 4.0, BTTS/O/U: 4.0)
- Valószínűség különbség tolerancia alatt
- Csak mai mérkőzések (opcionális)

### Logolás
```
INFO | Automatikus value bet kiválasztás indul...
INFO | 1X2 legjobb bet: FI#12345 HOME @ 2.10 (edge: 0.120, value: 0.126)
INFO | BTTS legjobb bet: FI#12346 YES @ 1.80 (edge: 0.080, value: 0.085)
INFO | O/U legjobb bet: FI#12347 OVER 2.5 @ 1.75 (edge: 0.100, value: 0.108)
INFO | Automatikus kiválasztás kész: 3 piac, összesen 3 ajánlás
```

## Használati Workflow

1. **Elemzés futtatás**: `/run` parancs a mérkőzések elemzéséhez
2. **Automatikus kiválasztás**: `/autobets` parancs a value bet-ek generálásához
3. **Üzenetek érkezése**: 3-4 Telegram üzenet érkezik (szelvények + összefoglaló)
4. **Döntéshozatal**: Részletes információk alapján fogadási döntés

## Megjegyzések

- A funkció a meglévő `select_best_tickets_enhanced()` funkcióra épül
- Kompatibilis minden meglévő szűrési és értékelési kritériummal
- Automatikus rate limiting (0.5s késleltetés üzenetek között)
- Teljes hibakezelés és visszajelzés
- Magyar nyelvű interface és magyarázatok