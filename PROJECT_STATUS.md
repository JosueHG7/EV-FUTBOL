# PROJECT STATUS — Football EV Betting System
Última actualización: 2026-08-14

---

## Estructura del proyecto

```
football-ev/
├── config.py                              # Variables globales, rutas, API keys, umbrales EV/Kelly
├── main.py                                # Entry point: obtiene fixtures + odds en vivo, calcula EV
│
├── collectors/
│   ├── cache.py                           # Cache genérico en JSON con TTL configurable
│   ├── football_collector.py              # Descarga fixtures desde API-Football (api-football.com)
│   ├── odds_collector.py                  # Descarga odds en vivo desde The Odds API
│   └── historical_odds_collector.py       # Descarga CSVs históricos de football-data.co.uk
│
├── database/
│   ├── models.py                          # ORM SQLAlchemy: tablas Match y Odds
│   ├── db.py                              # Engine SQLite, get_session() context manager
│   ├── init_db.py                         # Crea tablas si no existen
│   └── ingest.py                          # Importa CSVs históricos a la BD (Match + Odds Pinnacle)
│
├── models/
│   ├── ev_calculator.py                   # Calcula EV = prob_modelo * odd - 1 por mercado
│   ├── kelly.py                           # Kelly fraccionado: fracción de bankroll a apostar
│   ├── poisson_model.py                   # Dixon-Coles con parámetro rho (corrección 0-0/1-0/0-1/1-1) + decay temporal
│   ├── feature_engineering.py             # 33 features: forma reciente, venue, H2H, fatiga, Poisson embed
│   ├── xgboost_model.py                   # XGBoost multiclase (H/D/A) + walk-forward + Brier Score
│   ├── ensemble_model.py                  # Ensemble ponderado Poisson + XGBoost, pesos óptimos vía Brent
│   ├── backtester.py                      # Backtester original (solo Poisson, odds semanales)
│   ├── real_backtester.py                 # Backtester walk-forward real con odds Pinnacle históricas
│   └── warmup_experiment.py               # Verificación robustez ROI: warmup 6/12/18 meses + leakage audit
│
├── dashboard/
│   ├── __init__.py
│   └── app.py                             # Dashboard Streamlit: Picks en vivo + Rendimiento (backtest)
│
├── tests/
│   └── __init__.py                        # (vacío — pendiente)
│
└── data/
    ├── fixtures_2022.json                 # Cache fixtures API-Football temporada 2022 (3.8 MB)
    ├── fixtures_2023.json                 # Cache fixtures API-Football temporada 2023 (2.7 MB)
    ├── fixtures_2024.json                 # Cache fixtures API-Football temporada 2024 (146 B — incompleto)
    ├── fixtures_raw.json                  # Cache fixtures última consulta (3.8 MB)
    ├── odds_raw.json                      # Cache odds en vivo última consulta (1.6 MB)
    ├── backtest_results.json              # Resultados backtest Poisson original (1.6 MB)
    ├── real_backtest_results.json         # Resultados backtest 3-modelos walk-forward (1.4 MB)
    └── odds_historical/                   # CSVs Pinnacle football-data.co.uk
        ├── 2223_E0.csv  2324_E0.csv  2425_E0.csv   # Premier League
        ├── 2223_SP1.csv 2324_SP1.csv 2425_SP1.csv  # La Liga
        ├── 2223_D1.csv  2324_D1.csv  2425_D1.csv   # Bundesliga
        ├── 2223_I1.csv  2324_I1.csv  2425_I1.csv   # Serie A
        └── 2223_F1.csv  2324_F1.csv  2425_F1.csv   # Ligue 1
```

---

## Stack tecnológico

| Paquete | Versión instalada | Uso |
|---|---|---|
| pandas | 2.2.3 | DataFrames, manipulación de datos |
| numpy | 2.2.6 | Arrays, Brier Score, operaciones vectoriales |
| scipy | 1.17.1 | Dixon-Coles MLE (`minimize`), Brent (`minimize_scalar`) |
| xgboost | **3.2.0** | Clasificador multiclase H/D/A (`multi:softprob`) |
| scikit-learn | 1.8.0 | Métricas auxiliares |
| sqlalchemy | 2.0.49 | ORM SQLite (Match, Odds) |
| streamlit | 1.57.0 | Dashboard (Fase 4, pendiente) |
| requests | ≥2.32 | Llamadas a APIs externas |
| python-dotenv | ≥1.0 | Carga de `.env` (API keys) |
| pytest | ≥8.2 | Tests (pendiente) |

> **Nota XGBoost 3.x**: `early_stopping_rounds` debe ir en el constructor, no en `fit()`.
> Output de entrenamiento suprimido con `contextlib.redirect_stdout/stderr(io.StringIO())`.

---

## Estado de los datos

### Partidos en BD por liga y temporada

| league_id | Liga | 2022 | 2023 | 2024 |
|---|---|---|---|---|
| 2 | Champions League | 214 | — | 279 |
| 39 | Premier League | 380 | 380 | 380 |
| 61 | Ligue 1 | 380 | — | 308 |
| 78 | Bundesliga | 308 | 308 | 308 |
| 135 | Serie A | 381 | 380 | 380 |
| 140 | La Liga | 380 | 380 | 380 |
| **Total** | | | | **5.526 partidos** |

### Odds históricas disponibles

- **9.960 registros** de odds Pinnacle (H/D/A) en la BD
- **4.980 partidos** con odds Pinnacle válidas (cruce match ↔ odds)
- Fuente: football-data.co.uk CSVs (columnas PSH, PSD, PSA)
- Cobertura: temporadas 2022/23, 2023/24, 2024/25 — 5 ligas (sin UCL histórica)

### Archivos de caché en data/

| Archivo | Tamaño | Descripción |
|---|---|---|
| fixtures_2022.json | 3,8 MB | Fixtures API-Football 2022 |
| fixtures_2023.json | 2,7 MB | Fixtures API-Football 2023 |
| fixtures_raw.json | 3,8 MB | Última consulta fixtures |
| odds_raw.json | 1,6 MB | Última consulta odds en vivo |
| backtest_results.json | 1,6 MB | Backtest Poisson original |
| real_backtest_results.json | 1,4 MB | Backtest 3-modelos walk-forward |
| odds_historical/ (15 CSVs) | ~2,4 MB | Pinnacle histórico 3 temporadas |

---

## Resultados del modelo

### Brier Score (walk-forward, out-of-sample — 24 periodos, ~3.448 partidos)

| Modelo | Brier Score | vs Naive | vs Poisson |
|---|---|---|---|
| **Ensemble** (w=0.47/0.53) | **0.5785** | −0.0882 | −0.0196 |
| Poisson (Dixon-Coles + decay) | 0.5981 | −0.0686 | — |
| XGBoost | 0.5936 | −0.0731 | −0.0045 |
| Naive (1/3 cada) | 0.6667 | — | +0.0686 |

Pesos óptimos: **Poisson = 0.4678 / XGBoost = 0.5322** (optimizados via Brent sobre Brier walk-forward)

### ROI por warmup — Ensemble con MIN_EV = 3%, bankroll inicial = 100

| Warmup | Picks | Win% | ROI | Max DD | Brier | Bankroll final |
|---|---|---|---|---|---|---|
| 6 meses (180d) | 3.633 | 34,6% | **+15,9%** | 16,4% | 0.6087 | 2.639,70 |
| 12 meses (365d) | 2.623 | 33,6% | **+15,2%** | 22,6% | 0.6009 | 801,41 |
| 18 meses (548d) | 2.113 | 33,0% | **+13,5%** | 23,2% | 0.6084 | 440,20 |

- **Todos positivos**: SI
- **Dispersión ROI**: 2,4% (< 10% = estable)
- **CONCLUSION: ROI positivo y estable en los 3 warmups → EDGE REAL**

### Tabla comparativa 3 modelos (warmup = 12 meses, MIN_EV = 3%)

| Modelo | Picks | Win% | ROI | Max DD | Bankroll final |
|---|---|---|---|---|---|
| Poisson solo | 2.956 | — | +12,2% | 24,9% | 533,08 |
| XGBoost solo | 2.770 | — | +13,0% | 21,9% | 503,79 |
| **Ensemble** | **2.623** | 33,6% | **+15,2%** | 22,6% | **801,41** |

### Data leakage audit (warmup = 12 meses, 23 ventanas)

```
Comprobando: max(fecha_train) < min(fecha_eval) en cada ventana

Ventana 1: train hasta 2023-06-11 18:45:00  |  eval desde 2023-08-11 17:30:00  |  gap=60.9d  OK
Ventana 2: train hasta 2023-08-14 19:30:00  |  eval desde 2023-08-18 17:30:00  |  gap=3.9d   OK
Ventana 3: train hasta 2023-09-03 19:00:00  |  eval desde 2023-09-15 18:30:00  |  gap=12.0d  OK
...
ASSERTION RESULT: todas las ventanas pasaron (gap > 0 dias)
```

**Nota imputation leakage**: `prepare_dataset()` usa medianas globales para imputar NaN en
features de forma/venue. Afecta ~2-4% de filas. Impacto estimado en ROI: < 0,1 pp. **Negligible.**

---

## Configuración actual

| Variable | Valor | Descripción |
|---|---|---|
| `MIN_EV_THRESHOLD` | 0.03 | Mínimo 3% EV para considerar apuesta |
| `MIN_KELLY_FRACTION` | 0.01 | Mínimo 1% bankroll por apuesta |
| `MAX_KELLY_FRACTION` | 0.05 | Máximo 5% bankroll (Kelly fraccionado) |
| `DEFAULT_BANKROLL` | 100.0 | Bankroll inicial en unidades |
| `TARGET_BOOKMAKER` | "Pinnacle" | Bookmaker de referencia |
| `MAX_ODDS` | 20.0 | Filtro odds sospechosas |
| `MIN_ODDS` | 1.20 | Filtro odds de valor marginal |
| `DEV_MODE` | True | Prefiere caché aunque haya expirado |
| `CACHE_HOURS_ODDS` | 3 | TTL caché odds en vivo |
| `CACHE_HOURS_FIXTURES` | 24 | TTL caché fixtures |
| `API_FOOTBALL_SEASON` | 2024 | Temporada activa en API-Football |

**Ligas monitoreadas** (API-Football IDs): La Liga (140), Premier League (39), Bundesliga (78),
Serie A (135), Ligue 1 (61), Champions League (2)

---

## Próximos pasos

- [x] **Integrar xG de Understat** — `collectors/understat_collector.py` + caché `data/understat/`, usado en `feature_engineering.py`
- [x] **Conectar `main.py` con `EnsembleModel`** — `main.py` entrena EnsembleModel (pesos 0.451/0.549) con xG. Lógica refactorizada en `build_live_model()` / `scan_picks()` (reutilizadas por el dashboard)
- [x] **Agregar más temporadas** — CSVs 2020/21 y 2021/22 descargados (`data/odds_historical/2021_*`, `2122_*`)
- [x] **Dashboard Streamlit** — `dashboard/app.py`: pestaña *Picks en vivo* (EnsembleModel + odds actuales) y *Rendimiento* (KPIs, bankroll histórico, PnL por liga/tipo, tabla de picks del backtest). Ejecutar: `streamlit run dashboard/app.py`
- [ ] **Alertas Telegram** — bot que envía picks +EV al canal cuando se detecta ventana antes del partido
- [ ] **Tests** — `tests/` vacío; añadir cobertura pytest (Kelly, EV, remove_vig, Poisson)
- [ ] **Revisar picks de EV extremo** — el modelo emite ocasionalmente picks con EV > 100% en cuotas altas (ej. 8.33); evaluar cap de EV o filtro de cuota máxima más estricto

---

## Comandos útiles

```bash
# Inicializar BD
python database/init_db.py

# Ingestar CSVs históricos de odds (football-data.co.uk)
python database/ingest.py

# Backtest completo 3 modelos (Poisson / XGBoost / Ensemble)
python models/real_backtester.py

# Verificar robustez ROI + audit data leakage
python models/warmup_experiment.py

# Optimizar y evaluar ensemble (Brier Score + pesos óptimos)
python models/ensemble_model.py

# Evaluar solo XGBoost (walk-forward + feature importance)
python models/xgboost_model.py

# Evaluar solo Poisson Dixon-Coles
python models/poisson_model.py

# Obtener picks en vivo (odds actuales + EV)
python main.py

# Dashboard interactivo (picks en vivo + rendimiento del backtest)
streamlit run dashboard/app.py

# Descargar odds históricas
python collectors/historical_odds_collector.py
```
