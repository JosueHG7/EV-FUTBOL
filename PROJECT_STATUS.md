# PROJECT STATUS — Football Analysis Assistant
Última actualización: 2026-08-19

---

## Qué es este proyecto (importante — leer primero)

Empezó como un "sistema de apuestas de valor esperado (EV+)" que decía tener
+13-16% de ROI. **Una auditoría del backtester (2026-08-15) demostró que ese
edge NO era real** — era un bug de desalineación de datos (ver más abajo).
Corregido, el sistema **NO bate al mercado** (pierde ~el margen de la casa).

El proyecto se **reenfocó** a lo que el usuario realmente necesita: un
**asistente de análisis pre-partido** (decision-support), que automatiza el
trabajo manual que hace a diario — reunir forma, historia, H2H, y **promedios
rodantes de córners / tiros / tarjetas / goles** vs las líneas de la casa.
No es un tipster; el juicio final es del usuario.

---

## Arquitectura de datos

- **Fuente única: API-Football v3** (plan PRO del usuario, $19/mo, 7.500 req/día,
  300 req/min, sin auto-renovación). Reemplazó el stack multi-fuente anterior
  (football-data.co.uk CSVs + The Odds API + Understat).
- **BD: SQLite** (`database/football_ev.db`), vía SQLAlchemy ORM.
  **Migration-ready a PostgreSQL** (solo cambiar `DATABASE_URL`) — el usuario
  quiere acceso remoto/móvil a futuro; cuando llegue ese momento se migra.
- El colector respeta un **throttle (~3 req/s)** para no gatillar el firewall.

### Ligas cubiertas (29) — IDs API-Football

| ID | Liga | | ID | Liga |
|---|---|---|---|---|
| 113 | Suecia Allsvenskan | | 262 | México Liga MX |
| 114 | Suecia Superettan | | 253 | USA MLS |
| 244 | Finlandia Veikkausliiga | | 71 | Brasil Serie A |
| 245 | Finlandia Ykkönen | | 72 | Brasil Serie B |
| 164 | Islandia Úrvalsdeild | | 128 | Argentina Liga Prof. |
| 165 | Islandia 1. Deild | | 162 | Costa Rica Primera |
| 103 | Noruega Eliteserien | | 307 | Arabia Pro League |
| 104 | Noruega 1. Division | | 88 | Países Bajos Eredivisie |
| 94 | Portugal Primeira | | 89 | Países Bajos Eerste Divisie |
| 95 | Portugal Segunda | | 79 | Alemania 2. Bundesliga |
| 39 | Premier League | | 80 | Alemania 3. Liga |
| 140 | La Liga | | 40 | Inglaterra Championship |
| 135 | Serie A | | 207 | Suiza Super League |
| 78 | Bundesliga | | 144 | Bélgica Jupiler Pro League |
| 61 | Ligue 1 | | | |

- **~35.800 partidos** ingestados (temporadas 2023-2026), ~29.550 finalizados.
- **Cobertura de stats por-partido** (córners/tiros/tarjetas/xG) en 24 de las 29
  ligas — `STAT_LEAGUES` en el colector. **NO en Islandia, Costa Rica,
  Finlandia/Noruega 2ª** — ahí el dato no existe. Esas ligas muestran solo goles.

### Esquema (`database/models.py`)

- `matches` — fixtures + resultados (enriquecido: HT goals, venue, referee, round, country).
- `odds` — 1X2 por bookmaker.
- `market_odds` — flexible (over/under, BTTS, **córners** ya poblado; tarjetas a futuro): market/selection/line/odd.
- `model_predictions` — snapshot sellado del modelo + mercado + cuotas reales antes de cada partido próximo (incluye córners); base de la pestaña Predicciones.
- `team_statistics` — stats por-partido en formato largo (incluye `expected_goals`). Ampliable sin tocar esquema.
- `injuries`, `lineups` / `lineup_players`, `predictions` — tablas listas (se pueblan on-demand).
- `picks` — historial de apuestas del usuario (fase posterior).

---

## Componentes clave

| Módulo | Función |
|---|---|
| `refresh.py` | **El comando diario.** Pipeline de 6 pasos: resultados recientes → próximos + odds → stats por-partido → reentrena modelo → precalcula dossiers (`.pkl`) → sella snapshot de predicciones. Deja el dashboard con carga instantánea. |
| `collectors/apifootball_collector.py` | Fuente única. `rebuild` (histórico), `--upcoming` (fixtures próximos + odds), `--stats` (stats por-partido de equipos con partido próximo), `--results` (resultados recientes). Throttle incluido. |
| `models/poisson_model.py` | Poisson Dixon-Coles: 1X2, over/under, BTTS, matriz de marcadores, **proyección de córners vs línea**. |
| `models/match_analysis.py` | **Motor del dossier**: forma, H2H, descanso, xG, y **promedios rodantes** (últimos 5/10) de córners/tiros/tarjetas/atajadas/goles a favor y en contra. |
| `models/grading.py` | Califica cada predicción sellada contra el resultado real: acierto direccional **y** ROI (staking plano 1u a la cuota sellada) por mercado (1X2/O-U/BTTS/córners). |
| `dashboard/app.py` | Streamlit. Pestañas **🔍 Análisis** (triaje + fichas con promedios), **Resultados**, **📋 Predicciones** (bitácora de validación: acierto/ROI del modelo vs. mercado, con fila Total combinada). Lee todo de artefactos/BD, sin cómputo pesado en vivo. |
| `models/ensemble_model.py`, `models/xgboost_model.py` | Modelo Poisson+XGBoost (histórico; sus probabilidades de goal-markets salen sobreconfiadas → pendiente calibrar). |
| `models/real_backtester.py` | Backtester walk-forward (parametrizable por bookmaker). |

---

## El bug que desmontó el "edge" (referencia)

`build_features` hacía `sort_values("match_date")` **inestable**, reordenando
~24% de las filas (partidos con el mismo horario). El backtester pegaba las odds
a `feat_df` **por posición** → el 24% de partidos recibía las cuotas de OTRO
partido → EV/ROI inflados. **Fix: `sort_values(..., kind="stable")`.** Corregido,
Poisson/XGBoost/Ensemble dan ROI ~-5% (sin edge). Los "+15-22%" nunca existieron.
Probado también contra Bet365 (soft book) con filtros: suelo ~-4.7%, tampoco cruza cero.

---

## Estado actual

**Funciona:**
- BD única con las 29 ligas del usuario, esquema rico.
- `refresh.py` corre el pipeline diario completo y deja el dashboard con carga
  instantánea (artefactos `.pkl` precalculados, cacheados por mtime).
- Dashboard con triaje de partidos próximos (todas las ligas), fichas por
  partido (tabs + panel de un vistazo), pestaña de Resultados, y pestaña
  **📋 Predicciones**: bitácora de validación que sella modelo + mercado +
  cuotas reales antes del partido y las califica al finalizar (acierto y ROI,
  por mercado y en un **Total combinado**).
- **Córners vs línea de la casa** ya integrado: proyección del modelo, línea
  del mercado, cuota sellada del lado que llama el modelo, y se califica
  igual que los demás mercados.
- **Promedios rodantes** (córners/tiros/tarjetas/goles) en cada ficha — el método del usuario, con datos reales fiables.
- Bug de fuga de datos en los snapshots pre-partido (corregido 2026-08-16):
  predicciones contaminadas quedan marcadas, no borradas, y excluidas del cálculo de ROI/acierto.

**Pendiente:**
1. Comparar promedios vs **líneas de tarjetas** (córners ya está — falta tarjetas).
2. **Calibrar** las probabilidades del modelo (el "Over 2.5 94%").
3. Historial de picks del usuario (registro + rendimiento) — distinto de la
   bitácora de Predicciones, que califica las llamadas del *modelo*, no las
   apuestas reales del usuario.
4. Migración a PostgreSQL + acceso remoto/móvil.

**Aviso:** las señales de goles/1X2 (del modelo) aún NO están calibradas →
no fiables para apostar. Los **promedios rodantes SÍ** son datos reales y
usables hoy. El ROI en Predicciones es diagnóstico con muestra chica, no un
backtest cerrado.

---

## Comandos útiles

```bash
# Reconstruir la BD desde API-Football (histórico, 21 ligas × 4 temporadas)
python collectors/apifootball_collector.py

# Recolectar partidos próximos + odds (1X2/O-U/BTTS)
python collectors/apifootball_collector.py --upcoming --days 3

# Ingestar stats por-partido (córners/tiros/tarjetas) de equipos con partido próximo
python collectors/apifootball_collector.py --stats --recent 10

# Lanzar el dashboard
streamlit run dashboard/app.py

# Backtester (auditoría / referencia)
python models/real_backtester.py
```
