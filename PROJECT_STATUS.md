# PROJECT STATUS — Football Analysis Assistant
Última actualización: 2026-08-15

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

### Ligas cubiertas (21) — IDs API-Football

| ID | Liga | | ID | Liga |
|---|---|---|---|---|
| 113 | Suecia Allsvenskan | | 39 | Premier League |
| 114 | Suecia Superettan | | 140 | La Liga |
| 244 | Finlandia Veikkausliiga | | 135 | Serie A |
| 245 | Finlandia Ykkönen | | 78 | Bundesliga |
| 164 | Islandia Úrvalsdeild | | 61 | Ligue 1 |
| 165 | Islandia 1. Deild | | 262 | México Liga MX |
| 103 | Noruega Eliteserien | | 253 | USA MLS |
| 104 | Noruega 1. Division | | 71 | Brasil Serie A |
| 94 | Portugal Primeira | | 72 | Brasil Serie B |
| 95 | Portugal Segunda | | 128 | Argentina Liga Prof. |
| | | | 162 | Costa Rica Primera |

- **24.698 partidos** ingestados (temporadas 2023-2026), 20.899 finalizados, 462 equipos.
- **Cobertura de stats por-partido** (córners/tiros/tarjetas/xG) SOLO en las ligas
  más fuertes: Suecia, Finlandia 1ª, Noruega 1ª, Portugal, 5 grandes, México, MLS,
  Brasil, Argentina. **NO en Islandia, Costa Rica, Finlandia/Noruega 2ª** — ahí el
  dato no existe (`STAT_LEAGUES` en el colector). Esas ligas muestran solo goles.

### Esquema (`database/models.py`)

- `matches` — fixtures + resultados (enriquecido: HT goals, venue, referee, round, country).
- `odds` — 1X2 por bookmaker.
- `market_odds` — flexible (over/under, BTTS, y a futuro córners/tarjetas): market/selection/line/odd.
- `team_statistics` — stats por-partido en formato largo (incluye `expected_goals`). Ampliable sin tocar esquema.
- `injuries`, `lineups` / `lineup_players`, `predictions` — tablas listas (se pueblan on-demand).
- `picks` — historial de apuestas del usuario (fase posterior).

---

## Componentes clave

| Módulo | Función |
|---|---|
| `collectors/apifootball_collector.py` | Fuente única. `rebuild` (histórico), `--upcoming` (fixtures próximos + odds), `--stats` (stats por-partido de equipos con partido próximo). Throttle incluido. |
| `models/poisson_model.py` | Poisson Dixon-Coles: 1X2, over/under, BTTS, matriz de marcadores. |
| `models/match_analysis.py` | **Motor del dossier**: forma, H2H, descanso, xG, y **promedios rodantes** (últimos 5/10) de córners/tiros/tarjetas/atajadas/goles a favor y en contra. |
| `dashboard/app.py` | Streamlit. Pestaña **🔍 Análisis** (triaje + fichas con promedios). Lee todo de la BD. |
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
- BD única con las 21 ligas del usuario, esquema rico.
- Dashboard con triaje de partidos próximos (todas las ligas) y fichas por partido.
- **Promedios rodantes** (córners/tiros/tarjetas/goles) en cada ficha — el método del usuario, con datos reales fiables.

**Pendiente:**
1. Comparar promedios vs **líneas de córners/tarjetas** (recolectar esos mercados) + señales.
2. **Optimizar la carga** (entrenar modelo en un script / guardar en disco → dashboard instantáneo).
3. **Calibrar** las probabilidades del modelo (el "Over 2.5 94%").
4. Historial de picks (registro + rendimiento).
5. Migración a PostgreSQL + acceso remoto/móvil.

**Aviso:** las señales del triaje (del modelo) aún NO están calibradas → no fiables
para apostar. Los **promedios rodantes SÍ** son datos reales y usables hoy.

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
