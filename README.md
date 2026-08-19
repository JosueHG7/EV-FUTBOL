# Football EV — Asistente de Análisis Pre-Partido

Herramienta de **decision-support** para análisis pre-partido de fútbol: reúne
forma, H2H, descanso y **promedios rodantes** (córners, tiros, tarjetas, goles)
de un equipo vs. las líneas de la casa, y muestra dónde el modelo discrepa del
mercado.

**No es un tipster.** El proyecto arrancó como sistema "+EV" con un supuesto
edge de +13-22% ROI; una auditoría (2026-08-15) demostró que era un bug de
desalineación de datos en el backtester, no un edge real — corregido, el
modelo **no bate al mercado** (ROI ≈ -5%, el margen de la casa). Por eso el
proyecto se reenfocó: el modelo se trata como **diagnóstico** ("dónde discrepa
del mercado"), nunca como sugerencia de apuesta, y sus probabilidades de
goal-markets están **sin calibrar** (sobreconfiadas). El valor real hoy son
los promedios rodantes con datos verificables.

Ver [`PROJECT_STATUS.md`](PROJECT_STATUS.md) para la narrativa completa y el
roadmap pendiente, y [`CLAUDE.md`](CLAUDE.md) para arquitectura y comandos
detallados (pensado para asistir a Claude Code, pero sirve igual de guía).

## Stack

| Capa | Tecnología |
|------|-----------|
| Lenguaje | Python 3.12 (3.11+) |
| Datos | Pandas, PyArrow |
| Base de datos | SQLite + SQLAlchemy ORM |
| Modelos | Poisson Dixon-Coles + XGBoost (ensemble), scikit-learn/scipy |
| Dashboard | Streamlit + Plotly |
| Fuente de datos | API-Football v3 (fuente única, plan PRO) |

## Estructura

```
football-ev/
├── collectors/     # API-Football v3: histórico, próximos partidos + odds, stats
├── database/       # Modelos ORM (SQLAlchemy) y conexión SQLite
├── models/         # Poisson+XGBoost, dossier de partido, backtester, grading
├── dashboard/      # Interfaz Streamlit (Streamlit-free en models/, todo se lee de la BD)
├── data/           # Datos legacy de Understat (usados en entrenamiento histórico)
└── tests/          # Sin suite real todavía (solo __init__.py)
```

## Instalación

```bash
git clone https://github.com/JosueHG7/EV-FUTBOL.git
cd EV-FUTBOL
python -m venv venv
venv\Scripts\activate        # Windows; source venv/bin/activate en Mac/Linux
pip install -r requirements.txt
```

Creá un archivo `.env` en la raíz (no se sube a git) con tu key de
[API-Football](https://www.api-football.com/):

```
API_FOOTBALL_KEY=tu_key_aqui
```

La base de datos (`database/football_ev.db`) y los artefactos del modelo
(`*.pkl`) están gitignored — se generan localmente, no vienen en el repo (ver
Comandos abajo). Para pasar el proyecto a otra máquina rápido, también podés
copiar directamente el `.db` en vez de reconstruir el histórico desde cero.

## Comandos

```bash
python refresh.py                    # pipeline diario completo (ver CLAUDE.md)
streamlit run dashboard/app.py       # dashboard (carga instantánea con artefactos ya generados)

# Colección individual (refresh.py ya corre todo esto)
python collectors/apifootball_collector.py                       # reconstrucción completa de la BD
python collectors/apifootball_collector.py --upcoming --days 7   # próximos partidos + odds
python collectors/apifootball_collector.py --stats --recent 10   # stats por-partido de equipos próximos
python collectors/apifootball_collector.py --results --back 2    # resultados recientes + stats

python models/real_backtester.py     # backtester walk-forward (auditoría/referencia)
```

**Windows:** para matar Streamlit si queda colgado —
```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'streamlit' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

## Estado y roadmap

Ver [`PROJECT_STATUS.md`](PROJECT_STATUS.md) para el detalle de qué funciona
hoy y qué queda pendiente (líneas de córners/tarjetas, calibración,
historial de picks, migración a PostgreSQL).
