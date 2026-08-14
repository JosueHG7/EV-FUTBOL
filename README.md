# Football EV — Sistema de Apuestas con Valor Esperado Positivo

Herramienta de análisis que identifica apuestas deportivas de fútbol con **valor esperado positivo (+EV)** comparando las probabilidades implícitas de las casas de apuestas contra probabilidades calculadas por modelos predictivos propios.

## Concepto clave

Una apuesta tiene **+EV** cuando la probabilidad real de un evento es mayor que la probabilidad implícita en la cuota ofrecida:

```
EV = (prob_real × ganancia) - (1 - prob_real) × stake
```

Si `EV > 0`, la apuesta es rentable a largo plazo.

## Stack

| Capa | Tecnología |
|------|-----------|
| Lenguaje | Python 3.11+ |
| Datos | Pandas, PyArrow |
| Base de datos | SQLite + SQLAlchemy |
| Modelos | scikit-learn, scipy |
| Dashboard | Streamlit + Plotly |
| APIs | The Odds API, Football-Data.org |

## Estructura

```
football-ev/
├── collectors/     # Llamadas a APIs externas (cuotas, estadísticas)
├── database/       # Modelos ORM y conexión SQLite
├── models/         # Modelos predictivos (Poisson, Elo, ML)
├── data/           # Datos en CSV/Parquet
├── dashboard/      # Interfaz Streamlit
└── tests/          # Tests unitarios
```

## Instalación

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Copia `.env` y añade tus API keys:
```bash
copy .env .env.local
```

## Fases de desarrollo

- **Fase 1** — Estructura base y configuración ✅
- **Fase 2** — Collectors de cuotas y modelo Poisson bivariado
- **Fase 3** — Cálculo de EV y criterio de Kelly
- **Fase 4** — Dashboard Streamlit con alertas en tiempo real

## APIs gratuitas recomendadas

| API | Plan gratuito | Uso |
|-----|--------------|-----|
| [The Odds API](https://the-odds-api.com) | 500 req/mes | Cuotas en vivo de múltiples casas |
| [Football-Data.org](https://www.football-data.org) | 10 ligas | Resultados y estadísticas históricas |
| [API-Football (RapidAPI)](https://rapidapi.com/api-sports/api/api-football) | 100 req/día | Estadísticas avanzadas de equipos |
