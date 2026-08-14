from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "database" / "football_ev.db"

# Database
DATABASE_URL = f"sqlite:///{DB_PATH}"

# API Keys
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")

# Odds API (the-odds-api.com)
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# API-Football (api-football.com)
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
API_FOOTBALL_SEASON = 2024
LEAGUE_IDS = {
    "la_liga":          140,
    "premier_league":    39,
    "bundesliga":        78,
    "serie_a":          135,
    "ligue_1":           61,
    "champions_league":   2,
}
ODDS_API_REGIONS = "eu"           # eu, us, uk, au
ODDS_API_MARKETS = "h2h,spreads,totals"
ODDS_API_ODDS_FORMAT = "decimal"

# Cache
DEV_MODE = True             # True → prefiere cache aunque haya expirado
CACHE_HOURS_ODDS = 3        # Las cuotas cambian rápido
CACHE_HOURS_FIXTURES = 24   # Los fixtures son estables durante el día

# EV thresholds
MIN_EV_THRESHOLD = 0.03           # Mínimo 3% de EV para considerar una apuesta
MIN_KELLY_FRACTION = 0.01         # Mínimo 1% de bankroll por apuesta
MAX_KELLY_FRACTION = 0.05         # Máximo 5% de bankroll por apuesta (Kelly fraccionado)

# Bankroll
DEFAULT_BANKROLL = 100.0
TARGET_BOOKMAKER = "Pinnacle"
MAX_ODDS = 20.0   # odds mayores a esto son sospechosas
MIN_ODDS = 1.20   # odds menores a esto tienen valor marginal         # Bankroll inicial en unidades

# Ligas a seguir (IDs de the-odds-api)
TRACKED_LEAGUES = [
    "soccer_spain_la_liga",
    "soccer_epl",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
]
