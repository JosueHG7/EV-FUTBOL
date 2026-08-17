# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project actually is (read first)

It started as a "+EV betting system" but **pivoted to a pre-match analysis assistant** (decision-support, **not** a tipster). A 2026-08-15 audit proved the original "+13–22% edge" was a data-alignment bug, not a real edge — corrected, the model does **not** beat the market (ROI ~-5%, roughly the bookmaker margin).

Consequences that shape every change here:
- The model is **uncalibrated** and its probabilities are known to be overconfident (e.g. "Over 2.5 94%"). Treat model output as a **diagnostic** ("where the model disagrees with the market"), never as a suggested bet. The UI copy is deliberately honest about this — preserve that framing.
- The genuinely useful, real-data features are the **rolling averages / trends** (corners, shots on target, cards, goals, both-halves) vs the bookmaker lines — this automates the user's manual research method.
- `PROJECT_STATUS.md` is the living design doc; read it for the fuller narrative and the pending roadmap.

## Commands

```bash
python refresh.py                    # THE daily command — see the 6-step pipeline below
streamlit run dashboard/app.py       # launch the dashboard (loads artifacts, ~instant)

# Individual collection (refresh.py runs all of these; use directly for targeted updates)
python collectors/apifootball_collector.py                       # full DB rebuild (all leagues × seasons)
python collectors/apifootball_collector.py --upcoming --days 7   # upcoming fixtures + odds
python collectors/apifootball_collector.py --stats --recent 10   # per-match stats for upcoming teams
python collectors/apifootball_collector.py --results --back 2    # recent finished results + stats

python models/real_backtester.py     # walk-forward backtester (audit/reference only)
```

There is **no working test suite** — `tests/` contains only `__init__.py` despite `pytest` being in `requirements.txt`. "Smoke test" means `python -m py_compile <file>` plus running the affected script/dashboard.

### Windows environment gotchas
- Shell is **PowerShell**. `pkill` does not work; kill Streamlit with `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'streamlit' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }` and check the port with `Get-NetTCPConnection -LocalPort 8501`.
- The console is cp1252; scripts that print emojis/accents need `PYTHONIOENCODING=utf-8` (the entry scripts call `sys.stdout.reconfigure(encoding="utf-8")` for this).

## Architecture

**Single data source: API-Football v3** (PRO plan). The multi-source stack (football-data.co.uk CSVs, The Odds API, Understat) is legacy; `collectors/apifootball_collector.py` is the one that matters. It **throttles to ~3 req/s** — keep that or the firewall flags the account. `.env` holds `API_FOOTBALL_KEY` (and legacy `ODDS_API_KEY`); never commit it.

**Data flow:**
```
API-Football → collectors/ → SQLite (database/football_ev.db, SQLAlchemy ORM)
            → models/ (Poisson+XGBoost, dossier builder)
            → *.pkl artifacts → dashboard/app.py (Streamlit)
```

`refresh.py` orchestrates the whole daily update in 6 steps: (1) recent results + stats, (2) upcoming fixtures + odds, (3) per-match stats for upcoming teams, (4) train model → `models/trained_model.pkl`, (5) precompute all match dossiers → `models/analyzed.pkl`, (6) snapshot model predictions into the DB. Both `.pkl` files and the `.db` are gitignored.

### Key patterns (require reading multiple files)

- **Artifact + mtime caching.** Training and dossier-building are slow (~40s each), so `refresh.py` precomputes them to `.pkl`. The dashboard's cached loaders are **keyed on the artifact/DB file mtime** (`_load_model_cached`, `_analyzed_cached`, `_db_mtime()` in `dashboard/app.py`), so a fresh `refresh.py` is reflected automatically without a restart, and cold loads are instant. Never move heavy computation back into a live dashboard call.
- **Pure analysis module.** `models/analysis_builder.py` (`load_upcoming`, `build_analysis`, `signals`, `to_local`, `load_stats_df`) is **Streamlit-free on purpose** so both `refresh.py` and the dashboard share it. `models/match_analysis.py` `build_dossier()` produces the per-match dossier (model probs, market de-vig, goal internals, form/H2H/rest, corner projection); `team_trends()` computes the rolling trend hit-rates. Keep new pre-match analysis logic in these pure modules, not in `app.py`.
- **Prediction snapshot + grading.** `refresh.py` seals each upcoming match's model + market probabilities **and the real decimal odds** into `ModelPrediction` before kickoff. After the match, `models/grading.py` grades directional accuracy **and** flat-stake ROI against those sealed odds (the honest "would it have beaten the price?" test), shown in the 📋 Predicciones tab.
- **Model detail.** Over/Under 2.5 and BTTS come from the **Poisson component only**; 1X2 blends Poisson + XGBoost (`models/ensemble_model.py`). Poisson strengths are time-decayed (0.98/day) over all history — so early-season predictions carry last season's form; the dossier surfaces a "new-season small-sample" ⚠️ for this.
- **Times are naive-UTC in the DB;** the dashboard converts to the machine's local timezone via `analysis_builder.to_local` for all display/grouping.

### Schema (`database/models.py`)
`matches` (fixtures + results, incl. half-time goals), `odds` (1X2), `market_odds` (flexible: over/under, BTTS, corners — market/selection/line/odd), `team_statistics` (long format, per-match corners/shots/cards/xG), `model_predictions` (the snapshot log). `init_db()` in `database/db.py` runs `create_all` **plus** idempotent `ALTER TABLE` migrations (`_apply_column_migrations`) since there's no Alembic in the live path.

### Leagues (`collectors/apifootball_collector.py`)
Two dicts drive coverage: `LEAGUES` (29 leagues → id: (country, name)) and `STAT_LEAGUES` (the 24 that expose per-match corner/shot/card stats — the rest, e.g. Iceland/Costa Rica/2nd-tier Finland, only have goals, so only goal/half trends apply there).

**To add a league:** add it to `LEAGUES` (+ `STAT_LEAGUES` if it has stats — verify against the API first), backfill history with `rebuild(wipe=False, leagues={new})`, then run `refresh.py` to retrain and pick it up. `wipe=False` uses `session.merge` so it won't disturb existing data.

## The bug that removed the "edge" (don't reintroduce)

`build_features` used an **unstable** `sort_values("match_date")`, reordering ~24% of same-kickoff rows; the backtester then attached odds **by position**, giving a quarter of matches another match's odds and faking the ROI. Fix was `sort_values(..., kind="stable")` — that sort stability matters anywhere odds/features are joined positionally.
