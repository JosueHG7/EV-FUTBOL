"""
database/models.py

Schema for the Football analysis system. Single source: API-Football (v3).

Design notes
------------
- Kept engine-agnostic (standard SQLAlchemy types, no SQLite-only features) so a
  future migration SQLite → PostgreSQL is a DATABASE_URL change, not a rewrite.
- `matches` + `odds` are structured/queryable (model, form, filters run on them).
- Rich per-fixture data (injuries, lineups, statistics/xG, predictions) lives in
  dedicated tables, populated on demand for upcoming matches.
- IDs mirror API-Football's own IDs (fixture id, team id, player id) so re-ingest
  is idempotent via upsert/merge.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Index, Integer, String,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Core: matches (fixtures + results)
# ---------------------------------------------------------------------------

class Match(Base):
    __tablename__ = "matches"

    id:             Mapped[int] = mapped_column(Integer, primary_key=True)  # API-Football fixture id
    league_id:      Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    league_name:    Mapped[str] = mapped_column(String(100), nullable=False)
    country:        Mapped[str | None] = mapped_column(String(80), nullable=True)
    season:         Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    round:          Mapped[str | None] = mapped_column(String(80), nullable=True)

    home_team_id:   Mapped[int] = mapped_column(Integer, nullable=False)
    home_team_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    away_team_id:   Mapped[int] = mapped_column(Integer, nullable=False)
    away_team_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    match_date:     Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    status:         Mapped[str] = mapped_column(String(20), nullable=False, default="scheduled")

    home_goals:     Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals:     Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_goals_ht:  Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals_ht:  Mapped[int | None] = mapped_column(Integer, nullable=True)

    venue:          Mapped[str | None] = mapped_column(String(120), nullable=True)
    referee:        Mapped[str | None] = mapped_column(String(120), nullable=True)

    created_at:     Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at:     Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)

    odds:        Mapped[list["Odds"]]         = relationship(back_populates="match", cascade="all, delete-orphan")
    picks:       Mapped[list["Pick"]]         = relationship(back_populates="match", cascade="all, delete-orphan")
    injuries:    Mapped[list["Injury"]]       = relationship(back_populates="match", cascade="all, delete-orphan")
    lineups:     Mapped[list["Lineup"]]       = relationship(back_populates="match", cascade="all, delete-orphan")
    statistics:  Mapped[list["TeamStatistic"]] = relationship(back_populates="match", cascade="all, delete-orphan")
    prediction:  Mapped["Prediction | None"]  = relationship(back_populates="match", cascade="all, delete-orphan", uselist=False)
    model_prediction: Mapped["ModelPrediction | None"] = relationship(back_populates="match", cascade="all, delete-orphan", uselist=False)

    def __repr__(self) -> str:
        return f"<Match {self.home_team_name} vs {self.away_team_name} ({self.match_date.date()})>"


# ---------------------------------------------------------------------------
# Odds (per bookmaker; 1X2 today, extensible via `market`)
# ---------------------------------------------------------------------------

class Odds(Base):
    __tablename__ = "odds"

    id:           Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id:     Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False, index=True)
    bookmaker:    Mapped[str] = mapped_column(String(100), nullable=False)
    market:       Mapped[str] = mapped_column(String(30), nullable=False, default="1x2")

    home_win:     Mapped[float] = mapped_column(Float, nullable=False)
    draw:         Mapped[float] = mapped_column(Float, nullable=False)
    away_win:     Mapped[float] = mapped_column(Float, nullable=False)

    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    match: Mapped["Match"] = relationship(back_populates="odds")

    def __repr__(self) -> str:
        return f"<Odds match={self.match_id} {self.bookmaker} {self.home_win}/{self.draw}/{self.away_win}>"


# ---------------------------------------------------------------------------
# Market odds (flexible: over/under, BTTS, corners, cards, ... any market)
# ---------------------------------------------------------------------------

class MarketOdds(Base):
    __tablename__ = "market_odds"

    id:           Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id:     Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False, index=True)
    bookmaker:    Mapped[str] = mapped_column(String(60), nullable=False)
    market:       Mapped[str] = mapped_column(String(40), nullable=False)  # "ou_goals" | "btts" | "corners_ou" | "cards_ou"...
    selection:    Mapped[str] = mapped_column(String(30), nullable=False)  # "over" | "under" | "yes" | "no"
    line:         Mapped[float | None] = mapped_column(Float, nullable=True)  # 2.5, 9.5... (None for BTTS)
    odd:          Mapped[float] = mapped_column(Float, nullable=False)
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    __table_args__ = (Index("ix_marketodds_match_market", "match_id", "market"),)


# ---------------------------------------------------------------------------
# Injuries
# ---------------------------------------------------------------------------

class Injury(Base):
    __tablename__ = "injuries"

    id:           Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id:     Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False, index=True)
    team_id:      Mapped[int] = mapped_column(Integer, nullable=False)
    team_name:    Mapped[str | None] = mapped_column(String(100), nullable=True)
    player_id:    Mapped[int | None] = mapped_column(Integer, nullable=True)
    player_name:  Mapped[str | None] = mapped_column(String(120), nullable=True)
    type:         Mapped[str | None] = mapped_column(String(80), nullable=True)   # "Missing Fixture" / "Questionable"
    reason:       Mapped[str | None] = mapped_column(String(120), nullable=True)  # "Thigh Injury"...
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    match: Mapped["Match"] = relationship(back_populates="injuries")


# ---------------------------------------------------------------------------
# Lineups (formation + coach per team) and individual players
# ---------------------------------------------------------------------------

class Lineup(Base):
    __tablename__ = "lineups"

    id:           Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id:     Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False, index=True)
    team_id:      Mapped[int] = mapped_column(Integer, nullable=False)
    team_name:    Mapped[str | None] = mapped_column(String(100), nullable=True)
    formation:    Mapped[str | None] = mapped_column(String(20), nullable=True)
    coach:        Mapped[str | None] = mapped_column(String(120), nullable=True)
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    match:   Mapped["Match"] = relationship(back_populates="lineups")
    players: Mapped[list["LineupPlayer"]] = relationship(back_populates="lineup", cascade="all, delete-orphan")


class LineupPlayer(Base):
    __tablename__ = "lineup_players"

    id:          Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lineup_id:   Mapped[int] = mapped_column(ForeignKey("lineups.id"), nullable=False, index=True)
    player_id:   Mapped[int | None] = mapped_column(Integer, nullable=True)
    player_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    number:      Mapped[int | None] = mapped_column(Integer, nullable=True)
    pos:         Mapped[str | None] = mapped_column(String(10), nullable=True)   # G/D/M/F
    grid:        Mapped[str | None] = mapped_column(String(10), nullable=True)   # "4:2" pitch coord
    is_starter:  Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    lineup: Mapped["Lineup"] = relationship(back_populates="players")


# ---------------------------------------------------------------------------
# Team statistics per fixture (long format; includes expected_goals = xG)
# ---------------------------------------------------------------------------

class TeamStatistic(Base):
    __tablename__ = "team_statistics"

    id:           Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id:     Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False, index=True)
    team_id:      Mapped[int] = mapped_column(Integer, nullable=False)
    type:         Mapped[str] = mapped_column(String(60), nullable=False)   # "expected_goals", "Ball Possession"...
    value_num:    Mapped[float | None] = mapped_column(Float, nullable=True)
    value_str:    Mapped[str | None] = mapped_column(String(40), nullable=True)
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    match: Mapped["Match"] = relationship(back_populates="statistics")

    __table_args__ = (Index("ix_teamstat_match_team_type", "match_id", "team_id", "type"),)


# ---------------------------------------------------------------------------
# Predictions (API-Football's own model output)
# ---------------------------------------------------------------------------

class Prediction(Base):
    __tablename__ = "predictions"

    id:           Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id:     Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False, unique=True, index=True)
    winner_id:    Mapped[int | None] = mapped_column(Integer, nullable=True)
    winner_name:  Mapped[str | None] = mapped_column(String(100), nullable=True)
    pct_home:     Mapped[float | None] = mapped_column(Float, nullable=True)
    pct_draw:     Mapped[float | None] = mapped_column(Float, nullable=True)
    pct_away:     Mapped[float | None] = mapped_column(Float, nullable=True)
    advice:       Mapped[str | None] = mapped_column(String(160), nullable=True)
    under_over:   Mapped[str | None] = mapped_column(String(20), nullable=True)
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    match: Mapped["Match"] = relationship(back_populates="prediction")


# ---------------------------------------------------------------------------
# Model prediction snapshots — recorded BEFORE kickoff, graded after the result.
# A read-only observation log to measure (over time) whether the model's
# discrepancy vs market predicts anything. NOT a bet suggestion.
# ---------------------------------------------------------------------------

class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id:          Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id:    Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False, unique=True, index=True)
    snapshot_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    # 1X2 — model vs market (de-vigged)
    m_home:   Mapped[float | None] = mapped_column(Float, nullable=True)
    m_draw:   Mapped[float | None] = mapped_column(Float, nullable=True)
    m_away:   Mapped[float | None] = mapped_column(Float, nullable=True)
    mk_home:  Mapped[float | None] = mapped_column(Float, nullable=True)
    mk_draw:  Mapped[float | None] = mapped_column(Float, nullable=True)
    mk_away:  Mapped[float | None] = mapped_column(Float, nullable=True)

    # Over/Under 2.5 goals
    m_over25:  Mapped[float | None] = mapped_column(Float, nullable=True)
    mk_over25: Mapped[float | None] = mapped_column(Float, nullable=True)

    # BTTS
    m_btts_yes:  Mapped[float | None] = mapped_column(Float, nullable=True)
    mk_btts_yes: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Total corners
    corner_proj: Mapped[float | None] = mapped_column(Float, nullable=True)
    corner_line: Mapped[float | None] = mapped_column(Float, nullable=True)
    corner_std:  Mapped[float | None] = mapped_column(Float, nullable=True)

    # Cuotas decimales reales al momento del snapshot — permiten calcular ROI
    # (no solo acierto direccional). Congeladas antes del partido.
    o_home:         Mapped[float | None] = mapped_column(Float, nullable=True)
    o_draw:         Mapped[float | None] = mapped_column(Float, nullable=True)
    o_away:         Mapped[float | None] = mapped_column(Float, nullable=True)
    o_over25:       Mapped[float | None] = mapped_column(Float, nullable=True)
    o_under25:      Mapped[float | None] = mapped_column(Float, nullable=True)
    o_btts_yes:     Mapped[float | None] = mapped_column(Float, nullable=True)
    o_btts_no:      Mapped[float | None] = mapped_column(Float, nullable=True)
    o_corner_over:  Mapped[float | None] = mapped_column(Float, nullable=True)
    o_corner_under: Mapped[float | None] = mapped_column(Float, nullable=True)

    match: Mapped["Match"] = relationship(back_populates="model_prediction")


# ---------------------------------------------------------------------------
# Picks (the user's own bet log — populated in a later phase)
# ---------------------------------------------------------------------------

class Pick(Base):
    __tablename__ = "picks"

    id:                  Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id:            Mapped[int]   = mapped_column(ForeignKey("matches.id"), nullable=False, index=True)

    bet_type:            Mapped[str]   = mapped_column(String(30), nullable=False)  # home_win | draw | away_win | over25 | btts...
    model_probability:   Mapped[float] = mapped_column(Float, nullable=False)
    implied_probability: Mapped[float] = mapped_column(Float, nullable=False)
    ev:                  Mapped[float] = mapped_column(Float, nullable=False)
    kelly_fraction:      Mapped[float] = mapped_column(Float, nullable=False)
    odds:                Mapped[float] = mapped_column(Float, nullable=False)
    stake:               Mapped[float | None] = mapped_column(Float, nullable=True)

    result:              Mapped[str]   = mapped_column(String(20), nullable=False, default="pending")  # pending | won | lost | void
    pnl:                 Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at:          Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    match: Mapped["Match"] = relationship(back_populates="picks")

    def __repr__(self) -> str:
        return f"<Pick match={self.match_id} {self.bet_type} EV={self.ev:.3f} result={self.result}>"
