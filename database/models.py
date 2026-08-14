from datetime import datetime, timezone
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Match(Base):
    __tablename__ = "matches"

    id:             Mapped[int]      = mapped_column(Integer, primary_key=True)
    league_id:      Mapped[int]      = mapped_column(Integer, nullable=False)
    league_name:    Mapped[str]      = mapped_column(String(100), nullable=False)
    season:         Mapped[int]      = mapped_column(Integer, nullable=False)

    home_team_id:   Mapped[int]      = mapped_column(Integer, nullable=False)
    home_team_name: Mapped[str]      = mapped_column(String(100), nullable=False)
    away_team_id:   Mapped[int]      = mapped_column(Integer, nullable=False)
    away_team_name: Mapped[str]      = mapped_column(String(100), nullable=False)

    match_date:     Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status:         Mapped[str]      = mapped_column(String(20), nullable=False, default="scheduled")

    home_goals:     Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals:     Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at:     Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    odds:  Mapped[list["Odds"]] = relationship("Odds", back_populates="match", cascade="all, delete-orphan")
    picks: Mapped[list["Pick"]] = relationship("Pick", back_populates="match", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Match {self.home_team_name} vs {self.away_team_name} ({self.match_date.date()})>"


class Odds(Base):
    __tablename__ = "odds"

    id:                 Mapped[int]      = mapped_column(Integer, primary_key=True)
    match_id:           Mapped[int]      = mapped_column(ForeignKey("matches.id"), nullable=False)
    bookmaker:          Mapped[str]      = mapped_column(String(100), nullable=False)

    home_win:           Mapped[float]    = mapped_column(Float, nullable=False)
    draw:               Mapped[float]    = mapped_column(Float, nullable=False)
    away_win:           Mapped[float]    = mapped_column(Float, nullable=False)

    collected_at:       Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    match: Mapped["Match"] = relationship("Match", back_populates="odds")

    def __repr__(self) -> str:
        return f"<Odds match={self.match_id} {self.bookmaker} {self.home_win}/{self.draw}/{self.away_win}>"


class Pick(Base):
    __tablename__ = "picks"

    id:                   Mapped[int]   = mapped_column(Integer, primary_key=True)
    match_id:             Mapped[int]   = mapped_column(ForeignKey("matches.id"), nullable=False)

    bet_type:             Mapped[str]   = mapped_column(String(20), nullable=False)   # home_win | draw | away_win

    model_probability:    Mapped[float] = mapped_column(Float, nullable=False)
    implied_probability:  Mapped[float] = mapped_column(Float, nullable=False)
    ev:                   Mapped[float] = mapped_column(Float, nullable=False)
    kelly_fraction:       Mapped[float] = mapped_column(Float, nullable=False)
    odds:                 Mapped[float] = mapped_column(Float, nullable=False)

    result:               Mapped[str]   = mapped_column(String(20), nullable=False, default="pending")  # pending | won | lost

    created_at:           Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    match: Mapped["Match"] = relationship("Match", back_populates="picks")

    def __repr__(self) -> str:
        return f"<Pick match={self.match_id} {self.bet_type} EV={self.ev:.3f} result={self.result}>"
