import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import Session, sessionmaker

sys.path.append(str(Path(__file__).parent.parent))
import config
from database.models import Base

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(config.DATABASE_URL, echo=False)
    return _engine


# Columnas añadidas después de la creación inicial de la tabla. create_all() NO
# altera tablas existentes, así que las agregamos a mano (SQLite ADD COLUMN es
# barato e idempotente). {tabla: {columna: tipo_sql}}.
_ADDED_COLUMNS = {
    "model_predictions": {
        "o_home": "FLOAT", "o_draw": "FLOAT", "o_away": "FLOAT",
        "o_over25": "FLOAT", "o_under25": "FLOAT",
        "o_btts_yes": "FLOAT", "o_btts_no": "FLOAT",
        "o_corner_over": "FLOAT", "o_corner_under": "FLOAT",
        "card_proj": "FLOAT", "card_line": "FLOAT", "card_std": "FLOAT",
        "o_card_over": "FLOAT", "o_card_under": "FLOAT",
        "leak_flagged": "BOOLEAN NOT NULL DEFAULT 0",
    },
}


def _apply_column_migrations(engine: Engine) -> None:
    """Añade columnas nuevas a tablas ya existentes (no-op si la tabla no existe
    aún o si ya las tiene). Evita 'no such column' contra BD antiguas."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table, cols in _ADDED_COLUMNS.items():
            if table not in tables:
                continue  # create_all ya la creó completa
            existing = {c["name"] for c in inspector.get_columns(table)}
            for name, sqltype in cols.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {sqltype}"))


def init_db() -> None:
    """Crea todas las tablas si no existen y aplica migraciones de columnas.
    Seguro de llamar múltiples veces."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    _apply_column_migrations(engine)
    print(f"Base de datos inicializada: {config.DB_PATH}")


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager que provee una sesión y hace commit/rollback automático."""
    factory = sessionmaker(bind=get_engine())
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    init_db()
