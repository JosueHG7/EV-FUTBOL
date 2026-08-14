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


def init_db() -> None:
    """Crea todas las tablas si no existen. Seguro de llamar múltiples veces."""
    Base.metadata.create_all(get_engine())
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
