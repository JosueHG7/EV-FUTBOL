import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from database.db import get_engine, init_db
from database.models import Base
import config

if __name__ == "__main__":
    init_db()

    engine = get_engine()
    existing = engine.dialect.get_table_names(engine.connect())
    for table_name in Base.metadata.sorted_tables:
        status = "OK" if table_name.name in existing else "ERROR"
        print(f"  [{status}] {table_name.name}")

    size_kb = config.DB_PATH.stat().st_size / 1024
    print(f"\n  Ruta : {config.DB_PATH}")
    print(f"  Tamaño: {size_kb:.1f} KB")
