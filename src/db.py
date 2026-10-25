"""Database helpers: one code path for PostgreSQL and SQLite."""
import csv
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from config import DATABASE_URL, SQL_DIR

VIEW_NAMES = []  # populated from sql/views.sql


def get_engine(url: str | None = None):
    return create_engine(url or DATABASE_URL, future=True)


def is_postgres(engine) -> bool:
    return engine.dialect.name == "postgresql"


def _split_statements(sql: str) -> list[str]:
    sql = re.sub(r"--[^\n]*", "", sql)
    return [s.strip() for s in sql.split(";") if s.strip()]


def run_sql_file(engine, filename: str):
    sql = (SQL_DIR / filename).read_text()
    with engine.begin() as conn:
        for stmt in _split_statements(sql):
            conn.execute(text(stmt))


def load_named_queries(filename: str) -> dict[str, str]:
    """Parse a .sql file with `-- name: xyz` markers into {name: query}."""
    raw = (SQL_DIR / filename).read_text()
    parts = re.split(r"^--\s*name:\s*(\w+)\s*$", raw, flags=re.M)
    return {parts[i]: parts[i + 1].strip().rstrip(";") for i in range(1, len(parts), 2)}


def view_names() -> list[str]:
    raw = (SQL_DIR / "views.sql").read_text()
    return re.findall(r"CREATE VIEW\s+(\w+)", raw, flags=re.I)


def drop_views(engine):
    with engine.begin() as conn:
        for v in reversed(view_names()):
            conn.execute(text(f"DROP VIEW IF EXISTS {v}"))


def create_views(engine):
    drop_views(engine)
    run_sql_file(engine, "views.sql")


def read_sql(query: str, engine, params: dict | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


def _pg_copy(table, conn, keys, data_iter):
    """Fast bulk insert for PostgreSQL using COPY."""
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        buf = io.StringIO()
        writer = csv.writer(buf)
        for row in data_iter:
            writer.writerow([None if (isinstance(v, float) and v != v) else v for v in row])
        buf.seek(0)
        cols = ", ".join(f'"{k}"' for k in keys)
        name = f"{table.schema}.{table.name}" if table.schema else table.name
        cur.copy_expert(f"COPY {name} ({cols}) FROM STDIN WITH CSV", buf)


def write_df(df: pd.DataFrame, table: str, engine, if_exists: str = "replace"):
    df = df.replace([np.inf, -np.inf], np.nan)
    method = _pg_copy if is_postgres(engine) else None
    df.to_sql(table, engine, if_exists=if_exists, index=False,
              method=method, chunksize=None if method else 20_000)
