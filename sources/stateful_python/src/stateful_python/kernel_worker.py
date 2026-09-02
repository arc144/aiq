# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent scientific Python worker with direct read-only SQLite access."""

from __future__ import annotations

import ast
import contextlib
import io
import itertools
import json
import math
import re
import sqlite3
import statistics
import sys
import traceback
from collections import Counter
from collections import defaultdict
from datetime import date
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels.api as sm
from scipy import stats

_MAX_SQL_ROWS = 100_000
_DEFAULT_SQL_ROWS = 20_000
_MAX_SAMPLE_ROWS = 100
_MAX_VALUE_ROWS = 500


class _CappedStringIO(io.StringIO):
    """Capture output without retaining an unbounded cell transcript."""

    def __init__(self, max_chars: int) -> None:
        super().__init__()
        self.max_chars = max_chars
        self.captured_chars = 0
        self.truncated = False

    def write(self, value: str) -> int:
        text = str(value)
        remaining = max(0, self.max_chars - self.captured_chars)
        if len(text) > remaining:
            self.truncated = True
        if remaining:
            super().write(text[:remaining])
            self.captured_chars += min(len(text), remaining)
        return len(text)


def _combined_output(stdout: _CappedStringIO, stderr: _CappedStringIO, max_output_chars: int) -> str:
    printed = stdout.getvalue()
    warnings = stderr.getvalue()
    combined = printed + (("\n" if printed and warnings else "") + warnings if warnings else "")
    truncated = stdout.truncated or stderr.truncated or len(combined) > max_output_chars
    combined = combined[:max_output_chars]
    if truncated:
        combined += "\n... output truncated ..."
    return combined


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


class ReadOnlyDatabase:
    """Small, model-facing inspection surface for one benchmark SQLite file."""

    def __init__(self, path: Path, name: str) -> None:
        self.path = path.resolve()
        self.name = name
        uri = f"{self.path.as_uri()}?mode=ro&immutable=1"
        self.connection = sqlite3.connect(uri, uri=True)
        self.connection.execute("PRAGMA query_only = ON")

    def close(self) -> None:
        self.connection.close()

    def _objects(self) -> list[tuple[str, str]]:
        rows = self.connection.execute(
            """
            SELECT name, type
            FROM sqlite_schema
            WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        return [(str(name), str(object_type)) for name, object_type in rows]

    def _resolve_table(self, requested: str) -> str:
        matches = [name for name, _ in self._objects() if name.casefold() == requested.strip().casefold()]
        if not matches:
            available = ", ".join(name for name, _ in self._objects())
            raise KeyError(f"Unknown table {requested!r}. Available tables: {available}")
        return matches[0]

    def database_info(self) -> dict[str, Any]:
        """Return the selected database and its available object names."""

        objects = self._objects()
        return {
            "database_name": self.name,
            "database_file": self.path.name,
            "table_count": sum(object_type == "table" for _, object_type in objects),
            "view_count": sum(object_type == "view" for _, object_type in objects),
            "tables": [name for name, _ in objects],
        }

    def tables(self) -> pd.DataFrame:
        """List tables/views and their column counts without scanning table rows."""

        records = []
        for name, object_type in self._objects():
            columns = self.connection.execute(f"PRAGMA table_info({_quote_identifier(name)})").fetchall()
            records.append({"table": name, "type": object_type, "columns": len(columns)})
        return pd.DataFrame(records, columns=["table", "type", "columns"])

    def _description(self, table: str) -> pd.DataFrame:
        description_path = self.path.parent / "database_description" / f"{table}.csv"
        if not description_path.is_file():
            return pd.DataFrame()
        try:
            return pd.read_csv(description_path, encoding_errors="replace").fillna("")
        except Exception:
            return pd.DataFrame()

    def schema(self, table: str | None = None) -> pd.DataFrame:
        """Return physical columns, keys, and available BIRD column descriptions."""

        selected = [self._resolve_table(table)] if table else [name for name, _ in self._objects()]
        records: list[dict[str, Any]] = []
        for table_name in selected:
            descriptions = self._description(table_name)
            description_by_column: dict[str, dict[str, Any]] = {}
            if not descriptions.empty and "original_column_name" in descriptions:
                description_by_column = {
                    str(row["original_column_name"]).casefold(): row.to_dict() for _, row in descriptions.iterrows()
                }
            foreign_keys = {
                str(row[3]): f"{row[2]}.{row[4]}"
                for row in self.connection.execute(
                    f"PRAGMA foreign_key_list({_quote_identifier(table_name)})"
                ).fetchall()
            }
            for _, column_name, declared_type, not_null, default_value, primary_key in self.connection.execute(
                f"PRAGMA table_info({_quote_identifier(table_name)})"
            ).fetchall():
                description = description_by_column.get(str(column_name).casefold(), {})
                records.append(
                    {
                        "table": table_name,
                        "column": str(column_name),
                        "type": str(declared_type or ""),
                        "not_null": bool(not_null),
                        "primary_key": bool(primary_key),
                        "foreign_key": foreign_keys.get(str(column_name), ""),
                        "default": default_value,
                        "description": description.get("column_description", ""),
                        "data_format": description.get("data_format", ""),
                        "value_description": description.get("value_description", ""),
                    }
                )
        return pd.DataFrame(records)

    def relationships(self, table: str | None = None) -> pd.DataFrame:
        """List declared foreign-key relationships for all or one table."""

        selected = [self._resolve_table(table)] if table else [name for name, _ in self._objects()]
        records = []
        for table_name in selected:
            for row in self.connection.execute(f"PRAGMA foreign_key_list({_quote_identifier(table_name)})").fetchall():
                records.append(
                    {
                        "from_table": table_name,
                        "from_column": str(row[3]),
                        "to_table": str(row[2]),
                        "to_column": str(row[4]),
                    }
                )
        return pd.DataFrame(records, columns=["from_table", "from_column", "to_table", "to_column"])

    def sample(self, table: str, n: int = 5) -> pd.DataFrame:
        """Return a small raw sample so formats and grain can be inspected."""

        table_name = self._resolve_table(table)
        if not 1 <= int(n) <= _MAX_SAMPLE_ROWS:
            raise ValueError(f"n must be between 1 and {_MAX_SAMPLE_ROWS}")
        return self.sql(f"SELECT * FROM {_quote_identifier(table_name)} LIMIT {int(n)}", max_rows=int(n))

    def values(self, table: str, column: str, limit: int = 50) -> pd.DataFrame:
        """Return common raw values and counts for one verified table column."""

        table_name = self._resolve_table(table)
        columns = {
            str(row[1]).casefold(): str(row[1])
            for row in self.connection.execute(f"PRAGMA table_info({_quote_identifier(table_name)})").fetchall()
        }
        column_name = columns.get(column.strip().casefold())
        if column_name is None:
            raise KeyError(f"Unknown column {column!r} in table {table_name!r}")
        if not 1 <= int(limit) <= _MAX_VALUE_ROWS:
            raise ValueError(f"limit must be between 1 and {_MAX_VALUE_ROWS}")
        quoted_table = _quote_identifier(table_name)
        quoted_column = _quote_identifier(column_name)
        query = (
            f"SELECT {quoted_column} AS value, COUNT(*) AS count FROM {quoted_table} "
            f"GROUP BY {quoted_column} ORDER BY count DESC, value LIMIT {int(limit)}"
        )
        return self.sql(query, max_rows=int(limit))

    def sql(self, query: str, max_rows: int = _DEFAULT_SQL_ROWS) -> pd.DataFrame:
        """Execute one read-only SELECT/CTE and return exact rows as a DataFrame."""

        normalized = query.strip()
        if not normalized:
            raise ValueError("query is required")
        if not 1 <= int(max_rows) <= _MAX_SQL_ROWS:
            raise ValueError(f"max_rows must be between 1 and {_MAX_SQL_ROWS}")
        if not re.match(r"(?is)^(?:SELECT|WITH|EXPLAIN\s+QUERY\s+PLAN)\b", normalized):
            raise ValueError("sql() accepts only a read-only SELECT, WITH, or EXPLAIN QUERY PLAN statement")
        cursor = self.connection.execute(normalized)
        if cursor.description is None:
            raise ValueError("query returned no tabular result")
        rows = cursor.fetchmany(int(max_rows) + 1)
        truncated = len(rows) > int(max_rows)
        rows = rows[: int(max_rows)]
        frame = pd.DataFrame.from_records(rows, columns=[column[0] for column in cursor.description])
        frame.attrs.update(
            {
                "database_name": self.name,
                "row_count": len(frame),
                "truncated": truncated,
                "sql": normalized,
            }
        )
        return frame

    def query_plan(self, query: str) -> pd.DataFrame:
        """Inspect SQLite's plan for a read-only query before running an expensive join."""

        normalized = query.strip()
        if not re.match(r"(?is)^(?:SELECT|WITH)\b", normalized):
            raise ValueError("query_plan() accepts only SELECT or WITH statements")
        return self.sql(f"EXPLAIN QUERY PLAN {normalized}", max_rows=10_000)


def _compile_cell(code: str) -> tuple[Any | None, Any | None]:
    tree = ast.parse(code, mode="exec")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        statements = ast.Module(body=tree.body[:-1], type_ignores=[])
        expression = ast.Expression(body=tree.body[-1].value)
        return compile(statements, "<aiq-python>", "exec"), compile(expression, "<aiq-python>", "eval")
    return compile(tree, "<aiq-python>", "exec"), None


def _display(value: Any, max_output_chars: int) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.DataFrame):
        rendered = value.to_string(max_rows=60, max_cols=50, line_width=180)
        if value.attrs:
            rendered += f"\nattrs={value.attrs!r}"
    elif isinstance(value, pd.Series):
        rendered = value.to_string(max_rows=100)
    elif isinstance(value, np.ndarray):
        rendered = np.array2string(value, threshold=500, edgeitems=20)
    else:
        rendered = repr(value)
    if len(rendered) <= max_output_chars:
        return rendered
    return rendered[:max_output_chars] + "\n... output truncated ..."


def _visible_variables(namespace: dict[str, Any]) -> list[str]:
    hidden = {
        "Counter",
        "date",
        "database_info",
        "datetime",
        "defaultdict",
        "itertools",
        "json",
        "math",
        "np",
        "pd",
        "query_plan",
        "re",
        "relationships",
        "sample",
        "schema",
        "scipy",
        "sklearn",
        "sm",
        "sql",
        "statistics",
        "stats",
        "tables",
        "timedelta",
        "timezone",
        "values",
    }
    return sorted(name for name in namespace if not name.startswith("_") and name not in hidden)


def _execute(namespace: dict[str, Any], code: str, max_output_chars: int) -> dict[str, Any]:
    stdout = _CappedStringIO(max_output_chars)
    stderr = _CappedStringIO(max_output_chars)
    try:
        statements, expression = _compile_cell(code)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if statements is not None:
                exec(statements, namespace)
            value = eval(expression, namespace) if expression is not None else None
        return {
            "status": "ok",
            "output": _combined_output(stdout, stderr, max_output_chars),
            "result": _display(value, max_output_chars),
            "result_type": type(value).__name__ if value is not None else None,
            "variables": _visible_variables(namespace),
        }
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        return {
            "status": "error",
            "error": type(exc).__name__,
            "detail": str(exc)[:2_000],
            "traceback": "".join(traceback.format_exception(exc))[-4_000:],
            "output": _combined_output(stdout, stderr, max_output_chars),
            "variables": _visible_variables(namespace),
        }


def _new_namespace(database: ReadOnlyDatabase) -> dict[str, Any]:
    return {
        "__name__": "__aiq_analysis__",
        "Counter": Counter,
        "date": date,
        "database_info": database.database_info,
        "datetime": datetime,
        "defaultdict": defaultdict,
        "itertools": itertools,
        "json": json,
        "math": math,
        "np": np,
        "pd": pd,
        "query_plan": database.query_plan,
        "re": re,
        "relationships": database.relationships,
        "sample": database.sample,
        "schema": database.schema,
        "scipy": scipy,
        "sklearn": sklearn,
        "sm": sm,
        "sql": database.sql,
        "statistics": statistics,
        "stats": stats,
        "tables": database.tables,
        "timedelta": timedelta,
        "timezone": timezone,
        "values": database.values,
    }


def main() -> None:
    if len(sys.argv) != 4:
        raise ValueError("kernel worker requires <database_path> <database_name> <max_output_chars>")
    database_path = Path(sys.argv[1])
    database_name = sys.argv[2]
    max_output_chars = int(sys.argv[3])
    database = ReadOnlyDatabase(database_path, database_name)
    namespace = _new_namespace(database)
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("operation") == "close":
                    response = {"status": "ok", "operation": "close"}
                    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                    sys.stdout.flush()
                    return
                response = _execute(namespace, str(request.get("code") or ""), max_output_chars)
            except Exception as exc:
                response = {"status": "error", "error": type(exc).__name__, "detail": str(exc)[:2_000]}
            sys.stdout.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
            sys.stdout.flush()
    finally:
        database.close()


if __name__ == "__main__":
    main()
