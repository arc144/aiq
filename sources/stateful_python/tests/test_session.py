# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for persistent Python and direct read-only SQLite helpers."""

import json
import sqlite3
from pathlib import Path

import pytest
from stateful_python.session import PersistentPythonSession


def _database(path: Path) -> Path:
    database = path / "example.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(customer_id),
            amount TEXT,
            ordered_at TEXT
        );
        INSERT INTO customers VALUES (1, 'Alpha'), (2, 'Beta');
        INSERT INTO orders VALUES
            (1, 1, '1,200.50', '1/2/24'),
            (2, 1, '10.00', '2/3/24'),
            (3, 2, '50.00', '2/4/24');
        """
    )
    connection.close()
    return database


@pytest.mark.asyncio
async def test_variables_and_dataframes_persist_across_cells(tmp_path: Path) -> None:
    session = PersistentPythonSession(
        database_path=_database(tmp_path),
        database_name="example",
        working_directory=tmp_path,
    )
    try:
        first = json.loads(await session.execute("frame = pd.DataFrame({'value': [1, 2, 3]})\nframe"))
        second = json.loads(await session.execute("frame['value'].sum()"))
    finally:
        await session.aclose()

    assert first["status"] == "ok"
    assert "frame" in first["variables"]
    assert second["status"] == "ok"
    assert second["result"] == "np.int64(6)"


@pytest.mark.asyncio
async def test_direct_sql_helpers_expose_schema_rows_and_relationships(tmp_path: Path) -> None:
    session = PersistentPythonSession(
        database_path=_database(tmp_path),
        database_name="example",
        working_directory=tmp_path,
    )
    try:
        response = json.loads(
            await session.execute(
                'frame = sql("SELECT customer_id, COUNT(*) AS orders FROM orders GROUP BY customer_id")\n'
                "(database_info()['database_name'], int(frame['orders'].sum()), "
                "schema('orders').shape[0], relationships('orders').shape[0])"
            )
        )
    finally:
        await session.aclose()

    assert response["status"] == "ok"
    assert response["result"] == "('example', 3, 4, 1)"


@pytest.mark.asyncio
async def test_database_is_read_only(tmp_path: Path) -> None:
    session = PersistentPythonSession(
        database_path=_database(tmp_path),
        database_name="example",
        working_directory=tmp_path,
    )
    try:
        response = json.loads(await session.execute('sql("DELETE FROM orders")'))
    finally:
        await session.aclose()

    assert response["status"] == "error"
    assert response["error"] == "ValueError"
