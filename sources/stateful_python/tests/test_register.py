# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT registration tests for request-scoped persistent Python."""

import json
import sqlite3
from unittest.mock import MagicMock

import pytest
from stateful_python import register as register_module

from aiq_agent.agents.data_science.utils.analysis_runtime import begin_analysis_run
from aiq_agent.agents.data_science.utils.analysis_runtime import end_analysis_run


@pytest.mark.asyncio
async def test_registration_selects_request_database_and_returns_exact_rows(tmp_path) -> None:
    database = tmp_path / "example.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript("CREATE TABLE values_table (value INTEGER); INSERT INTO values_table VALUES (4), (9);")
    connection.close()
    config = register_module.StatefulPythonConfig(wall_timeout_seconds=60, database_root=tmp_path)
    registration = register_module.stateful_python.__wrapped__(config, MagicMock())
    function_info = await anext(registration)
    token = begin_analysis_run(database_name="example")
    try:
        response = json.loads(
            await function_info.single_fn("frame = sql('SELECT value FROM values_table')\nframe['value'].mean()")
        )
    finally:
        await end_analysis_run(token)
        await registration.aclose()

    assert response["status"] == "ok"
    assert response["result"] == "np.float64(6.5)"
