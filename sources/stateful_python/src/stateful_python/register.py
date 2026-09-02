# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register a request-scoped persistent Python kernel as a NAT function."""

import json
from pathlib import Path

from pydantic import ConfigDict
from pydantic import Field

from aiq_agent.agents.data_science.utils.analysis_runtime import get_analysis_run
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from .session import PersistentPythonSession
from .session import PythonSessionLimits


class StatefulPythonConfig(FunctionBaseConfig, name="stateful_python"):
    """Configuration for one persistent Python kernel per DS request."""

    model_config = ConfigDict(extra="forbid")

    wall_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    max_code_chars: int = Field(default=50_000, ge=1_000, le=500_000)
    max_output_chars: int = Field(default=50_000, ge=1_000, le=500_000)
    database_root: Path = Field(description="Root containing benchmark SQLite databases, searched recursively.")


def _database_index(root: Path) -> dict[str, Path]:
    resolved_root = root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"stateful_python database_root is not a directory: {resolved_root}")
    index: dict[str, Path] = {}
    for path in sorted(resolved_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".db", ".sqlite"}:
            continue
        key = path.stem.casefold()
        prior = index.get(key)
        if prior is not None and prior != path:
            raise ValueError(f"duplicate benchmark database name {path.stem!r}: {prior} and {path}")
        index[key] = path.resolve()
    if not index:
        raise ValueError(f"stateful_python found no SQLite databases under {resolved_root}")
    return index


@register_function(config_type=StatefulPythonConfig)
async def stateful_python(tool_config: StatefulPythonConfig, _builder: Builder):
    """Build one model-facing Python tool backed by request-owned kernels."""

    databases = _database_index(tool_config.database_root)
    limits = PythonSessionLimits(
        wall_timeout_seconds=tool_config.wall_timeout_seconds,
        max_code_chars=tool_config.max_code_chars,
        max_output_chars=tool_config.max_output_chars,
    )

    async def _run(code: str) -> str:
        """Execute Python in the persistent analysis kernel for this request.

        Variables, imports, DataFrames, and fitted objects persist across calls.
        NumPy (`np`), pandas (`pd`), SciPy (`scipy`, `stats`), scikit-learn
        (`sklearn`), and statsmodels (`sm`) are preloaded. The FDABench task's
        selected SQLite database is connected read-only. Inspect it with
        `database_info()`, `tables()`, `schema(table=None)`,
        `relationships(table=None)`, `sample(table, n=5)`, and
        `values(table, column, limit=50)`. Execute read-only analytical SQL with
        `sql(query, max_rows=20000)` and inspect expensive joins with
        `query_plan(query)`. `sql()` returns an exact pandas DataFrame that can be
        assigned to a persistent variable and analyzed in later Python calls.
        """

        run_state = get_analysis_run()
        if run_state is None:
            return '{"status":"error","error":"analysis_runtime_unavailable"}'
        database_name = (run_state.database_name or "").strip()
        if not database_name:
            return '{"status":"error","error":"database_name_unavailable"}'
        database_path = databases.get(database_name.casefold())
        if database_path is None:
            return json.dumps(
                {
                    "status": "error",
                    "error": "database_not_found",
                    "database_name": database_name,
                },
                separators=(",", ":"),
            )
        session = run_state.resources.get("stateful_python")
        if session is None:
            session = PersistentPythonSession(
                database_path=database_path,
                database_name=database_name,
                working_directory=run_state.root,
                limits=limits,
            )
            run_state.resources["stateful_python"] = session
        return await session.execute(code)

    yield FunctionInfo.from_fn(_run, description=_run.__doc__)
