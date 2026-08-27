"""Architectural invariants: workflow, not an agent."""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
import textwrap
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, StateGraph

import assist.graph.edges as edges_mod
from assist.config import settings
from assist.domain.catalog import Candidate
from assist.domain.constraints import ConstraintState
from assist.domain.context import ServerUserCtx
from assist.domain.enums import (
    DegradedReason,
    DeviceClass,
    MaturityRating,
    MediaType,
    Package,
    Route,
)
from assist.graph.build import (
    _assert_workflow_invariants,
    build_graph,
    default_mermaid_path,
    export_mermaid,
    passthrough,
)
from assist.graph.build import (
    _simple_cycles as build_simple_cycles,
)
from assist.graph.edges import after_guard, after_retrieve, route_reply
from assist.graph.state import PersonSoft, TurnState, empty_turn_state

EDGES_PATH = Path(edges_mod.__file__).resolve()
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_NODES_DIR = _SRC_ROOT / "assist" / "nodes"

_FORBIDDEN_IMPORTS = frozenset(
    {
        "assist.llm",
        "langchain_anthropic",
        "langchain_openai",
        "anthropic",
        "openai",
        "httpx",
        "redis",
        "elasticsearch",
        "asyncpg",
        "sqlalchemy",
        "aiohttp",
        "requests",
        "socket",
        "urllib",
        "http.client",
    }
)
_LLM_PROVIDER_IMPORTS = frozenset(
    {
        "langchain_anthropic",
        "langchain_openai",
        "langchain_google_genai",
        "langchain_community.chat_models",
        "anthropic",
        "openai",
    }
)
_TOOL_TYPE_NAMES = frozenset({"ToolNode", "AgentExecutor"})
_TOOL_CALL_NAMES = frozenset({"bind_tools", "ToolNode", "AgentExecutor", "with_structured_output"})
_TOOL_KWARG_CALLS = frozenset({"bind", "invoke", "ainvoke", "with_config", "bind_tools"})
_CYCLE_NODES = frozenset({"retrieve", "broaden_constraints"})
_IGNORE_PREFIX = "__"
_RECURSION_CONFIG: RunnableConfig = {"recursion_limit": 40}


def _ctx() -> ServerUserCtx:
    return ServerUserCtx(
        user_id="u1",
        profile_id="p1",
        geo="US",
        package=Package.BASIC,
        maturity_max=MaturityRating.PG_13,
        kids_flag=False,
        device_class=DeviceClass.WEB,
    )


def _candidate(catalog_id: str = "ttl_1") -> Candidate:
    return Candidate(
        catalog_id=catalog_id,
        title=catalog_id,
        media_type=MediaType.FILM,
        score=1.0,
    )


def _is_first_party_name(mod: str) -> bool:
    return mod == "assist" or mod.startswith("assist.")


def _module_name_from_path(path: Path, src_root: Path = _SRC_ROOT) -> str | None:
    try:
        rel = path.resolve().relative_to(src_root.resolve())
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if any(part in {"", ".", ".."} for part in parts):
        return None
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def _resolve_relative(current_module: str, level: int, module: str | None) -> str | None:
    parts = current_module.split(".")
    if level < 1 or level > len(parts):
        return None
    parent = parts[:-level]
    if module:
        return ".".join([*parent, *module.split(".")]) if parent else module
    return ".".join(parent) if parent else None


def _const_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _dotted_call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: list[str] = []
        current: ast.AST = func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return ".".join(reversed(parts))
    return None


def _dynamic_imported_module(node: ast.Call, current_module: str | None) -> str | None:
    name = _dotted_call_name(node.func)
    if name is None:
        return None
    short = name.rsplit(".", 1)[-1]
    if short not in {"import_module", "__import__"}:
        return None
    if not node.args:
        return None
    spec = _const_str(node.args[0])
    if spec is None:
        return None
    if not spec.startswith("."):
        return spec
    package = None
    if len(node.args) >= 2:
        package = _const_str(node.args[1])
    for kw in node.keywords:
        if kw.arg == "package":
            package = _const_str(kw.value)
    if package is None and current_module is not None:
        package = current_module.rsplit(".", 1)[0] if "." in current_module else current_module
    if not package:
        return None
    level = len(spec) - len(spec.lstrip("."))
    rest = spec.lstrip(".") or None
    return _resolve_relative(package + "._", level, rest)


def _imported_modules(path: Path, *, current_module: str | None = None) -> set[str]:
    """Imported names from AST: relatives resolved, importlib/__import__ included."""
    if current_module is None:
        current_module = _module_name_from_path(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()

    def add(mod: str | None) -> None:
        if mod:
            modules.add(mod)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if current_module is None:
                    add(node.module)
                    continue
                resolved = _resolve_relative(current_module, node.level, node.module)
                add(resolved)
                if resolved:
                    for alias in node.names:
                        add(f"{resolved}.{alias.name}")
            elif node.module is not None:
                add(node.module)
                if node.module == "assist" or node.module.startswith("assist."):
                    for alias in node.names:
                        add(f"{node.module}.{alias.name}")
        elif isinstance(node, ast.Call):
            add(_dynamic_imported_module(node, current_module))
    return modules


def _is_forbidden(mod: str) -> bool:
    for banned in _FORBIDDEN_IMPORTS:
        if mod == banned or mod.startswith(banned + "."):
            return True
    return False


def _is_llm_provider(mod: str) -> bool:
    for banned in _LLM_PROVIDER_IMPORTS:
        if mod == banned or mod.startswith(banned + "."):
            return True
    return False


def _module_source_path(mod: str, src_root: Path = _SRC_ROOT) -> Path | None:
    if not _is_first_party_name(mod):
        return None
    parts = mod.split(".")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    rel = Path(*parts)
    py_path = src_root / rel.with_suffix(".py")
    init_path = src_root / rel / "__init__.py"
    for candidate in (py_path, init_path):
        if not candidate.is_file():
            continue
        try:
            candidate.resolve().relative_to(src_root.resolve())
        except ValueError:
            return None
        return candidate
    return None


def _reachable_import_names(root_module: str, *, src_root: Path = _SRC_ROOT) -> set[str]:
    """First-party import graph from `root_module`. Does not walk site-packages."""
    seen: set[str] = set()
    names: set[str] = set()
    stack = [root_module]
    while stack:
        mod = stack.pop()
        if mod in seen:
            continue
        path = _module_source_path(mod, src_root)
        if path is None:
            continue
        seen.add(mod)
        imported = _imported_modules(path, current_module=mod)
        names.update(imported)
        for name in imported:
            if _is_first_party_name(name) and name not in seen:
                stack.append(name)
    return names


def _first_party_modules_loaded_by_import(module: str) -> set[str]:
    probe = (
        "import importlib, sys\n"
        "before = set(sys.modules)\n"
        "importlib.import_module(sys.argv[1])\n"
        "for name in sorted(set(sys.modules) - before):\n"
        "    print(name)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, module],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    loaded = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    return {name for name in loaded if _is_first_party_name(name)}


def _adjacency(compiled: Any) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = defaultdict(list)
    for edge in compiled.get_graph().edges:
        adj[edge.source].append(edge.target)
    return dict(adj)


def _simple_cycles(adj: Mapping[str, list[str]]) -> list[tuple[str, ...]]:
    """Enumerate directed simple cycles, including self-loops.

    Independent of assist.graph.build so a bug there cannot hide in this test.
    """
    found: list[tuple[str, ...]] = []
    for start in sorted(adj):
        stack: list[tuple[str, list[str], set[str]]] = [(start, [start], {start})]
        while stack:
            current, path, seen = stack.pop()
            for nxt in adj.get(current, ()):
                if nxt == start:
                    found.append(tuple(path))
                elif nxt not in seen and nxt > start:
                    stack.append((nxt, [*path, nxt], seen | {nxt}))
    return found


def _has_tools(obj: object, seen: set[int] | None = None) -> bool:
    if obj is None:
        return False
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return False
    seen.add(oid)

    name = type(obj).__name__
    if name in {"ToolNode", "AgentExecutor"}:
        return True

    tools = getattr(obj, "tools", None)
    if tools:
        return True
    kwargs = getattr(obj, "kwargs", None)
    if isinstance(kwargs, dict) and kwargs.get("tools"):
        return True

    for attr in ("bound", "runnable", "first", "last", "middle", "mapper", "func", "afunc"):
        child = getattr(obj, attr, None)
        if child is not None and child is not obj and _has_tools(child, seen):
            return True
    steps = getattr(obj, "steps", None)
    if isinstance(steps, (list, tuple)):
        for item in steps:
            if _has_tools(item, seen):
                return True
    return False


def _tree_uses_tools(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {
            "bind_tools",
            "with_structured_output",
        }:
            return True
        if isinstance(node, ast.Name) and node.id in _TOOL_TYPE_NAMES:
            return True
        if isinstance(node, ast.Call):
            dotted = _dotted_call_name(node.func) or ""
            short = dotted.rsplit(".", 1)[-1]
            if short in _TOOL_CALL_NAMES:
                return True
            if short in _TOOL_KWARG_CALLS and any(kw.arg == "tools" for kw in node.keywords):
                return True
    return False


def _source_uses_tools(obj: object) -> bool:
    try:
        target = inspect.unwrap(obj)  # type: ignore[arg-type]
        src = inspect.getsource(target)
    except (OSError, TypeError):
        return False
    tree = ast.parse(textwrap.dedent(src))
    return _tree_uses_tools(tree)


def _path_uses_tools(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _tree_uses_tools(tree)


def _provider_import_hits(path: Path, *, current_module: str | None = None) -> list[str]:
    modules = _imported_modules(path, current_module=current_module)
    return [mod for mod in sorted(modules) if _is_llm_provider(mod)]


def _node_callable(node: object) -> object:
    bound = getattr(node, "bound", node)
    fn = getattr(bound, "afunc", None) or getattr(bound, "func", None)
    if fn is None:
        return bound
    return inspect.unwrap(fn)


def _defining_module_file(obj: object) -> Path | None:
    module = inspect.getmodule(inspect.unwrap(obj))  # type: ignore[arg-type]
    filename = getattr(module, "__file__", None) if module is not None else None
    if not filename:
        return None
    return Path(filename).resolve()


def _is_under(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _iter_node_paths() -> list[Path]:
    if not _NODES_DIR.is_dir():
        return []
    return sorted(path for path in _NODES_DIR.rglob("*.py") if path.name != "__init__.py")


def _scratch_add_node(graph: StateGraph[TurnState], name: str, fn: Any) -> None:
    """LangGraph node overloads reject async defs under mypy; same seam as build._add_node."""
    graph.add_node(name, fn)


def test_edges_imports_nothing_from_llm() -> None:
    modules = _imported_modules(EDGES_PATH)
    llm_hits = [mod for mod in sorted(modules) if _is_forbidden(mod)]
    assert llm_hits == [], f"edges.py imported forbidden modules: {llm_hits}"
    assert not any(mod == "assist.llm" or mod.startswith("assist.llm.") for mod in modules)


def test_relative_llm_import_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "edges.py"
    path.write_text("from ..llm import gateway\nfrom .. import llm\n", encoding="utf-8")
    modules = _imported_modules(path, current_module="assist.graph.edges")
    hits = [mod for mod in sorted(modules) if _is_forbidden(mod)]
    assert "assist.llm" in modules
    assert hits, f"relative llm imports must be detected, got {sorted(modules)}"


def test_dynamic_llm_import_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "edges.py"
    path.write_text(
        "import importlib\n"
        "from importlib import import_module\n"
        "importlib.import_module('assist.llm')\n"
        "import_module('assist.llm.gateway')\n"
        "__import__('assist.llm')\n",
        encoding="utf-8",
    )
    modules = _imported_modules(path, current_module="assist.graph.edges")
    hits = [mod for mod in sorted(modules) if _is_forbidden(mod)]
    assert "assist.llm" in modules
    assert "assist.llm.gateway" in modules
    assert hits, f"importlib/__import__ llm must be detected, got {sorted(modules)}"


def test_transitive_llm_import_is_detected(tmp_path: Path) -> None:
    src = tmp_path / "src"
    graph_dir = src / "assist" / "graph"
    graph_dir.mkdir(parents=True)
    (src / "assist" / "__init__.py").write_text("", encoding="utf-8")
    (graph_dir / "__init__.py").write_text("", encoding="utf-8")
    (graph_dir / "helper.py").write_text("from assist.llm import gateway\n", encoding="utf-8")
    (graph_dir / "edges.py").write_text("from .helper import gateway\n", encoding="utf-8")
    imported = _reachable_import_names("assist.graph.edges", src_root=src)
    hits = [mod for mod in sorted(imported) if _is_forbidden(mod)]
    assert hits, f"transitive llm import must be detected, got {sorted(imported)}"


def test_edges_cannot_reach_forbidden_modules_transitively() -> None:
    imported = _reachable_import_names("assist.graph.edges")
    hits = [mod for mod in sorted(imported) if _is_forbidden(mod)]
    assert hits == [], f"edges import graph reached forbidden modules: {hits}"
    loaded = _first_party_modules_loaded_by_import("assist.graph.edges")
    loaded_hits = [mod for mod in sorted(loaded) if _is_forbidden(mod)]
    assert loaded_hits == [], f"importing edges loaded forbidden modules: {loaded_hits}"


def test_graph_compiles_without_checkpointer() -> None:
    compiled = build_graph()
    assert compiled.checkpointer is False
    assert compiled.get_graph() is not None


def test_only_cycle_is_retrieve_broaden() -> None:
    compiled = build_graph()
    stage_nodes = {name for name in compiled.nodes if not str(name).startswith(_IGNORE_PREFIX)}
    cycles = _simple_cycles(_adjacency(compiled))
    assert cycles, "expected the retrieve↔broaden cycle"

    participants: set[str] = set()
    for cycle in cycles:
        nodes = {name for name in cycle if name in stage_nodes}
        if not nodes:
            continue
        assert nodes == _CYCLE_NODES, f"unexpected cycle among {sorted(nodes)}: {cycle}"
        participants.update(nodes)
    assert participants == _CYCLE_NODES


def test_self_loop_is_a_cycle() -> None:
    adj = {"rank": ["rank"]}
    test_cycles = _simple_cycles(adj)
    build_cycles = build_simple_cycles(adj)
    assert test_cycles, "self-loop rank->rank must be a cycle"
    assert build_cycles, "build enumerator must treat a self-loop as a cycle"
    assert {node for cycle in test_cycles for node in cycle} == {"rank"}


def test_scratch_self_loop_fails_workflow_invariants() -> None:
    graph: StateGraph[TurnState] = StateGraph(TurnState)
    _scratch_add_node(graph, "retrieve", passthrough)
    _scratch_add_node(graph, "broaden_constraints", passthrough)
    _scratch_add_node(graph, "rank", passthrough)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "broaden_constraints")
    graph.add_edge("broaden_constraints", "retrieve")
    graph.add_edge("retrieve", "rank")
    graph.add_edge("rank", "rank")
    compiled = graph.compile(checkpointer=False)
    with pytest.raises(RuntimeError, match="cycle"):
        _assert_workflow_invariants(compiled)


async def test_stub_turn_runs_end_to_end() -> None:
    compiled = build_graph()
    result = await compiled.ainvoke(empty_turn_state(_ctx(), text="something cozy"))
    assert result["retrieve_attempts"] >= 1
    assert "ctx" in result
    # Empty-candidate stub path takes the bounded broaden cycle, then persist.
    assert result["retrieve_attempts"] == result["retrieve_max_attempts"]


async def test_retrieve_attempts_cap_is_enforced() -> None:
    compiled = build_graph()
    cap = 2
    visits = {"retrieve": 0, "broaden_constraints": 0}
    async for update in compiled.astream(
        empty_turn_state(_ctx(), retrieve_max_attempts=cap),
        stream_mode="updates",
    ):
        for name in update:
            if name in visits:
                visits[name] += 1
    assert visits["retrieve"] == cap
    assert visits["broaden_constraints"] == cap - 1

    cap_one = await compiled.ainvoke(empty_turn_state(_ctx(), retrieve_max_attempts=1))
    assert cap_one["retrieve_attempts"] == 1


@pytest.mark.parametrize(
    "payload",
    [{}, {"retrieve_attempts": 0}, {"retrieve_max_attempts": 99}],
    ids=["empty", "reset_attempts", "raise_cap"],
)
async def test_hostile_retrieve_cannot_bypass_cap(payload: dict[str, object]) -> None:
    async def hostile(_state: TurnState) -> dict[str, object]:
        return dict(payload)

    compiled = build_graph(node_overrides={"retrieve": hostile})
    cap = 2
    visits = {"retrieve": 0, "broaden_constraints": 0}
    try:
        async for update in compiled.astream(
            empty_turn_state(_ctx(), retrieve_max_attempts=cap),
            _RECURSION_CONFIG,
            stream_mode="updates",
        ):
            for name in update:
                if name in visits:
                    visits[name] += 1
    except GraphRecursionError as exc:
        pytest.fail(f"GraphRecursionError escaped for payload {payload}: {exc}")
    assert visits["retrieve"] == cap
    assert visits["broaden_constraints"] == cap - 1

    result = await compiled.ainvoke(
        empty_turn_state(_ctx(), retrieve_max_attempts=cap),
        _RECURSION_CONFIG,
    )
    assert result["retrieve_attempts"] == cap
    assert result["retrieve_max_attempts"] == cap


async def test_seeded_candidates_skip_broaden() -> None:
    compiled = build_graph()
    visits = {"retrieve": 0, "broaden_constraints": 0}
    async for update in compiled.astream(
        empty_turn_state(_ctx(), candidates=(_candidate(),)),
        stream_mode="updates",
    ):
        for name in update:
            if name in visits:
                visits[name] += 1
    assert visits["retrieve"] == 1
    assert visits["broaden_constraints"] == 0


async def test_blocked_turn_skips_retrieve() -> None:
    compiled = build_graph()
    result = await compiled.ainvoke(empty_turn_state(_ctx(), safety_blocked=True))
    assert result["retrieve_attempts"] == 0


def test_no_node_is_given_tools() -> None:
    """Backstop: neither model call is given tools.

    Catches:
    - Construction-time binding on the compiled node (ToolNode, AgentExecutor,
      a `.tools` attribute, or `kwargs['tools']`).
    - `bind_tools`, `ToolNode`, `AgentExecutor`, `with_structured_output`, or a
      `tools=` kwarg on bind/invoke in the registered callable's source.
    - Direct provider-client imports in that callable's module, and in
      `src/assist/nodes/`, other than through `assist.llm.gateway`.

    Does not catch:
    - `getattr(model, "bind_tools")(...)` or other computed attribute names.
    - `exec` / `eval`, or a helper outside `nodes/` that the callable imports
      under a name this scan does not follow.
    - Tools passed under a non-standard keyword.
    - A live provider call: stubs do no I/O, and real nodes are not invoked
      here (they may touch Redis, ES, or the gateway).
    """
    compiled = build_graph()
    llm_root = _SRC_ROOT / "assist" / "llm"
    for name, node in compiled.nodes.items():
        if str(name).startswith(_IGNORE_PREFIX):
            continue
        bound = getattr(node, "bound", node)
        assert not _has_tools(bound), f"node {name!r} has tools"
        impl = _node_callable(node)
        assert not _source_uses_tools(impl), f"node {name!r} binds tools in source"
        path = _defining_module_file(impl)
        if path is None or _is_under(llm_root, path):
            continue
        if _is_under(_SRC_ROOT, path):
            hits = _provider_import_hits(path)
            assert hits == [], f"node {name!r} imports provider clients: {hits}"

    for path in _iter_node_paths():
        assert not _path_uses_tools(path), f"{path.name} binds tools in source"
        hits = _provider_import_hits(path)
        assert hits == [], f"{path.name} imports provider clients: {hits}"


def test_call_time_bind_tools_is_detected() -> None:
    async def sneaky(_state: TurnState) -> dict[str, object]:
        model = type("M", (), {"bind_tools": lambda self, tools: self})()
        model.bind_tools([])
        return {}

    assert _has_tools(sneaky) is False
    assert _source_uses_tools(sneaky) is True

    graph: StateGraph[TurnState] = StateGraph(TurnState)
    _scratch_add_node(graph, "sneaky", sneaky)
    graph.add_edge(START, "sneaky")
    compiled = graph.compile(checkpointer=False)
    node = compiled.nodes["sneaky"]
    bound = getattr(node, "bound", node)
    assert _has_tools(bound) is False
    assert _source_uses_tools(_node_callable(node)) is True


def test_after_retrieve_predicates() -> None:
    ctx = _ctx()
    empty = empty_turn_state(ctx, retrieve_attempts=1, retrieve_max_attempts=2, candidates=())
    assert after_retrieve(empty) == "broaden"
    capped = empty_turn_state(ctx, retrieve_attempts=2, retrieve_max_attempts=2, candidates=())
    assert after_retrieve(capped) == "rank"
    hits = empty_turn_state(
        ctx,
        retrieve_attempts=1,
        retrieve_max_attempts=2,
        candidates=(_candidate(),),
    )
    assert after_retrieve(hits) == "rank"


def test_after_retrieve_ignores_cap_above_settings() -> None:
    ctx = _ctx()
    configured = int(settings.retrieve_max_attempts)
    raised = empty_turn_state(
        ctx,
        retrieve_attempts=configured,
        retrieve_max_attempts=configured + 97,
        candidates=(),
    )
    assert after_retrieve(raised) == "rank"
    still_open = empty_turn_state(
        ctx,
        retrieve_attempts=max(configured - 1, 0),
        retrieve_max_attempts=configured + 97,
        candidates=(),
    )
    assert after_retrieve(still_open) == "broaden"


def test_after_retrieve_tolerates_both_media_type_sentinels() -> None:
    ctx = _ctx()
    for media_type in (None, MediaType.ANY):
        state = empty_turn_state(
            ctx,
            constraints=ConstraintState(media_type=media_type),
            retrieve_attempts=1,
            retrieve_max_attempts=2,
            candidates=(),
        )
        assert after_retrieve(state) == "broaden"


def test_after_guard_and_route_reply() -> None:
    ctx = _ctx()
    assert after_guard(empty_turn_state(ctx)) == "intent"
    assert after_guard(empty_turn_state(ctx, safety_blocked=True)) == "refusal"
    assert after_guard(empty_turn_state(ctx, degraded_reason=DegradedReason.SAFETY_BLOCK)) == (
        "refusal"
    )
    assert route_reply(empty_turn_state(ctx)) == "template"
    assert route_reply(empty_turn_state(ctx, route=Route.GENERATIVE)) == "generative"
    assert route_reply(empty_turn_state(ctx, person_ambiguous=True)) == "clarify"
    assert route_reply(empty_turn_state(ctx, route=Route.SAFETY)) == "refusal"


def test_person_soft_hint_routes_to_people() -> None:
    from assist.graph.edges import after_merge

    ctx = _ctx()
    assert after_merge(empty_turn_state(ctx)) == "retrieve"
    hinted = empty_turn_state(ctx, person_soft=PersonSoft(free_hint="spy guy from the 90s"))
    assert after_merge(hinted) == "resolve_people"


def test_export_mermaid_writes_file(tmp_path: Path) -> None:
    target = tmp_path / "graph.mmd"
    mermaid = export_mermaid(target)
    assert target.is_file()
    text = target.read_text(encoding="utf-8")
    assert text == mermaid
    assert "retrieve" in text
    assert "broaden_constraints" in text
    assert "graph " in text.lower()


def test_committed_graph_mmd_matches_export(tmp_path: Path) -> None:
    generated = export_mermaid(tmp_path / "graph.mmd")
    committed = default_mermaid_path().read_text(encoding="utf-8")
    assert committed == generated
