"""T29 regression: the flattened IntentUpdateWire must actually compile and
decode against the live Anthropic API. The bug this guards against (a nested
FieldOp discriminated union compiling to a ~9.7KB grammar Anthropic rejects
with 400 invalid_request_error, "compiled grammar is too large") only shows
up against the real provider -- every other test in this suite injects a fake
model at the `with_structured_output` boundary, so it could not have caught
this. Skips cleanly with no credentials; deselect explicitly with
`-m "not live"`.
"""

from __future__ import annotations

import pytest

from assist.config import Settings
from assist.domain.constraints import ConstraintState
from assist.domain.enums import GenreId, MoodId
from assist.llm.gateway import get_chat_model, structured_output
from assist.llm.prompts import chat_prompt_template
from assist.nodes.intent import IntentUpdateWire

_settings = Settings()

pytestmark = pytest.mark.skipif(
    not (_settings.anthropic_api_key or "").strip(),
    reason="ANTHROPIC_API_KEY not configured; skipping live provider test",
)


@pytest.mark.live
@pytest.mark.asyncio
async def test_intent_wire_schema_accepted_by_live_provider() -> None:
    """Real call, real schema, no fake model. Only asserts the request was
    accepted and decoded -- not on which constraints the model chose, which
    would be flaky. One query only: a second live call in the same run/loop
    hits an unrelated httpx connection-pool flake, not a schema issue."""
    model = get_chat_model(settings=_settings)
    chain = chat_prompt_template("intent") | structured_output(
        IntentUpdateWire,
        retries=0,
        model=model,
        settings=_settings,
    )
    payload = {
        "text": "Czech movies",
        "constraints_json": ConstraintState.empty().model_dump_json(),
        "genre_ids": ", ".join(item.value for item in GenreId),
        "mood_ids": ", ".join(item.value for item in MoodId),
    }
    # No fallback: a schema/grammar rejection must surface as a failure here,
    # not be swallowed into a degrade path.
    wire = await chain.ainvoke(payload)
    assert isinstance(wire, IntentUpdateWire)
