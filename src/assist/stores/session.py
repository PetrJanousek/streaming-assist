"""Redis session + chip repository.

Chips live inside the session JSON (plan §4.3). One GET loads constraints,
history, and the issued-chip map; one SET writes them back. That is what
makes the AuthZ rule "client sends chip_id, server owns the delta" a
single round-trip rather than a second keyspace.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from redis.asyncio import Redis
from redis.commands.core import AsyncScript

from assist.config import settings
from assist.domain.constraints import ConstraintDelta, ConstraintState
from assist.domain.context import ServerUserCtx
from assist.domain.enums import DeviceClass, MaturityRating, Package, Route, SpeechAct
from assist.obs.logging import get_logger

TURN_HISTORY_CAP = 6

log = get_logger(__name__)

# Atomic create-or-replace that refuses to steal another profile's session.
# JSON lives as one value (plan §4.3); cjson is the bind check, not a schema.
_SAVE_LUA = """
local current = redis.call('GET', KEYS[1])
if current then
  local ok, obj = pcall(cjson.decode, current)
  if not ok or type(obj) ~= 'table' then
    return 0
  end
  if obj['user_id'] ~= ARGV[2] or obj['profile_id'] ~= ARGV[3] then
    return 0
  end
end
redis.call('SET', KEYS[1], ARGV[1], 'EX', tonumber(ARGV[4]))
return 1
"""


def session_key(session_id: str) -> str:
    return f"sess:{session_id}"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _as_str(raw: str | bytes) -> str:
    return raw.decode() if isinstance(raw, bytes) else raw


class ChipInvalid(Exception):
    """Unknown, expired, or not-issued chip_id. Maps to HTTP 400 chip_invalid."""

    error_type = "chip_invalid"

    def __init__(self, chip_id: str) -> None:
        self.chip_id = chip_id
        super().__init__(f"unknown or expired chip_id: {chip_id}")


class SessionBindError(Exception):
    """Session exists but is bound to a different user or profile."""

    error_type = "session_bind_rejected"

    def __init__(
        self,
        *,
        session_id: str,
        bound_user_id: str,
        bound_profile_id: str,
        user_id: str,
        profile_id: str,
    ) -> None:
        self.session_id = session_id
        self.bound_user_id = bound_user_id
        self.bound_profile_id = bound_profile_id
        self.user_id = user_id
        self.profile_id = profile_id
        super().__init__(f"session {session_id} is bound to another profile")


class ChipRecord(BaseModel):
    """Server-minted chip. The client never sees `delta`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chip_id: str
    label: str
    delta: ConstraintDelta
    speech_act: SpeechAct
    minted_turn: int
    expires_at: datetime


class TurnSummary(BaseModel):
    """Last-N conversational turn. Analytics events live in Postgres (T04)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    message_type: Literal["text", "chip"]
    text: str = ""
    reply: str = ""
    pick_ids: tuple[str, ...] = ()
    route: Route | None = None
    intent_source: str | None = None


class ServerCtxEcho(BaseModel):
    """Last-seen ServerUserCtx fields. Echo only — never AuthZ authority."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    geo: str
    package: Package
    maturity_max: MaturityRating
    kids: bool
    device_class: DeviceClass

    @classmethod
    def from_ctx(cls, ctx: ServerUserCtx) -> ServerCtxEcho:
        return cls(
            geo=ctx.geo,
            package=ctx.package,
            maturity_max=ctx.maturity_max,
            kids=ctx.kids_flag,
            device_class=ctx.device_class,
        )


class Session(BaseModel):
    """Sticky per-profile assist session. Frozen; methods return copies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str
    user_id: str
    profile_id: str
    created_at: datetime
    updated_at: datetime
    server_ctx_echo: ServerCtxEcho | None = None
    constraints: ConstraintState = Field(default_factory=ConstraintState)
    turns: tuple[TurnSummary, ...] = ()
    issued_chips: dict[str, ChipRecord] = Field(default_factory=dict)
    turn_count: int = 0

    @classmethod
    def create(
        cls,
        *,
        user_id: str,
        profile_id: str,
        session_id: str | None = None,
    ) -> Session:
        now = _utcnow()
        return cls(
            session_id=session_id or str(uuid4()),
            user_id=user_id,
            profile_id=profile_id,
            created_at=now,
            updated_at=now,
        )

    def lookup_chip(self, chip_id: str) -> ChipRecord:
        record = self.issued_chips.get(chip_id)
        if record is None:
            log.info("chip_invalid", session_id=self.session_id, chip_id=chip_id, reason="unknown")
            raise ChipInvalid(chip_id)
        expires_at = record.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if expires_at <= _utcnow():
            log.info("chip_invalid", session_id=self.session_id, chip_id=chip_id, reason="expired")
            raise ChipInvalid(chip_id)
        return record

    def mint_chip(
        self,
        *,
        label: str,
        delta: ConstraintDelta,
        speech_act: SpeechAct,
        ttl_s: int | None = None,
    ) -> tuple[Session, ChipRecord]:
        if not label:
            raise ValueError("chip label must be non-empty")
        ttl = settings.session_ttl_s if ttl_s is None else ttl_s
        record = ChipRecord(
            chip_id=f"c_{uuid4().hex}",
            label=label,
            delta=delta,
            speech_act=speech_act,
            minted_turn=self.turn_count,
            expires_at=_utcnow() + timedelta(seconds=ttl),
        )
        chips = dict(self.issued_chips)
        chips[record.chip_id] = record
        return self._touch(issued_chips=chips), record

    def append_turn(self, turn: TurnSummary) -> Session:
        turns = (*self.turns, turn)[-TURN_HISTORY_CAP:]
        return self._touch(turns=turns, turn_count=self.turn_count + 1)

    def with_constraints(self, constraints: ConstraintState) -> Session:
        return self._touch(constraints=constraints)

    def with_ctx_echo(self, ctx: ServerUserCtx) -> Session:
        return self._touch(server_ctx_echo=ServerCtxEcho.from_ctx(ctx))

    def _touch(self, **kwargs: object) -> Session:
        return self.model_copy(update={"updated_at": _utcnow(), **kwargs})


class SessionRepository:
    """Load/save sessions. Caller owns the Redis client (T13 lifespan)."""

    def __init__(self, redis: Redis, *, ttl_s: int | None = None) -> None:
        self._redis = redis
        self._ttl_s = settings.session_ttl_s if ttl_s is None else ttl_s
        self._save_script: AsyncScript = redis.register_script(_SAVE_LUA)

    async def load(self, session_id: str, user_id: str, profile_id: str) -> Session:
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if not user_id or not profile_id:
            raise ValueError("user_id and profile_id must be non-empty")
        key = session_key(session_id)
        raw = await self._redis.get(key)
        if raw is None:
            return Session.create(session_id=session_id, user_id=user_id, profile_id=profile_id)
        session = Session.model_validate_json(_as_str(raw))
        if session.user_id != user_id or session.profile_id != profile_id:
            log.warning(
                "session_bind_rejected",
                session_id=session_id,
                bound_user_id=session.user_id,
                bound_profile_id=session.profile_id,
                user_id=user_id,
                profile_id=profile_id,
            )
            raise SessionBindError(
                session_id=session_id,
                bound_user_id=session.user_id,
                bound_profile_id=session.profile_id,
                user_id=user_id,
                profile_id=profile_id,
            )
        # Sliding 24h: a live conversation must not die mid-watch.
        await self._redis.expire(key, self._ttl_s)
        return session

    async def save(self, session: Session) -> None:
        key = session_key(session.session_id)
        ok = await self._save_script(
            keys=[key],
            args=[
                session.model_dump_json(),
                session.user_id,
                session.profile_id,
                self._ttl_s,
            ],
        )
        if int(ok) != 1:
            log.warning(
                "session_bind_rejected",
                session_id=session.session_id,
                user_id=session.user_id,
                profile_id=session.profile_id,
                reason="save",
            )
            raise SessionBindError(
                session_id=session.session_id,
                bound_user_id="",
                bound_profile_id="",
                user_id=session.user_id,
                profile_id=session.profile_id,
            )
