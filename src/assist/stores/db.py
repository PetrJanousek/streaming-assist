"""Async Postgres catalog store: models, engine, and repositories.

Schema matches implementation-plan §4.1. `TurnEventRepository.record` is the
fire-and-forget path — it never raises to the caller (plan: degrade, never error).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    Text,
    func,
    select,
)
from sqlalchemy import (
    text as sql_text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, insert
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from assist.config import Settings, settings
from assist.domain.catalog import Person, Title
from assist.domain.enums import (
    CreditRole,
    DegradedReason,
    DeviceClass,
    GenreId,
    MaturityRating,
    MediaType,
    MoodId,
    Package,
    Route,
    SpeechAct,
)
from assist.obs.logging import get_logger

log = get_logger("assist.stores.db")

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def postgres_async_url(dsn: str) -> str:
    """Coerce a Postgres DSN to the asyncpg SQLAlchemy dialect."""
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn
    for prefix in ("postgresql+psycopg2://", "postgresql+psycopg://", "postgresql://"):
        if dsn.startswith(prefix):
            return "postgresql+asyncpg://" + dsn.removeprefix(prefix)
    return dsn


def create_db_engine(
    dsn: str,
    *,
    pool_size: int,
    max_overflow: int,
) -> AsyncEngine:
    return create_async_engine(
        postgres_async_url(dsn),
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
    )


# ---------------------------------------------------------------------------
# ORM rows
# ---------------------------------------------------------------------------


class TitleRow(Base):
    __tablename__ = "titles"

    catalog_id: Mapped[str] = mapped_column(Text, primary_key=True)
    media_type: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    synopsis: Mapped[str] = mapped_column(Text, nullable=False, server_default=sql_text("''"))
    release_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    runtime_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    seasons: Mapped[int | None] = mapped_column(Integer, nullable=True)
    maturity_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    origins: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=sql_text("'{}'")
    )
    genres: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=sql_text("'{}'")
    )
    local_original: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_text("false")
    )
    pop_28d: Mapped[float] = mapped_column(Float, nullable=False, server_default=sql_text("0"))
    enrichment: Mapped[dict[str, object] | None] = mapped_column(JSONB, nullable=True)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PersonRow(Base):
    __tablename__ = "people"

    person_id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    name_norm: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    roles: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=sql_text("'{}'")
    )
    credit_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=sql_text("0"))
    active_year_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    active_year_max: Mapped[int | None] = mapped_column(Integer, nullable=True)
    popularity: Mapped[float] = mapped_column(Float, nullable=False, server_default=sql_text("0"))


class CreditRow(Base):
    __tablename__ = "credits"

    catalog_id: Mapped[str] = mapped_column(
        Text, ForeignKey("titles.catalog_id", ondelete="CASCADE"), primary_key=True
    )
    person_id: Mapped[str] = mapped_column(
        Text, ForeignKey("people.person_id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[str] = mapped_column(Text, primary_key=True)


class AvailabilityRow(Base):
    __tablename__ = "availability"

    catalog_id: Mapped[str] = mapped_column(
        Text, ForeignKey("titles.catalog_id", ondelete="CASCADE"), primary_key=True
    )
    package: Mapped[str] = mapped_column(Text, primary_key=True)
    geo: Mapped[str] = mapped_column(Text, primary_key=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    playable: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_text("false")
    )


class TaxonomyRow(Base):
    __tablename__ = "taxonomy"

    kind: Mapped[str] = mapped_column(Text, primary_key=True)
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    synonyms: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=sql_text("'{}'")
    )


class PhraseBankRow(Base):
    __tablename__ = "phrase_bank"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    speech_act: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    template: Mapped[str] = mapped_column(Text, nullable=False)


class ProfileRow(Base):
    __tablename__ = "profiles"

    profile_id: Mapped[str] = mapped_column(Text, primary_key=True)
    token: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    maturity_max: Mapped[str] = mapped_column(Text, nullable=False)
    kids: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=sql_text("false"))
    geo: Mapped[str] = mapped_column(Text, nullable=False)
    package: Mapped[str] = mapped_column(Text, nullable=False)
    device_class: Mapped[str] = mapped_column(Text, nullable=False)


class GoldenQueryRow(Base):
    __tablename__ = "golden_queries"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    expect_ids: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=sql_text("'{}'")
    )
    expect_class: Mapped[str] = mapped_column(Text, nullable=False)
    slice: Mapped[str] = mapped_column(Text, nullable=False)


class TurnEventRow(Base):
    __tablename__ = "turn_events"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=sql_text("gen_random_uuid()"),
    )
    trace_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(Text, nullable=False)
    route: Mapped[str] = mapped_column(Text, nullable=False)
    intent_source: Mapped[str] = mapped_column(Text, nullable=False)
    degraded_reason: Mapped[str] = mapped_column(Text, nullable=False)
    stage_latency_ms: Mapped[dict[str, object]] = mapped_column(
        JSONB, nullable=False, server_default=sql_text("'{}'::jsonb")
    )
    tokens_in: Mapped[int] = mapped_column(Integer, nullable=False, server_default=sql_text("0"))
    tokens_out: Mapped[int] = mapped_column(Integer, nullable=False, server_default=sql_text("0"))
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, server_default=sql_text("0"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


# ---------------------------------------------------------------------------
# Store records (DB-only fields that domain types do not carry)
# ---------------------------------------------------------------------------


class TitleRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: Title
    enrichment: dict[str, object] | None = None
    indexed_at: datetime | None = None


class CreditRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    catalog_id: str
    person_id: str
    role: CreditRole


class AvailabilityWindow(BaseModel):
    model_config = ConfigDict(frozen=True)

    catalog_id: str
    package: Package
    geo: str
    window_start: datetime
    window_end: datetime
    playable: bool


class TaxonomyEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    id: str
    label: str
    synonyms: tuple[str, ...] = ()


class PhraseTemplate(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    speech_act: SpeechAct
    kind: str
    template: str


class ProfileRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile_id: str
    token: str
    maturity_max: MaturityRating
    kids: bool
    geo: str
    package: Package
    device_class: DeviceClass


class GoldenQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    expect_ids: tuple[str, ...] = ()
    expect_class: str
    slice: str


class TurnEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    trace_id: str
    session_id: str
    route: Route
    intent_source: str
    degraded_reason: DegradedReason
    stage_latency_ms: dict[str, int] = Field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime | None = None


def _moods_from_enrichment(enrichment: Mapping[str, object] | None) -> tuple[MoodId, ...]:
    if not enrichment:
        return ()
    raw = enrichment.get("moods")
    if not isinstance(raw, list):
        return ()
    moods: list[MoodId] = []
    for item in raw:
        if isinstance(item, str):
            moods.append(MoodId(item))
    return tuple(moods)


def _title_from_row(row: TitleRow) -> Title:
    return Title(
        catalog_id=row.catalog_id,
        media_type=MediaType(row.media_type),
        title=row.title,
        synopsis=row.synopsis,
        release_year=row.release_year,
        runtime_min=row.runtime_min,
        seasons=row.seasons,
        maturity_rank=row.maturity_rank,
        origins=tuple(row.origins or ()),
        genres=tuple(GenreId(g) for g in (row.genres or ())),
        moods=_moods_from_enrichment(row.enrichment),
        local_original=row.local_original,
        pop_28d=row.pop_28d,
    )


def _enrichment_for_write(
    title: Title, enrichment: Mapping[str, object] | None
) -> tuple[dict[str, object] | None, bool]:
    """Return (payload, replace_on_conflict).

    Omit `enrichment` on an update so T10 re-runs cannot wipe T11's JSON.
    Moods still land on INSERT when the caller only passed a Title.
    """
    if enrichment is None:
        if title.moods:
            return {"moods": [m.value for m in title.moods]}, False
        return None, False
    payload: dict[str, object] = dict(enrichment)
    if title.moods:
        payload["moods"] = [m.value for m in title.moods]
    return payload, True


def _person_from_row(row: PersonRow) -> Person:
    return Person(
        person_id=row.person_id,
        name=row.name,
        name_norm=row.name_norm,
        roles=tuple(CreditRole(r) for r in (row.roles or ())),
        credit_count=row.credit_count,
        active_year_min=row.active_year_min,
        active_year_max=row.active_year_max,
        popularity=row.popularity,
    )


# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------


class TitleRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(
        self,
        title: Title,
        *,
        enrichment: Mapping[str, object] | None = None,
        indexed_at: datetime | None = None,
    ) -> None:
        enrichment_payload, replace_enrichment = _enrichment_for_write(title, enrichment)
        stmt = insert(TitleRow).values(
            catalog_id=title.catalog_id,
            media_type=title.media_type.value,
            title=title.title,
            synopsis=title.synopsis,
            release_year=title.release_year,
            runtime_min=title.runtime_min,
            seasons=title.seasons,
            maturity_rank=title.maturity_rank,
            origins=list(title.origins),
            genres=[g.value for g in title.genres],
            local_original=title.local_original,
            pop_28d=title.pop_28d,
            enrichment=enrichment_payload,
            indexed_at=indexed_at,
        )
        updates = {
            "media_type": stmt.excluded.media_type,
            "title": stmt.excluded.title,
            "synopsis": stmt.excluded.synopsis,
            "release_year": stmt.excluded.release_year,
            "runtime_min": stmt.excluded.runtime_min,
            "seasons": stmt.excluded.seasons,
            "maturity_rank": stmt.excluded.maturity_rank,
            "origins": stmt.excluded.origins,
            "genres": stmt.excluded.genres,
            "local_original": stmt.excluded.local_original,
            "pop_28d": stmt.excluded.pop_28d,
        }
        if replace_enrichment:
            updates["enrichment"] = stmt.excluded.enrichment
        if indexed_at is not None:
            updates["indexed_at"] = stmt.excluded.indexed_at
        stmt = stmt.on_conflict_do_update(index_elements=["catalog_id"], set_=updates)
        await self._session.execute(stmt)

    async def get(self, catalog_id: str) -> Title | None:
        stored = await self.get_stored(catalog_id)
        return None if stored is None else stored.title

    async def get_stored(self, catalog_id: str) -> TitleRecord | None:
        row = await self._session.get(TitleRow, catalog_id)
        if row is None:
            return None
        enrichment = dict(row.enrichment) if row.enrichment is not None else None
        return TitleRecord(
            title=_title_from_row(row), enrichment=enrichment, indexed_at=row.indexed_at
        )


class PersonRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, person: Person) -> None:
        stmt = insert(PersonRow).values(
            person_id=person.person_id,
            name=person.name,
            name_norm=person.name_norm,
            roles=[r.value for r in person.roles],
            credit_count=person.credit_count,
            active_year_min=person.active_year_min,
            active_year_max=person.active_year_max,
            popularity=person.popularity,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["person_id"],
            set_={
                "name": stmt.excluded.name,
                "name_norm": stmt.excluded.name_norm,
                "roles": stmt.excluded.roles,
                "credit_count": stmt.excluded.credit_count,
                "active_year_min": stmt.excluded.active_year_min,
                "active_year_max": stmt.excluded.active_year_max,
                "popularity": stmt.excluded.popularity,
            },
        )
        await self._session.execute(stmt)

    async def get(self, person_id: str) -> Person | None:
        row = await self._session.get(PersonRow, person_id)
        return None if row is None else _person_from_row(row)


class CreditRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, credit: CreditRecord) -> None:
        stmt = (
            insert(CreditRow)
            .values(
                catalog_id=credit.catalog_id,
                person_id=credit.person_id,
                role=credit.role.value,
            )
            .on_conflict_do_nothing(index_elements=["catalog_id", "person_id", "role"])
        )
        await self._session.execute(stmt)

    async def list_for_title(self, catalog_id: str) -> list[CreditRecord]:
        result = await self._session.execute(
            select(CreditRow).where(CreditRow.catalog_id == catalog_id).order_by(CreditRow.role)
        )
        return [
            CreditRecord(
                catalog_id=row.catalog_id,
                person_id=row.person_id,
                role=CreditRole(row.role),
            )
            for row in result.scalars()
        ]


class AvailabilityRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, window: AvailabilityWindow) -> None:
        stmt = insert(AvailabilityRow).values(
            catalog_id=window.catalog_id,
            package=window.package.value,
            geo=window.geo,
            window_start=window.window_start,
            window_end=window.window_end,
            playable=window.playable,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["catalog_id", "package", "geo"],
            set_={
                "window_start": stmt.excluded.window_start,
                "window_end": stmt.excluded.window_end,
                "playable": stmt.excluded.playable,
            },
        )
        await self._session.execute(stmt)

    async def get(self, catalog_id: str, package: Package, geo: str) -> AvailabilityWindow | None:
        row = await self._session.get(AvailabilityRow, (catalog_id, package.value, geo))
        if row is None:
            return None
        return AvailabilityWindow(
            catalog_id=row.catalog_id,
            package=Package(row.package),
            geo=row.geo,
            window_start=row.window_start,
            window_end=row.window_end,
            playable=row.playable,
        )

    async def list_for_package_geo(
        self, catalog_ids: list[str], package: Package, geo: str
    ) -> list[AvailabilityWindow]:
        """Batch lookup for one package+geo. Missing ids are omitted, not invented."""
        if not catalog_ids:
            return []
        result = await self._session.execute(
            select(AvailabilityRow).where(
                AvailabilityRow.catalog_id.in_(catalog_ids),
                AvailabilityRow.package == package.value,
                AvailabilityRow.geo == geo,
            )
        )
        return [
            AvailabilityWindow(
                catalog_id=row.catalog_id,
                package=Package(row.package),
                geo=row.geo,
                window_start=row.window_start,
                window_end=row.window_end,
                playable=row.playable,
            )
            for row in result.scalars()
        ]


class TaxonomyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, entry: TaxonomyEntry) -> None:
        stmt = insert(TaxonomyRow).values(
            kind=entry.kind,
            id=entry.id,
            label=entry.label,
            synonyms=list(entry.synonyms),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["kind", "id"],
            set_={
                "label": stmt.excluded.label,
                "synonyms": stmt.excluded.synonyms,
            },
        )
        await self._session.execute(stmt)

    async def get(self, kind: str, id: str) -> TaxonomyEntry | None:
        row = await self._session.get(TaxonomyRow, (kind, id))
        if row is None:
            return None
        return TaxonomyEntry(
            kind=row.kind,
            id=row.id,
            label=row.label,
            synonyms=tuple(row.synonyms or ()),
        )

    async def list_by_kind(self, kind: str) -> list[TaxonomyEntry]:
        result = await self._session.execute(
            select(TaxonomyRow).where(TaxonomyRow.kind == kind).order_by(TaxonomyRow.id)
        )
        return [
            TaxonomyEntry(
                kind=row.kind,
                id=row.id,
                label=row.label,
                synonyms=tuple(row.synonyms or ()),
            )
            for row in result.scalars()
        ]


class PhraseBankRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, phrase: PhraseTemplate) -> None:
        stmt = insert(PhraseBankRow).values(
            id=phrase.id,
            speech_act=phrase.speech_act.value,
            kind=phrase.kind,
            template=phrase.template,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "speech_act": stmt.excluded.speech_act,
                "kind": stmt.excluded.kind,
                "template": stmt.excluded.template,
            },
        )
        await self._session.execute(stmt)

    async def get(self, phrase_id: str) -> PhraseTemplate | None:
        row = await self._session.get(PhraseBankRow, phrase_id)
        if row is None:
            return None
        return PhraseTemplate(
            id=row.id,
            speech_act=SpeechAct(row.speech_act),
            kind=row.kind,
            template=row.template,
        )

    async def list_by_speech_act(self, speech_act: SpeechAct) -> list[PhraseTemplate]:
        result = await self._session.execute(
            select(PhraseBankRow)
            .where(PhraseBankRow.speech_act == speech_act.value)
            .order_by(PhraseBankRow.id)
        )
        return [
            PhraseTemplate(
                id=row.id,
                speech_act=SpeechAct(row.speech_act),
                kind=row.kind,
                template=row.template,
            )
            for row in result.scalars()
        ]


class ProfileRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, profile: ProfileRecord) -> None:
        stmt = insert(ProfileRow).values(
            profile_id=profile.profile_id,
            token=profile.token,
            maturity_max=profile.maturity_max.value,
            kids=profile.kids,
            geo=profile.geo,
            package=profile.package.value,
            device_class=profile.device_class.value,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["profile_id"],
            set_={
                "token": stmt.excluded.token,
                "maturity_max": stmt.excluded.maturity_max,
                "kids": stmt.excluded.kids,
                "geo": stmt.excluded.geo,
                "package": stmt.excluded.package,
                "device_class": stmt.excluded.device_class,
            },
        )
        await self._session.execute(stmt)

    async def get(self, profile_id: str) -> ProfileRecord | None:
        row = await self._session.get(ProfileRow, profile_id)
        return None if row is None else _profile_from_row(row)

    async def get_by_token(self, token: str) -> ProfileRecord | None:
        result = await self._session.execute(select(ProfileRow).where(ProfileRow.token == token))
        row = result.scalar_one_or_none()
        return None if row is None else _profile_from_row(row)


def _profile_from_row(row: ProfileRow) -> ProfileRecord:
    return ProfileRecord(
        profile_id=row.profile_id,
        token=row.token,
        maturity_max=MaturityRating(row.maturity_max),
        kids=row.kids,
        geo=row.geo,
        package=Package(row.package),
        device_class=DeviceClass(row.device_class),
    )


class GoldenQueryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, query: GoldenQuery) -> None:
        stmt = insert(GoldenQueryRow).values(
            id=query.id,
            text=query.text,
            expect_ids=list(query.expect_ids),
            expect_class=query.expect_class,
            slice=query.slice,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "text": stmt.excluded.text,
                "expect_ids": stmt.excluded.expect_ids,
                "expect_class": stmt.excluded.expect_class,
                "slice": stmt.excluded.slice,
            },
        )
        await self._session.execute(stmt)

    async def get(self, query_id: str) -> GoldenQuery | None:
        row = await self._session.get(GoldenQueryRow, query_id)
        if row is None:
            return None
        return GoldenQuery(
            id=row.id,
            text=row.text,
            expect_ids=tuple(row.expect_ids or ()),
            expect_class=row.expect_class,
            slice=row.slice,
        )

    async def list_all(self) -> list[GoldenQuery]:
        result = await self._session.execute(select(GoldenQueryRow).order_by(GoldenQueryRow.id))
        return [
            GoldenQuery(
                id=row.id,
                text=row.text,
                expect_ids=tuple(row.expect_ids or ()),
                expect_class=row.expect_class,
                slice=row.slice,
            )
            for row in result.scalars()
        ]


class TurnEventRepository:
    """Analytics writes. Failures are logged and swallowed — never a request error."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def record(self, event: TurnEvent) -> None:
        try:
            async with self._session_factory() as session:
                session.add(
                    TurnEventRow(
                        id=event.id,
                        trace_id=event.trace_id,
                        session_id=event.session_id,
                        route=event.route.value,
                        intent_source=event.intent_source,
                        degraded_reason=event.degraded_reason.value,
                        stage_latency_ms=dict(event.stage_latency_ms),
                        tokens_in=event.tokens_in,
                        tokens_out=event.tokens_out,
                        cost_usd=event.cost_usd,
                    )
                )
                await session.commit()
        except Exception:
            # Fire-and-forget: a missed analytics row must not fail the turn.
            log.exception(
                "turn_event_write_failed",
                trace_id=event.trace_id,
                session_id=event.session_id,
            )

    async def get(self, event_id: UUID) -> TurnEvent | None:
        async with self._session_factory() as session:
            row = await session.get(TurnEventRow, event_id)
            if row is None:
                return None
            latency_raw = row.stage_latency_ms or {}
            latency = {str(k): int(v) for k, v in latency_raw.items() if isinstance(v, int | float)}
            return TurnEvent(
                id=row.id,
                trace_id=row.trace_id,
                session_id=row.session_id,
                route=Route(row.route),
                intent_source=row.intent_source,
                degraded_reason=DegradedReason(row.degraded_reason),
                stage_latency_ms=latency,
                tokens_in=row.tokens_in,
                tokens_out=row.tokens_out,
                cost_usd=row.cost_usd,
                created_at=row.created_at,
            )


# ---------------------------------------------------------------------------
# Database facade
# ---------------------------------------------------------------------------


class Database:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine
        self.session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            engine, expire_on_commit=False
        )

    @classmethod
    def from_settings(cls, cfg: Settings | None = None) -> Database:
        resolved = cfg or settings
        return cls(
            create_db_engine(
                resolved.postgres_dsn,
                pool_size=resolved.postgres_pool_size,
                max_overflow=resolved.postgres_max_overflow,
            )
        )

    @classmethod
    def from_dsn(
        cls,
        dsn: str,
        *,
        pool_size: int = 5,
        max_overflow: int = 10,
    ) -> Database:
        return cls(create_db_engine(dsn, pool_size=pool_size, max_overflow=max_overflow))

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def dispose(self) -> None:
        await self.engine.dispose()

    def titles(self, session: AsyncSession) -> TitleRepository:
        return TitleRepository(session)

    def people(self, session: AsyncSession) -> PersonRepository:
        return PersonRepository(session)

    def credits(self, session: AsyncSession) -> CreditRepository:
        return CreditRepository(session)

    def availability(self, session: AsyncSession) -> AvailabilityRepository:
        return AvailabilityRepository(session)

    def taxonomy(self, session: AsyncSession) -> TaxonomyRepository:
        return TaxonomyRepository(session)

    def phrase_bank(self, session: AsyncSession) -> PhraseBankRepository:
        return PhraseBankRepository(session)

    def profiles(self, session: AsyncSession) -> ProfileRepository:
        return ProfileRepository(session)

    def golden_queries(self, session: AsyncSession) -> GoldenQueryRepository:
        return GoldenQueryRepository(session)

    @property
    def turn_events(self) -> TurnEventRepository:
        # Own session factory so a failed analytics write cannot abort the request txn.
        return TurnEventRepository(self.session_factory)


# Imported by tests that assert every catalog table exists after migrate.
CATALOG_TABLES: tuple[str, ...] = (
    "titles",
    "people",
    "credits",
    "availability",
    "taxonomy",
    "phrase_bank",
    "profiles",
    "golden_queries",
    "turn_events",
)
