"""initial catalog schema (implementation-plan §4.1)

Revision ID: 0001
Revises:
Create Date: 2026-08-27
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "titles",
        sa.Column("catalog_id", sa.Text(), nullable=False),
        sa.Column("media_type", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("synopsis", sa.Text(), server_default=sa.text("''"), nullable=False),
        sa.Column("release_year", sa.Integer(), nullable=True),
        sa.Column("runtime_min", sa.Integer(), nullable=True),
        sa.Column("seasons", sa.Integer(), nullable=True),
        sa.Column("maturity_rank", sa.Integer(), nullable=False),
        sa.Column(
            "origins",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column(
            "genres",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column("local_original", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("pop_28d", sa.Float(), server_default=sa.text("0"), nullable=False),
        sa.Column("enrichment", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("catalog_id", name="pk_titles"),
    )

    op.create_table(
        "people",
        sa.Column("person_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("name_norm", sa.Text(), nullable=False),
        sa.Column(
            "roles",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column("credit_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("active_year_min", sa.Integer(), nullable=True),
        sa.Column("active_year_max", sa.Integer(), nullable=True),
        sa.Column("popularity", sa.Float(), server_default=sa.text("0"), nullable=False),
        sa.PrimaryKeyConstraint("person_id", name="pk_people"),
    )
    op.create_index("ix_people_name_norm", "people", ["name_norm"], unique=False)

    op.create_table(
        "credits",
        sa.Column("catalog_id", sa.Text(), nullable=False),
        sa.Column("person_id", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["catalog_id"],
            ["titles.catalog_id"],
            name="fk_credits_catalog_id_titles",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["person_id"],
            ["people.person_id"],
            name="fk_credits_person_id_people",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("catalog_id", "person_id", "role", name="pk_credits"),
    )

    op.create_table(
        "availability",
        sa.Column("catalog_id", sa.Text(), nullable=False),
        sa.Column("package", sa.Text(), nullable=False),
        sa.Column("geo", sa.Text(), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("playable", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.ForeignKeyConstraint(
            ["catalog_id"],
            ["titles.catalog_id"],
            name="fk_availability_catalog_id_titles",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("catalog_id", "package", "geo", name="pk_availability"),
    )

    op.create_table(
        "taxonomy",
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column(
            "synonyms",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("kind", "id", name="pk_taxonomy"),
    )

    op.create_table(
        "phrase_bank",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("speech_act", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("template", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_phrase_bank"),
    )
    op.create_index("ix_phrase_bank_speech_act", "phrase_bank", ["speech_act"], unique=False)

    op.create_table(
        "profiles",
        sa.Column("profile_id", sa.Text(), nullable=False),
        sa.Column("token", sa.Text(), nullable=False),
        sa.Column("maturity_max", sa.Text(), nullable=False),
        sa.Column("kids", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("geo", sa.Text(), nullable=False),
        sa.Column("package", sa.Text(), nullable=False),
        sa.Column("device_class", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("profile_id", name="pk_profiles"),
        sa.UniqueConstraint("token", name="uq_profiles_token"),
    )

    op.create_table(
        "golden_queries",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column(
            "expect_ids",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column("expect_class", sa.Text(), nullable=False),
        sa.Column("slice", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_golden_queries"),
    )

    op.create_table(
        "turn_events",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("trace_id", sa.Text(), nullable=False),
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("route", sa.Text(), nullable=False),
        sa.Column("intent_source", sa.Text(), nullable=False),
        sa.Column("degraded_reason", sa.Text(), nullable=False),
        sa.Column(
            "stage_latency_ms",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("tokens_in", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("tokens_out", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("cost_usd", sa.Float(), server_default=sa.text("0"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_turn_events"),
    )
    op.create_index("ix_turn_events_trace_id", "turn_events", ["trace_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_turn_events_trace_id", table_name="turn_events")
    op.drop_table("turn_events")
    op.drop_table("golden_queries")
    op.drop_table("profiles")
    op.drop_index("ix_phrase_bank_speech_act", table_name="phrase_bank")
    op.drop_table("phrase_bank")
    op.drop_table("taxonomy")
    op.drop_table("availability")
    op.drop_table("credits")
    op.drop_index("ix_people_name_norm", table_name="people")
    op.drop_table("people")
    op.drop_table("titles")
