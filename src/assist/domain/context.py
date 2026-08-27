"""Trusted per-request user context. Client hints never land here."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from assist.domain.enums import DeviceClass, MaturityRating, Package


class ServerUserCtx(BaseModel):
    """AuthZ floor for a turn. Frozen so no node can widen entitlements."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    user_id: str
    profile_id: str
    geo: str
    package: Package
    maturity_max: MaturityRating
    kids_flag: bool
    device_class: DeviceClass
