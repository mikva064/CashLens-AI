"""Internal integration contract. The web service owns job IDs and authentication."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class RuleSource(StrictModel):
    document: str = Field(min_length=1)
    section: str = Field(min_length=1)
    url: str = Field(pattern=r"^https?://", min_length=10)

    @field_validator("url")
    @classmethod
    def valid_source_url(cls, value: str) -> str:
        if any(c.isspace() for c in value):
            raise ValueError("Source URL cannot contain whitespace")
        return str(HttpUrl(value))


class Rule(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    title: str = Field(min_length=1)
    observable_condition: str = Field(min_length=10)
    source: RuleSource
    verified: bool = False


class EvidenceFrame(StrictModel):
    timestamp_sec: float = Field(ge=0)
    image_path: Path


class Violation(StrictModel):
    type: str = Field(min_length=1)
    timecode_sec: float = Field(ge=0)
    description: str = Field(min_length=1)
    rule_id: str
    source: RuleSource
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)


class ObservationDefinition(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    title: str = Field(min_length=1)
    observable_condition: str = Field(min_length=10)
    description_template: str = Field(min_length=1, max_length=1000)
    review_reason: str = Field(min_length=10, max_length=1600)


class Observation(StrictModel):
    observation_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    type: str = Field(min_length=1)
    timecode_sec: float = Field(ge=0)
    description: str = Field(min_length=1)
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)
    review_reason: str = Field(min_length=10)


class AnalysisResult(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    verdict: Literal["violation", "no_violation", "requires_review"]
    violations: list[Violation]
    observations: list[Observation] = Field(default_factory=list)

    @model_validator(mode="after")
    def verdict_matches_events(self):
        expected = ("violation" if self.violations else
                    "requires_review" if self.observations else "no_violation")
        if self.verdict != expected:
            raise ValueError("verdict must agree with qualified violations and review observations")
        return self
