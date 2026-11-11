"""Typed contracts between the agents.

Every hand-off between two prompts is one of these models, so the pipeline code
never parses free text: the API constrains the model to the schema and Pydantic
validates it again on our side.
"""
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal[
    "adventure", "friendship", "funny", "cozy", "mystery", "curiosity", "comfort"
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StoryBrief(_Strict):
    """The planner's output: what story to tell and how."""

    category: Category = Field(description="The single best-fitting story category.")
    adjustments: list[str] = Field(
        description="Each change made to keep the request suitable for ages 5-10, "
        "with the gentler replacement. Empty when the request was fine as asked."
    )
    must_keep: list[str] = Field(
        description="Concrete details from the request the story has to honor "
        "(names, animals, places, wishes)."
    )
    title: str
    hero: str = Field(description="Who the story follows, with one endearing trait.")
    companions: list[str]
    setting: str
    wish: str = Field(description="What the hero wants, in a child's terms.")
    gentle_obstacle: str = Field(
        description="What stands in the way. Low stakes; nobody is in real danger."
    )
    beats: list[str] = Field(
        description="Five beats in order: cozy opening, the wish appears, tries and "
        "a small setback, the kind or clever turn, sleepy landing."
    )
    quiet_takeaway: str = Field(
        description="The feeling or idea the child keeps. Shown, never preached."
    )
    refrain: str = Field(
        description="A short, rhythmic line that can come back two or three times."
    )


class Criterion(_Strict):
    evidence: str = Field(
        description="One or two sentences citing what in the story earns the score."
    )
    score: int = Field(description="Integer from 1 (poor) to 5 (excellent).")


class Verdict(_Strict):
    """The judge's output. Code, not the judge, decides pass or fail from it."""

    safety: Criterion
    age_fit: Criterion
    story_arc: Criterion
    delight: Criterion
    read_aloud: Criterion
    bedtime_landing: Criterion
    request_fit: Criterion
    keep: list[str] = Field(
        description="What already works and a revision should not lose."
    )
    fixes: list[str] = Field(
        description="Specific, actionable edits, most important first. Empty if none."
    )

    CRITERIA: ClassVar[tuple[str, ...]] = (
        "safety", "age_fit", "story_arc", "delight",
        "read_aloud", "bedtime_landing", "request_fit",
    )

    def scores(self) -> dict[str, int]:
        return {name: getattr(self, name).score for name in self.CRITERIA}

    @property
    def average(self) -> float:
        values = self.scores().values()
        return sum(values) / len(values)

    @property
    def is_safe(self) -> bool:
        """Hard gate: a story that is not clearly safe is never shown."""
        return self.safety.score >= 5 and self.age_fit.score >= 4

    def passes(self, bar: float) -> bool:
        return self.is_safe and min(self.scores().values()) >= 3 and self.average >= bar
