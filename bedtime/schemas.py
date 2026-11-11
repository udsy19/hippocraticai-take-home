"""Typed contracts between the agents.

Every hand-off between two prompts is one of these models, so the pipeline code
never parses free text: the API constrains the model to the schema and Pydantic
validates it again on our side.
"""
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Category = Literal[
    "adventure", "friendship", "funny", "cozy", "mystery", "curiosity", "comfort"
]


Score = Literal[1, 2, 3, 4, 5]
SafetyResult = Literal["pass", "fail", "uncertain"]
MAX_FIXES = 5


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


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

    _required_text = field_validator("title", "hero", "setting", "wish")(_non_blank)

    @field_validator("beats")
    @classmethod
    def _enough_beats(cls, beats: list[str]) -> list[str]:
        # The API guarantees the shape of the JSON, not that it is usable: a
        # brief without a real arc is rejected here and the planner is retried.
        beats = [beat for beat in beats if beat.strip()]
        if not 3 <= len(beats) <= 7:
            raise ValueError("a brief needs three to seven beats")
        return beats


class Criterion(_Strict):
    evidence: str = Field(
        description="One or two sentences citing what in the story earns the score."
    )
    score: Score = Field(description="From 1 (poor) to 5 (excellent).")


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

    @field_validator("fixes")
    @classmethod
    def _cap_fixes(cls, fixes: list[str]) -> list[str]:
        return [fix for fix in fixes if fix.strip()][:MAX_FIXES]

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
        """The judge's half of the publication gate. 4 means a small, harmless nit."""
        return self.safety.score >= 4 and self.age_fit.score >= 3

    @property
    def passes(self) -> bool:
        """Good enough to stop revising: spotless on safety, no weak criterion."""
        return self.safety.score == 5 and min(self.scores().values()) >= 4


class SafetyCheck(_Strict):
    evidence: str = Field(description="The line or moment this result rests on.")
    result: SafetyResult


class SafetyReview(_Strict):
    """A second, narrower reviewer that reads only the story.

    It has to agree with the judge before anything is shown. Anything other
    than a clean pass on every category blocks publication (fail closed).
    """

    frightening_content: SafetyCheck
    violence_or_cruelty: SafetyCheck
    risky_imitation: SafetyCheck
    mature_themes: SafetyCheck
    distress_left_unresolved: SafetyCheck
    false_promises_or_advice: SafetyCheck

    CATEGORIES: ClassVar[tuple[str, ...]] = (
        "frightening_content", "violence_or_cruelty", "risky_imitation",
        "mature_themes", "distress_left_unresolved", "false_promises_or_advice",
    )

    @property
    def cleared(self) -> bool:
        return all(getattr(self, name).result == "pass" for name in self.CATEGORIES)

    def concerns(self) -> list[str]:
        return [
            f"Safety ({name.replace('_', ' ')}): {getattr(self, name).evidence}"
            for name in self.CATEGORIES
            if getattr(self, name).result != "pass"
        ]
