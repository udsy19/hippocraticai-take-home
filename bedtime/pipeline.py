"""Orchestration: plan -> write -> judge -> revise, with code-owned control flow.

The models produce content and assessments; this module owns every decision
(pass or revise, which draft wins, when to stop, whether a story may be shown).
"""
from dataclasses import dataclass, field
from typing import Callable

from . import prompts
from .llm import LLM, ModelDeclined
from .schemas import StoryBrief, Verdict

PASS_BAR = 4.3  # mean rubric score a draft needs to stop the loop early
MAX_REQUEST_CHARS = 2000

SORRY = (
    "I couldn't make that one into a bedtime story that felt right for tonight. "
    "Shall we try a different idea? Maybe an animal, a place, or something that "
    "happened today."
)


@dataclass
class Draft:
    story: str
    verdict: Verdict


@dataclass
class Session:
    """Everything known about one story, so feedback rounds can build on it."""

    request: str
    minutes: int
    brief: StoryBrief
    drafts: list[Draft] = field(default_factory=list)
    feedback: list[str] = field(default_factory=list)
    best: Draft | None = None  # the draft currently shown to the listener

    @property
    def story(self) -> str:
        return self.best.story if self.best else SORRY


class Storyteller:
    def __init__(
        self,
        llm: LLM,
        max_revisions: int = 2,
        on_event: Callable[[str], None] = lambda message: None,
    ):
        self.llm = llm
        self.max_revisions = max_revisions
        self.on_event = on_event

    def tell(self, request: str, minutes: int = 5) -> Session:
        """Create a story for a new request."""
        request = request.strip()[:MAX_REQUEST_CHARS] or "Surprise me with a story."
        self.on_event("Planning the story...")
        brief = self.llm.structured(
            prompts.PLANNER_SYSTEM, prompts.planner_prompt(request), StoryBrief
        )
        session = Session(request=request, minutes=minutes, brief=brief)
        self.on_event(f"Writing a {brief.category} story: {brief.title}")
        story = self.llm.text(
            prompts.STORYTELLER_SYSTEM, prompts.storyteller_prompt(brief, minutes)
        )
        session.best = self._refine(session, story)
        return session

    def revise(self, session: Session, feedback: str) -> Session:
        """Apply the listener's requested change to the story they just heard."""
        if session.best is None:
            return session
        session.feedback.append(feedback.strip()[:MAX_REQUEST_CHARS])
        self.on_event("Reworking the story with your idea...")
        story = self.llm.text(
            prompts.STORYTELLER_SYSTEM,
            prompts.revision_prompt(
                session.brief, session.minutes, session.best.story,
                verdict=None, listener_feedback=session.feedback[-1],
            ),
        )
        # If no safe version of the change exists, the listener keeps the story
        # they already had instead of losing it.
        session.best = self._refine(session, story) or session.best
        return session

    def _refine(self, session: Session, story: str) -> Draft | None:
        """Judge a draft and revise until it passes or the budget runs out.

        Returns the best safe draft from this run, or None if none was safe.
        """
        candidates: list[Draft] = []
        for round_number in range(self.max_revisions + 1):
            draft = Draft(story, self._judge(session, story))
            session.drafts.append(draft)
            candidates.append(draft)
            self.on_event(_summary(round_number, draft.verdict))
            if draft.verdict.passes(PASS_BAR) or round_number == self.max_revisions:
                break
            self.on_event("Revising with the editor's notes...")
            story = self.llm.text(
                prompts.STORYTELLER_SYSTEM,
                prompts.revision_prompt(
                    session.brief, session.minutes, story,
                    verdict=draft.verdict, listener_feedback=None,
                ),
            )
        # A revision can score lower than its predecessor, so choose the best
        # safe draft instead of assuming the last one is the best.
        safe = [draft for draft in candidates if draft.verdict.is_safe]
        return max(safe, key=lambda draft: draft.verdict.average, default=None)

    def _judge(self, session: Session, story: str) -> Verdict:
        return self.llm.structured(
            prompts.JUDGE_SYSTEM,
            prompts.judge_prompt(
                session.request, session.brief, session.minutes, story, session.feedback
            ),
            Verdict,
            effort="high",  # the judge is the quality gate; give it room to think
        )


def _summary(round_number: int, verdict: Verdict) -> str:
    scores = " ".join(f"{name}={score}" for name, score in verdict.scores().items())
    status = "pass" if verdict.passes(PASS_BAR) else "revise"
    return f"Editor review {round_number + 1}: {verdict.average:.1f}/5 [{status}] {scores}"


__all__ = ["Storyteller", "Session", "Draft", "ModelDeclined", "SORRY"]
