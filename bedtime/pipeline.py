"""Orchestration: plan -> write -> review -> revise, with code-owned control flow.

The models produce content and assessments; this module owns every decision
(pass or revise, which draft wins, when to stop, whether a story may be shown).
"""
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from . import prompts
from .llm import LLM, InvalidRequest, ModelDeclined, ModelUnavailable, StoryError
from .schemas import SafetyReview, StoryBrief, Verdict

MAX_REQUEST_CHARS = 2000
MINUTES = range(2, 11)
MAX_REVISIONS = 5

SORRY = (
    "I couldn't make that one into a bedtime story that felt right for tonight. "
    "Shall we try a different idea? Maybe an animal, a place, or something that "
    "happened today."
)


@dataclass
class Draft:
    """One story and everything known about it. Immutable once reviewed."""

    story: str
    verdict: Verdict
    safety: SafetyReview
    words: int
    length_ok: bool
    format_problems: list[str] = field(default_factory=list)

    @property
    def publishable(self) -> bool:
        """May a family see this? The judge and the safety reviewer must agree,
        and the text must be readable aloud exactly as written."""
        return (
            bool(self.story.strip())
            and not self.format_problems
            and self.verdict.is_safe
            and self.safety.cleared
        )

    @property
    def passes(self) -> bool:
        """Good enough to stop revising."""
        return self.publishable and self.verdict.passes and self.length_ok


@dataclass
class Session:
    """Everything known about one story, so feedback rounds can build on it."""

    request: str
    minutes: int
    brief: StoryBrief
    drafts: list[Draft] = field(default_factory=list)  # every draft reviewed
    feedback: list[str] = field(default_factory=list)  # accepted changes only
    rejected_feedback: list[str] = field(default_factory=list)
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
        if not 0 <= max_revisions <= MAX_REVISIONS:
            raise InvalidRequest(f"revisions must be between 0 and {MAX_REVISIONS}")
        self.llm = llm
        self.max_revisions = max_revisions
        self.on_event = on_event

    def tell(self, request: str, minutes: int = 5) -> Session:
        """Create a story for a new request."""
        if minutes not in MINUTES:
            raise InvalidRequest(
                f"minutes must be between {MINUTES.start} and {MINUTES.stop - 1}"
            )
        request = _checked(request, "request") or "Surprise me with a story."
        self.on_event("Planning the story...")
        brief = self._structured(
            prompts.PLANNER_SYSTEM, prompts.planner_prompt(request), StoryBrief
        )
        session = Session(request=request, minutes=minutes, brief=brief)
        self.on_event(f"Writing a {brief.category} story: {brief.title}")
        story = self.llm.text(
            prompts.STORYTELLER_SYSTEM, prompts.storyteller_prompt(brief, minutes)
        )
        session.best = self._refine(session, story, session.feedback)
        return session

    def revise(self, session: Session, feedback: str) -> bool:
        """Apply the listener's requested change to the story they just heard.

        Transactional: the feedback and the new story are committed together,
        and only if a publishable version exists. Otherwise the session is
        left exactly as it was, apart from a note of the rejected request.
        Returns True if the story changed.
        """
        feedback = _checked(feedback, "feedback")
        if session.best is None or not feedback:
            return False
        self.on_event("Reworking the story with your idea...")
        proposed = [*session.feedback, feedback]
        try:
            story = self.llm.text(
                prompts.STORYTELLER_SYSTEM,
                prompts.revision_prompt(
                    session.brief, session.minutes, session.best.story,
                    keep=[], fixes=[], listener_feedback=feedback,
                ),
            )
            candidate = self._refine(session, story, proposed)
        except ModelDeclined:
            candidate = None
        if candidate is None:
            session.rejected_feedback.append(feedback)
            return False
        session.feedback = proposed
        session.best = candidate
        return True

    def _refine(self, session: Session, story: str, feedback: list[str]) -> Draft | None:
        """Review a draft and revise until it passes or the budget runs out.

        Returns the best publishable draft from this run, or None. A draft
        that cannot be reviewed is never shown (fail closed).
        """
        candidates: list[Draft] = []
        for round_number in range(self.max_revisions + 1):
            try:
                draft = self._review(session, story, feedback)
            except ModelDeclined:
                self.on_event("A review could not be completed; discarding that draft.")
                break
            session.drafts.append(draft)
            candidates.append(draft)
            self.on_event(_summary(round_number, draft))
            if draft.passes or round_number == self.max_revisions:
                break
            self.on_event("Revising with the editor's notes...")
            try:
                story = self.llm.text(
                    prompts.STORYTELLER_SYSTEM,
                    prompts.revision_prompt(
                        session.brief, session.minutes, story,
                        keep=draft.verdict.keep, fixes=_fixes(draft, session.minutes),
                        listener_feedback=None,
                    ),
                )
            except ModelDeclined:
                break
        # A revision can score lower than its predecessor, so choose the best
        # publishable draft instead of assuming the last one is the best.
        publishable = [draft for draft in candidates if draft.publishable]
        return max(
            publishable,
            key=lambda d: (d.verdict.safety.score, d.length_ok, d.verdict.average),
            default=None,
        )

    def _review(self, session: Session, story: str, feedback: list[str]) -> Draft:
        """Run the quality judge and the safety reviewer side by side."""
        with ThreadPoolExecutor(max_workers=2) as pool:
            verdict = pool.submit(
                self._structured,
                prompts.JUDGE_SYSTEM,
                prompts.judge_prompt(
                    session.request, session.brief, session.minutes, story, feedback
                ),
                Verdict,
                "high",  # the reviewers are the gate; give them room to think
            )
            safety = pool.submit(
                self._structured,
                prompts.SAFETY_SYSTEM, prompts.safety_prompt(story), SafetyReview, "high",
            )
            verdict, safety = verdict.result(), safety.result()
        words = _body_words(story)
        low, high = prompts.word_range(session.minutes)
        return Draft(
            story, verdict, safety, words, low <= words <= high, _format_problems(story)
        )

    def _structured(self, system, prompt, schema, effort="medium"):
        """A structured call, retried once if the reply was unusable."""
        try:
            return self.llm.structured(system, prompt, schema, effort)
        except ModelDeclined:
            return self.llm.structured(system, prompt, schema, effort)


def _checked(text: str, name: str) -> str:
    text = text.strip()
    if len(text) > MAX_REQUEST_CHARS:
        raise InvalidRequest(
            f"That {name} is {len(text)} characters; please keep it under "
            f"{MAX_REQUEST_CHARS} so nothing gets cut off."
        )
    return text


def _body_words(story: str) -> int:
    """Words in the story, not counting the title line."""
    _title, _, body = story.strip().partition("\n")
    return len(body.split())


def _format_problems(story: str) -> list[str]:
    """Things a parent cannot read aloud. Found by pattern, not by a model."""
    problems = []
    placeholders = re.findall(r"\[[^\]\n]{1,40}\]|\{[^}\n]{1,40}\}", story)
    if placeholders:
        problems.append(
            f"Format: remove the unfilled placeholders {', '.join(sorted(set(placeholders)))}; "
            "invent a specific name or detail instead."
        )
    if re.search(r"[*`]|^\s*#", story, re.M):
        problems.append(
            "Format: plain text only. Remove markdown such as *asterisks*, # headings, "
            "bullets or backticks; write sound effects as ordinary words."
        )
    return problems


def _fixes(draft: Draft, minutes: int) -> list[str]:
    """What the writer is told to change: safety, format, length, then craft."""
    fixes = draft.safety.concerns() + draft.format_problems
    if not draft.length_ok:
        low, high = prompts.word_range(minutes)
        change = "cut it down" if draft.words > high else "add to the middle of the story"
        fixes.append(
            f"Length: the story is {draft.words} words and must be between {low} and "
            f"{high}; {change}."
        )
    return fixes + draft.verdict.fixes


def _summary(round_number: int, draft: Draft) -> str:
    scores = " ".join(f"{name}={score}" for name, score in draft.verdict.scores().items())
    status = "pass" if draft.passes else "revise" if draft.publishable else "blocked"
    safety = "clear" if draft.safety.cleared else "concerns"
    length = "ok" if draft.length_ok else "off"
    if draft.format_problems:
        length += " | format: problems"
    return (
        f"Review {round_number + 1}: {draft.verdict.average:.1f}/5 [{status}] {scores} "
        f"| safety check: {safety} | {draft.words} words: {length}"
    )


__all__ = [
    "Storyteller", "Session", "Draft", "SORRY",
    "StoryError", "InvalidRequest", "ModelDeclined", "ModelUnavailable",
]
