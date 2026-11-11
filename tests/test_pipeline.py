"""Offline tests for the control flow. A scripted fake stands in for the model."""
import unittest

from bedtime.pipeline import SORRY, Storyteller
from bedtime.schemas import Criterion, StoryBrief, Verdict

BRIEF = StoryBrief(
    category="friendship", adjustments=[], must_keep=["Alice", "Bob the cat"],
    title="Alice and Bob", hero="Alice", companions=["Bob"], setting="a rooftop garden",
    wish="to find Bob's lost bell", gentle_obstacle="it rolled under the fence",
    beats=["a", "b", "c", "d", "e"], quiet_takeaway="friends wait for each other",
    refrain="Jingle, jingle, where did you go?",
)


def verdict(score: int, safety: int = 5, age_fit: int = 5) -> Verdict:
    values = {name: Criterion(evidence="e", score=score) for name in Verdict.CRITERIA}
    values["safety"] = Criterion(evidence="e", score=safety)
    values["age_fit"] = Criterion(evidence="e", score=age_fit)
    return Verdict(**values, keep=["the refrain"], fixes=["slow the ending"])


class FakeLLM:
    """Returns scripted verdicts in order and numbers each story it writes."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.prompts = []
        self.stories = 0

    def text(self, system, prompt, effort="medium"):
        self.prompts.append(prompt)
        self.stories += 1
        return f"story {self.stories}"

    def structured(self, system, prompt, schema, effort="medium"):
        return BRIEF if schema is StoryBrief else self.verdicts.pop(0)


class PipelineTest(unittest.TestCase):
    def test_good_first_draft_is_not_rewritten(self):
        llm = FakeLLM([verdict(5)])
        session = Storyteller(llm).tell("Alice and Bob")
        self.assertEqual(session.story, "story 1")
        self.assertEqual(llm.stories, 1)

    def test_weak_draft_is_revised_with_the_judges_notes(self):
        llm = FakeLLM([verdict(3), verdict(5)])
        session = Storyteller(llm).tell("Alice and Bob")
        self.assertEqual(session.story, "story 2")
        self.assertIn("slow the ending", llm.prompts[1])
        self.assertIn("story 1", llm.prompts[1])

    def test_best_draft_wins_when_a_revision_gets_worse(self):
        llm = FakeLLM([verdict(4), verdict(3), verdict(3)])
        session = Storyteller(llm, max_revisions=2).tell("Alice and Bob")
        self.assertEqual(session.story, "story 1")
        self.assertEqual(len(session.drafts), 3)

    def test_unsafe_drafts_are_never_shown(self):
        llm = FakeLLM([verdict(5, safety=4), verdict(5, safety=3)])
        session = Storyteller(llm, max_revisions=1).tell("something spooky")
        self.assertIsNone(session.best)
        self.assertEqual(session.story, SORRY)

    def test_high_average_cannot_outvote_the_safety_gate(self):
        self.assertFalse(verdict(5, safety=4).passes(4.3))
        self.assertFalse(verdict(5, age_fit=3).passes(4.3))

    def test_listener_feedback_reaches_the_writer_and_is_rejudged(self):
        llm = FakeLLM([verdict(5), verdict(5)])
        teller = Storyteller(llm)
        session = teller.revise(teller.tell("Alice and Bob"), "add a dragon")
        self.assertEqual(session.story, "story 2")
        self.assertIn("add a dragon", llm.prompts[-1])

    def test_unsafe_feedback_keeps_the_previous_story(self):
        llm = FakeLLM([verdict(5), verdict(5, safety=2), verdict(5, safety=2)])
        teller = Storyteller(llm, max_revisions=1)
        session = teller.revise(teller.tell("Alice and Bob"), "make it terrifying")
        self.assertEqual(session.story, "story 1")


if __name__ == "__main__":
    unittest.main()
