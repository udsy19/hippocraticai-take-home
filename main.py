"""
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

1. Calibrate the reviewers against people. The judge and the safety reviewer are the
   same model that wrote the story, so I would hand-label around 50 stories (including
   deliberately flawed ones) and measure how often each reviewer agrees with a human
   before trusting their scores further. evals/run.py already gives the harness; it
   needs labelled data and a larger fixture set so the win rates carry error bars.
2. Track tokens and cost per stage next to the call counts and latency already
   recorded, then test cheaper settings (lower judge effort, skipping the second
   review when the first draft passes cleanly) against the eval instead of guessing.
3. An --age flag that narrows 5-10 into three bands with their own vocabulary and
   length targets, checked against the same eval.
"""
import argparse
import os
import sys

from bedtime.llm import ClaudeLLM
from bedtime.metrics import MeteredLLM
from bedtime.pipeline import (
    MAX_REVISIONS, SORRY, InvalidRequest, ModelDeclined, ModelUnavailable,
    Session, Storyteller,
)

example_requests = "A story about a girl named Alice and her best friend Bob, who happens to be a cat."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tell a bedtime story for ages 5-10.")
    parser.add_argument("request", nargs="?", help="what the story should be about")
    parser.add_argument("--minutes", type=int, default=5, choices=range(2, 11),
                        metavar="2-10", help="read-aloud length (default 5)")
    parser.add_argument("--revisions", type=int, default=2,
                        choices=range(0, MAX_REVISIONS + 1), metavar=f"0-{MAX_REVISIONS}",
                        help="max review-driven rewrites per story (default 2)")
    parser.add_argument("--backend", choices=["api", "agent-sdk"], default="api",
                        help="Messages API with ANTHROPIC_API_KEY (default), or the "
                        "Claude Agent SDK using your Claude Code login")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show the plan, the judge's scores and its notes")
    return parser.parse_args()


def make_llm(backend: str):
    if backend == "api":
        return ClaudeLLM()
    from bedtime.agent_sdk_llm import AgentSDKLLM  # optional dependency
    return AgentSDKLLM()


def show(session: Session, verbose: bool, first: bool = False) -> None:
    if first and session.brief.adjustments and session.best:
        print("\nA note for the grown-up: I made this gentler for bedtime.")
        for change in session.brief.adjustments:
            print(f"  - {change}")
    print(f"\n{session.story}\n")
    if verbose and session.best:
        draft = session.best
        print("--- editor's review of this version ---")
        for name in draft.verdict.CRITERIA:
            criterion = getattr(draft.verdict, name)
            print(f"{name:>16} {criterion.score}/5  {criterion.evidence}")
        print(f"{'safety check':>16} {'clear' if draft.safety.cleared else 'concerns'}")
        print(f"{'length':>16} {draft.words} words")
        print(f"drafts reviewed: {len(session.drafts)}\n")


def main() -> int:
    args = parse_args()
    if args.backend == "api" and not os.getenv("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY, or use --backend agent-sdk.", file=sys.stderr)
        return 1
    llm = MeteredLLM(make_llm(args.backend))
    progress = (lambda message: print(f"  [{message}]", file=sys.stderr))
    teller = Storyteller(
        llm, max_revisions=args.revisions,
        on_event=progress if args.verbose else (lambda message: None),
    )
    request = args.request or input("What kind of story do you want to hear? ")
    print("\nOne moment while I think of a good one...")
    try:
        session = teller.tell(request, minutes=args.minutes)
        show(session, args.verbose, first=True)
        while session.best and sys.stdin.isatty():
            feedback = input("Want anything changed? (press Enter to say goodnight) ")
            if not feedback.strip():
                break
            try:
                changed = teller.revise(session, feedback)
            except InvalidRequest as err:
                print(f"\n{err}")
                continue
            if not changed:
                print("\nI couldn't find a cozy way to do that, so I kept our story as it was.")
            show(session, args.verbose)
    except InvalidRequest as err:
        print(f"\n{err}")
        return 2
    except ModelDeclined:
        print(f"\n{SORRY}")
    except ModelUnavailable as err:
        print(f"\n{err}", file=sys.stderr)
        return 1
    finally:
        if args.verbose:
            print(f"  [{llm.metrics.summary()}]", file=sys.stderr)
    print("Goodnight. Sweet dreams.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
