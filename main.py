"""
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

1. An offline eval harness: a fixed set of ~30 requests (plain, vague, unsuitable,
   prompt-injection, real-worry "comfort" cases) run through the pipeline, with the
   judge's scores logged per prompt version, so prompt changes are measured instead
   of eyeballed. I would also hand-label a few stories to check that the judge's
   scores agree with a human's before trusting it as the gate.
2. A second, narrower safety judge that runs in parallel with the craft judge and
   votes independently, since one model grading its own family's writing is the
   weakest link in the system.
3. Light personalization across nights: remember the child's favorite characters
   and refrain so tomorrow's story can continue tonight's.
"""
import argparse
import os
import sys

import anthropic

from bedtime.llm import ClaudeLLM
from bedtime.pipeline import ModelDeclined, SORRY, Session, Storyteller

example_requests = "A story about a girl named Alice and her best friend Bob, who happens to be a cat."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tell a bedtime story for ages 5-10.")
    parser.add_argument("request", nargs="?", help="what the story should be about")
    parser.add_argument("--minutes", type=int, default=5, choices=range(2, 11),
                        metavar="2-10", help="read-aloud length (default 5)")
    parser.add_argument("--revisions", type=int, default=2,
                        help="max judge-driven rewrites per story (default 2)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show the plan, the judge's scores and its notes")
    return parser.parse_args()


def show(session: Session, verbose: bool) -> None:
    if session.brief.adjustments and session.best and not session.feedback:
        print("\nA note for the grown-up: I made this gentler for bedtime.")
        for change in session.brief.adjustments:
            print(f"  - {change}")
    print(f"\n{session.story}\n")
    if verbose and session.best:
        verdict = session.best.verdict
        print("--- editor's review of this version ---")
        for name in verdict.CRITERIA:
            criterion = getattr(verdict, name)
            print(f"{name:>16} {criterion.score}/5  {criterion.evidence}")
        print(f"drafts written: {len(session.drafts)}\n")


def main() -> int:
    args = parse_args()
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment first.", file=sys.stderr)
        return 1
    progress = (lambda message: print(f"  [{message}]", file=sys.stderr))
    request = args.request or input("What kind of story do you want to hear? ")
    print("\nOne moment while I think of a good one...")
    try:
        teller = Storyteller(
            ClaudeLLM(), max_revisions=args.revisions,
            on_event=progress if args.verbose else (lambda message: None),
        )
        session = teller.tell(request, minutes=args.minutes)
        show(session, args.verbose)
        while session.best and sys.stdin.isatty():
            feedback = input("Want anything changed? (press Enter to say goodnight) ")
            if not feedback.strip():
                break
            before = session.best
            session = teller.revise(session, feedback)
            if session.best is before:
                print("\nI couldn't find a cozy way to do that, so I kept our story as it was.")
            show(session, args.verbose)
    except ModelDeclined:
        print(f"\n{SORRY}")
    except anthropic.AuthenticationError:
        print("The Anthropic API rejected ANTHROPIC_API_KEY.", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError:
        print("Couldn't reach the Anthropic API. Check your connection.", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as err:
        print(f"The Anthropic API returned an error ({err.status_code}).", file=sys.stderr)
        return 1
    print("Goodnight. Sweet dreams.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
