"""Validate issue-link syntax using a GitHub event JSON file (no network or secrets)."""

import argparse
import json
from pathlib import Path

from repository_checks import issue_references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    args = parser.parse_args()
    event = json.loads(args.event.read_text())
    if not issue_references(event, args.repository):
        raise SystemExit("PR body requires an explicit issue link, e.g. Refs #15")
    print("PR issue-link syntax passed (issue existence/review enforcement is separate)")


if __name__ == "__main__":
    main()
