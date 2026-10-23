#!/usr/bin/env python3
"""Compares the open-loop ranking (open_loop_report.json, by ADE) and the
closed-loop ranking (closed_loop_report.json, by collision+off-route
score), and counts how many of the 3 ranked positions disagree. This is
the number the resume's project 2 bullet calls out: displacement error
alone, used to pick a checkpoint, can pick a different winner than the
metric that actually matters once the model is driving."""
import json


def disagreement_count(ranking_a: list[str], ranking_b: list[str]) -> int:
    return sum(1 for a, b in zip(ranking_a, ranking_b) if a != b)


def main():
    with open("open_loop_report.json") as f:
        open_loop = json.load(f)
    with open("closed_loop_report.json") as f:
        closed_loop = json.load(f)

    open_ranking = open_loop["ranking"]
    closed_ranking = closed_loop["ranking"]
    n_disagree = disagreement_count(open_ranking, closed_ranking)

    print(f"open-loop ranking (best ADE first):              {open_ranking}")
    print(f"closed-loop ranking (best collision+off-route):  {closed_ranking}")
    print(f"disagreement: {n_disagree} of {len(open_ranking)} ranked positions differ")

    report = {
        "open_loop_ranking": open_ranking,
        "closed_loop_ranking": closed_ranking,
        "n_positions": len(open_ranking),
        "n_disagree": n_disagree,
    }
    with open("rank_compare_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("wrote rank_compare_report.json")


if __name__ == "__main__":
    main()
