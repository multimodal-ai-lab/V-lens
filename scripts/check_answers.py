"""Which two-step questions does a model answer correctly in text? (the question selection of Table 1, Table 5 and
Supp. Table S2)

A two-step question is used for a model only if that model answers it correctly: then it took the middle step. The model
answers each question greedily, as asked, plus "Answer in a few words." An answer counts as correct when it contains the
expected answer (numbers also as words, "8" or "eight"); every other answer is listed for a check by eye. The selections
used for the paper, after that check, are configs/test_sets/val.json (heldout_items) and configs/test_sets/test.json
(fresh2_items); the drawing and scoring scripts read those files.

    python scripts/check_answers.py --model bagel --items fresh2_items
"""

import argparse
import json
import re

import yaml

from vlens import prompts, results
from vlens.models import load

NUMBERS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
           "8": "eight", "9": "nine", "10": "ten", "12": "twelve", "20": "twenty"}


def matches(reply: str, answer: str) -> bool:
    reply, answer = reply.lower(), str(answer).lower()
    forms = {answer} | ({NUMBERS[answer]} if answer in NUMBERS else set())
    return any(re.search(rf"(?<![\w]){re.escape(f)}(?![\w])", reply) for f in forms)


def main():
    ap = argparse.ArgumentParser(description="Greedy text answers to the two-step questions, matched automatically.")
    ap.add_argument("--model", default="bagel")
    ap.add_argument("--items", default="heldout_items")
    args = ap.parse_args()
    # each item set keeps its own answers (heldout_answers, fresh2_answers)
    out = results.folder(f"{args.items.replace('_items', '')}_answers")
    items = yaml.safe_load((prompts.PROMPTS / f"{args.items}.yaml").read_text())["two_hop"]
    m = load(args.model)
    rows = []
    for i, it in enumerate(items):
        reply = m.answer(it["prompt"] + " Answer in a few words.", max_new_tokens=40)
        rows.append(dict(item=i, concept=it["concept"], prompt=it["prompt"], answer=it["answer"], reply=reply,
                         correct=matches(reply, it["answer"])))
        print(f"{i:2d} {'ok ' if rows[-1]['correct'] else 'NO '} {it['answer']!r:14} {reply[:90]!r}", flush=True)
    results.save_numbers(out / f"answers_{args.model}.json", dict(model=args.model, rows=rows,
                                                                  n_correct=sum(r["correct"] for r in rows)))
    print(f"{sum(r['correct'] for r in rows)} of {len(rows)} correct by the automatic match")


if __name__ == "__main__":
    main()
