"""Run one judge (vlens/judges.py) over a set of cell pictures and store its answers (Sec. 4.1, Supp. B).

--set cells:<folder> judges the pictures of scripts/draw_cells.py (or of another drawing script that writes records in
the same format) in $VLENS_OUT/<folder>/ (all models, or --model). Three questions:
- default, the choice question (lens-variant and ablation tables): the item's concept, the first literal meaning of its
  cue words (so a web is not counted as a spider; for a word with two meanings, its other meaning, so the panel can tell
  them apart), three other concept names drawn at random, and "none of these", shuffled by the picture's file name;
- --subject, the main-subject question of the paper (Sec. 4.1): the intended object, the alternative, "both, as main
  subjects", "neither, as a main subject"; answers go to their own set (<set>_subject);
- --pair, an earlier two-thing question (the truth, its alternative, both, neither); answers go to <set>_pair.
The answer is the letter with the highest probability (one forward pass), plus a short open answer. --shard i --of n
splits the pictures by a hash of the file name; --kind keeps one kind of item. Pictures this judge has already answered
on the same set are skipped. Answers: $VLENS_OUT/judges/answers_<judge>_<set>[...].json.

    python scripts/run_judge.py --judge qwen3vl --set cells:fresh2_cells        # environment: sensenova
    python scripts/run_judge.py --judge gemma3 --set cells:fresh2_cells         # environment: sensenova
    python scripts/run_judge.py --judge llavaov2 --set cells:fresh2_cells       # environment: judges
"""

import argparse
import json
import random
import time
import zlib

from PIL import Image

from vlens import prompts, results
from vlens.judges import (Judge, answer_files, pair_options, pair_question, shuffled_options, subject_options,
                          subject_question)

FOLDER = results.RESULTS / "judges"


def cells_tasks(folder: str, model: str | None) -> list[dict]:
    import yaml
    from vlens.prompts import PROMPTS
    base = results.RESULTS / folder
    names = prompts.concept_names()
    item_files = {}
    tasks = []
    for model_dir in sorted(p for p in base.iterdir() if p.is_dir() and (model is None or p.name == model)):
        seen = set()
        for rec_file in sorted(model_dir.glob("records*.json")):
            for r in json.loads(rec_file.read_text()):
                if r["file"] in seen:
                    continue
                seen.add(r["file"])
                if r["items"] not in item_files:
                    item_files[r["items"]] = yaml.safe_load((PROMPTS / f"{r['items']}.yaml").read_text())
                item = item_files[r["items"]][r["kind"]][r["item"]]
                if r["kind"] == "double_meaning":  # the word's other meaning is always offered, so the panel can tell them apart
                    meanings = [m["name"] if isinstance(m, dict) else m for m in (item["meaning_a"], item["meaning_b"])]
                    literal = next(m for m in meanings if m != r["truth"])
                elif r["kind"] == "reads":  # texts with named reads: the item names its alternative
                    literal = item.get("alt")
                else:
                    literal = (item.get("literals") or [None])[0]
                rng = random.Random(r["file"])
                fixed = [n for n in (r["truth"], literal) if n]
                others = rng.sample([n for n in names if n not in fixed], 5 - len(fixed))
                tasks.append(dict(r, path=str(base / r["file"]), literal=literal,
                                  options=shuffled_options(fixed + others, r["file"])))
    return tasks


def main():
    ap = argparse.ArgumentParser(description="Run one judge over the cell pictures of one output folder.")
    ap.add_argument("--judge", required=True, choices=["qwen3vl", "gemma3", "llavaov2"])
    ap.add_argument("--set", required=True, help="cells:<folder>, a folder of $VLENS_OUT with cell pictures")
    ap.add_argument("--model", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--kind", default=None, help="cells only: two_hop or double_meaning (default: both)")
    ap.add_argument("--items", default=None, help="cells only: keep pictures of these item files (comma separated)")
    ap.add_argument("--ways", default=None, help="cells only: keep pictures drawn these ways (comma separated)")
    ap.add_argument("--subject", action="store_true", help="the main-subject question (the truth, its alternative, "
                                                          "both, neither, as MAIN subjects)")
    ap.add_argument("--pair", action="store_true", help="the earlier two-thing question (the truth, its alternative, "
                                                       "both, neither); answers go to a separate set")
    args = ap.parse_args()
    if not args.set.startswith("cells:"):
        raise SystemExit("--set must be cells:<folder>")
    tasks = cells_tasks(args.set.split(":", 1)[1], args.model)
    if args.items:
        tasks = [t for t in tasks if t.get("items") in args.items.split(",")]
    if args.ways:
        tasks = [t for t in tasks if t.get("way") in args.ways.split(",")]
    set_tag = args.set + ("_pair" if args.pair else "") + ("_subject" if args.subject else "")
    if args.subject:
        tasks = [dict(t, options=subject_options(t["truth"], t["literal"], t["file"])) for t in tasks if t.get("literal")]
    if args.pair:  # only pictures whose item names an alternative can be judged as a pair
        tasks = [dict(t, options=pair_options(t["truth"], t["literal"], t["file"])) for t in tasks if t.get("literal")]
    if args.kind:
        tasks = [t for t in tasks if t.get("kind") == args.kind]
    # A picture's shard follows from its file name, not its place in the list: the list grows while pictures are still
    # being drawn, and shards started at different times would otherwise overlap and leave gaps.
    tasks = [t for t in tasks if zlib.crc32(t["file"].encode()) % args.of == args.shard]
    done = set()  # pictures this judge has already answered in an earlier run on the same set, any shard
    for f in answer_files(FOLDER, args.judge, set_tag):
        done |= {a["file"] for a in json.loads(f.read_text())}
    tasks = [t for t in tasks if t["file"] not in done]
    print(f"{len(tasks)} pictures to judge ({len(done)} judged before)", flush=True)
    if not tasks:
        return
    FOLDER.mkdir(parents=True, exist_ok=True)
    judge = Judge(args.judge)
    t0 = time.time()
    answers = []
    for k, t in enumerate(tasks):
        image = Image.open(t["path"])
        question = subject_question if args.subject else pair_question if args.pair else None
        choice = judge.choose(image, t["options"], question) if question else judge.choose(image, t["options"])
        answers.append(dict(t, open=judge.open(image), chosen=choice.chosen,
                            probs=dict(zip(choice.options, [round(p, 4) for p in choice.probs]))))
        if k % 25 == 0:
            print(f"[{time.time() - t0:6.0f}s] {k}/{len(tasks)} {t['file']}: {answers[-1]['chosen']!r} / "
                  f"{answers[-1]['open']!r} (truth {t['truth']})", flush=True)
    tag = (set_tag.replace(":", "_") + (f"_{args.model}" if args.model else "")
           + (f"_{args.shard}of{args.of}" if args.of > 1 else "") + f"_{time.strftime('%H%M%S')}")  # never overwrite
    (FOLDER / f"answers_{args.judge}_{tag}.json").write_text(json.dumps(answers, indent=1))
    print("->", FOLDER / f"answers_{args.judge}_{tag}.json")


if __name__ == "__main__":
    main()
