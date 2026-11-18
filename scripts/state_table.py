"""Table 4 and Supp. Table S20 for the state stories (CPU). Per model, arm and training seed, over the training stories
in $VLENS_OUT/state_train/<model>/<arm>_<set>_s<seed>/ (untrained: 0_<set>), from scripts/state_tuning.py replies, eval
and judge. --set v5 (default, the paper's Table 4): the 80 stories of state_questions_v5.yaml (40 scenarios x happened /
cancelled); --set v2: the 40 stories of the second set.
  as written / reversed: the parsed one-word reply names the right option with the options in that order ("ab" / "ba");
  both orders: right in both ("correct text" of Table 4); right pictures: the lens picture of the subject (drawing seed
  0) shows the right state for at least two of the three judges ("correct picture"; each judge's latest answer counts);
  other-state pictures: it shows the scenario's other state.
Table 4 gives the shares in percent, as mean and sample standard deviation over the three training seeds, rounded half
up to one decimal. The paper's arms (--set v5): 0, V600 (right state, 600 updates), S600 (wrong state, 600 updates) for
both models.

    python scripts/state_table.py --model show_o2
    python scripts/state_table.py --model hunyuan
"""
import argparse
import glob
import json
import statistics
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal

from vlens import results

D = results.RESULTS / "state_train"
JUDGES = ("qwen3vl", "gemma3", "llavaov2")
PAPER_ARMS = {"v5": dict(show_o2="0,V600,S600", hunyuan="0,V600,S600"),
              "v2": dict(show_o2="0,V600,S600", hunyuan="0,V,S600")}


def votes(model):
    v = defaultdict(dict)   # file -> judge -> chosen (later files override earlier ones)
    for j in JUDGES:
        for f in sorted(glob.glob(str(D / f"judge_{j}_{model}_*.json"))):
            for a in json.load(open(f)):
                v[a["file"]][j] = a["chosen"]
    return v


def row(model, folder, v):
    d = D / model / folder
    if not (d / "replies.json").exists():
        return None
    rep = defaultdict(dict)
    for x in json.load(open(d / "replies.json")):
        if x["split"] == "train":
            rep[(x["id"], x["version"])][x["order"]] = x["answer"] == x["right"]   # the parsed one-word answer
    # one records file, or one per shard; the table counts drawing seed 0 only
    recs = [r for f in sorted(d.glob("records*.json")) if "extra" not in f.name for r in json.load(open(f))
            if r.get("seed", 0) == 0]
    pic, alt = {}, {}
    for r in recs:
        if r["split"] != "train":
            continue
        a = v.get(r["file"], {})
        pic[(r["id"], r["version"])] = len(a) == len(JUDGES) and sum(c == r["truth"] for c in a.values()) >= 2
        alt[(r["id"], r["version"])] = len(a) == len(JUDGES) and sum(c == r["other"] for c in a.values()) >= 2
    judged = sum(len(v.get(r["file"], {})) == len(JUDGES) for r in recs if r["split"] == "train")
    return dict(written=sum(o.get("ab", False) for o in rep.values()), reversed=sum(o.get("ba", False) for o in rep.values()),
                both=sum(o.get("ab", False) and o.get("ba", False) for o in rep.values()), pictures=sum(pic.values()),
                other_pictures=sum(alt.values()), stories=len(rep), judged=judged,
                train_pictures=sum(r["split"] == "train" for r in recs))


def one_decimal(x):
    """Rounded half up, as the paper prints it (29 of 80 stories = 36.25 -> 36.3)."""
    return Decimal(repr(round(x, 9))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hunyuan", choices=["show_o2", "hunyuan"])
    ap.add_argument("--set", default="v5", choices=["v5", "v2"])
    ap.add_argument("--arms", default="", help="comma separated (default: the paper's arms of the model)")
    a = ap.parse_args()
    arms = (a.arms or PAPER_ARMS[a.set][a.model]).split(",")
    v = votes(a.model)
    out = {}
    for arm in arms:
        for s in ([None] if arm == "0" else [0, 1, 2]):
            folder = f"0_{a.set}" if arm == "0" else f"{arm}_{a.set}_s{s}"
            r = row(a.model, folder, v)
            if r:
                out[folder] = r
                print(f"{folder:12s} as written {r['written']:2d}  reversed {r['reversed']:2d}  both {r['both']:2d}  "
                      f"right pictures {r['pictures']:2d}  other-state pictures {r['other_pictures']:2d}"
                      f"   (stories {r['stories']}, judged {r['judged']}/{r['train_pictures']})")
    print("Table 4 (% of stories; trained arms: mean +- s.d. over training seeds)")
    for arm in arms:
        rows = [r for k, r in out.items() if (k == f"0_{a.set}" if arm == "0" else k.startswith(f"{arm}_{a.set}_s"))]
        if not rows:
            continue
        text = [100 * r["both"] / r["stories"] for r in rows]
        pics = [100 * r["pictures"] / r["stories"] for r in rows]
        sd = lambda xs: f" +- {one_decimal(statistics.stdev(xs))}" if len(xs) > 1 else ""   # noqa: E731
        print(f"  {arm:5s} correct text {one_decimal(statistics.mean(text))}{sd(text)}   correct picture "
              f"{one_decimal(statistics.mean(pics))}{sd(pics)}")
    (D / f"table_{a.set}_{a.model}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
