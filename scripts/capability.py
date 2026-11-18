"""General capabilities after lens tuning (Supp. F.3, Table S19): ordinary image generation on 120 GenEval prompts
scored by the three judges, and text knowledge on 300 MMLU-Pro questions, for training seed 0 of text supervision (T),
ordinary drawing supervision (G) and lens tuning (V) against each untrained model (base). BAGEL, SenseNova-U1 and
HunyuanImage-3.0, with the adapters of scripts/lens_tuning.py ($VLENS_DATA/<store>/<arm>_s0.pt), on throughout.

Benchmarks:
- geneval: 120 GenEval prompts (20 from each of its six categories; items/geneval120.json, fixed by `prep` from the
  prompt file of the GenEval repository at a pinned commit). One picture per prompt through the model's ordinary
  text-to-image path with the adapters on throughout (read_prompt, then draw with drawing seed 0 and the model's own
  default settings). Scored by the three judges instead of GenEval's object detector: one yes/no question per
  requirement of the prompt (each object present, the exact count, each colour, the relative position;
  vlens.judges.Judge.choose with the options "yes", "no"). A judge accepts a picture when it answers "yes" to every
  question of the prompt; the prompt passes when at least two of the three judges accept it.
- mmlu_pro: the 300 MMLU-Pro questions of scripts/benchmarks.py (run `python scripts/benchmarks.py prep` first), asked
  and scored by its run_benches() (the option whose letter is the most likely first reply token).

Steps:
  prep    (CPU) items/geneval120.json: per category, sorted(random.Random(0).sample(<line numbers>, 20))
  run     (GPU) one model and checkpoint: --benches mmlu_pro,geneval [--shard k --of n] (the shard splits only the
          pictures); pictures in <model>/<arm>/geneval/<id>.png, answers in <model>/<arm>/mmlu_pro.jsonl; resumes.
          --tag smoke: one prompt per category and the first --limit questions, into smoke/
  judge   (GPU) one judge over every picture drawn so far: judges/<judge>/<model>_<arm>.jsonl, one line per question;
          resumes (one process per judge at a time: it is the only writer of its files)
  score   (CPU) table.json, table.txt and summary.json (base value and change with its 95% paired bootstrap interval per
          model and checkpoint; GenEval: prompts resampled within categories)
  status  (CPU) one JSON line: adapter files, pictures, answers, judged pictures

    python scripts/benchmarks.py prep
    python scripts/capability.py prep
    python scripts/capability.py run --model bagel --arm base
    python scripts/capability.py run --model bagel --arm V
    python scripts/capability.py judge --judge qwen3vl      # also gemma3, llavaov2
    python scripts/capability.py score
"""

import argparse
import contextlib
import hashlib
import json
import random
import re
import time

from vlens import paths, results

bench = results.load_script("benchmarks")

OUT = results.RESULTS / "capability"
ITEMS = OUT / "items"
GENEVAL_COMMIT = "d927da8e42fde2b1b5cd743da4df5ff83c1654ff"   # the last commit touching the prompt file
GENEVAL_URL = f"https://raw.githubusercontent.com/djghosh13/geneval/{GENEVAL_COMMIT}/prompts/evaluation_metadata.jsonl"
GENEVAL_SHA256 = "5c48e0813e812e3c373fa5c8ed07a8f0a483be30272b4427b0559c8048e67c13"   # of that file (553 prompts)
CATEGORIES = ("single_object", "two_object", "counting", "colors", "position", "color_attr")
CATEGORY_NAMES = dict(single_object="single object", two_object="two objects", counting="counting", colors="colours",
                      position="position", color_attr="colour attribution")
PER_CATEGORY = 20
DRAW_SEED = 0
MODELS = ("bagel", "sensenova", "hunyuan")
STORE = dict(bagel="vsup", sensenova="vsup_sensenova", hunyuan="vsup_hunyuan")   # lens_tuning.py's MODELS[...]["store"]
ADAPTER_LAYERS = dict(bagel=8, sensenova=24, hunyuan=9)   # lens_tuning.py's range(insertion layer)
ARMS = ("base", "T", "V", "G")
JUDGES = ("qwen3vl", "gemma3", "llavaov2")
NUMBER = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
RELATION = {"left of": "to the left of", "right of": "to the right of", "above": "above", "below": "below"}


# ------------------------------------------------------------------------------------------------------------ items

def article(noun):
    return "an" if noun[0] in "aeiou" else "a"


def checks(row):
    """The yes/no questions for one GenEval prompt, one per requirement GenEval's own evaluation checks: every object
    present (counting: exactly that many, as its include count n with exclude count n + 1 demands), every colour, the
    relative position."""
    inc = row["include"]
    if row["tag"] == "counting":
        (o,) = inc
        n = o["count"]
        assert row.get("exclude") == [{"class": o["class"], "count": n + 1}], row
        plural = re.fullmatch(rf"a photo of {NUMBER[n]} (.+)", row["prompt"]).group(1)
        return [dict(kind="count", question=f"Are there exactly {NUMBER[n]} {plural} in this picture?")]
    assert not row.get("exclude"), row
    out = []
    for o in inc:
        assert o["count"] == 1, row
        out.append(dict(kind="object", question=f"Is there {article(o['class'])} {o['class']} in this picture?"))
    for o in inc:
        if "color" in o:
            out.append(dict(kind="colour", question=f"Is the {o['class']} in this picture {o['color']}?"))
    for o in inc:
        if "position" in o:
            rel, j = o["position"]
            side = " (as seen by the viewer)" if rel in ("left of", "right of") else ""
            out.append(dict(kind="position",
                            question=f"Is the {o['class']} {RELATION[rel]} the {inc[j]['class']} in this picture{side}?"))
    return out


def step_prep(args):
    import urllib.request
    ITEMS.mkdir(parents=True, exist_ok=True)
    raw = ITEMS / "evaluation_metadata.jsonl"
    if not raw.exists():
        raw.write_bytes(urllib.request.urlopen(GENEVAL_URL, timeout=120).read())
    data = raw.read_bytes()
    if hashlib.sha256(data).hexdigest() != GENEVAL_SHA256:
        raise SystemExit(f"{raw} is not the pinned GenEval prompt file")
    rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    items = []
    for tag in CATEGORIES:
        lines = [i for i, r in enumerate(rows) if r["tag"] == tag]
        for i in sorted(random.Random(0).sample(lines, PER_CATEGORY)):
            r = rows[i]
            items.append(dict(id=f"{tag}_{i:03d}", line=i, category=tag, prompt=r["prompt"], include=r["include"],
                              exclude=r.get("exclude", []), checks=checks(r)))
    text = json.dumps(items, indent=1)
    f = ITEMS / "geneval120.json"
    if f.exists() and f.read_text() != text:
        raise SystemExit(f"{f} exists and differs: the items are fixed")
    f.write_text(text)
    meta = ITEMS / "geneval120_source.json"
    if not meta.exists():
        meta.write_text(json.dumps(dict(
            source=GENEVAL_URL, commit=GENEVAL_COMMIT, sha256=hashlib.sha256(data).hexdigest(), prompts=len(rows),
            per_category=PER_CATEGORY, rule="per category in the order " + ", ".join(CATEGORIES)
            + ": sorted(random.Random(0).sample(<line numbers of that category>, 20))"), indent=1))
    print(f"{len(items)} prompts, {sum(len(it['checks']) for it in items)} questions per judge and checkpoint ->", f)
    for it in items[::PER_CATEGORY]:
        print(" ", it["id"], it["prompt"], "|", " / ".join(c["question"] for c in it["checks"]))


def geneval_items(smoke=False):
    items = json.loads((ITEMS / "geneval120.json").read_text())
    return items[::PER_CATEGORY] if smoke else items


def lines(f):
    """The complete JSON lines of a file another job may be appending to (read only; a cut-off last line is skipped)."""
    out = []
    if f.exists():
        for line in f.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


# -------------------------------------------------------------------------------------------------------------- run

def sha256(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def step_run(args):
    import torch
    from vlens.lora import add_lora
    from vlens.models import load

    model, arm = args.model, args.arm
    smoke = args.tag == "smoke"
    out = (OUT / "smoke" if smoke else OUT) / model / arm
    out.mkdir(parents=True, exist_ok=True)
    todo = args.benches.split(",")
    assert set(todo) <= {"mmlu_pro", "geneval"}, todo
    assert smoke or not args.limit, "a limited run writes only into smoke/"
    m = load(model)
    adapters, info = None, dict(model=model, arm=arm, adapters=None, draw_seed=DRAW_SEED)
    if arm != "base":   # as scripts/lens_tuning.py with_adapters(m, arm, 0), checking the adapter layers
        f = paths.DATA / STORE[model] / f"{arm}_s0.pt"
        state = torch.load(f)
        assert state.get("arm", arm) == arm and state.get("seed", 0) == 0, (f, state.get("arm"), state.get("seed"))
        layers = sorted({int(k.split(".")[1]) for k in state["adapters"]})
        assert layers == list(range(ADAPTER_LAYERS[model])), layers
        adapters = add_lora(m, layers, rank=state["rank"], alpha=state["alpha"])
        adapters.load_state_dict(state["adapters"])
        size = sum(float(p.detach().float().norm()) ** 2 for p in adapters.parameters()) ** 0.5
        info.update(adapters=f"{STORE[model]}/{arm}_s0.pt", sha256=sha256(f), updates=state.get("update"),
                    rank=state["rank"], alpha=state["alpha"], weight_norm=size)
        print(f"adapters {info['adapters']}: layers 0-{layers[-1]}, {len(adapters.modules)} projections, rank "
              f"{state['rank']}, alpha {state['alpha']}, {state.get('update')} updates, weight norm {size:.3f}", flush=True)
    (out / "checkpoint.json").write_text(json.dumps(info, indent=1))
    for b in todo:
        if b == "mmlu_pro":
            bench.run_benches(m, model, adapters, out, ["mmlu_pro"], args.limit)
        else:
            draw_geneval(m, adapters, out / "geneval", geneval_items(smoke)[args.shard::args.of])


def draw_geneval(m, adapters, folder, items):
    """One picture per prompt, adapters on throughout, as scripts/lens_tuning.py step_generate draws."""
    folder.mkdir(parents=True, exist_ok=True)
    on = adapters.on if adapters is not None else contextlib.nullcontext
    t0, n = time.time(), 0
    for it in items:
        f = folder / f"{it['id']}.png"
        if f.exists():
            continue
        with on():
            reading = m.read_prompt(it["prompt"])
            picture = m.to_picture(m.draw(reading, seed=DRAW_SEED)[0])
        part = f.with_name(f.name + ".part")   # renamed when complete, so a stopped job leaves no broken picture
        picture.save(part, format="PNG")
        part.replace(f)
        n += 1
        if n <= 2 or n % 10 == 0:
            print(f"[{time.time() - t0:6.0f}s] geneval {n}: {it['prompt']!r}", flush=True)
    have = sum((folder / (it["id"] + ".png")).exists() for it in items)
    print(f"geneval: {have} of {len(items)} pictures in {folder}", flush=True)


# ------------------------------------------------------------------------------------------------------------ judge

def yes_no(question):
    """Judge.choose's question for the options ["yes", "no"] (letters A and B)."""
    return lambda options: (question + "\n" + "\n".join(f"{L}) {o}" for L, o in zip("AB", options))
                            + "\nAnswer with the letter only.")


def step_judge(args):
    from PIL import Image
    from vlens.judges import Judge

    smoke = args.tag == "smoke"
    base = OUT / "smoke" if smoke else OUT
    items = geneval_items(smoke)
    judge, t0, n = None, time.time(), 0
    for model in MODELS:
        for arm in ARMS:
            folder = base / model / arm / "geneval"
            if not folder.is_dir():
                continue
            f = base / "judges" / args.judge / f"{model}_{arm}.jsonl"
            f.parent.mkdir(parents=True, exist_ok=True)
            done = {(r["id"], r["check"]) for r in bench.records(f)}
            todo = [it for it in items if (folder / (it["id"] + ".png")).exists()
                    and any((it["id"], k) not in done for k in range(len(it["checks"])))]
            if not todo:
                continue
            if judge is None:
                judge = Judge(args.judge)
                print(f"[{time.time() - t0:6.0f}s] {args.judge} loaded", flush=True)
            with f.open("a") as fh:
                for it in todo:
                    image = Image.open(folder / (it["id"] + ".png")).convert("RGB")
                    for k, c in enumerate(it["checks"]):
                        if (it["id"], k) in done:
                            continue
                        ch = judge.choose(image, ["yes", "no"], question=yes_no(c["question"]))
                        fh.write(json.dumps(dict(id=it["id"], check=k, kind=c["kind"], question=c["question"],
                                                 p_yes=ch.probs[0], yes=ch.chosen == "yes")) + "\n")
                        fh.flush()
                    n += 1
            print(f"[{time.time() - t0:6.0f}s] {args.judge}: {model} {arm} judged, {n} pictures so far", flush=True)
    print(f"{args.judge}: {n} pictures judged in this job")


# ------------------------------------------------------------------------------------------------------------ score

def verdicts(base, model, arm, items):
    """{id: dict(judges={judge: accepted every requirement}, passed=at least two judges accept)} for the pictures that
    every judge answered completely."""
    ans = {j: {} for j in JUDGES}
    for j in JUDGES:
        for r in lines(base / "judges" / j / f"{model}_{arm}.jsonl"):
            ans[j].setdefault(r["id"], {})[r["check"]] = r["yes"]
    out = {}
    for it in items:
        n = len(it["checks"])
        if all(len(ans[j].get(it["id"], {})) == n for j in JUDGES):
            acc = {j: all(ans[j][it["id"]].values()) for j in JUDGES}
            out[it["id"]] = dict(judges=acc, passed=sum(acc.values()) >= 2)
    return out


def stratified_bootstrap(a, b, strata, n=10000, seed=0):
    """Mean over strata of mean(b - a), with its 95% percentile interval: prompts resampled with replacement within
    each category (paired), so every resample keeps 20 prompts per category, as the mean of the category rates does."""
    import numpy as np
    d = np.asarray(b, float) - np.asarray(a, float)
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in dict.fromkeys(strata.tolist())]
    point = float(np.mean([d[g].mean() for g in groups]))
    means = np.zeros(n)
    for g in groups:
        means += d[g][rng.integers(0, len(g), size=(n, len(g)))].mean(1)
    means /= len(groups)
    return point, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def step_score(args):
    items = geneval_items()
    category = {it["id"]: it["category"] for it in items}
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    table = dict(geneval=[], mmlu_pro=[])
    for model in MODELS:
        base_v = verdicts(OUT, model, "base", items)
        for arm in ARMS:
            v = verdicts(OUT, model, arm, items)
            if not v:
                continue
            row = dict(model=model, arm=arm, n=len(v), of=len(items), complete=len(v) == len(items),
                       pass_rate={c: mean([x["passed"] for i, x in v.items() if category[i] == c]) for c in CATEGORIES},
                       judge_accepts={j: mean([x["judges"][j] for x in v.values()]) for j in JUDGES},
                       unanimous=mean([len(set(x["judges"].values())) == 1 for x in v.values()]))
            rates = [r for r in row["pass_rate"].values() if r is not None]
            row["mean"] = mean(rates)
            if arm != "base":
                shared = [it["id"] for it in items if it["id"] in v and it["id"] in base_v]
                if shared:
                    a = [base_v[i]["passed"] for i in shared]
                    b = [v[i]["passed"] for i in shared]
                    d, lo, hi = stratified_bootstrap(a, b, [category[i] for i in shared])
                    row.update(paired_n=len(shared), base_mean=mean([mean([base_v[i]["passed"] for i in shared
                                                                          if category[i] == c])
                                                                     for c in CATEGORIES
                                                                     if any(category[i] == c for i in shared)]),
                               diff=d, ci95=[lo, hi], diff_category={})
                    for c in CATEGORIES:
                        ids = [i for i in shared if category[i] == c]
                        if ids:
                            d, lo, hi = bench.bootstrap([base_v[i]["passed"] for i in ids], [v[i]["passed"] for i in ids])
                            row["diff_category"][c] = dict(n=len(ids), diff=d, ci95=[lo, hi])
            table["geneval"].append(row)

        base_rows = {r["id"]: r for r in bench.scored(OUT / model / "base" / "mmlu_pro.jsonl")}
        for arm in ARMS:
            rows = bench.scored(OUT / model / arm / "mmlu_pro.jsonl")
            if not rows:
                continue
            row = dict(model=model, arm=arm, n=len(rows), of=300, complete=len(rows) == 300,
                       accuracy=mean([r["correct"] for r in rows]), reply_accuracy=mean([r["reply_correct"] for r in rows]),
                       unparsed=sum(r["pred"] is None for r in rows))
            if arm != "base":
                shared = [r for r in rows if r["id"] in base_rows]
                if shared:
                    d, lo, hi = bench.bootstrap([base_rows[r["id"]]["correct"] for r in shared], [r["correct"] for r in shared])
                    row.update(paired_n=len(shared), base_accuracy=mean([base_rows[r["id"]]["correct"] for r in shared]),
                               diff=d, ci95=[lo, hi])
                    d, lo, hi = bench.bootstrap([base_rows[r["id"]]["reply_correct"] for r in shared],
                                                [r["reply_correct"] for r in shared])
                    row.update(base_reply_accuracy=mean([base_rows[r["id"]]["reply_correct"] for r in shared]),
                               reply_diff=d, reply_ci95=[lo, hi])
            table["mmlu_pro"].append(row)
    (OUT / "table.json").write_text(json.dumps(table, indent=1))
    (OUT / "summary.json").write_text(json.dumps(summary(table), indent=1))

    pct = lambda x: "    -" if x is None else f"{100 * x:5.1f}"
    ci = lambda r, d="diff", c="ci95": (f"{100 * r[d]:+5.1f} [{100 * r[c][0]:+5.1f}, {100 * r[c][1]:+5.1f}]"
                                        if d in r else "")
    out = ["GenEval prompts with VLM scoring (120 prompts, 20 per category, one picture each, drawing seed 0).",
           "Pass rate in %: a prompt passes when at least two of the three judges (qwen3vl, gemma3, llavaov2) answer 'yes' "
           "to every requirement question.",
           "mean = mean of the six category rates; diff = checkpoint - base in points, 95% paired bootstrap interval "
           "(10,000 resamples of prompts within categories).",
           f"{'model':10s} {'ckpt':5s} {'n':>4s} " + " ".join(f"{c[:6]:>6s}" for c in CATEGORIES)
           + f" {'mean':>6s}  {'diff [95%]':>23s}  judges accept (q/g/l), unanimous"]
    for r in table["geneval"]:
        note = "" if r["complete"] else f"  partial {r['n']}/{r['of']}"
        out.append(f"{r['model']:10s} {r['arm']:5s} {r['n']:4d} " + " ".join(f"{pct(r['pass_rate'][c]):>6s}" for c in CATEGORIES)
                   + f" {pct(r['mean']):>6s}  {ci(r):>23s}  "
                   + "/".join(pct(r["judge_accepts"][j]).strip() for j in JUDGES) + f", {pct(r['unanimous']).strip()}" + note)
    out += ["", "Per-category differences to base (points, 95% paired bootstrap over that category's 20 prompts):",
            f"{'model':10s} {'ckpt':5s} " + " ".join(f"{CATEGORY_NAMES[c]:>24s}" for c in CATEGORIES)]
    for r in table["geneval"]:
        if r.get("diff_category"):
            out.append(f"{r['model']:10s} {r['arm']:5s} " + " ".join(
                f"{ci(r['diff_category'][c]) if c in r['diff_category'] else '':>24s}" for c in CATEGORIES))
    out += ["", "MMLU-Pro (300 questions of scripts/benchmarks.py). Accuracy in %: the option whose letter is the most "
                "likely first reply token; reply = parsed greedy reply (unparsed counted wrong).",
            f"{'model':10s} {'ckpt':5s} {'n':>4s} {'acc':>6s}  {'diff [95%]':>23s}  {'reply':>6s}  {'reply diff [95%]':>23s}"
            f"  unparsed"]
    for r in table["mmlu_pro"]:
        note = "" if r["complete"] else f"  partial {r['n']}/{r['of']}"
        out.append(f"{r['model']:10s} {r['arm']:5s} {r['n']:4d} {pct(r['accuracy'])}  {ci(r):>23s}  "
                   f"{pct(r['reply_accuracy'])}  {ci(r, 'reply_diff', 'reply_ci95'):>23s}  {r['unparsed']:5d}{note}")
    (OUT / "table.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


def verdict(ci95):
    """The rule fixed before evaluation: harmed / improved when the 95% interval lies entirely below / above 0."""
    return "harmed" if ci95[1] < 0 else "improved" if ci95[0] > 0 else "no detectable change"


def summary(table):
    """Supp. Table S19: per model and checkpoint the base value, the checkpoint's value and the change from base with its 95% interval, for GenEval-120 (mean of the six category
    pass rates) and MMLU-Pro-300 (first-token accuracy), in percent and points; with the rule's verdict."""
    out = {}
    for key, measure, value, base_value in (("geneval", "geneval120_mean_pass", "mean", "base_mean"),
                                            ("mmlu_pro", "mmlu_pro300_first_token", "accuracy", "base_accuracy")):
        for r in table[key]:
            e = dict(value=round(100 * r[value], 2) if r[value] is not None else None, n=r["n"], of=r["of"],
                     complete=r["complete"])
            if r["arm"] == "base":
                e["base"] = e["value"]
            elif "diff" in r:
                e.update(base=round(100 * r[base_value], 2), diff=round(100 * r["diff"], 2),
                         ci95=[round(100 * x, 2) for x in r["ci95"]], paired_n=r["paired_n"],
                         verdict=verdict(r["ci95"]) if r["complete"] else "incomplete")
            out.setdefault(r["model"], {}).setdefault(r["arm"], {})[measure] = e
    order = {m: i for i, m in enumerate(MODELS)}
    return {m: {a: out[m][a] for a in ARMS if a in out[m]} for m in sorted(out, key=order.get)}


# ----------------------------------------------------------------------------------------------------------- status

def step_status(args):
    items = geneval_items()
    now = time.time()
    st = dict(adapters={}, sets={}, judged={j: {} for j in JUDGES})
    for model in MODELS:
        for arm in ARMS:
            key = f"{model}.{arm}"
            f = paths.DATA / STORE[model] / f"{arm}_s0.pt"
            st["adapters"][key] = 1e9 if arm == "base" else (now - f.stat().st_mtime if f.exists() else None)
            folder = OUT / model / arm / "geneval"
            have = [(folder / (it["id"] + ".png")).exists() for it in items]
            st["sets"][key] = dict(pictures=sum(have), shards2=[sum(have[0::2]), sum(have[1::2])],
                                   mmlu=len({r["id"] for r in lines(OUT / model / arm / "mmlu_pro.jsonl")}))
            for j in JUDGES:
                done = {(r["id"], r["check"]) for r in lines(OUT / "judges" / j / f"{model}_{arm}.jsonl")}
                st["judged"][j][key] = sum(h and all((it["id"], k) in done for k in range(len(it["checks"])))
                                           for it, h in zip(items, have))
    print(json.dumps(st))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "run", "judge", "score", "status"])
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--benches", default="mmlu_pro,geneval")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--judge", choices=JUDGES)
    ap.add_argument("--limit", type=int, default=0, help="run --tag smoke: only the first n MMLU-Pro questions")
    ap.add_argument("--tag", default="", help="smoke: one prompt per category, into smoke/")
    args = ap.parse_args()
    dict(prep=step_prep, run=step_run, judge=step_judge, score=step_score, status=step_status)[args.step](args)


if __name__ == "__main__":
    main()
