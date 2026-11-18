"""The 300 MMLU-Pro questions of the capability check (Supp. F.3, Table S19) and the shared code that asks and scores
multiple-choice questions; scripts/capability.py imports this file.

prep (CPU) writes the fixed item file $VLENS_OUT/benchmarks/items/mmlu_pro.json: 300 questions of the MMLU-Pro test split
(TIGER-Lab/MMLU-Pro, dataset revision pinned below) drawn with random.Random(0). Every question is asked greedily with
"Answer with the letter of the correct option."; the scored answer is the option whose letter is the most likely first
token of the reply (read by a hook on the text head during the same greedy pass; a tie goes to the parsed reply), and
the parsed free reply is kept as well. HunyuanImage answers with its one-word system prompt (the first token must hold
a letter or digit). Adapters are on for the whole question pass and the reply.

    python scripts/benchmarks.py prep
"""

import argparse
import contextlib
import hashlib
import json
import random
import re
import time

from vlens import paths, results

OUT = results.RESULTS / "benchmarks"
ITEMS = OUT / "items"
IMAGES = paths.DATA / "benchmarks" / "images"   # picture questions only (none in MMLU-Pro)
HUNYUAN_SYSTEM = "You are a helpful assistant. Answer every question with exactly one word. Never explain your answer."
LETTERS = "ABCDEFGHIJ"
MMLU_PRO = dict(repo="TIGER-Lab/MMLU-Pro", revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",
                file="data/test-00000-of-00001.parquet")
#: sha256 of the item file the paper's numbers used (json.dumps(items, indent=1))
MMLU_PRO_ITEMS_SHA256 = "1d4c9aae8b578200c7e7f04d597f97c057afa3b76b968aee9de90362eedee268"


# ------------------------------------------------------------------------------------------------------------ items

def step_prep(args):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    ITEMS.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(hf_hub_download(MMLU_PRO["repo"], MMLU_PRO["file"], repo_type="dataset",
                                         revision=MMLU_PRO["revision"])).to_pylist()
    mmlu = [dict(id=f"mmlu_pro_{r['question_id']}", category=r["category"], question=r["question"],
                 options=list(r["options"]), answer=r["answer"])
            for r in (rows[i] for i in sorted(random.Random(0).sample(range(len(rows)), 300)))]
    assert all(LETTERS[len(it["options"]) - 1] >= it["answer"] for it in mmlu)
    text = json.dumps(mmlu, indent=1)
    (ITEMS / "mmlu_pro.json").write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    print("mmlu_pro", len(mmlu), "items, sha256", sha, "(the paper's items)" if sha == MMLU_PRO_ITEMS_SHA256
          else "(DIFFERS from the paper's items)")


def load_items(bench):
    return json.loads((ITEMS / f"{bench}.json").read_text())


def prompt(bench, it):
    if bench == "pope":
        return f"{it['question']} Answer with yes or no."
    if bench == "mme":   # MME's own questions already end "Please answer yes or no."
        return it["question"]
    options = "\n".join(f"{L}. {o}" for L, o in zip(LETTERS, it["options"]))
    tail = "\nAnswer with the letter of the correct option."
    if bench == "mmlu_pro":
        return f"Question: {it['question']}\nOptions:\n{options}{tail}"
    raise ValueError(bench)


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def parse(bench, reply, it):
    """yes/no: the first "yes" or "no" as a word. Letters: a reply that is the letter (alone, or followed by ".", ")",
    ":" or a space), then "answer is X" / "option X", then the text of exactly one option (the reply equal to it, or
    starting with it). None when nothing matches (counted wrong)."""
    if bench in ("pope", "mme"):
        m = re.search(r"\b(yes|no)\b", reply.lower())
        return m.group(1) if m else None
    letters = LETTERS[:len(it["options"])]
    s = reply.strip()
    if s.strip(" .()*").upper() in letters and len(s.strip(" .()*")) == 1:
        return s.strip(" .()*").upper()
    m = re.match(rf"^\**\(?([{letters}])\)?(?=$|[.,:;)*]|\s*\n)", s)   # "B", "B.", "(B)", "B: ..."; not "A cat"
    if m:
        return m.group(1)
    m = re.search(rf"(?:answer|option)\s*(?:is)?\s*:?\s*\**\(?([{letters}])\)?(?=$|[\s.,:;)*])", s, re.I)
    if m:
        return m.group(1).upper()
    r, opts = _norm(s), [_norm(o) for o in it["options"]]
    for test in (lambda o: r == o, lambda o: re.match(rf"^{re.escape(o)}\b", r)):
        hits = [L for L, o in zip(letters, opts) if o and test(o)]
        if len(hits) == 1 and sum(bool(o) and re.search(rf"\b{re.escape(o)}\b", r) is not None for o in opts) == 1:
            return hits[0]
    return None


# -------------------------------------------------------------------------------------------------------------- run

def pick(score, pred):
    """The option with the highest first-token score; on a tie (bf16 logits, e.g. 21.0 and 21.0) the one the greedy
    reply gave, if it is among the tied, else the first tied option."""
    best = max(score.values())
    tied = [o for o, s in score.items() if s == best]
    return pred if pred in tied else tied[0]


def records(f):
    """The finished items of a record file (a line cut off by a stopped job is dropped and the file rewritten)."""
    if not f.exists():
        return []
    good, bad = [], 0
    for line in f.read_text().splitlines():
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        f.write_text("".join(json.dumps(r) + "\n" for r in good))
    return good


def run_benches(m, model, adapters, out, todo, limit=0):
    """Ask every item of the benchmarks `todo` (the first `limit` of each when set) with the adapters on, one JSON line
    per item in out/<bench>.jsonl, resuming; True when every benchmark is complete."""
    from PIL import Image

    on = adapters.on if adapters is not None else contextlib.nullcontext

    if model == "hunyuan":
        ask = lambda text, image: m.reply(text, HUNYUAN_SYSTEM, image=image, max_new_tokens=8, require_word=True)
    else:
        ask = lambda text, image: (m.answer(text, max_new_tokens=16) if image is None
                                   else m.answer_image(image, text, max_new_tokens=16))
    # The first call of the text head during a reply scores the reply's first token (BAGEL: the start-marker step of
    # generate_text; Show-o2: its first full pass; SenseNova-U1 and HunyuanImage: generate()'s prefill), as in
    # scripts/state_tuning.py margin_with. The scored answer is the option whose token is most likely there.
    head = [mod for name, mod in m.net.named_modules() if name.split(".")[-1] == "lm_head"][-1]
    first = []
    hook = head.register_forward_hook(lambda mod, i, o: first.append(o.detach().reshape(-1, o.shape[-1])[-1].float().cpu())
                                      if not first else None)
    tk = m.tokenizer
    spell = lambda w: sorted({tk.encode(v, add_special_tokens=False)[0] for v in (w, " " + w)})
    option_ids = {L: spell(L) for L in LETTERS}
    option_ids.update(yes=spell("Yes") + spell("yes"), no=spell("No") + spell("no"))
    print("option tokens:", {k: [tk.decode([i]) for i in v] for k, v in option_ids.items()}, flush=True)

    def choose(bench, it, pred):
        opts = ["yes", "no"] if bench in ("pope", "mme") else list(LETTERS[:len(it["options"])])
        score = {o: max(float(first[0][i]) for i in option_ids[o]) for o in opts}
        return pick(score, pred), score

    complete = True
    for bench in todo:
        items = load_items(bench)
        if limit:
            items = items[:limit]
        f = out / f"{bench}.jsonl"
        done = {r["id"] for r in records(f)}
        t0, n = time.time(), 0
        with f.open("a") as fh:
            for it in items:
                if it["id"] in done:
                    continue
                image = Image.open(IMAGES / it["image"]).convert("RGB") if it.get("image") else None
                first.clear()
                with on():
                    reply = ask(prompt(bench, it), image)
                pred = parse(bench, reply, it)
                choice, score = choose(bench, it, pred)
                fh.write(json.dumps(dict(id=it["id"], reply=reply, pred=pred, choice=choice, answer=it["answer"],
                                         correct=choice == it["answer"], reply_correct=pred == it["answer"],
                                         scores=score)) + "\n")
                fh.flush()
                n += 1
                if n <= 3 or n % 50 == 0:
                    print(f"[{time.time() - t0:6.0f}s] {bench} {n}: {reply!r} -> parsed {pred}, first token {choice} "
                          f"(answer {it['answer']})", flush=True)
        got = records(f)
        acc = sum(r["correct"] for r in got) / max(1, len(got))
        print(f"{bench}: {len(got)} of {len(items)} items, accuracy {acc:.3f} (first token), reply accuracy "
              f"{sum(r['reply_correct'] for r in got) / max(1, len(got)):.3f}, unparsed replies "
              f"{sum(r['pred'] is None for r in got)}, reply and first token disagree "
              f"{sum(r['pred'] is not None and r['pred'] != r['choice'] for r in got)}", flush=True)
        complete &= {r["id"] for r in got} >= {it["id"] for it in items}
    hook.remove()
    return complete


# ------------------------------------------------------------------------------------------------------------ score

def bootstrap(a, b, n=10000, seed=0):
    """95% percentile interval of mean(b - a) over items resampled with replacement (paired)."""
    import numpy as np
    d = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def scored(f):
    """Records with the reply parsed again and the first-token answer re-derived from the stored scores by pick()."""
    rows = records(f)
    items = {it["id"]: it for it in load_items(f.stem)} if rows else {}
    for r in rows:   # the reply parsed again (the parser also accepts replies like "answer is:\n\n**F.")
        r["pred"] = parse(f.stem, r["reply"], items[r["id"]])
        r["reply_correct"] = r["pred"] == r["answer"]
        r["choice"] = pick(r["scores"], r["pred"])
        r["correct"] = r["choice"] == r["answer"]
    return rows


def main():
    ap = argparse.ArgumentParser(description="The 300 MMLU-Pro items of the capability check (the run and the scoring "
                                             "are in scripts/capability.py).")
    ap.add_argument("step", choices=["prep"])
    args = ap.parse_args()
    step_prep(args)


if __name__ == "__main__":
    main()
