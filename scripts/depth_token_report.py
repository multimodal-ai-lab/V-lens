"""Summaries of the every-token depth scores (Sec. 4.3, Fig. 4, Supp. D: Table S11): per model and layer, the describing
word, the best later token and its parts (question words, instruction, chat format, the token before the answer, the
answer token), the peak layers, which kinds of tokens show the object, and what the answer token's pictures show.
CPU only; reads scores.json and answer_token_pictures.json of scripts/depth_through_answer.py score.

    python scripts/depth_token_report.py [--models bagel,show_o2,sensenova,hunyuan]
"""
import argparse
import json
from collections import Counter, defaultdict

from vlens import results

O = results.RESULTS / "alltokens_answer"
R = {"bagel": 8, "show_o2": 27, "sensenova": 24, "hunyuan": 9}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", default="bagel,show_o2,sensenova,hunyuan")
    args = ap.parse_args()
    s = json.loads((O / "scores.json").read_text())
    pics = json.loads((O / "answer_token_pictures.json").read_text())
    for m in args.models.split(","):
        if m not in s:
            continue
        r = s[m]
        c, n = r["curves"], r["counted"]
        L = r["layers"]
        print(f"\n== {m}: counted {n}, cells waiting {r['pending_cells']}")
        print("layer " + " ".join(f"{l:>3}" for l in L))
        for name in ("desc", "best_later_old", "best_later", "best_later_no_instruction", "best_later_to_preanswer",
                     "best_question", "best_instruction", "best_chat", "preanswer", "answer_token"):
            row = r["desc"] if name == "desc" else c[name]
            print(f"{name[:26]:26s} " + " ".join(f"{row[str(l)] if str(l) in row else row[l]:>3}" for l in L))
        old, new = c["best_later_old"], c["best_later"]
        g = lambda d, l: d[str(l)] if str(l) in d else d[l]  # noqa: E731
        peak_old = max(g(old, l) for l in L)
        peak_new = max(g(new, l) for l in L)
        print(f"peak question-only {peak_old}/{n} at {[l for l in L if g(old, l) == peak_old]}; through the answer "
              f"{peak_new}/{n} at {[l for l in L if g(new, l) == peak_new]}; at R={R[m]}: {g(old, R[m])} and {g(new, R[m])}")
        # which kinds win: question-layer cells by kind, and questions found only by the positions after the question
        kinds = Counter()
        for l in L:
            for k, v in r["wins"][str(l) if str(l) in r["wins"] else l].items():
                kinds[k] += v
        print("question-layer hits by kind (all layers):", dict(kinds.most_common()))
        toks = Counter()
        for l in L:
            for q, ts in r["tokens_showing"][str(l) if str(l) in r["tokens_showing"] else l].items():
                for t in ts:
                    toks[t] += 1
        print("tokens showing it most often:", toks.most_common(10))
        sole = {l: v for l, v in r["sole_new"].items() if v}
        print("questions only the positions after the question find:", sole)
        print("answer token pictures, judges' majority:", r["answer_token_choices"])
        by = defaultdict(Counter)
        for p in pics:
            if p["file"].startswith(m + "/"):
                by[p["verdict"]][p["layer"]] += 1
        for v, cnt in by.items():
            print(f"  {v}: by layer {dict(sorted(cnt.items()))}")


if __name__ == "__main__":
    main()
