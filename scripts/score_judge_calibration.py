"""Score the judge calibration (paper Sec. 4.1, Supp. B, Table S5).

On the labeled pictures of scripts/draw_judge_calibration.py (configs/prompts/subject_calibration.yaml), the share of
pictures where the panel majority (at least two of the three judges) gives the right answer, overall and per kind
(background object, two main subjects, one main subject, neither candidate). Scored for the main-subject question
(run_judge.py --subject, the question of the paper) and, when its answers exist, for the simpler two-thing question
(run_judge.py --pair) as a comparison. The thresholds fixed before any experiment are 0.90 overall and 0.80 on the
background cases. Writes $VLENS_OUT/subject_calibration/scores.json.

    python scripts/score_judge_calibration.py
"""

import argparse
import json
import re
from collections import Counter, defaultdict

import yaml

from vlens import prompts, results
from vlens.judges import BOTH, NEITHER, SUBJECT_BOTH, SUBJECT_NEITHER, answer_files

J = results.RESULTS / "judges"
KIND = {"bg": "background", "two": "two subjects", "one": "single", "none": "neither"}


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    rows = yaml.safe_load((prompts.PROMPTS / "subject_calibration.yaml").read_text())["reads"]
    out = {}
    for mode, both, neither in (("subject", SUBJECT_BOTH, SUBJECT_NEITHER), ("pair", BOTH, NEITHER)):
        votes = defaultdict(dict)
        for j in ("qwen3vl", "gemma3", "llavaov2"):
            for f in answer_files(J, j, f"cells:subject_calibration_{mode}"):
                for a in json.loads(f.read_text()):
                    votes[a["file"]][j] = a["chosen"]
        per = defaultdict(lambda: [0, 0])
        wrong = []
        for file, v in votes.items():
            if len(v) < 3:
                continue
            item = rows[int(re.search(r"subject_calibration_(\d+)_", file)[1])]
            right = {"truth": item["truth"], "both": both, "neither": neither}[item["expect"]]
            top, n = Counter(v.values()).most_common(1)[0]
            ok = n >= 2 and top == right
            kind = KIND[item["id"].split("_")[0]]
            per[kind][0] += ok
            per[kind][1] += 1
            per["all"][0] += ok
            per["all"][1] += 1
            if not ok:
                wrong.append((file, item["id"], dict(v)))
        out[mode] = {k: dict(right=a, n=b, share=a / b) for k, (a, b) in per.items()}
        print(mode, {k: f"{a}/{b}" for k, (a, b) in per.items()})
        for w in wrong[:12]:
            print("   wrong:", w)
    passed = out["subject"]["all"]["share"] >= 0.90 and out["subject"]["background"]["share"] >= 0.80
    out["main_subject_question_passes"] = passed
    print("main-subject question passes its calibration:", passed)
    (results.folder("subject_calibration") / "scores.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
