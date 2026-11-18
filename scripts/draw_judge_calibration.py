"""Draw the labeled calibration pictures of the main-subject judge question (paper Sec. 4.1, Supp. B, Table S5).

Each request of configs/prompts/subject_calibration.yaml (one main subject, two main subjects, a main subject with a
distracting background object, or neither candidate) is drawn as the model normally draws it, seeds 0 and 1. SenseNova-U1
draws the request without its "a photo of " prefix (its system message already asks for an image). The records have the
format of scripts/draw_cells.py (kind "reads"), so scripts/run_judge.py --set cells:subject_calibration judges them, and
scripts/score_judge_calibration.py scores the panel.

    python scripts/draw_judge_calibration.py --model bagel        # also show_o2, sensenova
"""

import argparse
import json

import yaml

from vlens import prompts, results
from vlens.models import load


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="bagel")
    args = ap.parse_args()
    rows = yaml.safe_load((prompts.PROMPTS / "subject_calibration.yaml").read_text())["reads"]
    out = results.folder("subject_calibration") / args.model
    out.mkdir(parents=True, exist_ok=True)
    m = load(args.model)
    recs = []
    for i, r in enumerate(rows):
        request = r["request"] if args.model != "sensenova" else r["request"].removeprefix("a photo of ")
        reading = m.read_prompt(request)
        for seed in (0, 1):
            name = f"subject_calibration_{i:03d}_x_L0_full_s{seed}.png"
            if not (out / name).exists():
                m.to_picture(m.draw(reading, seed=seed)[0]).save(out / name)
            recs.append(dict(file=f"{args.model}/{name}", model=args.model, items="subject_calibration", kind="reads",
                             item=i, text=request, position="x", word="x", token=None, layer=0, way="full", seed=seed,
                             truth=r["truth"]))
        print(f"{r['id']} done", flush=True)
    (out / "records_subject_calibration.json").write_text(json.dumps(recs, indent=1))


if __name__ == "__main__":
    main()
