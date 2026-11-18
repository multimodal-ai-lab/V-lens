"""Screen items by each model's own text answers, before any picture is drawn; also the answer-matching helpers
(correct, ALIASES, NUMBERS) that scripts/lens_tuning.py uses to score its text answers.

For every text with an `answer` in the items file (kind "reads"), the model answers greedily. Questions get a
short-answer instruction; reference texts get 'What does "it" refer to in the last sentence? Answer with one word.' and
must name the target. A reply is correct when it contains the expected answer as a word (a number also as a word,
"yes"/"no" as the first word). A pair is kept for a model only when both of its texts are answered correctly. Writes
$VLENS_OUT/confirm_cells/screen_<items>_<model>[_name].json.

    python scripts/screen_answers.py --model bagel --items followup_matched --mode name
"""

import argparse
import json
import re

import yaml

from vlens import prompts, results
from vlens.models import load

SHORT = " Answer with just the answer, a number or one word, and nothing else."
REFERENCE = ' What does "it" refer to in the last sentence? Answer with one word.'
NAME = 'Question: "{text}" Which animal or object is this question about? Answer with the name only.'
#: names that count besides the target's own when the model names a target (fixed before any picture was drawn).
#: "chicken" (rooster) and "peafowl" (peacock) are less specific but accepted; porcupine (hedgehog), crane or heron
#: (flamingo), alligator (crocodile), mitten (glove) and generic words (canopy, screen, blade, portrait, fruit) are not.
ALIASES = {"rooster": ["cock", "cockerel", "male chicken", "chicken"],
           "peacock": ["peafowl", "Indian peafowl", "Indian peacock", "male peafowl"],
           "bicycle": ["bike"], "smartphone": ["phone", "mobile phone", "cell phone", "cellphone"],
           "umbrella": ["brolly"], "refrigerator": ["fridge"], "violin": ["fiddle"],
           "sailboat": ["sailing boat", "sailing vessel", "sloop", "ketch"], "fan": ["electric fan", "desk fan"],
           "bee": ["honeybee", "honey bee"], "cactus": ["prickly pear", "prickly pear cactus"],
           "piano": ["grand piano", "upright piano"], "racket": ["racquet", "tennis racket", "tennis racquet"],
           "pyramid": ["Egyptian pyramid", "Great Pyramid", "Great Pyramid of Giza"], "otter": ["river otter"],
           "platypus": ["duck-billed platypus"], "lamp": ["table lamp", "bedside lamp"], "toaster": ["pop-up toaster"],
           "baseball bat": ["bat"], "bat": ["flying mammal"],
           # for the context-test items
           "rhinoceros": ["rhino"], "remote control": ["remote", "TV remote"], "airplane": ["plane", "aeroplane",
           "aircraft", "jet"], "helicopter": ["chopper"], "paintbrush": ["paint brush", "brush"], "backpack":
           ["rucksack", "school bag", "schoolbag"], "bus": ["school bus"], "watering can": ["watering-can"],
           "teapot": ["kettle"], "jacket": ["windbreaker", "parka"], "newspaper": ["paper"], "lemon": ["lime"],
           "tennis racket": ["racket", "racquet", "tennis racquet"], "crocodile": ["alligator", "croc"],
           "laptop": ["notebook computer", "laptop computer"], "whale": ["blue whale", "humpback"]}
NUMBERS = {str(i): w for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                                            "nine", "ten", "eleven", "twelve"])}


def correct(reply: str, answer) -> bool:
    """`answer` is one accepted answer or a list of them (spellings, synonyms); plurals count."""
    if isinstance(answer, list):
        return any(correct(reply, a) for a in answer)
    r = reply.lower().strip()
    a = str(answer).lower().strip()
    if a in ("yes", "no"):
        return re.match(rf"\W*{a}\b", r) is not None
    words = [a] + ([NUMBERS[a]] if a in NUMBERS else [])
    return any(re.search(rf"(?<![\w-]){re.escape(w)}(?:s|es)?(?![\w-])", r) for w in words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bagel")
    ap.add_argument("--items", required=True)
    ap.add_argument("--mode", default="answer", choices=["answer", "name"],
                    help="answer: the question itself (reference texts: what 'it' refers to); name: which thing a "
                         "matched question is about (the eligibility screen; reference texts are asked as in answer)")
    args = ap.parse_args()
    items = yaml.safe_load((prompts.PROMPTS / f"{args.items}.yaml").read_text())["reads"]
    m = load(args.model)
    rows = []
    for i, it in enumerate(items):
        if it.get("answer") is None:
            continue
        reference = it.get("family") == "reference"
        if reference:
            reply = m.answer(it["text"] + REFERENCE, max_new_tokens=40)
            ok = correct(reply, [it["truth"]] + ALIASES.get(it["truth"], []))
        elif args.mode == "name":
            reply = m.answer(NAME.format(text=it["text"]), max_new_tokens=40)
            ok = correct(reply, [it["truth"]] + ALIASES.get(it["truth"], []))
        else:
            reply = m.answer(it["text"] + SHORT, max_new_tokens=40)
            ok = correct(reply, it["answer"])
        rows.append(dict(item=i, id=it["id"], pair=it.get("pair"), role=it.get("role"), truth=it["truth"],
                         answer=it["truth"] if reference or args.mode == "name" else it["answer"], reply=reply,
                         correct=ok))
        print(f"{it['id']:>24}: {reply!r} ({'right' if ok else 'WRONG'}, expected {rows[-1]['answer']!r})", flush=True)
    pairs = {}
    for r in rows:
        pairs.setdefault(r["pair"], []).append(r["correct"])
    kept = sorted(p for p, v in pairs.items() if len(v) == 2 and all(v))
    out = results.folder("confirm_cells") / f"screen_{args.items}_{args.model}{'_name' if args.mode == 'name' else ''}.json"
    out.write_text(json.dumps(dict(model=args.model, items=args.items, kept_pairs=kept, n_pairs=len(pairs), rows=rows),
                              indent=1))
    print(f"kept {len(kept)} of {len(pairs)} pairs -> {out}")


if __name__ == "__main__":
    main()
