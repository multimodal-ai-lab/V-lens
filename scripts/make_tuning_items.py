"""Build the texts of the lens-tuning study on pronouns (Sec. 4.6, Supp. F.1, item table in Supp. G) from the noun table
configs/prompts/vsup_nouns.yaml (seed 20261019). Writes configs/prompts/vsup_items.yaml (the released file was made by
this script; rerunning it reproduces it).

Every text names two things of different categories and ends with a verb that selects one of them, followed by "it":
"A dog and a cake were in the kitchen. She ate it." Each pair is used with either thing selected and with both noun
orders, so a sentence never tells which thing is meant by which nouns it contains.

Splits (per category, fixed seed): 8 training nouns, 3 development nouns, 3 test nouns.
- train: 60 pairs of training nouns x 2 referents x 2 orders x 4 sentence forms (960 texts), training verbs.
- dev:   20 pairs of development nouns, training verbs and forms (one form per pair).
- test1: 20 new pairs of training nouns, held-out sentence forms, training verbs ("Phrases" in Table 3).
- test2: 20 new pairs of training nouns, held-out verbs, training forms ("Verbs").
- test3: 20 pairs of held-out nouns (never trained), training verbs and forms ("Nouns").
Test and dev texts: 2 referents x 2 orders, one form per pair (80 texts per group). A "unit" is one pair in one order
and form; it succeeds when both of its texts (the two referents) succeed.

    python scripts/make_tuning_items.py
"""

import argparse
import itertools
import random
from collections import Counter

import yaml

from vlens import prompts

SEED = 20261019
REFERENCE = ' What does "it" refer to in the last sentence? Answer with one word.'  # as scripts/screen_answers.py


def article(x: str) -> str:
    return ("an " if x[0] in "aeiou" else "a ") + x


def pick_pairs(nouns, cat, table, n, rng, taken):
    """n pairs of nouns of different, compatible categories, balancing how often each noun is used."""
    pool = [(x, y) for x, y in itertools.combinations(nouns, 2)
            if cat[x] != cat[y] and cat[y] in table[cat[x]]["pairs_with"] and cat[x] in table[cat[y]]["pairs_with"]
            and frozenset((x, y)) not in taken]
    rng.shuffle(pool)
    used = Counter(x for p in taken for x in p)
    chosen = []
    for _ in range(n):
        x, y = min(pool, key=lambda p: (used[p[0]] + used[p[1]], max(used[p[0]], used[p[1]])))
        pool.remove((x, y))
        chosen.append((x, y))
        used.update((x, y))
        taken.add(frozenset((x, y)))
    return chosen


def texts_for(group, i, pair, forms, verb_of, cfg, cat, rng):
    out = []
    for order in (0, 1):
        x, y = pair if order == 0 else pair[::-1]
        for f in forms:
            place = rng.choice(cfg["places"])
            for ref in pair:
                other = pair[1] if ref == pair[0] else pair[0]
                verb = verb_of(ref, f)
                text = cfg["forms"][f].format(A_x=article(x).capitalize(), a_x=article(x), a_y=article(y), place=place,
                                              verb=verb)
                a, b = (ref, other) if rng.random() < 0.5 else (other, ref)
                out.append(dict(
                    id=f"{group}_{i:02d}_o{order}_f{f}_{ref.replace(' ', '-')}", group=group, pair=f"{group}_{i:02d}",
                    unit=f"{group}_{i:02d}_o{order}_f{f}", order=order, form=f, text=text, truth=ref, alt=other,
                    category=cat[ref], alt_category=cat[other], verb=verb,
                    reads=[dict(name="it", word="it")],
                    questions=dict(
                        name=text + REFERENCE,
                        choice=f'{text} In the last sentence, is "it" the {a} or the {b}? Answer with one word.',
                        yes=f'{text} About "it" in the last sentence: {cfg["categories"][cat[ref]]["question"]} '
                            'Answer yes or no.',
                        no=f'{text} About "it" in the last sentence: {cfg["categories"][cat[other]]["question"]} '
                           'Answer yes or no.')))
    return out


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    cfg = yaml.safe_load((prompts.PROMPTS / "vsup_nouns.yaml").read_text())
    table = cfg["categories"]
    rng = random.Random(SEED)
    cat, split = {}, {}
    for c, spec in table.items():
        nouns = list(spec["nouns"])
        rng.shuffle(nouns)
        for k, x in enumerate(nouns):
            cat[x] = c
            split[x] = "train" if k < 8 else "dev" if k < 11 else "test"
    by = {s: sorted(x for x in cat if split[x] == s) for s in ("train", "dev", "test")}
    taken = set()
    train_pairs = pick_pairs(by["train"], cat, table, 60, rng, taken)
    test1_pairs = pick_pairs(by["train"], cat, table, 20, rng, taken)
    test2_pairs = pick_pairs(by["train"], cat, table, 20, rng, taken)
    dev_pairs = pick_pairs(by["dev"], cat, table, 20, rng, set())
    test3_pairs = pick_pairs(by["test"], cat, table, 20, rng, set())

    train_verb = lambda ref, f: table[cat[ref]]["verbs"][f % 2]
    held_verb = lambda ref, f: table[cat[ref]]["heldout_verb"]
    items = []
    for i, p in enumerate(train_pairs):
        items += texts_for("train", i, p, cfg["train_forms"], train_verb, cfg, cat, rng)
    for group, pairs, forms, verb in (("dev", dev_pairs, cfg["train_forms"], train_verb),
                                      ("test1", test1_pairs, cfg["heldout_forms"], train_verb),
                                      ("test2", test2_pairs, cfg["train_forms"], held_verb),
                                      ("test3", test3_pairs, cfg["train_forms"], train_verb)):
        for i, p in enumerate(pairs):
            items += texts_for(group, i, p, [rng.choice(forms)], verb, cfg, cat, rng)

    # The wrong-picture arm (S): a fixed permutation of the training nouns that sends every category's nouns to the nouns
    # of the next category (so the wrong target is never a sensible reading), drawn until it also avoids each noun's
    # partners in the training texts.
    train_nouns = by["train"]
    others = {x: {it["alt"] for it in items if it["group"] == "train" and it["truth"] == x} for x in train_nouns}
    cats = list(table)
    for attempt in range(100000):
        wrong = {}
        for c, d in zip(cats, cats[1:] + cats[:1]):
            src, dst = [x for x in train_nouns if cat[x] == c], [x for x in train_nouns if cat[x] == d]
            rng.shuffle(dst)
            wrong.update(zip(src, dst))
        if all(wrong[x] not in others[x] for x in train_nouns):
            break
    else:
        raise RuntimeError("no permutation found")
    out = dict(seed=SEED, split=split, category=cat, wrong=wrong,
               pairs={g: [list(p) for p in ps] for g, ps in (("train", train_pairs), ("dev", dev_pairs),
                                                             ("test1", test1_pairs), ("test2", test2_pairs),
                                                             ("test3", test3_pairs))},
               neutral=cfg["neutral"], general_qa=cfg["general_qa"], reads=items)
    (prompts.PROMPTS / "vsup_items.yaml").write_text(yaml.safe_dump(out, sort_keys=False, width=140))
    print(Counter(it["group"] for it in items), "| permutation found after", attempt + 1, "draws")
    print({s: len(v) for s, v in by.items()}, "| train nouns used:", Counter(x for p in train_pairs for x in p).most_common()[-3:])
    for g in ("train", "test1", "test2", "test3"):
        print(next(it["text"] for it in items if it["group"] == g))


if __name__ == "__main__":
    main()
