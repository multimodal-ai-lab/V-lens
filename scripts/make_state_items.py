"""Build configs/prompts/state_questions_v2.yaml, the second set of state stories (Sec. 4.7), from the first set
(state_questions.yaml). Per scenario four stories, of which the paper uses the first two:
  happened            the target changes                                  (truth: outcome)
  cancelled           the target does not change                          (truth: start)
  distract_cancelled  the target does not change, a nearby object does    (truth: start)
  distract_happened   the target changes, the nearby object does not      (truth: outcome)
The nearby object is the next scenario of the same family. The question names the target ("Is the slice of bread now
white or brown?"), and every story is asked with the two options in both orders, so a model that favours one position
cannot look right. The released yaml was made by this script.

    python scripts/make_state_items.py
"""

import argparse
from collections import defaultdict

import yaml

from vlens import prompts

# the question with the target named, so the distractor stories stay unambiguous
NAMED = {
    "ice": "Is the ice cube now solid or liquid?", "freeze": "Is the water in the glass now liquid or solid?",
    "butter": "Is the butter now solid or liquid?", "chocolate": "Is the chocolate bar now hard or soft?",
    "snowman": "Is the snowman still standing or has it disappeared?", "candle": "Is the candle now tall or short?",
    "marshmallow": "Is the marshmallow now white or brown?", "toast": "Is the slice of bread now white or brown?",
    "popcorn": "Are the corn kernels now small or big?", "pancake": "Is the pancake batter now liquid or solid?",
    "egg_fry": "Is the egg white now clear or white?", "steak": "Is the steak's outside now red or brown?",
    "seed": "Is there now a plant or no plant where the seed is?", "tadpole": "Does the tadpole now have legs or no legs?",
    "caterpillar": "Does the caterpillar now have wings or no wings?", "banana": "Is the banana now green or brown?",
    "puppy": "Is the puppy now small or big?", "hen_egg": "Can what is in the hen's egg now walk or not walk?",
    "autumn": "Are the oak tree's leaves green or orange?", "egg_drop": "Is the raw egg now whole or broken?",
    "glass": "Is the drinking glass now whole or in pieces?", "balloon": "Is the balloon now round or flat?",
    "paper": "Is the sheet of paper now white or black?", "log": "Is the log now one piece or several pieces?",
    "apple_eat": "Is the red apple now round or thin?", "pencil": "Is the pencil now long or short?",
    "paint_green": "Is the blue paint now blue or green?", "paint_pink": "Is the red paint now red or pink?",
    "shirt": "Is the white shirt now white or pink?", "nail": "Is the iron nail now grey or orange?",
    "grapes": "Are the grapes now smooth or wrinkled?", "sunburn": "Is the man's skin now pale or red?",
    "puddle": "Is the road with the puddle now wet or dry?", "car_snow": "Is the red car now red or white?",
    "pumpkin": "Does the pumpkin now have a face or no face?", "sandcastle": "Is the sandcastle still standing or washed away?",
    "lake": "Is the lake's surface now water or ice?", "flower": "Is the flower now fresh or wilted?",
    "bath": "Are the girl's clothes now white or brown?", "ice_cream": "Is the ice cream now dripping or firm?",
}


def main():
    argparse.ArgumentParser(description="Build configs/prompts/state_questions_v2.yaml.").parse_args()
    c = yaml.safe_load((prompts.PROMPTS / "state_questions.yaml").read_text())
    fam = defaultdict(list)
    for it in c["items"]:
        fam[it["family"]].append(it)
    items = []
    for f, its in fam.items():
        for k, it in enumerate(its):
            other = its[(k + 1) % len(its)]
            assert it["id"] in NAMED, it["id"]
            items.append(dict(
                id=it["id"], family=f, other=other["id"], options=it["options"], answer=it["answer"],
                outcome=it["outcome"], start=it["start"], question=NAMED[it["id"]],
                stories=dict(happened=it["happened"], cancelled=it["cancelled"],
                             distract_cancelled=f"{it['cancelled']} Nearby, {other['happened'][0].lower()}{other['happened'][1:]}",
                             distract_happened=f"{it['happened']} Nearby, {other['cancelled'][0].lower()}{other['cancelled'][1:]}")))
    out = dict(template=c["template"], truth={"happened": "outcome", "cancelled": "start", "distract_cancelled": "start",
                                              "distract_happened": "outcome"}, items=items)
    head = ("# State questions, second set (built by scripts/make_state_items.py from state_questions.yaml; do not edit by\n"
            "# hand). Four stories per scenario, the question names the target, options asked in both orders.\n")
    (prompts.PROMPTS / "state_questions_v2.yaml").write_text(head + yaml.safe_dump(out, sort_keys=False, width=200))
    print(len(items), "scenarios,", 4 * 2 * len(items), "texts")


if __name__ == "__main__":
    main()
