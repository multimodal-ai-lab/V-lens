"""Judges that say what a picture shows: multimodal models that look at the picture and answer questions.

Every judge sees only the picture, never the text it was decoded from, the model or the layer, and answers two ways:
- open: "What is the main thing shown in this picture?" in at most three words;
- choice: a lettered list of names in an order shuffled per picture, plus "none of these"; the answer is read from the
  judge's own probabilities for the letters (one forward pass, no sampling), so it is exact and gives a confidence.

The panel (three different makers):
    qwen3vl   Qwen/Qwen3-VL-30B-A3B-Instruct                  one 80 GB GPU
    gemma3    google/gemma-3-27b-it                           one 80 GB GPU
    llavaov2  lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct  (its own remote code)
Qwen3-VL and Gemma-3 load with transformers 4.57 (the SenseNova-U1 environment, requirements/sensenova.txt); LLaVA-
OneVision-2 needs transformers 5 (requirements/judges.txt). In bf16, Qwen3-VL needs about 61 GB and Gemma-3 about 54 GB;
on smaller GPUs device_map="auto" spreads them over two.
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from vlens import paths

PANEL = {
    "qwen3vl": dict(repo="Qwen/Qwen3-VL-30B-A3B-Instruct", remote_code=False),
    "gemma3": dict(repo="google/gemma-3-27b-it", remote_code=False),
    "llavaov2": dict(repo="lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct", remote_code=True),
}
OPEN_QUESTION = "What is the main thing shown in this picture? Answer with one to three words and nothing else."
NONE = "none of these"


def answer_files(folder: Path, judge: str, set_name: str) -> list[Path]:
    """The answer files one judge wrote on one picture set (scripts/run_judge.py names them
    answers_<judge>_<set>[_<model>][_<i>of<n>][_<HHMMSS>].json). Matched exactly, because a plain prefix match would also
    take a set whose name starts the same way."""
    pattern = re.compile(rf"answers_{re.escape(judge)}_{re.escape(set_name.replace(':', '_'))}"
                         r"(_(bagel|show_o2|sensenova|liquid|janus_pro|hunyuan))?(_\d+of\d+)?(_\d{6})?\.json")
    return sorted(f for f in folder.glob(f"answers_{judge}_*.json") if pattern.fullmatch(f.name))


def choice_question(options: list[str]) -> str:
    lines = [f"{letter}) {name}" for letter, name in zip(string.ascii_uppercase, options)]
    return ("Which of these does the picture show? If none of them, choose the last option.\n" + "\n".join(lines)
            + "\nAnswer with the letter only.")


BOTH, NEITHER = "both of these", "neither of these"


def pair_question(options: list[str]) -> str:
    """An earlier two-thing question (two things, both, or neither); the paper uses subject_question."""
    lines = [f"{letter}) {name}" for letter, name in zip(string.ascii_uppercase, options)]
    return ("Which of these is visibly depicted in this picture? Written words or letters in the picture do not count.\n"
            + "\n".join(lines) + "\nAnswer with the letter only.")


def pair_options(target: str, other: str, key: str) -> list[str]:
    """The two things in an order fixed by `key` (the picture's id), then "both of these", then "neither of these"."""
    order = [target, other]
    random.Random(key).shuffle(order)
    return order + [BOTH, NEITHER]


SUBJECT_BOTH, SUBJECT_NEITHER = "both, as main subjects", "neither, as a main subject"


def subject_question(options: list[str]) -> str:
    """The main-subject question of the paper (Sec. 4.1, Supp. B): which candidate is a MAIN subject; background objects
    and written words do not count."""
    lines = [f"{letter}) {name}" for letter, name in zip(string.ascii_uppercase, options)]
    return ("Which of these is depicted as a main subject of the picture? Ignore incidental background objects and "
            "written words.\n" + "\n".join(lines) + "\nAnswer with the letter only.")


def subject_options(target: str, other: str, key: str) -> list[str]:
    """The two things in an order fixed by `key`, then "both, as main subjects", then "neither, as a main subject"."""
    order = [target, other]
    random.Random(key).shuffle(order)
    return order + [SUBJECT_BOTH, SUBJECT_NEITHER]


def shuffled_options(names: list[str], key: str) -> list[str]:
    """The names in an order fixed by `key` (the picture's id), with "none of these" always last."""
    order = list(names)
    random.Random(key).shuffle(order)
    return order + [NONE]


@dataclass
class Choice:
    options: list[str]
    probs: list[float]  # renormalised over the option letters

    @property
    def chosen(self) -> str:
        return self.options[max(range(len(self.probs)), key=self.probs.__getitem__)]


class Judge:
    def __init__(self, name: str):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        spec = PANEL[name]
        path = paths.snapshot(spec["repo"])
        self.name = name
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=spec["remote_code"])
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, trust_remote_code=spec["remote_code"], dtype=torch.bfloat16, device_map="auto").eval()
        self.device = self.model.device  # inputs go where the first layers are
        self.tokenizer = self.processor.tokenizer

    def _inputs(self, image: Image.Image, question: str):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image.convert("RGB")], return_tensors="pt", padding=True)
        return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    @torch.no_grad()
    def open(self, image: Image.Image, max_new_tokens: int = 10) -> str:
        inputs = self._inputs(image, OPEN_QUESTION)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    @torch.no_grad()
    def choose(self, image: Image.Image, options: list[str], question=choice_question) -> Choice:
        inputs = self._inputs(image, question(options))
        logits = self.model(**inputs).logits[0, -1].float()
        letters = [self.tokenizer.encode(letter, add_special_tokens=False)[0]
                   for letter in string.ascii_uppercase[:len(options)]]
        probs = torch.softmax(logits[letters], dim=0).tolist()
        return Choice(options=options, probs=probs)
