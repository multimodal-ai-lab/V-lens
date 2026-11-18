"""State stories: the items' texts and the answer helpers used by scripts/state_tuning.py (Sec. 4.7), plus two
text-only screens of the stories.

Items: configs/prompts/state_questions.yaml (the first set: 40 scenarios, each in two versions, the change happened or
was cancelled, the question about "it") and configs/prompts/state_questions_v2.yaml (the second set, built by
scripts/make_state_items.py: the question names the thing, both option orders).

Steps (each writes into $VLENS_OUT/state_questions/[v2/]<model>/):
  answers  the greedy text answer to "<story> <question> Answer with one word: <a> or <b>." (HunyuanImage in its direct
           mode, drawing and reasoning starts forbidden). Right when the first option word in the reply is the right one.
  scores   the log-probabilities of both options as the reply's first token (scores_text.json); a story's
           position-free margin is the mean over its option orders of log p(right) - log p(wrong). HunyuanImage's
           training scenarios (TRAIN in scripts/state_tuning.py) were chosen from this screen on the first set.

    python scripts/state_questions.py scores --model hunyuan
"""

import argparse
import json
import re
import time

import yaml

from vlens import prompts, results

OUT = results.RESULTS / "state_questions"
SET = "v1"   # --set v2: configs/prompts/state_questions_v2.yaml (scripts/make_state_items.py), results in .../v2/<model>
VERSIONS = ("happened", "cancelled")
# HunyuanImage answers in one word with this system prompt; its default system prompt makes it reason in plain text and
# not answer within a few tokens
HUNYUAN_SYSTEM = "You are a helpful assistant. Answer every question with exactly one word. Never explain your answer."


def cfg():
    name = "state_questions.yaml" if SET == "v1" else f"state_questions_{SET}.yaml"
    return yaml.safe_load((prompts.PROMPTS / name).read_text())


def texts(c):
    """(item, version, story, full question text, right answer, right picture, other picture)."""
    if "stories" in c["items"][0]:   # v2: four stories, the target named, both option orders
        for it in c["items"]:
            for v, story in it["stories"].items():
                k = 0 if c["truth"][v] == "outcome" else 1
                truth, other = (it["outcome"], it["start"]) if k == 0 else (it["start"], it["outcome"])
                for order, (a, b) in (("ab", it["options"]), ("ba", it["options"][::-1])):
                    q = c["template"].format(story=story, question=it["question"], a=a, b=b)
                    yield it, f"{v}_{order}", story, q, it["answer"][k], truth, other
        return
    for it in c["items"]:
        for k, v in enumerate(VERSIONS):
            q = c["template"].format(story=it[v], question=it["question"], a=it["options"][0], b=it["options"][1])
            truth, other = (it["outcome"], it["start"]) if v == "happened" else (it["start"], it["outcome"])
            yield it, v, it[v], q, it["answer"][k], truth, other


def first_option(reply, options):
    hits = [(m.start(), o) for o in options for m in re.finditer(rf"\b{re.escape(o)}\b", reply.lower())]
    return min(hits)[1] if hits else None


def marks(reading, story, question):
    """Token positions: the story's last full stop, the question's subject, the question mark."""
    strs, full, starts = reading.token_strs, "", []
    for s in strs:
        starts.append(len(full))
        full += s
    s0 = full.rfind(story)
    q0 = full.find(question, s0 + len(story))
    def at(char_index):   # the token covering a character
        return max(p for p, st in enumerate(starts) if st <= char_index)
    out = {"story_end": at(s0 + len(story) - 1), "q_mark": at(q0 + len(question) - 1)}
    words = list(re.finditer(r"\S+", question))
    k = next((i for i, w in enumerate(words) if w.group().lower() in ("now", "still") and i > 1), 2)
    subj = words[k - 1] if k > 1 else words[1]
    out["q_subject"] = at(q0 + subj.end() - 1)
    return out


def step_answers(args, m):
    c, out = cfg(), OUT / args.model
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for it, v, story, text, right, truth, other in texts(c):
        t0 = time.time()
        if args.model == "hunyuan":
            from transformers import LogitsProcessorList, SuppressTokensLogitsProcessor
            tk = m.net._tokenizer
            banned = [tk.convert_tokens_to_ids(tk.boi_token), tk.convert_tokens_to_ids(tk.think_token)]
            inputs = m.net.prepare_model_inputs(prompt=text, mode="gen_text", bot_task="auto", max_new_tokens=8,
                                                system_prompt=HUNYUAN_SYSTEM)
            reply = m.net.generate(**inputs, decode_text=True, do_sample=False,
                                   logits_processor=LogitsProcessorList([SuppressTokensLogitsProcessor(banned, device=m.device)]))
            reply = reply[0] if isinstance(reply, list) else reply
            for t in ("<answer>", "</answer>", "<|endoftext|>"):
                reply = reply.replace(t, "")
        else:
            reply = m.answer(text, max_new_tokens=12)
        got = first_option(reply, it["options"])
        rows.append(dict(id=it["id"], family=it["family"], version=v, right_answer=right, answer=got, right=got == right,
                         reply=reply.strip()[:200], seconds=round(time.time() - t0, 1)))
        print(it["id"], v, right, "->", got, flush=True)
    (out / "answers.json").write_text(json.dumps(rows, indent=1))


def option_logprobs(m, model, text, options):
    """Log-probability of each option word as the first token of the reply (the best of its spellings: with or without a
    leading space, lower or capitalised), read by a hook on the output head during a one-token answer. Position-free
    answers come from averaging over both option orders (the models favour one option position)."""
    import torch
    head = [mod for name, mod in m.net.named_modules() if name.split(".")[-1] == "lm_head"][-1]
    got = []
    h = head.register_forward_hook(lambda mod, i, o: got.append(o.detach()) if not got else None)
    try:
        if model == "hunyuan":
            inputs = m.net.prepare_model_inputs(prompt=text, mode="gen_text", bot_task="auto", max_new_tokens=1,
                                                system_prompt=HUNYUAN_SYSTEM)
            m.net.generate(**inputs, decode_text=False, do_sample=False)
        else:
            m.answer(text, max_new_tokens=1)
    finally:
        h.remove()
    logits = got[0].float().reshape(-1, got[0].shape[-1])[-1]
    lp = torch.log_softmax(logits, -1)
    tok = getattr(m, "tokenizer", None) or m.net._tokenizer
    out = {}
    for o in options:
        ids = {tok.encode(v, add_special_tokens=False)[0] for v in (o, " " + o, o.capitalize(), " " + o.capitalize())}
        out[o] = max(float(lp[i]) for i in ids)
    return out


def step_scores(args, m):
    """scores.json: per text, the log-probabilities of both options; the position-free score of a story is the mean over
    its option orders of log p(right) - log p(wrong)."""
    c, out = cfg(), OUT / args.model
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for it, v, story, text, right, truth, other in texts(c):
        lp = option_logprobs(m, args.model, text, it["options"])
        wrong = [o for o in it["options"] if o != right][0]
        rows.append(dict(id=it["id"], family=it["family"], version=v, right_answer=right, logp=lp,
                         margin=lp[right] - lp[wrong]))
        print(it["id"], v, round(rows[-1]["margin"], 2), flush=True)
    (out / "scores_text.json").write_text(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["answers", "scores"])
    ap.add_argument("--model", default="bagel")
    ap.add_argument("--set", default="v1", choices=["v1", "v2"])
    args = ap.parse_args()
    global SET, OUT
    SET = args.set
    if SET != "v1":
        OUT = OUT / SET
    from vlens.models import load
    m = load(args.model)
    {"answers": step_answers, "scores": step_scores}[args.step](args, m)


if __name__ == "__main__":
    main()
