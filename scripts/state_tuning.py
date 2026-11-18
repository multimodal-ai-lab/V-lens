"""Lens tuning on state stories (Sec. 4.7, Table 4, the states figure, Supp. Tables S20 and S21): supervise the
question subject's state, carried by the V-lens into the drawing request, with the model's own pictures of the right
(or, as a control, the wrong) state, and measure the text answers; plus the state interchange test.

Items: configs/prompts/state_questions.yaml (first set, 40 scenarios with questions about "it"),
state_questions_v2.yaml (second set, --set v2: the question names the thing; scripts/make_state_items.py) and
state_questions_v5.yaml (--set v5, Table 4: the second set's 20 scenarios and 20 more, 80 stories). Each story is
asked with both option orders. Training scenarios: on --set v5 every scenario; on --set v2 TRAIN below (20 scenarios,
both models); on the first set Show-o2 trains on every scenario except TEST, 28 scenarios = 56 stories.
The supervised state is the question's subject entering the insertion layer R, read in the format the model answers
in. Target pictures: the frozen model's own drawing of "a photo of <state>" (four drawing seeds for Show-o2, two for
HunyuanImage). Loss: the model's denoising loss with noise level uniform on [0.4, 1], no guidance, through the frozen
drawing request; no answer labels. Adapters (vlens/lora.py, rank 8, scaling 16): Show-o2 layers 0-26 (attention and
feed-forward projections), HunyuanImage layers 0-8 (attention and the shared MLP; texts read with its one-word system
prompt). AdamW, learning rate 1e-4, batches of 8 texts.
Arms: 0 none; V right-state pictures; S wrong-state pictures (the other state of the same scenario); T text (the right
option word as the reply's first token; Show-o2 only, not in the paper). --tag names a run (V600 = 600 updates).

Steps:
  targets     the frozen model draws "a photo of <state>" for every state (--target-seeds), latents kept; --set v5 draws
              only the states that the earlier targets file (latents.pt) lacks, into latents_v5.pt
  train       one arm, one training seed -> $VLENS_DATA/state_train/<model>/<arm><tag>[_v2|_v5]_s<seed>.pt
  eval        with an arm's adapters: margins.json (log p(right) - log p(wrong) of the reply's first token, every text and
              order; the states figure shows 1 / (1 + exp(-m)) with m averaged over both orders) and the V-lens picture
              of the question's subject at R (--picture-seeds) with records for judging
  replies     replies.json: the greedy one-word reply to every text with an arm's adapters (Table 4's "correct text")
  judge       three judges (main subject: the right state, the other state, both, neither)
  swap        the interchange test: the subject's states from the trained model put into the untrained one while it
              answers (--swap-layers all: every layer 0 to R; 27: only the state entering R) -> swap_<arm>_<layers>.json
  swap_table  Supp. Table S21 from the swap files: stories right (margin averaged over both orders above zero), mean
              margin over the training texts, and the carried share (interchange - original) / (trained - original)
  smoke       one target and three loss evaluations (time, memory, gradient) before a model's first training run
  pictures    more drawing seeds of the lens picture for chosen stories (--stories id:version,...)

Table 4, the states figure and Supp. Table S20 (80 stories, 600 updates for both targets in both models):
    python scripts/state_tuning.py targets --model show_o2                 # the earlier states (latents.pt)
    python scripts/state_tuning.py targets --model show_o2 --set v5        # the states those lack (latents_v5.pt)
    python scripts/state_tuning.py eval --model show_o2 --set v5 --arm 0
    python scripts/state_tuning.py replies --model show_o2 --set v5 --arm 0
    python scripts/state_tuning.py train --model show_o2 --set v5 --arm V --tag 600 --updates 600 --seed 0   # seeds 0-2
    python scripts/state_tuning.py eval --model show_o2 --set v5 --arm V --tag 600 --seed 0
    python scripts/state_tuning.py replies --model show_o2 --set v5 --arm V --tag 600 --seed 0
    (the same with --arm S for the wrong-state pictures)
    python scripts/state_tuning.py judge --model show_o2 --judge qwen3vl          # also gemma3, llavaov2
    python scripts/state_table.py --model show_o2
HunyuanImage: the same commands with --model hunyuan, targets with --target-seeds 2 (on --set v2 first, then --set v5;
optionally --shard i --of n), eval with --picture-seeds 0 (optionally --shard i --of n).
Interchange (Supp. Table S21), Show-o2 on the first set:
    python scripts/state_tuning.py train --model show_o2 --arm V --tag 600 --updates 600 --seed 0   # seeds 0-2
    python scripts/state_tuning.py swap --model show_o2 --arm V --tag 600 --seed 0 --swap-layers all+27
    python scripts/state_tuning.py swap_table --model show_o2 --arm V --tag 600
"""

import argparse
import json
import random
import time
import zlib
from collections import defaultdict

import torch
import yaml

from vlens import paths, prompts, readout, results
from vlens.carry import CopyTransform
from vlens.hidden import find_word, replace_at
from vlens.lora import add_lora

OUT = results.RESULTS / "state_train"
CONFIG = dict(bagel=dict(R=8, request="a photo of a thing", named="a photo of {}"),
              show_o2=dict(R=27, request="a photo of a thing", named="a photo of {}"),
              sensenova=dict(R=24, request="thing", named="{}"),
              hunyuan=dict(R=9, request="a photo of a thing", named="a photo of {}"))
TEST = ("ice", "candle", "toast", "steak", "tadpole", "banana", "egg_drop", "balloon", "paint_green", "nail",
        "car_snow", "lake")   # two scenarios per family, fixed before any training
PANEL = ("qwen3vl", "gemma3", "llavaov2")
# HunyuanImage's training scenarios (and, on the second set, every model's): the 14 scenarios with a wrong position-free
# answer in a plain story (scripts/state_questions.py scores), plus 6 whose states its lens or drawing shows clearly; the
# other 20 scenarios are held out
TRAIN = dict(hunyuan=("autumn", "car_snow", "hen_egg", "marshmallow", "popcorn", "puddle", "toast", "chocolate",
                      "egg_drop", "lake", "log", "nail", "sandcastle", "snowman",
                      "pumpkin", "egg_fry", "banana", "tadpole", "caterpillar", "butter"))

sq = results.load_script("state_questions")


SET = "v1"   # --set v2: the second set's plain stories, whose question names the thing ("Is the ice cube now solid ...")


def cfg():
    return yaml.safe_load((prompts.PROMPTS / ("state_questions.yaml" if SET == "v1" else f"state_questions_{SET}.yaml")).read_text())


def is_train(model, i):
    """On the 80-story set every scenario trains; on the second set every model trains on HunyuanImage's scenarios (the
    same stories and questions for Show-o2 and HunyuanImage)."""
    if SET == "v5":
        return True
    ids = TRAIN.get(model) or (TRAIN["hunyuan"] if SET == "v2" else None)
    return i in ids if ids else i not in TEST


def examples(c, split, model=None):
    """(item, version, order, story, text, right answer, right state) for train or test scenarios, both orders."""
    for it in c["items"]:
        if is_train(model, it["id"]) == (split == "test"):
            continue
        for k, v in enumerate(("happened", "cancelled")):
            story = it[v] if SET == "v1" else it["stories"][v]
            state = it["outcome"] if v == "happened" else it["start"]
            wrong_state = it["start"] if v == "happened" else it["outcome"]
            for order, (a, b) in (("ab", it["options"]), ("ba", it["options"][::-1])):
                text = c["template"].format(story=story, question=it["question"], a=a, b=b)
                yield dict(id=it["id"], version=v, order=order, story=story, question=it["question"], text=text,
                           options=it["options"], right=it["answer"][k], state=state, wrong_state=wrong_state)


def store(model):
    return paths.DATA / "state_train" / model


EXTRACT = {}   # model -> extraction layer, when --extract sets one other than R


def layer_of(model):
    return EXTRACT.get(model, CONFIG[model]["R"])


class Lens:
    """The V-lens from the extraction layer (R unless --extract) to the insertion layer R: centred, each coordinate
    scaled by the ratio of the two layers' spreads when they differ, at the typical distance."""

    def __init__(self, m, model):
        s = CopyTransform(model).s
        self.R, self.L = CONFIG[model]["R"], layer_of(model)
        assert s["R"] == self.R, (s["R"], self.R)
        self.mean = s["mean"][self.R].to(m.device).float()
        self.mean_L = s["mean"][self.L].to(m.device).float()
        self.ratio = None
        if self.L != self.R:
            sigma = readout.all_sigma(m, s, model)
            self.ratio = (sigma[self.R] / sigma[self.L]).to(m.device).float()
        self.spread = float(s["spread"])
        self.request = CONFIG[model]["request"]
        self.slot = find_word(m.read_prompt(self.request), "thing")[-1]

    def __call__(self, h):
        x = h.float().to(self.mean.device) - self.mean_L
        if self.ratio is not None:
            x = x * self.ratio
        return self.mean + x * (self.spread / x.norm())

    def reading(self, m, h, keep_grad=False):
        return m.read_prompt(self.request, keep_grad=keep_grad, edit=replace_at(self.R, [self.slot], self(h)[None]))


def answer_reading(m, model, text, keep_grad=False):
    """The text read as answer() reads it, so its states are the ones the answer comes from."""
    if model == "show_o2":
        ids, start = m._chat_ids(text)
        return m._read_ids(ids, text, start, keep_grad, None)
    if model == "hunyuan":   # the pass stops at R when training
        return m.read(text, keep_grad=keep_grad, system=sq.HUNYUAN_SYSTEM, upto=layer_of(model) if keep_grad else None)
    return m.read(text, keep_grad=keep_grad)


def source_state(m, model, ex, keep_grad=False):
    r = answer_reading(m, model, ex["text"], keep_grad)
    pos = sq.marks(r, ex["story"], ex["question"])["q_subject"]
    return r.hidden[layer_of(model)][pos]


def on(adapters):
    import contextlib
    return adapters.on() if adapters is not None else contextlib.nullcontext()


def step_targets(args, m):
    c = cfg()
    states = sorted({it[k] for it in c["items"] for k in ("outcome", "start")
                     if (args.model not in TRAIN and SET == "v1") or is_train(args.model, it["id"])})
    if SET == "v5":   # only the states the earlier targets lack (the earlier states keep their pictures)
        have = set(_load_latents(args.model, "latents"))
        states = [st for st in states if st not in have]
    states = [st for k, st in enumerate(states) if k % args.of == args.shard]
    out = OUT / args.model / "targets"
    out.mkdir(parents=True, exist_ok=True)
    latents = {}
    for st in states:
        reading = m.read_prompt(CONFIG[args.model]["named"].format(st))
        zs = []
        for seed in range(args.target_seeds):
            res = m.draw(reading, seed=seed)
            z = res[0] if args.model == "show_o2" else m.last_latents if args.model == "hunyuan" else None
            pic = m.to_picture(res[0])
            pic.save(out / f"{st.replace(' ', '-')}_s{seed}.png")
            zs.append((z[0] if z.dim() == 4 else z).cpu() if z is not None else m.from_picture(pic).cpu())
        latents[st] = torch.stack(zs)
        print(st, tuple(latents[st].shape), flush=True)
    store(args.model).mkdir(parents=True, exist_ok=True)
    pre = "latents_v5" if SET == "v5" else "latents"
    torch.save(latents, store(args.model) / (f"{pre}.pt" if args.of == 1 else f"{pre}_{args.shard}of{args.of}.pt"))


def _load_latents(model, pre):
    """One latents file, or its shards' files merged (empty if none exists)."""
    one = store(model) / f"{pre}.pt"
    if one.exists():
        return torch.load(one)
    merged = {}
    for f in sorted(store(model).glob(f"{pre}_[0-9]*of*.pt")):
        merged.update(torch.load(f))
    return merged


def load_latents(model):
    """The targets' latents. --set v5: the earlier states from latents.pt and the added ones from latents_v5.pt."""
    merged = _load_latents(model, "latents")
    if SET == "v5":
        merged = {**merged, **_load_latents(model, "latents_v5")}
        need = {it[k] for it in cfg()["items"] for k in ("outcome", "start")}
        assert need <= set(merged), sorted(need - set(merged))
    return merged


def loss_V(m, model, lens, ex, ctx, adapters):
    with on(adapters):
        h = source_state(m, model, ex, keep_grad=True)
    reading = lens.reading(m, h, keep_grad=True)       # adapters off: request and drawing on the frozen model
    pool = ctx["latents"][ex["wrong_state" if ctx.get("swap") else "state"]]
    z = pool[ctx["rng"].randrange(len(pool))].to(m.device)
    if model == "show_o2" and z.dim() == 3:    # Show-o2's latents come from its sampler without the batch axis
        z = z[None]
    level = 0.4 + 0.6 * ctx["rng"].random()
    noise = torch.randn(z.shape, generator=ctx["torch_rng"]).to(m.device)
    return m.denoise_loss(reading, z, level, noise)


def loss_T(m, model, lens, ex, ctx, adapters):
    """Cross-entropy of the right option word as the reply's first token (the best-scoring spelling)."""
    if model != "show_o2":
        raise NotImplementedError("arm T is wired for Show-o2 first")
    with on(adapters), torch.enable_grad():
        ids, _ = m._chat_ids(ex["text"])
        logits = m.net.showo(input_ids=torch.tensor([ids], device=m.device), use_cache=False).logits[0, -1]
    lp = torch.log_softmax(logits.float(), -1)
    tok = m.tokenizer
    ids = lambda o: sorted({tok.encode(w, add_special_tokens=False)[0] for w in (o, " " + o, o.capitalize(), " " + o.capitalize())})
    right = torch.logsumexp(lp[ids(ex["right"])], 0)
    wrong = torch.logsumexp(lp[ids([o for o in ex["options"] if o != ex["right"]][0])], 0)
    return -(right - torch.logsumexp(torch.stack([right, wrong]), 0))


def step_train(args, m):
    c = cfg()
    lens = Lens(m, args.model)
    layers = range(layer_of(args.model))
    adapters = add_lora(m, layers, rank=args.rank, alpha=args.alpha, seed=args.seed)
    params = adapters.parameters()
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    train = list(examples(c, "train", args.model))
    ctx = dict(rng=random.Random(args.seed), torch_rng=torch.Generator().manual_seed(args.seed))
    if args.arm in ("V", "S"):   # S: the same lens training on pictures of the wrong state (control)
        ctx["latents"] = load_latents(args.model)
        ctx["swap"] = args.arm == "S"
    loss_fn = dict(V=loss_V, S=loss_V, T=loss_T)[args.arm]
    order = random.Random(1000 + args.seed)
    history, t0 = [], time.time()
    for update in range(args.updates):
        total = 0.0
        for ex in (train[order.randrange(len(train))] for _ in range(args.batch)):
            loss = loss_fn(m, args.model, lens, ex, ctx, adapters) / args.batch
            loss.backward()
            total += float(loss)
        opt.step()
        opt.zero_grad()
        history.append(total)
        if update % 10 == 0 or update == args.updates - 1:
            print(f"[{time.time() - t0:6.0f}s] update {update}: loss {total:.4f}", flush=True)
    store(args.model).mkdir(parents=True, exist_ok=True)
    torch.save(dict(adapters=adapters.state_dict(), arm=args.arm, seed=args.seed, updates=args.updates, history=history,
                    rank=args.rank, alpha=args.alpha, lr=args.lr, batch=args.batch, texts=len(train)),
               store(args.model) / f"{arm_name(args)}_s{args.seed}.pt")
    print("->", store(args.model) / f"{arm_name(args)}_s{args.seed}.pt")


def arm_name(args):
    """The arm, with --tag (V600: 600 updates) and the extraction layer when --extract sets one (V_L24, 0_L24)."""
    return args.arm + args.tag + ("" if SET == "v1" else f"_{SET}") + (f"_L{args.extract}" if args.extract else "")


def with_adapters(m, model, arm, seed):
    if arm.split("_")[0] == "0":
        return None
    state = torch.load(store(model) / f"{arm}_s{seed}.pt")
    adapters = add_lora(m, range(layer_of(model)), rank=state["rank"], alpha=state["alpha"])
    adapters.load_state_dict(state["adapters"])
    return adapters


def step_eval(args, m):
    """With --of n (HunyuanImage, slow pictures): shard i draws every n-th picture text; shard 0 also scores every text."""
    c = cfg()
    lens = Lens(m, args.model)
    adapters = with_adapters(m, args.model, arm_name(args), args.seed)
    name = arm_name(args) + (f"_s{args.seed}" if args.arm != "0" else "")
    out = OUT / args.model / name
    out.mkdir(parents=True, exist_ok=True)
    tag = "" if args.of == 1 else f"_{args.shard}of{args.of}"
    rows, recs, k = [], [], -1
    for split in args.split.split(","):
        for ex in examples(c, split, args.model):
            if args.shard == 0:
                with on(adapters):
                    lp = sq.option_logprobs(m, args.model, ex["text"], ex["options"])
                wrong = [o for o in ex["options"] if o != ex["right"]][0]
                rows.append(dict(split=split, **{k: ex[k] for k in ("id", "version", "order", "right")},
                                 margin=lp[ex["right"]] - lp[wrong]))
            if ex["order"] != "ab" or args.no_pictures:
                continue
            k += 1
            if k % args.of != args.shard:
                continue
            with on(adapters):
                h = source_state(m, args.model, ex).detach()
            reading = lens.reading(m, h)
            for seed in map(int, args.picture_seeds.split(",")):
                f = f"{ex['id']}_{ex['version']}_s{seed}.png"
                if not (out / f).exists():
                    m.to_picture(m.draw(reading, seed=seed)[0]).save(out / f)
                other = [s for s in (cfg_item(c, ex["id"])["outcome"], cfg_item(c, ex["id"])["start"]) if s != ex["state"]][0]
                recs.append(dict(file=f"{args.model}/{name}/{f}", split=split, id=ex["id"], version=ex["version"],
                                 seed=seed, truth=ex["state"], other=other))
            (out / f"records{tag}.json").write_text(json.dumps(recs, indent=1))
        if args.shard == 0:
            (out / "margins.json").write_text(json.dumps(rows, indent=1))
        (out / f"records{tag}.json").write_text(json.dumps(recs, indent=1))
        print(split, "done", flush=True)


def hunyuan_reply(m, text):
    """HunyuanImage's one-word reply as the state questions take it (scripts/state_questions.py, step answers): the
    one-word system prompt, greedy, image and thinking tokens suppressed."""
    from transformers import LogitsProcessorList, SuppressTokensLogitsProcessor
    tk = m.net._tokenizer
    banned = [tk.convert_tokens_to_ids(tk.boi_token), tk.convert_tokens_to_ids(tk.think_token)]
    inputs = m.net.prepare_model_inputs(prompt=text, mode="gen_text", bot_task="auto", max_new_tokens=8,
                                        system_prompt=sq.HUNYUAN_SYSTEM)
    with torch.no_grad():
        reply = m.net.generate(**inputs, decode_text=True, do_sample=False,
                               logits_processor=LogitsProcessorList([SuppressTokensLogitsProcessor(banned, device=m.device)]))
    reply = reply[0] if isinstance(reply, list) else reply
    for t in ("<answer>", "</answer>", "<|endoftext|>"):
        reply = reply.replace(t, "")
    return reply


def step_replies(args, m):
    """The model's greedy reply to every text (both splits, both orders) with an arm's adapters, for the figure's
    before/after answers: replies.json beside the arm's margins."""
    c = cfg()
    adapters = with_adapters(m, args.model, arm_name(args), args.seed)
    name = arm_name(args) + (f"_s{args.seed}" if args.arm != "0" else "")
    out = OUT / args.model / name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in args.split.split(","):
        for ex in examples(c, split, args.model):
            with on(adapters):
                reply = hunyuan_reply(m, ex["text"]) if args.model == "hunyuan" else m.answer(ex["text"], max_new_tokens=8)
            rows.append(dict(split=split, **{k: ex[k] for k in ("id", "version", "order", "right")},
                             reply=reply.strip()[:120], answer=sq.first_option(reply, ex["options"])))
        print(split, "done", flush=True)
    (out / "replies.json").write_text(json.dumps(rows, indent=1))


def margin_with(m, model, ex, edit):
    """Position-free margin material: log p(right) - log p(wrong) of the reply's first token for this text, with `edit`
    applied to the question's hidden states while answering (sq.option_logprobs, plus the edit)."""
    head = [mod for name, mod in m.net.named_modules() if name.split(".")[-1] == "lm_head"][-1]
    got = []
    h = head.register_forward_hook(lambda mod, i, o: got.append(o.detach()) if not got else None)
    try:
        if model == "hunyuan":   # option_logprobs' one-token pass; the edit acts on the prompt's full pass
            inputs = m.net.prepare_model_inputs(prompt=ex["text"], mode="gen_text", bot_task="auto", max_new_tokens=1,
                                                system_prompt=sq.HUNYUAN_SYSTEM)
            if edit is not None:
                m._arm("edit", edit=edit)
            try:
                m.net.generate(**inputs, decode_text=False, do_sample=False)
            finally:
                m._armed = False
        else:
            m.answer(ex["text"], max_new_tokens=1, edit=edit)
    finally:
        h.remove()
    lp = torch.log_softmax(got[0].float().reshape(-1, got[0].shape[-1])[-1], -1)
    tok = m.tokenizer
    best = lambda o: max(float(lp[i]) for i in {tok.encode(v, add_special_tokens=False)[0]
                                                 for v in (o, " " + o, o.capitalize(), " " + o.capitalize())})
    wrong = [o for o in ex["options"] if o != ex["right"]][0]
    return best(ex["right"]) - best(wrong)


def step_swap(args, m):
    """The interchange test: is the supervised subject state itself what carries the answer change?
    For every text, the subject's states from the original model (h0) and from the trained one (hV), then six answer
    passes: original; trained; original with the subject's states replaced by hV; trained with them replaced by h0; and
    the two self-swaps (implementation checks). --swap-layers 27 replaces only the state entering the insertion layer
    (the supervised one); "all" replaces the subject's state entering layers 0 to R (every layer in Show-o2, R = 27);
    "every" every layer of the model (HunyuanImage: 0 to 31, R = 9). Several settings in one run, separated by "+"
    (the original and trained passes are shared). Writes swap_<arm>_<layers>.json per setting. Show-o2 and HunyuanImage
    (its answer pass is option_logprobs' generate() with the edit armed on the prompt's pass)."""
    if args.model not in ("show_o2", "hunyuan"):
        raise NotImplementedError("the interchange test is wired for Show-o2 and HunyuanImage")
    import contextlib
    c = cfg()
    adapters = with_adapters(m, args.model, arm_name(args), args.seed)
    R = CONFIG[args.model]["R"]
    settings = args.swap_layers.split("+")
    layer_sets = {s: list(range(R + 1)) if s == "all" else list(range(m.n_layers)) if s == "every"
                  else [int(x) for x in s.split(",")] for s in settings}

    def swap(donor, pos, layers):
        vecs = {l: donor.hidden[l][pos].detach() for l in layers}

        def edit(l, h):
            if l not in vecs:
                return h
            h = h.clone()
            h[pos] = vecs[l].to(h.dtype).to(h.device)
            return h
        return edit

    def margin(edit, adapt, ex):
        with (on(adapters) if adapt else contextlib.nullcontext()), torch.no_grad():
            return margin_with(m, args.model, ex, edit)

    rows, t0 = {s: [] for s in settings}, time.time()
    for split in args.split.split(","):
        for k, ex in enumerate(examples(c, split, args.model)):
            if args.limit and k >= args.limit:
                break
            with torch.no_grad():
                r0 = answer_reading(m, args.model, ex["text"])
                with on(adapters):
                    rV = answer_reading(m, args.model, ex["text"])
            pos = sq.marks(r0, ex["story"], ex["question"])["q_subject"]
            base = {"original": margin(None, False, ex), "trained": margin(None, True, ex)}
            for s in settings:
                L = layer_sets[s]
                conds = {"original+trained_state": (swap(rV, pos, L), False),
                         "trained+original_state": (swap(r0, pos, L), True),
                         "original+self": (swap(r0, pos, L), False), "trained+self": (swap(rV, pos, L), True)}
                row = dict(split=split, **{k: ex[k] for k in ("id", "version", "order", "right")}, **base)
                for name, (edit, adapt) in conds.items():
                    row[name] = margin(edit, adapt, ex)
                rows[s].append(row)
            if k == 0:
                print(f"[{time.time() - t0:5.0f}s] first text {ex['id']} subject {r0.token_strs[pos]!r}:",
                      {s: {n: round(v, 4) for n, v in rows[s][-1].items() if isinstance(v, float)} for s in settings},
                      flush=True)
        print(f"[{time.time() - t0:5.0f}s] {split} done", flush=True)
    name = arm_name(args) + f"_s{args.seed}"
    for s in settings:
        out = OUT / args.model / f"swap_{name}_{s.replace(',', '-')}{args.out_tag}.json"
        out.write_text(json.dumps(rows[s], indent=1))
        for split in args.split.split(","):
            rs = [r for r in rows[s] if r["split"] == split]
            print(s, split, {k: round(sum(r[k] for r in rs) / len(rs), 3) for k in rs[0] if isinstance(rs[0][k], float)},
                  flush=True)


def step_pictures(args, m):
    """More drawing seeds of the lens picture for chosen stories (--stories toast:happened,car_snow:happened), with an
    arm's adapters; pictures beside the arm's others, records in records_extra.json."""
    c = cfg()
    lens = Lens(m, args.model)
    adapters = with_adapters(m, args.model, arm_name(args), args.seed)
    name = arm_name(args) + (f"_s{args.seed}" if args.arm != "0" else "")
    out = OUT / args.model / name
    out.mkdir(parents=True, exist_ok=True)
    wanted = {tuple(x.split(":")) for x in args.stories.split(",")}
    recs = []
    for split in ("train", "test"):
        for ex in examples(c, split, args.model):
            if ex["order"] != "ab" or (ex["id"], ex["version"]) not in wanted:
                continue
            with on(adapters):
                h = source_state(m, args.model, ex).detach()
            reading = lens.reading(m, h)
            other = [s for s in (cfg_item(c, ex["id"])["outcome"], cfg_item(c, ex["id"])["start"]) if s != ex["state"]][0]
            for seed in map(int, args.picture_seeds.split(",")):
                f = f"{ex['id']}_{ex['version']}_s{seed}.png"
                if not (out / f).exists():
                    m.to_picture(m.draw(reading, seed=seed)[0]).save(out / f)
                recs.append(dict(file=f"{args.model}/{name}/{f}", split=split, id=ex["id"], version=ex["version"],
                                 seed=seed, truth=ex["state"], other=other))
            print(ex["id"], ex["version"], "done", flush=True)
    (out / "records_extra.json").write_text(json.dumps(recs, indent=1))


def step_smoke(args, m):
    """One target drawing and three training steps (levels 0.2, 0.5, 0.9) on the first training text: loss, gradient on
    the adapters, seconds and GPU memory, before a model's first training run."""
    c = cfg()
    lens = Lens(m, args.model)
    ex = next(examples(c, "train", args.model))
    t0 = time.time()
    m.draw(m.read_prompt(CONFIG[args.model]["named"].format(ex["state"])), seed=0)
    z = m.last_latents
    print("target", ex["state"], tuple(z.shape), f"{time.time() - t0:.0f}s", flush=True)
    adapters = add_lora(m, range(layer_of(args.model)), rank=args.rank, alpha=args.alpha, seed=0)
    params = adapters.parameters()
    for p in params:
        p.requires_grad_(True)
    g = torch.Generator().manual_seed(0)
    for level in (0.2, 0.5, 0.9):
        for p in params:
            p.grad = None
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(d)
        t0 = time.time()
        with on(adapters):
            h = source_state(m, args.model, ex, keep_grad=True)
        t1 = time.time()
        loss = m.denoise_loss(lens.reading(m, h, keep_grad=True), z, level, torch.randn(z.shape, generator=g))
        t2 = time.time()
        loss.backward()
        t3 = time.time()
        gnorm = sum(float(p.grad.norm()) ** 2 for p in params if p.grad is not None) ** 0.5
        mem = [round(torch.cuda.max_memory_allocated(d) / 2 ** 30, 1) for d in range(torch.cuda.device_count())]
        print(f"level {level}: loss {float(loss):.4f}, adapter grad {gnorm:.3e}, read {t1 - t0:.1f}s, "
              f"loss {t2 - t1:.1f}s, backward {t3 - t2:.1f}s, peak GiB {mem}", flush=True)


def cfg_item(c, i):
    return next(it for it in c["items"] if it["id"] == i)


def step_swap_table(args):
    """Supp. Table S21 (CPU): for each training seed's swap files (swap_<arm>_s<seed>_<layers>.json), over the training
    stories: stories answered right (the margin averaged over both option orders is above zero) by the original model,
    the trained model and the original model with the trained subject states; the mean margins over the training texts;
    and the carried share (interchange margin - original) / (trained margin - original)."""
    folder = OUT / args.model
    out = {}
    for f in sorted(folder.glob(f"swap_{arm_name(args)}_s*_*.json")):
        rows = [r for r in json.loads(f.read_text()) if r["split"] == "train"]
        if not rows:
            continue

        def right(key):
            per = defaultdict(list)
            for r in rows:
                per[(r["id"], r["version"])].append(r[key])
            return sum(sum(v) / len(v) > 0 for v in per.values()), len(per)

        mean = lambda key: sum(r[key] for r in rows) / len(rows)   # noqa: E731
        o, t, x = mean("original"), mean("trained"), mean("original+trained_state")
        out[f.stem] = dict(stories=right("original")[1], original_right=right("original")[0],
                           trained_right=right("trained")[0], interchange_right=right("original+trained_state")[0],
                           original_margin=o, trained_margin=t, interchange_margin=x,
                           carried_share=(x - o) / (t - o) if t != o else None)
        r = out[f.stem]
        share = "" if r["carried_share"] is None else f"; carried {100 * r['carried_share']:.0f}%"
        print(f"{f.stem}: right {r['original_right']} -> trained {r['trained_right']}, interchange "
              f"{r['interchange_right']} of {r['stories']}; mean margin {o:+.3f} -> {t:+.3f}, interchange {x:+.3f}"
              + share)
    (folder / f"swap_table_{arm_name(args)}.json").write_text(json.dumps(out, indent=1))


def step_judge(args):
    from PIL import Image

    from vlens.judges import Judge, subject_options, subject_question
    recs = []
    for f in (OUT / args.model).glob("*/records*.json"):
        recs += json.loads(f.read_text())
    done = set()
    for f in OUT.glob(f"judge_{args.judge}_{args.model}_*.json"):
        done |= {d["file"] for d in json.loads(f.read_text())}
    todo = [r for r in recs if r["file"] not in done and (OUT / r["file"]).exists()
            and zlib.crc32(r["file"].encode()) % args.of == args.shard]
    print(f"{len(todo)} pictures to judge", flush=True)
    if not todo:
        return
    j = Judge(args.judge)
    f = OUT / f"judge_{args.judge}_{args.model}_{args.shard}of{args.of}_{time.strftime('%H%M%S')}.json"
    ans = []
    for r in todo:
        opts = subject_options(r["truth"], r["other"], r["file"])
        ans.append(dict(file=r["file"], chosen=j.choose(Image.open(OUT / r["file"]), opts, subject_question).chosen))
    f.write_text(json.dumps(ans, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["targets", "train", "eval", "judge", "smoke", "replies", "swap", "swap_table",
                                     "pictures"])
    ap.add_argument("--model", default="show_o2", choices=sorted(CONFIG))
    ap.add_argument("--arm", default="V", choices=["0", "V", "T", "S"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--judge", default="qwen3vl")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--no-pictures", action="store_true")
    ap.add_argument("--target-seeds", type=int, default=4)
    ap.add_argument("--set", default="v1", choices=["v1", "v2", "v5"])
    ap.add_argument("--stories", default="", help="pictures step: id:version pairs, comma separated")
    ap.add_argument("--swap-layers", default="27", help='swap step: layers whose subject state is exchanged, "all" '
                    '(0 to R) or "every" (every layer); several settings separated by "+"')
    ap.add_argument("--limit", type=int, default=0, help="swap step: only the first n texts of each split (smoke test)")
    ap.add_argument("--out-tag", default="", help="swap step: added to the output file names (smoke test)")
    ap.add_argument("--tag", default="", help="added to the arm's name, e.g. 600 for a longer run (V600_s0)")
    ap.add_argument("--extract", type=int, default=None, help="extraction layer, if not R (adapters on the layers before "
                                                              "it; not used for the paper)")
    ap.add_argument("--picture-seeds", default="0,1")
    ap.add_argument("--split", default="train,test")
    args = ap.parse_args()
    if args.step == "judge":
        return step_judge(args)
    global SET
    SET = args.set
    if args.extract:
        EXTRACT[args.model] = args.extract
    if args.step == "swap_table":
        return step_swap_table(args)
    from vlens.models import load
    m = load(args.model)
    dict(targets=step_targets, train=step_train, eval=step_eval, smoke=step_smoke, replies=step_replies, swap=step_swap,
         pictures=step_pictures)[args.step](args, m)


if __name__ == "__main__":
    main()
