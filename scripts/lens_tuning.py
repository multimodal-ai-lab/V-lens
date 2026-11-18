"""Lens tuning on pronouns (Sec. 4.6, Table 3, Supp. F.1 and F.2): train small LoRA adapters on the text weights so
that the V-lens picture of "it" shows its referent, and the comparison arms. Items: configs/prompts/vsup_items.yaml
(scripts/make_tuning_items.py). Models: BAGEL (insertion layer 8, adapters on layers 0-7), SenseNova-U1 (24, layers
0-23), HunyuanImage-3.0 (9, layers 0-8; attention and the shared MLP).

The adapters (vlens/lora.py, rank 8, scaling 16, on the query, key, value, output, gate, up and down projections) sit on
the layers before the insertion layer, so they change the state of "it" entering it. They are on only while the source
text is read; the drawing request and the drawing itself run on the frozen model. Arms (same texts, same order, same
number of updates):
- 0  no training ("None");
- T  text: next-word loss on the answer to 'What does "it" refer to in the last sentence?' ("The <referent>.";
     HunyuanImage: "<referent></answer>" after its one-word system prompt, the format of its text answers);
- V  lens tuning: the model's own denoising loss (noise level uniform on [0.4, 1], no guidance) on a picture of the
     referent that the frozen model drew from "a photo of a <referent>", with the V-lens picture of "it" as the drawing
     request;
- G  ordinary drawing: the same loss and pictures as V, but the drawing request is the text itself followed by
     "A photo of it." (no lens), adapters on throughout;
- R  word state: 1 - cosine between the centred state of "it" and the referent word's own centred state (frozen model,
     averaged over the four neutral sentences of the item file), in the space the lens uses;
- S  wrong pictures: V with the pictures of a fixed wrong noun per referent (vsup_items.yaml, "wrong").
At the insertion layer the V-lens (coordinate scaling) reduces to centring and setting the typical radius, so the lens
here is mu_R + s_R (h - mu_R) / |h - mu_R|.

Steps:
- targets:  draw "a photo of a <noun>" for every training noun, drawing seeds 0-5, as judgeable records (folder
            vsup_targets; HunyuanImage also keeps each picture's final latent);
- latents:  keep up to four pictures per noun that at least two of the three judges name as the main subject (needs the
            judges' answers on vsup_targets, scripts/run_judge.py --subject);
- states:   the training nouns' own states at the insertion layer (for arm R);
- check:    the adapters change the source state and leave the request untouched; gradient against a finite difference
            along the gradient; time per update for each arm;
- train:    one arm, one training seed -> $VLENS_DATA/<store>/<arm>_s<seed>.pt;
- draw:     V-lens pictures of "it" (drawing seeds 0-2) with an arm's adapters, folder vsup_cells;
- text:     the four text questions (name, choice, yes, no) on every text and the 40 general questions, with an arm's
            adapters -> $VLENS_OUT/vsup_pilot/text_[<model>_]<arm>[_s<seed>].json; arm 0 on all groups is also the screen
            of the training texts (only texts the untrained model names correctly are trained on);
- generate: ordinary drawing from the text plus "A photo of it." with an arm's adapters on (no lens), drawing seeds 0
            and 1, folder vsup_gen.

    python scripts/lens_tuning.py targets --model bagel
    python scripts/lens_tuning.py text --model bagel --arm 0 --groups train,dev,test1,test2,test3
    python scripts/lens_tuning.py latents --model bagel
    python scripts/lens_tuning.py states --model bagel
    python scripts/lens_tuning.py train --model bagel --arm V --seed 0
    python scripts/lens_tuning.py draw --model bagel --arm V --seed 0
    python scripts/lens_tuning.py text --model bagel --arm V --seed 0
    python scripts/lens_tuning.py generate --model bagel --arm V --seed 0
"""

import argparse
import contextlib
import json
import random
import re
import time

import torch
import yaml

from vlens import paths, prompts, results
from vlens.carry import CopyTransform
from vlens.hidden import find_word, replace_at
from vlens.lora import add_lora
from vlens.models import load, transport_name
from vlens.transport import Transport

# Per model (configure()): the drawing request, the named request, the insertion layer (reading "it" there, the moved
# copy and copy:diag coincide), the adapter layers (all layers before it), where adapters and latents are stored, and the
# prefix of the text-answer files.
MODELS = dict(bagel=dict(request="a photo of a thing", named="a photo of a {}", layer=8, store="vsup", text="text_"),
              sensenova=dict(request="thing", named="{}", layer=24, store="vsup_sensenova", text="text_sensenova_"),
              hunyuan=dict(request="a photo of a thing", named="a photo of a {}", layer=9, store="vsup_hunyuan",
                           text="text_hunyuan_"))
HUNYUAN_SYSTEM = "You are a helpful assistant. Answer every question with exactly one word. Never explain your answer."
MODEL = "bagel"
REQUEST, NAMED, LAYER, STORE, TEXT = "a photo of a thing", "a photo of a {}", 8, paths.DATA / "vsup", "text_"
LORA_LAYERS = range(LAYER)
PILOT = results.RESULTS / "vsup_pilot"
REFERENCE = ' What does "it" refer to in the last sentence? Answer with one word.'
GEN_TAIL = " A photo of it."
ARMS = ("0", "T", "V", "R", "S", "G")
# the output folders of the targets, the lens pictures and ordinary drawing
SETS = dict(targets="vsup_targets", cells="vsup_cells", gen="vsup_gen")


def configure(model):
    global MODEL, REQUEST, NAMED, LAYER, STORE, TEXT, LORA_LAYERS
    c = MODELS[model]
    MODEL, REQUEST, NAMED, LAYER, TEXT = model, c["request"], c["named"], c["layer"], c["text"]
    STORE, LORA_LAYERS = paths.DATA / c["store"], range(c["layer"])


def items():
    return yaml.safe_load((prompts.PROMPTS / "vsup_items.yaml").read_text())


def word_end(reading, word: str, occurrence: int = -1) -> int:
    """The last token of `word` as a whole word ("it" does not match "kitchen"), as scripts/draw_cells.py reads."""
    starts, text = [], ""
    for s in reading.token_strs:
        starts.append(len(text))
        text += s
    hits = [h.start() for h in re.finditer(rf"(?<![\w-]){re.escape(word)}(?![\w-])", text)]
    begin = hits[occurrence]
    end = begin + len(word)
    return [p for p, s in enumerate(starts) if s < end and s + len(reading.token_strs[p]) > begin][-1]


def on(adapters):
    """Adapters on (none: the frozen model)."""
    return adapters.on() if adapters is not None else contextlib.nullcontext()


class Lens:
    """The copy at the insertion layer: mean + typical size * (h - mean) / |h - mean| (differentiable)."""

    def __init__(self, m):
        if MODEL == "bagel":
            t = Transport.load(transport_name("bagel"))
            assert t.R == LAYER, t.R
            self.mean = t.mean[LAYER].to(m.device).float()
            self.spread = float(t.meta["spread"])
        else:  # the statistics of the paper's V-lens (copy:diag), which at the insertion layer is this same copy
            s = CopyTransform(MODEL).s
            assert s["R"] == LAYER, s["R"]
            self.mean = s["mean"][LAYER].to(m.device).float()
            self.spread = float(s["spread"])
        self.slot = find_word(m.read_prompt(REQUEST), "thing")[-1]

    def __call__(self, h):
        x = h.float().to(self.mean.device) - self.mean   # HunyuanImage returns read states on the CPU
        return self.mean + x * (self.spread / x.norm())

    def request(self, m, h, keep_grad=False):
        return m.read_prompt(REQUEST, keep_grad=keep_grad, edit=replace_at(LAYER, [self.slot], self(h)[None]))


def source_state(m, text, adapters=None, keep_grad=False):
    """The state of "it" entering the insertion layer, the source text read with the adapters on."""
    with on(adapters):   # HunyuanImage's graph-keeping pass stops at, and keeps only, the layer asked for
        src = (m.read(text, keep_grad=True, upto=LAYER) if keep_grad and MODEL == "hunyuan"
               else m.read(text, keep_grad=keep_grad))
    return src.hidden[LAYER][word_end(src, "it")]


# ---------------------------------------------------------------------------------------------------------- losses

def loss_T(m, lens, ex, ctx, adapters):
    if MODEL == "sensenova":  # question and answer in one pass, in the chat format answer() uses; every answer token
        reply = f"The {ex['truth']}."  # is predicted from the position before it
        with on(adapters):
            r = m._read(m._chat_query(ex["text"] + REFERENCE) + reply, reply, True, None)
        logits = m.lm.lm_head(m.lm.model.norm(r.final)).float()
        pos = torch.nonzero(r.content).flatten().to(logits.device)
        ids = torch.tensor(r.token_ids, device=logits.device)
        return torch.nn.functional.cross_entropy(logits[pos - 1], ids[pos])
    if MODEL == "hunyuan":   # the question as the text step asks it (the "name" form, which is the text plus
        with on(adapters):   # REFERENCE, with HUNYUAN_SYSTEM); the referent noun as the model writes its one-word
            return m.reply_loss(ex["questions"]["name"], ex["truth"], HUNYUAN_SYSTEM)   # answers, then "</answer>"
    with on(adapters):
        first = m.read(ex["text"] + REFERENCE, keep_grad=True)
        reply = m.read(f"The {ex['truth']}.", previous=first.cache, keep_grad=True)
    logits = m.next_word_scores(reply.final).float()
    ids = torch.tensor(reply.token_ids, device=logits.device)
    return torch.nn.functional.cross_entropy(logits[:-1], ids[1:])


def target(m, ctx, noun):
    """One teacher picture of `noun`, one noise level (uniform on [0.4, 1]) and fresh noise."""
    pool = ctx["latents"][noun]
    g = ctx["rng"]
    z = pool[g.randrange(len(pool))].to(m.device)
    level = 0.4 + 0.6 * g.random()
    noise = torch.randn(z.shape, generator=ctx["torch_rng"]).to(m.device)
    return z, level, noise


def loss_V(m, lens, ex, ctx, adapters, noun=None):
    noun = noun or ex["truth"]
    h = source_state(m, ex["text"], adapters, keep_grad=True)
    reading = lens.request(m, h, keep_grad=True)  # adapters off: the request and the drawing run on the frozen model
    return m.denoise_loss(reading, *target(m, ctx, noun))


def loss_G(m, lens, ex, ctx, adapters):
    """Ordinary generation: the text plus "A photo of it." is the drawing request, adapters on throughout."""
    with on(adapters):
        reading = m.read_prompt(ex["text"] + GEN_TAIL, keep_grad=True)
        return m.denoise_loss(reading, *target(m, ctx, ex["truth"]))


def loss_S(m, lens, ex, ctx, adapters):
    return loss_V(m, lens, ex, ctx, adapters, noun=ctx["wrong"][ex["truth"]])


def loss_R(m, lens, ex, ctx, adapters):
    h = source_state(m, ex["text"], adapters, keep_grad=True)
    x = h.float() - lens.mean
    return 1 - torch.nn.functional.cosine_similarity(x, ctx["word_dirs"][ex["truth"]].to(x.device), dim=0)


LOSSES = dict(T=loss_T, V=loss_V, S=loss_S, R=loss_R, G=loss_G)


def context(m, lens, data, arm, seed):
    ctx = dict(rng=random.Random(seed), torch_rng=torch.Generator().manual_seed(seed), wrong=data["wrong"])
    if arm in ("V", "S", "G"):
        ctx["latents"] = torch.load(STORE / "latents.pt")
    if arm == "R":
        states = torch.load(STORE / "word_states.pt")
        ctx["word_dirs"] = {n: torch.nn.functional.normalize(s, dim=0) for n, s in states.items()}
    return ctx


def approved(data):
    """Training texts the frozen model resolves in text (the screen, `text` step with arm 0)."""
    screen = json.loads((PILOT / f"{TEXT}0.json").read_text())["rows"]
    if (PILOT / f"{TEXT}0_train.json").exists():   # the training texts screened apart (--forms name --tag train)
        screen += json.loads((PILOT / f"{TEXT}0_train.json").read_text())["rows"]
    ok = {r["id"] for r in screen if r["name"]}
    return [it for it in data["reads"] if it["group"] == "train" and it["id"] in ok]


# ----------------------------------------------------------------------------------------------------------- steps

def step_targets(args, m):
    data = items()
    nouns = sorted({it["truth"] for it in data["reads"] if it["group"] == "train"})
    cat = data["category"]
    rng = random.Random(0)
    rows = [dict(id=f"target_{n.replace(' ', '-')}", text=NAMED.format(n), truth=n,
                 alt=rng.choice([x for x in nouns if cat[x] != cat[n]])) for n in nouns]
    if MODEL == "bagel":  # the item file the judges read; the alternatives are the same for every model
        (prompts.PROMPTS / "vsup_targets.yaml").write_text(yaml.safe_dump(dict(reads=rows), sort_keys=False))
    out = results.folder(SETS["targets"]) / MODEL
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for i, r in list(enumerate(rows))[args.shard::args.of]:   # --shard/--of split the nouns
        reading = m.read_prompt(r["text"])
        for seed in range(6):
            file = f"vsup_target_{i:03d}_s{seed}.png"
            # HunyuanImage cannot encode a picture back (no from_picture): its final latent is kept next to the picture
            keep = MODEL == "hunyuan" and not (out / file.replace(".png", ".pt")).exists()
            if not (out / file).exists() or keep:
                m.to_picture(m.draw(reading, seed=seed)[0]).save(out / file)
                if MODEL == "hunyuan":
                    torch.save(m.last_latents, out / file.replace(".png", ".pt"))
            records.append(dict(file=f"{MODEL}/{file}", model=MODEL, items="vsup_targets", kind="reads", item=i,
                                text=r["text"], position="named", word=r["truth"], layer=None, way="named", seed=seed,
                                truth=r["truth"]))
        print(i, r["truth"], flush=True)
    tag = "" if args.of == 1 else f"_{args.shard}of{args.of}"
    (out / f"records_vsup_targets{tag}.json").write_text(json.dumps(records, indent=1))


def step_latents(args, m):
    from PIL import Image
    from vlens.judges import answer_files
    rows = yaml.safe_load((prompts.PROMPTS / "vsup_targets.yaml").read_text())["reads"]
    folder = results.folder(SETS["targets"])
    votes = {}
    for j in ("qwen3vl", "gemma3", "llavaov2"):
        for f in answer_files(results.RESULTS / "judges", j, f"cells:{SETS['targets']}_subject"):
            for a in json.loads(f.read_text()):
                votes.setdefault(a["file"], {})[j] = a["chosen"] == a["truth"]
    latents, kept = {}, {}
    for i, r in enumerate(rows):
        good = [s for s in range(6) if sum(votes.get(f"{MODEL}/vsup_target_{i:03d}_s{s}.png", {}).values()) >= 2][:4]
        kept[r["truth"]] = good
        if good:
            if MODEL == "hunyuan":   # the latents the targets step kept ([1, ...] each)
                latents[r["truth"]] = torch.cat([torch.load(folder / MODEL / f"vsup_target_{i:03d}_s{s}.pt")
                                                 for s in good])
            else:
                latents[r["truth"]] = torch.stack([m.from_picture(Image.open(folder / MODEL / f"vsup_target_{i:03d}_s{s}.png"))
                                                   .cpu() for s in good])
    STORE.mkdir(parents=True, exist_ok=True)
    torch.save(latents, STORE / "latents.pt")
    PILOT.mkdir(parents=True, exist_ok=True)
    (PILOT / ("targets_kept.json" if MODEL == "bagel" else f"targets_kept_{MODEL}.json")).write_text(json.dumps(kept, indent=1))
    print({k: len(v) for k, v in kept.items()})


def step_states(args, m):
    data = items()
    lens = Lens(m)
    nouns = sorted({it["truth"] for it in data["reads"] if it["group"] == "train"})
    states = {}
    for n in nouns:
        xs = []
        for s in data["neutral"]:
            a = ("an " if n[0] in "aeiou" else "a ") + n
            text = s.format(a_x=a, x=n)
            src = m.read(text)
            pos = word_end(src, n.split()[-1])
            x = src.hidden[LAYER][pos].float() - lens.mean
            xs.append(x / x.norm())
        states[n] = torch.stack(xs).mean(0).cpu()
    STORE.mkdir(parents=True, exist_ok=True)
    torch.save(states, STORE / "word_states.pt")
    print(len(states), "word states ->", STORE / "word_states.pt")


def step_check(args, m):
    data = items()
    lens = Lens(m)
    adapters = add_lora(m, LORA_LAYERS, rank=8, alpha=16, seed=0)
    ex = next(it for it in data["reads"] if it["group"] == "train")
    print("example:", ex["text"], "->", ex["truth"])
    base_src = source_state(m, ex["text"]).float()
    base_req = m.read_prompt(REQUEST).hidden
    for p in adapters.parameters():  # B away from zero, so the adapters do something
        if p.shape[1] == 8:
            torch.nn.init.normal_(p, std=1e-3)
    on_src = source_state(m, ex["text"], adapters).float()
    off_src = source_state(m, ex["text"]).float()
    off_req = m.read_prompt(REQUEST).hidden
    print(f"source state at layer {LAYER}: |on - base| / |base| = {float((on_src - base_src).norm() / base_src.norm()):.4f}, "
          f"|off - base| = {float((off_src - base_src).norm()):.2e}")
    print("request without injection, adapters off, largest change over layers:",
          max(float((a - b).abs().max()) for a, b in zip(base_req, off_req)))

    # Gradient against a central difference along the gradient of the adapter weights (V loss, fixed noise). The latent
    # is the first target picture of the targets step (HunyuanImage, which has no from_picture: its kept final latent).
    STORE.mkdir(parents=True, exist_ok=True)
    if MODEL == "hunyuan":
        z = torch.load(sorted((results.folder(SETS["targets"]) / MODEL).glob("*.pt"))[0])[0]
    else:
        z = m.from_picture(__import__("PIL.Image", fromlist=["Image"]).open(
            sorted((results.folder(SETS["targets"]) / MODEL).glob("*.png"))[0])).cpu()
    ctx = dict(latents={ex["truth"]: z[None]}, rng=random.Random(0), torch_rng=torch.Generator().manual_seed(0))
    if MODEL == "hunyuan":   # arm T: the frozen model's own greedy reply next to the teacher-forced target
        q = ex["questions"]["name"]
        reply = m.answer_one_word(q, HUNYUAN_SYSTEM, require_word=True)
        tk = m.tokenizer
        print(f"arm T: question {q!r}; frozen greedy reply {reply!r}, tokens {tk.convert_ids_to_tokens(m._gen)}; target "
              f"tokens {tk.convert_ids_to_tokens(tk.encode(ex['truth'], add_special_tokens=False))} + ['</answer>']; "
              f"frozen loss {float(m.reply_loss(q, ex['truth'], HUNYUAN_SYSTEM)):.4f}", flush=True)

    def fixed_loss(arm):
        ctx["rng"], ctx["torch_rng"] = random.Random(0), torch.Generator().manual_seed(0)
        return LOSSES[arm](m, lens, ex, ctx, adapters)

    params = adapters.parameters()
    for arm in ("V", "T") if MODEL == "hunyuan" else ("V",):
        loss = fixed_loss(arm)
        loss.backward()
        # Along the gradient itself: a random direction gives a derivative far below the bf16 noise of the loss.
        # a plain number: HunyuanImage's adapters sit on several GPUs, so their gradients cannot be summed as tensors
        norm = sum(float((p.grad.float() ** 2).sum()) for p in params) ** 0.5
        direction = [p.grad / norm for p in params]
        analytic = float(norm)
        print(f"{arm} loss {float(loss):.4f}; gradient norm {analytic:.3e}")
        for eps in [float(e) for e in args.eps.split(",")]:
            with torch.no_grad():
                for p, d in zip(params, direction):
                    p.add_(eps * d)
                plus = float(fixed_loss(arm))
                for p, d in zip(params, direction):
                    p.add_(-2 * eps * d)
                minus = float(fixed_loss(arm))
                for p, d in zip(params, direction):
                    p.add_(eps * d)
            print(f"  eps {eps:g}: finite difference {(plus - minus) / (2 * eps):.4e}, gradient {analytic:.4e}", flush=True)
        for p in params:
            p.grad = None

    # Time per update (16 examples) for each arm.
    torch.save({ex["truth"]: z[None]}, STORE / "check_latents.pt")
    states = {ex["truth"]: torch.randn(lens.mean.shape[-1])}   # the hidden size (HunyuanImage has no d_model)
    for arm in ("V", "T") if MODEL == "hunyuan" else ("T", "V", "R", "G"):
        ctx = dict(rng=random.Random(0), torch_rng=torch.Generator().manual_seed(0), latents={ex["truth"]: z[None]},
                   word_dirs=states, wrong={ex["truth"]: ex["truth"]})
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(4):
            (LOSSES[arm](m, lens, ex, ctx, adapters) / 16).backward()
        torch.cuda.synchronize()
        print(f"arm {arm}: {(time.time() - t0) / 4:.2f} s per example, {(time.time() - t0) / 4 * 16 * 600 / 3600:.1f} h "
              f"for 600 updates of 16; peak memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        for p in params:
            p.grad = None


def step_train(args, m):
    data = items()
    lens = Lens(m)
    adapters = add_lora(m, LORA_LAYERS, rank=args.rank, alpha=args.alpha, seed=args.seed)
    params = adapters.parameters()
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    train = approved(data)
    order = random.Random(1000 + args.seed)
    ctx = context(m, lens, data, args.arm, args.seed)
    if args.arm in ("V", "S"):
        need = {ex["truth"] for ex in train} if args.arm == "V" else {data["wrong"][ex["truth"]] for ex in train}
        missing = need - set(ctx["latents"])
        train = [ex for ex in train if (ex["truth"] if args.arm == "V" else data["wrong"][ex["truth"]]) in ctx["latents"]]
        print("nouns without verified target pictures (their texts are left out):", sorted(missing))
    out = STORE / f"{args.arm}_s{args.seed}.pt"
    history = []
    t0 = time.time()
    for update in range(args.updates):
        batch = [train[order.randrange(len(train))] for _ in range(args.batch)]
        total = 0.0
        for ex in batch:
            loss = LOSSES[args.arm](m, lens, ex, ctx, adapters) / args.batch
            loss.backward()
            total += float(loss)
        opt.step()
        opt.zero_grad()
        history.append(total)
        if update % 25 == 0 or update == args.updates - 1:
            print(f"[{time.time() - t0:6.0f}s] update {update}: loss {total:.4f}", flush=True)
        if update in (args.updates // 2 - 1, args.updates - 1):
            STORE.mkdir(parents=True, exist_ok=True)
            torch.save(dict(adapters=adapters.state_dict(), arm=args.arm, seed=args.seed, update=update + 1,
                            history=history, rank=args.rank, alpha=args.alpha, lr=args.lr, batch=args.batch,
                            texts=len(train)), out if update == args.updates - 1 else out.with_suffix(".half.pt"))
    print("->", out)


def with_adapters(m, arm, seed):
    """The adapters of a trained arm (or none for arm 0), switched off until used."""
    if arm == "0":
        return None
    state = torch.load(STORE / f"{arm}_s{seed}.pt")
    adapters = add_lora(m, LORA_LAYERS, rank=state["rank"], alpha=state["alpha"])
    adapters.load_state_dict(state["adapters"])
    return adapters


def step_draw(args, m):
    data = items()
    lens = Lens(m)
    adapters = with_adapters(m, args.arm, args.seed)
    name = f"{MODEL}_{args.arm}" + (f"_s{args.seed}" if args.arm != "0" else "")
    out = results.folder(SETS["cells"]) / name
    out.mkdir(parents=True, exist_ok=True)
    groups = args.groups.split(",")
    records = []
    t0 = time.time()
    todo = [(i, it) for i, it in enumerate(data["reads"]) if it["group"] in groups]
    tag = args.groups.replace(",", "-") + ("" if args.of == 1 else f"_{args.shard}of{args.of}")
    for i, it in todo[args.shard::args.of]:
        h = source_state(m, it["text"], adapters)
        reading = lens.request(m, h)
        for seed in map(int, args.draw_seeds.split(",")):
            file = f"vsup_{it['id']}_L{LAYER}_s{seed}.png"
            if not (out / file).exists():
                m.to_picture(m.draw(reading, seed=seed)[0]).save(out / file)
            records.append(dict(file=f"{name}/{file}", model=MODEL, arm=args.arm, train_seed=args.seed,
                                items="vsup_items", kind="reads", item=i, text=it["text"], position="it", word="it",
                                layer=LAYER, way="copy:diag", seed=seed, truth=it["truth"], group=it["group"],
                                unit=it["unit"]))
        if len(records) % 60 == 0:
            print(f"[{time.time() - t0:6.0f}s] {len(records)} pictures", flush=True)
            (out / f"records_{tag}.json").write_text(json.dumps(records, indent=1))
    (out / f"records_{tag}.json").write_text(json.dumps(records, indent=1))
    print("->", out)


def step_text(args, m):
    screen = results.load_script("screen_answers")
    data = items()
    adapters = with_adapters(m, args.arm, args.seed)
    groups = args.groups.split(",")
    forms = args.forms.split(",")   # all four by default; the training-text screen needs only "name"
    rows = []
    t0 = time.time()
    for it in data["reads"]:
        if it["group"] not in groups:
            continue
        q = it["questions"]
        with on(adapters):
            if MODEL == "hunyuan":
                reply = {k: m.answer_one_word(q[k], HUNYUAN_SYSTEM, require_word=True) for k in forms}
            else:
                reply = {k: m.answer(q[k], max_new_tokens=12) for k in forms}
        names = [it["truth"]] + screen.ALIASES.get(it["truth"], [])
        alt = [it["alt"]] + screen.ALIASES.get(it["alt"], [])
        right = dict(name=lambda r: screen.correct(r, names) and not screen.correct(r, alt),
                     choice=lambda r: screen.correct(r, names) and not screen.correct(r, alt),
                     yes=lambda r: r.lower().startswith("yes"), no=lambda r: r.lower().startswith("no"))
        rows.append(dict(id=it["id"], group=it["group"], unit=it["unit"], truth=it["truth"], reply=reply,
                         **{k: right[k](reply[k]) for k in forms}))
        if len(rows) % 100 == 0:
            print(f"[{time.time() - t0:6.0f}s] {len(rows)} texts", flush=True)
    general = []
    for question, answer in data["general_qa"]:
        with on(adapters):
            reply = (m.answer_one_word(question, HUNYUAN_SYSTEM, require_word=True) if MODEL == "hunyuan"
                     else m.answer(question + " Answer with one word.", max_new_tokens=12))
        digit = {w: d for d, w in screen.NUMBERS.items()}.get(answer)  # "7" counts for "seven"
        general.append(dict(question=question, reply=reply, right=screen.correct(reply, [answer] + [digit] * bool(digit))))
    summary = {g: {k: sum(r[k] for r in rows if r["group"] == g) / max(1, sum(r["group"] == g for r in rows))
                   for k in forms} for g in groups}
    summary["general"] = sum(r["right"] for r in general) / len(general)
    print(json.dumps(summary, indent=1))
    PILOT.mkdir(parents=True, exist_ok=True)
    tag = args.arm + (f"_s{args.seed}" if args.arm != "0" else "") + (f"_{args.tag}" if args.tag else "")
    (PILOT / f"{TEXT}{tag}.json").write_text(json.dumps(dict(summary=summary, rows=rows, general=general), indent=1))


def step_generate(args, m):
    """Ordinary image generation, no lens: the fine-tuned model (adapters on throughout) draws the text itself followed by
    "A photo of it." (drawing seeds 0 and 1). Does training on "it" change what the model draws when asked directly?
    Judged like the lens pictures (main subject: the referent, the other thing, both, neither)."""
    data = items()
    adapters = with_adapters(m, args.arm, args.seed)
    name = f"{MODEL}_{args.arm}" + (f"_s{args.seed}" if args.arm != "0" else "")
    out = results.folder(SETS["gen"]) / name
    out.mkdir(parents=True, exist_ok=True)
    groups = args.groups.split(",")
    tag = args.groups.replace(",", "-") + ("" if args.of == 1 else f"_{args.shard}of{args.of}")
    records = []
    t0 = time.time()
    todo = [(i, it) for i, it in enumerate(data["reads"]) if it["group"] in groups]
    for i, it in todo[args.shard::args.of]:
        prompt = it["text"] + GEN_TAIL
        with on(adapters):
            reading = m.read_prompt(prompt)
            for seed in (0, 1):
                file = f"vsupgen_{it['id']}_s{seed}.png"
                if not (out / file).exists():
                    m.to_picture(m.draw(reading, seed=seed)[0]).save(out / file)
                records.append(dict(file=f"{name}/{file}", model=MODEL, arm=args.arm, train_seed=args.seed,
                                    items="vsup_items", kind="reads", item=i, text=prompt, position="prompt", word="it",
                                    layer=None, way="generate", seed=seed, truth=it["truth"], group=it["group"],
                                    unit=it["unit"]))
        if len(records) % 40 == 0:
            print(f"[{time.time() - t0:6.0f}s] {len(records)} pictures", flush=True)
            (out / f"records_{tag}.json").write_text(json.dumps(records, indent=1))
    (out / f"records_{tag}.json").write_text(json.dumps(records, indent=1))
    print("->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["targets", "latents", "states", "check", "train", "draw", "text", "generate"])
    ap.add_argument("--model", default="bagel", choices=sorted(MODELS))
    ap.add_argument("--arm", default="0", choices=ARMS)
    ap.add_argument("--seed", type=int, default=0, help="training seed")
    ap.add_argument("--groups", default="dev,test1,test2,test3")
    ap.add_argument("--draw-seeds", default="0,1,2")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--forms", default="name,choice,yes,no", help="text: the question forms asked")
    ap.add_argument("--tag", default="", help="text: suffix of the output file (a separate screen)")
    ap.add_argument("--updates", type=int, default=600, help="600 for BAGEL and SenseNova-U1, 200 for HunyuanImage")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--eps", default="0.3,1,3", help="check: step sizes of the finite difference along the gradient")
    args = ap.parse_args()
    configure(args.model)
    # HunyuanImage's latents step only reads the latents the targets step kept: no model, runs on a CPU
    m = None if (args.step, args.model) == ("latents", "hunyuan") else load(args.model)
    dict(targets=step_targets, latents=step_latents, states=step_states, check=step_check, train=step_train,
         draw=step_draw, text=step_text, generate=step_generate)[args.step](args, m)


if __name__ == "__main__":
    main()
