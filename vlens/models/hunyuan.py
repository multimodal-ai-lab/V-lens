"""HunyuanImage-3.0-Instruct (Tencent, 80B mixture of experts, 13B active, 32 decoder layers, width 4096) through the same
interface as the other models, so scripts/calibrate_insertion_layer.py and the lens scripts run on it unchanged. Own
environment (requirements/hunyuan.txt), four 80 GB GPUs (device_map="auto").

How the model draws (its own pipeline, hunyuan_image_3_pipeline.py): at the first denoising step the transformer runs
over the whole sequence (system prompt, user prompt, image tokens) and caches it; every later step runs the image tokens
against that cache. Classifier-free guidance doubles the batch: a conditional row with the prompt and an unconditional
row without it. So a hidden state is changed by a pre-hook on decoder layer l during that first full pass, in the
conditional row only (the row whose tokens match the prompt's own reading); later steps read the changed cache.

- read(text): the text as a user message (gen_text, its chat template and system prompt), hidden states entering every
  layer from the prefill pass of a one-token generation.
- read_prompt(text, edit): the drawing request as the drawing itself reads it, from the first full pass of a one-step
  drawing (cached per text); with `edit`, the same reading carrying the edit for draw().
- draw(reading, size, seed): generate_image(bot_task="image", direct drawing without thinking or rewriting), the model's
  own guidance settings, STEPS denoising steps (set through generation_config.diff_infer_steps,
  the only place the model takes it from); returns [picture tensor HxWx3, uint8]. The final latents (in the model's
  own scale) are kept in `last_latents`, as training targets.
- read(text, keep_grad=True, upto=l) and denoise_loss(reading, x0, level, noise): for training through the lens
  (scripts/state_tuning.py). The read keeps the graph and stops at layer l; the loss is the flow-matching loss of
  the first denoising pass on x0 noised to `level`, for the conditional row carrying the reading's edit.
- reply_loss(text, reply, system): the cross-entropy of a given one-word answer in the model's own answer format, with
  the graph kept (text supervision, scripts/lens_tuning.py arm T).
"""

from __future__ import annotations

import functools
import re

import numpy as np
import torch
from PIL import Image

from vlens import paths
from vlens.models.base import TextReading

STEPS = 50          # the model's own default; generate_image ignores a step argument, the count is generation_config's
SIZE = 512


class _NotFirst:
    """A logits processor that bars some tokens at the first generated position only."""

    def __init__(self, token_ids):
        self.token_ids, self.first = torch.as_tensor(list(token_ids)), True

    def __call__(self, input_ids, scores):
        if self.first:
            scores[:, self.token_ids.to(scores.device)] = float("-inf")
            self.first = False
        return scores


class _Stop(Exception):
    """Ends a pass once a training step has what it needs (the state at one layer, or the loss)."""


class HunyuanImageModel:
    name = "HunyuanImage-3.0"

    def __init__(self, steps: int = STEPS, size: int = SIZE):
        from transformers import AutoModelForCausalLM
        # the downloaded checkpoint (README.md): the model's remote code fails on a folder name with dots
        path = str(paths.MODELS / "HunyuanImage3Instruct")
        # VLENS_GPU_GIB: optionally cap the weights per GPU so that "auto" spreads them over more GPUs and leaves room
        # for training activations (smaller GPUs); unset, the loading is unchanged
        import os
        import torch
        cap = os.environ.get("VLENS_GPU_GIB")
        mem = {i: f"{cap}GiB" for i in range(torch.cuda.device_count())} if cap else None
        self.net = AutoModelForCausalLM.from_pretrained(path, attn_implementation="sdpa", trust_remote_code=True,
                                                        torch_dtype="auto", device_map="auto", moe_impl="eager",
                                                        moe_drop_tokens=True, max_memory=mem)
        if not hasattr(self.net.config, "model_version"):  # the released config lacks it; load_tokenizer reads it
            self.net.config.model_version = None
        self.net.load_tokenizer(path)
        self.net.eval()
        for p in self.net.parameters():   # frozen; only adapters added for training take gradients
            p.requires_grad_(False)
        self.tokenizer = self.net._tokenizer
        self.layers = self.net.model.layers
        self.n_layers = len(self.layers)
        self.steps, self.size = steps, size
        self.device = next(self.net.parameters()).device
        self._armed, self._ids, self._mode = False, None, None   # what the hooks do during the next full pass
        self._capture, self._edit, self._match, self._rows = None, None, None, []
        self._prompts = {}
        self.net.register_forward_pre_hook(self._top, with_kwargs=True)
        for l, layer in enumerate(self.layers):
            layer.register_forward_pre_hook(self._pre(l), with_kwargs=True)
        self.layers[-1].register_forward_hook(self._last)   # the last layer's output, for the text lenses
        self._last_out = None

    # ------------------------------------------------------------------ hooks
    def _top(self, module, args, kwargs):
        ids = kwargs.get("input_ids")
        if self._armed and self._mode == "follow" and ids is not None and ids.shape[1] == 1 and self._ids is not None:
            self._gen.append(int(ids[0, 0]))          # a generated token, read back as the next step's input
            return None
        if self._armed and ids is not None and ids.shape[1] > 1:
            self._ids = ids.detach().cpu()
            if self._match is None:
                self._rows = [0]
            else:
                self._rows = [b for b in range(self._ids.shape[0]) if self._ids.shape[1] >= len(self._match)
                              and self._ids[b, :len(self._match)].tolist() == self._match]
        return None

    def _pre(self, l):
        def hook(module, args, kwargs):
            if not self._armed or self._ids is None:
                return None
            h = args[0] if args else kwargs["hidden_states"]
            if self._mode == "follow":                # every pass of a reply: the prompt, then one token at a time
                row = h[self._rows[0]].detach().float().cpu()
                if h.shape[1] == self._ids.shape[1]:
                    self._capture[l] = row
                elif h.shape[1] == 1:
                    self._steps[l].append(row)
                return None
            if h.shape[1] != self._ids.shape[1]:
                return None
            if self._mode == "grab":
                if l == self._grab_at:
                    self._capture[l] = h[self._rows[0]]   # with its graph, on this layer's GPU
                    self._armed = False
                    raise _Stop
                return None
            if self._mode == "capture":
                self._capture[l] = h[self._rows[0]].detach().float().cpu()
            elif self._mode == "edit" and self._edit is not None and self._rows:
                h = h.clone()
                for b in self._rows:
                    h[b] = self._edit(l, h[b])
                if args:
                    args = (h,) + tuple(args[1:])
                else:
                    kwargs = dict(kwargs, hidden_states=h)
            if l == self.n_layers - 1:
                self._armed = False   # only the first full pass
            return args, kwargs
        return hook

    def _last(self, module, args, out):
        """The last decoder layer's output (before the final norm) of the prefill pass, in capture mode."""
        if self._mode == "capture" and self._ids is not None:
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] == self._ids.shape[1]:
                self._last_out = h[self._rows[0]].detach().float().cpu()
        return None

    def logit_lens(self, hidden: torch.Tensor) -> torch.Tensor:
        """Next-word scores read straight off a hidden state (final norm + text head)."""
        with torch.no_grad():
            head = self.net.lm_head
            h = hidden.to(head.weight.device, head.weight.dtype)
            return head(self.net.model.ln_f(h)).float()

    def _arm(self, mode, match=None, edit=None):
        self._armed, self._ids, self._mode, self._match, self._edit = True, None, mode, match, edit
        self._gen, self._steps = [], [[] for _ in range(self.n_layers)]
        self._capture = [None] * self.n_layers
        self._rows = []

    def _reading(self, text, ids, extra_cache=None):
        strs = [self.tokenizer.decode([i]) for i in ids]
        full = "".join(strs)
        start = full.rfind(text)
        content = torch.zeros(len(ids), dtype=torch.bool)
        if start >= 0:
            pos = 0
            for p, s in enumerate(strs):
                if pos < start + len(text) and pos + len(s) > start and s.strip():
                    content[p] = True
                pos += len(s)
        return TextReading(text=text, token_ids=list(ids), token_strs=strs, hidden=list(self._capture), content=content,
                           cache=dict(edit=None, **(extra_cache or {})), last_output=self._last_out)

    # ------------------------------------------------------------------ reading
    def read(self, text: str, keep_grad: bool = False, system: str | None = None, upto: int | None = None) -> TextReading:
        """`system`: a system prompt in place of the default one (the one-word answers use one). With keep_grad the
        pass keeps its graph and stops at layer `upto` (default the last); only that layer's states are kept."""
        inputs = self.net.prepare_model_inputs(prompt=text, mode="gen_text", bot_task="auto", max_new_tokens=1,
                                               **({} if system is None else dict(system_prompt=system)))
        if not keep_grad:
            with torch.no_grad():
                self._arm("capture")
                self.net.generate(**inputs, decode_text=True, do_sample=False)
                self._armed = False
        else:
            self._arm("grab")
            self._grab_at = self.n_layers - 1 if upto is None else upto
            orig = self.net.forward

            @functools.wraps(orig)   # generate() checks its arguments against the forward's signature
            def forward(*a, **kw):   # generate() runs under no_grad; the pass itself keeps the graph
                with torch.enable_grad():
                    return orig(*a, **kw)
            self.net.forward = forward
            try:
                self.net.generate(**inputs, decode_text=True, do_sample=False)
            except _Stop:
                pass
            finally:
                del self.net.forward
                self._armed = False
        return self._reading(text, self._ids[self._rows[0]].tolist())

    @torch.no_grad()
    def read_prompt(self, text: str, *, edit=None, keep_grad: bool = False) -> TextReading:
        if text not in self._prompts:
            self._arm("capture")
            self._match = None
            self.net.generation_config.diff_infer_steps = 1   # only the first full pass is read
            self.net.generate_image(prompt=text, seed=0, image_size=f"{self.size}x{self.size}", bot_task="image",
                                    verbose=0)
            self._armed = False
            # the conditional row is the one whose tokens spell the prompt
            rows = [b for b in range(self._ids.shape[0]) if text in self.tokenizer.decode(self._ids[b].tolist())]
            if not rows:
                raise RuntimeError(f"no row of the first pass contains {text!r}")
            self._rows = rows
            # read the conditional row again (the capture above took row 0, which may be the unconditional one)
            if rows[0] != 0:
                self._arm("capture")
                self._match = self._ids[rows[0]].tolist()
                self.net.generation_config.diff_infer_steps = 1
                self.net.generate_image(prompt=text, seed=0, image_size=f"{self.size}x{self.size}", bot_task="image",
                                        verbose=0)
                self._armed = False
            self._prompts[text] = self._reading(text, self._ids[rows[0]].tolist())
        base = self._prompts[text]
        return TextReading(text=base.text, token_ids=base.token_ids, token_strs=base.token_strs, hidden=base.hidden,
                           content=base.content, cache=dict(edit=edit))

    def read_chat(self, text: str, *, edit=None) -> TextReading:
        return self.read(text)

    # ------------------------------------------------------------------ drawing
    @torch.no_grad()
    def draw(self, reading: TextReading, size: int | None = None, seed: int = 0, **kw):
        size = size or self.size
        self._arm("edit", match=reading.token_ids, edit=reading.cache.get("edit"))
        self.net.generation_config.diff_infer_steps = self.steps
        vae, latents = self.net.vae, []
        decode = vae.decode

        def keep(z, *a, **kw):   # the pipeline divides by the scaling factor just before decoding
            latents.append(z.detach())
            return decode(z, *a, **kw)
        vae.decode = keep
        try:
            _, samples = self.net.generate_image(prompt=reading.text, seed=seed, image_size=f"{size}x{size}",
                                                 bot_task="image", verbose=0)
        finally:
            del vae.decode
        self._armed = False
        if reading.cache.get("edit") is not None and not self._rows:
            raise RuntimeError("the edit found no conditional row to apply to")
        z = latents[0].float()
        z = z.squeeze(2) if z.dim() == 5 else z
        if getattr(vae.config, "shift_factor", None):
            z = z - vae.config.shift_factor
        if getattr(vae.config, "scaling_factor", None):
            z = z * vae.config.scaling_factor
        self.last_latents = z[:1].cpu()
        picture = samples[0] if isinstance(samples, (list, tuple)) else samples
        return [torch.from_numpy(np.asarray(picture.convert("RGB")).copy())]

    def denoise_loss(self, reading: TextReading, x0: torch.Tensor, level: float, noise: torch.Tensor) -> torch.Tensor:
        """Flow-matching loss of one denoising pass, as the pipeline's scheduler defines the flow (reverse=True, Euler):
        x_t = (1 - level) x0 + level noise at timestep level * 1000, target noise - x0. A one-step generate_image whose
        first full pass gets x_t in both guidance rows; the conditional row (the one carrying the reading's edit) gives
        the loss, the pass keeps its graph and stops there."""
        x_t = (1 - level) * x0.float() + level * noise.float()
        target = noise.float() - x0.float()
        orig = self.net.forward

        @functools.wraps(orig)
        def forward(*a, **kw):
            if kw.get("images") is None or not kw.get("first_step"):
                return orig(*a, **kw)
            images = kw["images"]
            kw["images"] = x_t.to(images.device, images.dtype).reshape(1, *images.shape[1:]).expand_as(images).contiguous()
            kw["timesteps"] = torch.full_like(kw["timesteps"], level * 1000.0)
            with torch.enable_grad():
                pred = orig(*a, **kw)["diffusion_prediction"]
                self._loss = ((pred[self._rows[0]].float() - target.to(pred.device).reshape(pred.shape[1:])) ** 2).mean()
            raise _Stop
        self._loss = None
        self._arm("edit", match=reading.token_ids, edit=reading.cache.get("edit"))
        self.net.generation_config.diff_infer_steps = 1
        self.net.forward = forward
        try:
            self.net.generate_image(prompt=reading.text, seed=0, image_size=f"{self.size}x{self.size}", bot_task="image",
                                    verbose=0)
        except _Stop:
            pass
        finally:
            del self.net.forward
            self._armed = False
        if self._loss is None or (reading.cache.get("edit") is not None and not self._rows):
            raise RuntimeError("the denoising pass found no conditional row")
        return self._loss

    def _follow(self, text, system, max_new_tokens=8, require_word=False):
        """A greedy reply in the one-word format of the state questions (`system`, image and thinking tokens
        suppressed), with the hidden states of the prompt and of every reply token as it is read back. Returns the
        reply (answer tags removed) and a TextReading over the prompt plus the reply tokens whose states exist."""
        from transformers import LogitsProcessorList, SuppressTokensLogitsProcessor
        tk = self.tokenizer
        banned = [tk.convert_tokens_to_ids(tk.boi_token), tk.convert_tokens_to_ids(tk.think_token)]
        procs = [SuppressTokensLogitsProcessor(banned, device=self.device)]
        if require_word:   # the reply must start with a word: no closing tag and no token without a letter or digit
            procs.append(_NotFirst(self._no_word_ids()))
        inputs = self.net.prepare_model_inputs(prompt=text, mode="gen_text", bot_task="auto",
                                               max_new_tokens=max_new_tokens, system_prompt=system)
        self._arm("follow")
        self._gen, self._steps = [], [[] for _ in range(self.n_layers)]
        try:
            with torch.no_grad():
                out = self.net.generate(**inputs, decode_text=True, do_sample=False,
                                        logits_processor=LogitsProcessorList(procs))
        finally:
            self._armed = False
        reply = out[0] if isinstance(out, list) else out
        for t in ("<answer>", "</answer>", "<|endoftext|>"):
            reply = reply.replace(t, "")
        if len(self._gen) != len(self._steps[0]):
            raise RuntimeError(f"{len(self._gen)} reply tokens but {len(self._steps[0])} captured states")
        ids = self._ids[self._rows[0]].tolist() + self._gen
        hidden = [torch.cat([self._capture[l]] + self._steps[l]) if self._steps[l] else self._capture[l]
                  for l in range(self.n_layers)]
        strs = [tk.decode([i]) for i in ids]
        n_prompt = self._ids.shape[1]
        content = torch.zeros(len(ids), dtype=torch.bool)
        specials = {"<answer>", "</answer>", "<|endoftext|>"}
        for q in range(n_prompt, len(ids)):
            content[q] = bool(strs[q].strip()) and strs[q].strip() not in specials
        return reply.strip(), TextReading(text=text, token_ids=ids, token_strs=strs, hidden=hidden, content=content,
                                          cache=dict(edit=None))

    def _no_word_ids(self):
        """Token ids that cannot start a word: the closing answer tag and every token without a letter or digit."""
        if getattr(self, "_no_word", None) is None:
            tk = self.tokenizer
            ids = [i for i, t in enumerate(tk.convert_ids_to_tokens(list(range(len(tk)))))
                   if t is None or not re.search(r"[A-Za-z0-9]", t)]
            self._no_word = sorted(set(ids) | {tk.convert_tokens_to_ids("</answer>")})
        return self._no_word

    @torch.no_grad()
    def reply(self, text: str, system: str, image=None, max_new_tokens: int = 8, require_word: bool = True) -> str:
        """The greedy reply answer_one_word gives (same prompt, same suppressed image and thinking tokens, same
        first-token rule), without recording hidden states, and optionally about a picture (used by
        scripts/benchmarks.py): `image` (a PIL picture) goes to the model's own input preparation
        as a conditioning picture (its image processor, centre crop, VAE and ViT encodings; seed 0 for the VAE sample),
        in gen_text mode. Adapters switched on around this call act on every token, the picture's included."""
        from transformers import LogitsProcessorList, SuppressTokensLogitsProcessor
        tk = self.tokenizer
        banned = [tk.convert_tokens_to_ids(tk.boi_token), tk.convert_tokens_to_ids(tk.think_token)]
        procs = [SuppressTokensLogitsProcessor(banned, device=self.device)]
        if require_word:
            procs.append(_NotFirst(self._no_word_ids()))
        extra = {} if image is None else dict(image=image.convert("RGB"), seed=0)
        inputs = self.net.prepare_model_inputs(prompt=text, mode="gen_text", bot_task="auto",
                                               max_new_tokens=max_new_tokens, system_prompt=system, **extra)
        self._armed = False   # the recording hooks stay idle
        out = self.net.generate(**inputs, decode_text=True, do_sample=False, logits_processor=LogitsProcessorList(procs))
        out = out[0] if isinstance(out, list) else out
        for t in ("<answer>", "</answer>", "<|endoftext|>"):
            out = out.replace(t, "")
        return out.strip()

    def answer_one_word(self, text: str, system: str, max_new_tokens: int = 8, require_word: bool = False) -> str:
        """require_word: on the pronoun questions of scripts/lens_tuning.py the greedy first token is often "</answer>", an
        empty reply, and after picture training often ".". The closing tag and every token without a letter or digit are then barred from the first position only."""
        return self._follow(text, system, max_new_tokens, require_word)[0]

    def reply_loss(self, text: str, reply: str, system: str | None = None) -> torch.Tensor:
        """Mean cross-entropy of `reply` as the answer to `text`, teacher-forced in one pass that keeps its graph (text
        supervision of adapters, scripts/lens_tuning.py arm T). The question is read exactly as answer_one_word
        reads it (gen_text, `system`; the prompt ends with "Assistant: "), and the model writes a one-word answer straight
        after that, closed by its answer tag (for example "belt", "</answer>"). So the reply's own
        tokens and "</answer>" are appended to the prompt, and only they are scored. The pass is the prefill of a
        generate() call (the read's route) and stops once it has the logits."""
        tk = self.tokenizer
        target = tk.encode(reply, add_special_tokens=False) + [tk.convert_tokens_to_ids("</answer>")]
        inputs = self.net.prepare_model_inputs(prompt=text, mode="gen_text", bot_task="auto",
                                               max_new_tokens=len(target) + 1,   # the cache's room covers the reply
                                               **({} if system is None else dict(system_prompt=system)))
        n = inputs["input_ids"].shape[1]
        ids = torch.tensor([target], device=inputs["input_ids"].device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], ids], 1)
        inputs["position_ids"] = torch.arange(n + len(target), device=ids.device)[None]
        inputs["max_new_tokens"] = 1   # as in read(): one step, of which only the prefill pass runs
        orig = self.net.forward

        @functools.wraps(orig)   # generate() checks its arguments against the forward's signature
        def forward(*a, **kw):   # generate() runs under no_grad; the pass and the loss keep the graph
            with torch.enable_grad():
                logits = orig(*a, **kw).logits[0, n - 1:-1]   # position p predicts token p + 1
                self._loss = torch.nn.functional.cross_entropy(logits.float(), ids[0].to(logits.device))
            raise _Stop
        self._loss, self._armed = None, False   # the hooks stay idle
        self.net.forward = forward
        try:
            self.net.generate(**inputs, decode_text=True, do_sample=False)
        except _Stop:
            pass
        finally:
            del self.net.forward
        if self._loss is None:
            raise RuntimeError("the teacher-forced pass gave no loss")
        return self._loss

    def read_reply(self, text: str, reply: str, system: str | None = None, max_new_tokens: int = 8) -> TextReading:
        """The question and the model's own greedy reply, as answer_one_word gives it (the reply is regenerated, so it
        must match); only the reply's tokens are marked as content."""
        again, reading = self._follow(text, system, max_new_tokens)
        if again != reply.strip():
            raise RuntimeError(f"the regenerated reply {again!r} differs from {reply!r}")
        return reading

    def to_picture(self, x) -> Image.Image:
        if isinstance(x, Image.Image):
            return x
        return Image.fromarray(x.cpu().numpy().astype("uint8"))

    @torch.no_grad()
    def answer(self, text: str, max_new_tokens: int = 512, allow: bool = False) -> str:
        """The final answer after the model's reasoning (it thinks first in this mode, up to minutes per answer). Off by
        default (allow=True switches it on); the experiments use answer_one_word() and reply()."""
        if not allow:
            raise NotImplementedError("reasoning answers are off by default; pass allow=True")
        inputs = self.net.prepare_model_inputs(prompt=text, mode="gen_text", bot_task="auto",
                                               max_new_tokens=max_new_tokens)
        out = self.net.generate(**inputs, decode_text=True, do_sample=False)
        out = out[0] if isinstance(out, list) else out
        return out.split("</think>")[-1].split("<answer>")[-1].replace("</answer>", "").replace("<|endoftext|>", "").strip()
