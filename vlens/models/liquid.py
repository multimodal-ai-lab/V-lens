"""Liquid-7B (Junfeng5/Liquid_V1_7B), a unified model that draws AUTOREGRESSIVELY: one Gemma-7B transformer (28 layers,
width 3072) reads text and writes 1,024 image tokens one by one, from a vocabulary that is 256,000 text tokens plus
8,192 image codes of the Chameleon VQGAN (32 x 32 codes, decoded to 512 px; code in third_party/chameleon, fetched by
third_party/setup.sh).

How it draws: the request is "<text> Generate an image based on this description.<boi>"; every image token is sampled
from guidance-mixed log-probabilities, uncond + 7 * (cond - uncond), where the unconditional request is
"<unconditional><boi>", then temperature 0.99, top-k 4096, top-p 0.96, restricted to the image codes. The conditional and
unconditional requests keep their own key/value caches, so an edit of the request (the V-lens slot) is applied by a
pre-hook on decoder layer l during the conditional request's first pass only; every image token reads the changed
cache, exactly as the denoising models' image side reads the changed text states.

- read(text): the text as a user turn of Liquid's Gemma chat format, hidden states entering every layer.
- read_prompt(text, edit): the drawing request as the drawing reads it; with `edit`, the same reading carrying the edit.
- draw(reading, seed): 1,024 image tokens as above, decoded; returns [picture tensor HxWx3, uint8].
- answer(text): a greedy reply in the chat format.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from PIL import Image

from vlens import paths
from vlens.models.base import TextReading

MODEL_ID = "Junfeng5/Liquid_V1_7B"
INSTRUCTION = " Generate an image based on this description."
UNCONDITIONAL = "<unconditional>"
TEXT_VOCAB, IMAGE_VOCAB, N_IMAGE_TOKENS = 256_000, 8_192, 1_024
GUIDANCE, TEMPERATURE, TOP_K, TOP_P = 7.0, 0.99, 4096, 0.96
VQGAN = paths.MODELS / "chameleon_vqgan"          # vqgan.ckpt and vqgan.yaml (README.md)
CODE = paths.REPO / "third_party"


class LiquidModel:
    name = "Liquid-7B"

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.net = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                                        attn_implementation="sdpa").cuda().eval()
        if str(CODE) not in sys.path:
            sys.path.insert(0, str(CODE))
        from chameleon.image_tokenizer import ImageTokenizer
        self.vq = ImageTokenizer(cfg_path=str(VQGAN / "vqgan.yaml"), ckpt_path=str(VQGAN / "vqgan.ckpt"), device="cuda")
        self.device = torch.device("cuda")
        self.layers = self.net.model.layers
        self.n_layers = len(self.layers)
        # the image codes start right after the text vocabulary; checked rather than assumed (codes are shifted by
        # len(tokenizer) while sampling is constrained at 256,000, so the two must agree)
        assert len(self.tokenizer) == TEXT_VOCAB and self.net.config.vocab_size == TEXT_VOCAB + IMAGE_VOCAB
        self._mode, self._capture, self._edit = None, None, None
        for l, layer in enumerate(self.layers):
            layer.register_forward_pre_hook(self._pre(l), with_kwargs=True)

    # ------------------------------------------------------------------ hooks
    def _pre(self, l):
        def hook(module, args, kwargs):
            if self._mode is None:
                return None
            h = args[0] if args else kwargs["hidden_states"]
            if self._mode == "capture":
                self._capture[l] = h[0].detach().float().cpu()
                return None
            if self._mode == "edit" and self._edit is not None:
                h = self._edit(l, h[0])[None]
                if args:
                    return (h,) + tuple(args[1:]), kwargs
                return args, dict(kwargs, hidden_states=h)
            return None
        return hook

    def _ids(self, text):
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    @torch.no_grad()
    def _read(self, text, formatted):
        ids = self._ids(formatted)
        self._mode, self._capture = "capture", [None] * self.n_layers
        try:
            self.net(input_ids=ids)
        finally:
            self._mode = None
        ids = ids[0].tolist()
        strs = [self.tokenizer.decode([i]) for i in ids]
        start = "".join(strs).find(text)
        content = torch.zeros(len(ids), dtype=torch.bool)
        if start >= 0:
            pos = 0
            for p, s in enumerate(strs):
                if pos < start + len(text) and pos + len(s) > start and s.strip():
                    content[p] = True
                pos += len(s)
        return TextReading(text=text, token_ids=ids, token_strs=strs, hidden=list(self._capture), content=content,
                           cache=dict(edit=None))

    # ------------------------------------------------------------------ reading
    def chat(self, text):
        return f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"

    def read(self, text: str) -> TextReading:
        return self._read(text, self.chat(text))

    def read_chat(self, text: str, *, edit=None) -> TextReading:
        return self.read(text)

    def read_prompt(self, text: str, *, edit=None, keep_grad: bool = False) -> TextReading:
        key = ("prompt", text)
        if not hasattr(self, "_prompts"):
            self._prompts = {}
        if key not in self._prompts:
            self._prompts[key] = self._read(text, text + INSTRUCTION + "<boi>")
        base = self._prompts[key]
        return TextReading(text=base.text, token_ids=base.token_ids, token_strs=base.token_strs, hidden=base.hidden,
                           content=base.content, cache=dict(edit=edit))

    # ------------------------------------------------------------------ drawing
    @torch.no_grad()
    def draw(self, reading: TextReading, size: int | None = None, seed: int = 0, **kw):
        from transformers import DynamicCache
        g = torch.Generator(device=self.device).manual_seed(seed)
        cond = torch.tensor([reading.token_ids], device=self.device)
        uncond = self._ids(UNCONDITIONAL + "<boi>")
        cc, uc = DynamicCache(), DynamicCache()
        self._mode, self._edit = "edit", reading.cache.get("edit")
        try:
            lc = self.net(input_ids=cond, past_key_values=cc, use_cache=True).logits[:, -1]
        finally:
            self._mode, self._edit = None, None
        lu = self.net(input_ids=uncond, past_key_values=uc, use_cache=True).logits[:, -1]
        codes = []
        for _ in range(N_IMAGE_TOKENS):
            c = torch.log_softmax(lc.float(), -1)
            u = torch.log_softmax(lu.float(), -1)
            s = (u + GUIDANCE * (c - u))[:, TEXT_VOCAB:TEXT_VOCAB + IMAGE_VOCAB] / TEMPERATURE
            kth = torch.topk(s, TOP_K, dim=-1).values[:, -1:]
            s = s.masked_fill(s < kth, float("-inf"))
            p = torch.softmax(s, -1)
            sp, si = torch.sort(p, descending=True)
            drop = sp.cumsum(-1) - sp > TOP_P
            sp = sp.masked_fill(drop, 0)
            code = si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=g))
            codes.append(int(code))
            tok = code + TEXT_VOCAB
            lc = self.net(input_ids=tok, past_key_values=cc, use_cache=True).logits[:, -1]
            lu = self.net(input_ids=tok, past_key_values=uc, use_cache=True).logits[:, -1]
        picture = self.vq.pil_from_img_toks(torch.tensor(codes, device=self.device))
        if size and picture.size[0] != size:
            picture = picture.resize((size, size), Image.BICUBIC)
        return [torch.from_numpy(np.asarray(picture.convert("RGB")).copy())]

    def to_picture(self, x) -> Image.Image:
        if isinstance(x, Image.Image):
            return x
        return Image.fromarray(x.cpu().numpy().astype("uint8"))

    @torch.no_grad()
    def answer(self, text: str, max_new_tokens: int = 40) -> str:
        ids = self._ids(self.chat(text))
        out = self.net.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False,
                                eos_token_id=[self.tokenizer.eos_token_id,
                                              self.tokenizer.convert_tokens_to_ids("<end_of_turn>")])
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
