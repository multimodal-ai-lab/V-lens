"""Janus-Pro-7B (deepseek-ai/Janus-Pro-7B; upstream code installed as the package `janus`, see requirements/janus.txt)
behind the common model interface.

How Janus-Pro draws (upstream's generation example): one language model (a Llama of 30 layers, width 4096) reads the
chat-formatted request, "<|User|>: <text>\\n\\n<|Assistant|>:<begin_of_image>", and then writes 576 discrete image codes
(24 x 24, a VQ tokenizer of 16 px patches, 384 px) one after another; each code is predicted by a separate generation head
on the last hidden state and fed back through its own embedding. Guidance: a second row with every request token but the
first and the last replaced by padding, logits mixed as uncond + 5 * (cond - uncond), temperature 1. So, as in Liquid,
the request is read once into a key/value cache that every code reads; an edit (the V-lens slot) is applied by a pre-hook
on the decoder layers during the first pass of the conditional row only.

- read(text): the text as a user turn in the chat format (as answer() reads it); hidden states entering every layer.
- read_prompt(text, edit): the drawing request as the drawing reads it; with `edit`, the same reading carrying the edit.
- draw(reading, seed): 576 codes as above, decoded; returns [picture tensor HxWx3, uint8].
- answer(text): a greedy reply in the chat format.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from vlens import paths
from vlens.models.base import TextReading

REPO = "deepseek-ai/Janus-Pro-7B"
CFG, TEMPERATURE, N_CODES, SIDE = 5.0, 1.0, 576, 384


def _snapshot():
    return paths.snapshot(REPO)


class JanusProModel:
    name = "Janus-Pro-7B"

    def __init__(self):
        from janus.models import MultiModalityCausalLM, VLChatProcessor
        path = str(_snapshot())
        self.proc = VLChatProcessor.from_pretrained(path)
        self.tokenizer = self.proc.tokenizer
        self.net = MultiModalityCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
        self.lm = self.net.language_model
        self.layers = self.lm.model.layers
        self.n_layers = len(self.layers)
        self.device = torch.device("cuda")
        self._mode, self._capture, self._edit = None, None, None
        for l, layer in enumerate(self.layers):
            layer.register_forward_pre_hook(self._pre(l), with_kwargs=True)

    # ------------------------------------------------------------------ hooks (row 0 only)
    def _pre(self, l):
        def hook(module, args, kwargs):
            if self._mode is None:
                return None
            h = args[0] if args else kwargs["hidden_states"]
            if self._mode == "capture":
                self._capture[l] = h[0].detach().float().cpu()
                return None
            if self._mode == "edit" and self._edit is not None:
                h = h.clone()
                h[0] = self._edit(l, h[0])
                if args:
                    return (h,) + tuple(args[1:]), kwargs
                return args, dict(kwargs, hidden_states=h)
            return None
        return hook

    def _format(self, text, gen):
        conv = [{"role": "<|User|>", "content": text}, {"role": "<|Assistant|>", "content": ""}]
        s = self.proc.apply_sft_template_for_multi_turn_prompts(conversations=conv, sft_format=self.proc.sft_format,
                                                                 system_prompt="")
        return s + self.proc.image_start_tag if gen else s

    @torch.no_grad()
    def _read(self, text, formatted):
        ids = self.tokenizer.encode(formatted)
        self._mode, self._capture = "capture", [None] * self.n_layers
        try:
            self.lm.model(inputs_embeds=self.lm.get_input_embeddings()(torch.tensor([ids], device=self.device)))
        finally:
            self._mode = None
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
    def read(self, text: str) -> TextReading:
        return self._read(text, self._format(text, False))

    def read_chat(self, text: str, *, edit=None) -> TextReading:
        return self.read(text)

    def read_prompt(self, text: str, *, edit=None, keep_grad: bool = False) -> TextReading:
        if not hasattr(self, "_prompts"):
            self._prompts = {}
        if text not in self._prompts:
            self._prompts[text] = self._read(text, self._format(text, True))
        b = self._prompts[text]
        return TextReading(text=b.text, token_ids=b.token_ids, token_strs=b.token_strs, hidden=b.hidden,
                           content=b.content, cache=dict(edit=edit))

    # ------------------------------------------------------------------ drawing
    @torch.no_grad()
    def draw(self, reading: TextReading, size: int | None = None, seed: int = 0, **kw):
        g = torch.Generator(device=self.device).manual_seed(seed)
        ids = torch.tensor(reading.token_ids, device=self.device)
        unc = ids.clone()
        unc[1:-1] = self.proc.pad_id
        # the conditional and the unconditional row run separately (each batch of one, as the reading does): batched,
        # bf16 sums come out slightly different from the reading's, and one changed code sends the drawing elsewhere
        # (with batched rows, replacing a state by itself changed pixels)
        self._mode, self._edit = "edit", reading.cache.get("edit")
        try:
            oc = self.lm.model(inputs_embeds=self.lm.get_input_embeddings()(ids[None]), use_cache=True)
        finally:
            self._mode, self._edit = None, None
        ou = self.lm.model(inputs_embeds=self.lm.get_input_embeddings()(unc[None]), use_cache=True)
        codes = []
        for i in range(N_CODES):
            if i:
                oc = self.lm.model(inputs_embeds=step, use_cache=True, past_key_values=oc.past_key_values)
                ou = self.lm.model(inputs_embeds=step, use_cache=True, past_key_values=ou.past_key_values)
            lc = self.net.gen_head(oc.last_hidden_state[:, -1, :]).float()[0]
            lu = self.net.gen_head(ou.last_hidden_state[:, -1, :]).float()[0]
            mixed = lu + CFG * (lc - lu)
            nxt = torch.multinomial(torch.softmax(mixed / TEMPERATURE, -1), 1, generator=g)
            codes.append(int(nxt))
            step = self.net.prepare_gen_img_embeds(nxt.view(-1)).unsqueeze(1)
        dec = self.net.gen_vision_model.decode_code(torch.tensor([codes], dtype=torch.int, device=self.device),
                                                    shape=[1, 8, SIDE // 16, SIDE // 16])
        arr = np.clip((dec.float().cpu().numpy()[0].transpose(1, 2, 0) + 1) / 2 * 255, 0, 255).astype("uint8")
        picture = Image.fromarray(arr)
        if size and size != SIDE:
            picture = picture.resize((size, size), Image.BICUBIC)
        return [torch.from_numpy(np.asarray(picture).copy())]

    def to_picture(self, x) -> Image.Image:
        if isinstance(x, Image.Image):
            return x
        return Image.fromarray(x.cpu().numpy().astype("uint8"))

    @torch.no_grad()
    def answer(self, text: str, max_new_tokens: int = 40, seed: int | None = None, temperature: float = 0.7) -> str:
        """Greedy reply; with `seed`, a sampled one (temperature) from that seed."""
        ids = torch.tensor([self.tokenizer.encode(self._format(text, False))], device=self.device)
        sample = dict(do_sample=False) if seed is None else dict(do_sample=True, temperature=temperature)
        if seed is not None:
            torch.manual_seed(seed)
        out = self.lm.generate(inputs_embeds=self.lm.get_input_embeddings()(ids), attention_mask=torch.ones_like(ids),
                               pad_token_id=self.tokenizer.eos_token_id, bos_token_id=self.tokenizer.bos_token_id,
                               eos_token_id=self.tokenizer.eos_token_id, max_new_tokens=max_new_tokens, **sample)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
