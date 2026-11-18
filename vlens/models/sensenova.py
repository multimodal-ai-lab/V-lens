"""SenseNova-U1-8B-MoT behind the common model interface. Runs in its own environment (it needs transformers >= 4.57;
see requirements/sensenova.txt).

How SenseNova-U1 draws (the parts this code relies on; built like BAGEL):

- Every one of the 42 layers holds two copies of the weights: one for text tokens, one for image tokens. The text is
  read once by the text copy and fills a key/value cache. Each layer's keys and values come from the hidden state
  *entering* that layer.
- The picture is drawn straight in pixels (no VAE): 32 x 32 pixel patches, one image token each (512 px = 256 tokens).
  At every step the image tokens pass through the image copy of every layer and read the text's cache there.
- A drawing prompt is a chat message with a fixed system message of about 240 tokens about image generation:
      <|im_start|>system ... <|im_end|> <|im_start|>user {prompt}<|im_end|> <|im_start|>assistant <think></think> <img>
  The empty prompt used for guidance is a short chat with no system message.
- Time runs from 0 (noise) to 1 (picture). The head predicts the finished picture x0, turned into a velocity
  v = (x0 - z) / (1 - t). The noise level used here is 1 - t. Guidance is the plain mix v_empty + g (v_prompt - v_empty).

Two kinds of reading:
- read(text): a plain text, read as the start of a chat message ("<|im_start|>user\\n" + text). The text's own tokens
  are the content. Used for the texts whose states are read and for the lens statistics.
- read_prompt(text): a drawing prompt in the full drawing format. draw() needs this kind.

Without the system message (system=False, or vlens.models.no_system_message(m); not used for the paper's numbers):
read_prompt() then reads
      <|im_start|>user {prompt}<|im_end|> <|im_start|>assistant <think></think> <img>
i.e. upstream's format with the system message left out (as upstream's own empty prompt does). The empty prompt used for
guidance is the same either way.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from vlens import paths
from vlens.models.base import Edit, TextReading

REPOS = {"U1": "sensenova/SenseNova-U1-8B-MoT"}

# Upstream's command-line defaults (examples/t2i/inference.py), except the size: upstream draws at 2048 px.
DRAW_DEFAULTS = dict(size=512, steps=50, shift=3.0, guidance=4.0, guidance_interval=(0.0, 1.0))
PLAIN_START = "<|im_start|>user\n"


def snapshot(version: str) -> Path:
    return paths.snapshot(REPOS[version])


class SenseNovaModel:
    def __init__(self, version: str = "U1", device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 system: bool = True):
        import sensenova_u1
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        sensenova_u1.set_attn_backend("sdpa")
        path = snapshot(version)
        self.name = f"sensenova_{version.lower()}"
        self.device = torch.device(device)
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.net = AutoModel.from_pretrained(path, config=AutoConfig.from_pretrained(path), dtype=dtype).eval()
        self.net.to(self.device)
        self.net.requires_grad_(False)
        self.net.config.t_eps = 0.02  # what upstream's t2i_generate sets
        self.lm = self.net.language_model
        self.layers = self.lm.model.layers
        self.n_layers = len(self.layers)
        self.d_model = self.lm.config.hidden_size
        self.text_vocab = self.lm.config.vocab_size
        self._empty: Optional[TextReading] = None
        self.system_message = system  # read_prompt() with upstream's generation system message (default) or without

    # ------------------------------------------------------------------ the text side

    def _read(self, query: str, text: str, keep_grad: bool, edit: Optional[Edit]) -> TextReading:
        enc = self.tokenizer(query, return_offsets_mapping=True)
        start = query.rfind(text) if text else len(query)
        content = torch.tensor([text != "" and start <= a < start + len(text) for a, _ in enc["offset_mapping"]])
        ids, indexes, mask = self.net._build_t2i_text_inputs(self.tokenizer, query)
        hidden: list[Optional[torch.Tensor]] = [None] * self.n_layers
        last: list[Optional[torch.Tensor]] = [None]

        def make(layer_index: int):
            def hook(module, args, kwargs):
                if kwargs.get("exist_image_gen_tokens"):
                    return None  # an image pass: never touched
                row = args[0][0]
                if edit is not None:
                    row = edit(layer_index, row)
                hidden[layer_index] = row
                return (row[None], *args[1:]), kwargs
            return hook

        def keep_last(module, args, kwargs, output):
            last[0] = (output[0] if isinstance(output, tuple) else output)[0]

        handles = [layer.register_forward_pre_hook(make(i), with_kwargs=True) for i, layer in enumerate(self.layers)]
        handles.append(self.layers[-1].register_forward_hook(keep_last, with_kwargs=True))
        grad_mode = torch.enable_grad() if keep_grad else torch.no_grad()
        try:
            with grad_mode:
                emb = self.lm.model.embed_tokens(ids)
                if keep_grad:
                    emb = emb.detach().requires_grad_(True)
                out = self.lm.model(inputs_embeds=emb, indexes=indexes, attention_mask=mask, use_cache=True)
        finally:
            for handle in handles:
                handle.remove()
        token_ids = ids[0].tolist()
        return TextReading(text=text, token_ids=token_ids, token_strs=[self.tokenizer.decode([i]) for i in token_ids],
                           hidden=hidden, content=content, cache={"kv": out.past_key_values, "text_len": ids.shape[1]},
                           final=out.last_hidden_state[0], last_output=last[0])

    def read(self, text: str, *, keep_grad: bool = False, edit: Optional[Edit] = None) -> TextReading:
        """A plain text, read as the start of a chat message."""
        return self._read(PLAIN_START + text, text, keep_grad, edit)

    def read_prompt(self, text: str, *, keep_grad: bool = False, edit: Optional[Edit] = None) -> TextReading:
        """A drawing prompt, in the full drawing format (system message, user message, empty thinking, <img>)."""
        from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN

        query = self.net._build_t2i_query(text, system_message=SYSTEM_MESSAGE_FOR_GEN if self.system_message else None,
                                          append_text="<think>\n\n</think>\n\n<img>")
        return self._read(query, text, keep_grad, edit)

    def read_nothing(self) -> TextReading:
        """The empty prompt used for guidance (upstream: a short chat with no system message)."""
        if self._empty is None:
            self._empty = self._read(self.net._build_t2i_query("", append_text="<img>"), "", False, None)
        return self._empty

    def logit_lens(self, hidden: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.lm.lm_head(self.lm.model.norm(hidden.to(self.dtype))).float()

    def _chat_query(self, text: str) -> str:
        return self.net._build_t2i_query(text, append_text="<think>\n\n</think>\n\n")

    def read_chat(self, text: str, *, edit: Optional[Edit] = None) -> TextReading:
        """A question read the way answer() reads it, so edits can name its positions."""
        return self._read(self._chat_query(text), text, False, edit)

    def read_reply(self, question: str, reply: str) -> TextReading:
        """The question and the model's own reply, as answer() produced them; the reply's tokens are the content."""
        return self._read(self._chat_query(question) + reply, reply, False, None)

    @torch.no_grad()
    def answer(self, text: str, max_new_tokens: int = 40, edit: Optional[Edit] = None) -> str:
        """Greedy text reply in chat format, without thinking (as upstream skips it: an empty <think></think>).
        `edit` changes the hidden states of the question (positions as in read_chat) on the pass that reads it; the
        reply's own tokens are not edited."""
        self.net.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        enc = self.tokenizer(self._chat_query(text), return_tensors="pt").to(self.device)
        n = enc.input_ids.shape[1]

        def make(layer_index: int):
            def hook(module, args, kwargs):
                if kwargs.get("exist_image_gen_tokens") or args[0].shape[1] != n:
                    return None  # an image pass, or one new token of the reply
                return (edit(layer_index, args[0][0])[None], *args[1:]), kwargs
            return hook

        handles = [layer.register_forward_pre_hook(make(i), with_kwargs=True)
                   for i, layer in enumerate(self.layers)] if edit is not None else []
        try:
            out = self.net.generate(pixel_values=None, input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                                    max_new_tokens=max_new_tokens, do_sample=False,
                                    eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        finally:
            for handle in handles:
                handle.remove()
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    @torch.no_grad()
    def answer_image(self, image: Image.Image, question: str, max_new_tokens: int = 16) -> str:
        """Greedy text reply to a question about a picture (used by scripts/benchmarks.py), as upstream's VQA example does it (examples/vqa/inference.py and the model's chat()
        @c35c83b): load_image_native with its default sizes (native resolution, sides rounded to 32 px), the question as
        "<image>\\n" + question in the model's chat template, <image> replaced by <img> + one <IMG_CONTEXT> per merged
        patch + </img>, generate() with the picture's features in place of the context tokens. One change, the same as
        answer(): the reply starts after an empty <think></think>, so the model answers without thinking. Adapters
        switched on around this call act on every token, the picture's included."""
        from sensenova_u1.models.neo_unify.utils import load_image_native

        net, tk = self.net, self.tokenizer
        net.img_context_token_id = tk.convert_tokens_to_ids("<IMG_CONTEXT>")
        net.img_start_token_id = tk.convert_tokens_to_ids("<img>")
        pixel_values, grid_hw = load_image_native(image)
        pixel_values, grid_hw = pixel_values.to(self.device, dtype=self.dtype), grid_hw.to(self.device)
        query = self._chat_query("<image>\n" + question)
        n = int(grid_hw[0, 0] * grid_hw[0, 1] * net.downsample_ratio ** 2)
        query = query.replace("<image>", "<img>" + "<IMG_CONTEXT>" * n + "</img>", 1)
        enc = tk(query, return_tensors="pt").to(self.device)
        out = net.generate(pixel_values=pixel_values, input_ids=enc.input_ids, grid_hw=grid_hw,
                           attention_mask=enc.attention_mask, max_new_tokens=max_new_tokens, do_sample=False,
                           eos_token_id=tk.convert_tokens_to_ids("<|im_end|>"))
        return tk.decode(out[0], skip_special_tokens=True).strip()

    # ------------------------------------------------------------------ the image side

    def _grid(self, size: int) -> tuple[int, int, int]:
        merge = int(1 / self.net.downsample_ratio)
        tokens_side = size // (self.net.patch_size * merge)
        return merge, tokens_side, size // self.net.patch_size

    def noise_scale(self, size: int) -> float:
        net = self.net
        merge, _, grid = self._grid(size)
        scale = net.noise_scale
        if net.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
            scale = math.sqrt(grid * grid / merge ** 2 / float(net.noise_scale_base_image_seq_len)) * float(net.noise_scale)
            if net.noise_scale_mode == "dynamic_sqrt":
                scale = math.sqrt(scale)
        return min(scale, net.noise_scale_max_value)

    def noise(self, size: int = DRAW_DEFAULTS["size"], seed: int = 0) -> torch.Tensor:
        """Starting picture [1, 3, size, size], drawn exactly as upstream (generator on the GPU, bfloat16)."""
        g = torch.Generator(self.device).manual_seed(seed)
        return self.noise_scale(size) * torch.randn((1, 3, size, size), device=self.device, dtype=self.dtype, generator=g)

    def _velocity(self, reading: TextReading, x: torch.Tensor, t: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Upstream's per-step body (modeling_neo_chat.py, t2i_generate) for a batch of pictures x [b, 3, size, size]
        that all read `reading`'s cache (batched by draw_many). Returns (patches z, velocity)."""
        net = self.net
        merge, side, grid = self._grid(size)
        n, b = side * side, x.shape[0]
        z = net.patchify(x, net.patch_size * merge)
        image_input = net.patchify(x, net.patch_size, channel_first=True)
        grid_hw = torch.tensor([[grid, grid]] * b, device=self.device)
        embeds = net.extract_feature(image_input.view(b * grid * grid, -1), gen_model=True, grid_hw=grid_hw).view(b, n, -1)
        t_exp = t.expand(b * n)
        temb = net.fm_modules["timestep_embedder"](t_exp).view(b, n, -1)
        if net.add_noise_scale_embedding:
            scale = torch.full_like(t_exp, self.noise_scale(size) / net.noise_scale_max_value)
            temb = temb + net.fm_modules["noise_scale_embedder"](scale).view(b, n, -1)
        index = net._build_t2i_image_indexes(side, side, reading.cache["text_len"], device=self.device)
        v = net._t2i_predict_v(embeds + temb, index, {"full_attention": None}, reading.cache["kv"], t, z,
                               image_token_num=n, timestep_embeddings=temb, image_size=(size, size))
        return z, v

    @torch.no_grad()
    def draw(self, reading: TextReading, size: int = DRAW_DEFAULTS["size"], seed: int = 0,
             steps: int = DRAW_DEFAULTS["steps"], shift: float = DRAW_DEFAULTS["shift"],
             guidance: float = DRAW_DEFAULTS["guidance"], guidance_interval: tuple = DRAW_DEFAULTS["guidance_interval"],
             ) -> tuple[torch.Tensor, list[tuple[float, torch.Tensor]]]:
        """Upstream's sampler for one picture. `reading` must come from read_prompt(). Returns the final picture
        [1, 3, size, size] in [-1, 1] and, for every step, (noise level, guess of the finished picture)."""
        from sensenova_u1.models.neo_unify.modeling_neo_chat import clear_flash_kv_cache, prepare_flash_kv_cache

        net = self.net
        merge, side, _ = self._grid(size)
        empty = self.read_nothing()
        readings = [reading, empty] if guidance > 1 else [reading]
        for r in readings:
            prepare_flash_kv_cache(r.cache["kv"], current_len=side * side, batch_size=1)
        x = self.noise(size, seed)
        ts = net._apply_time_schedule(torch.linspace(0.0, 1.0, steps + 1, device=self.device), side * side, shift)
        lo, hi = guidance_interval
        guesses = []
        try:
            for i in range(steps):
                t, t_next = ts[i], ts[i + 1]
                z, v = self._velocity(reading, x, t, size)
                if guidance > 1 and lo <= t <= hi:
                    _, v_u = self._velocity(empty, x, t, size)
                    v = v_u + guidance * (v - v_u)
                guesses.append((1.0 - float(t), net.unpatchify(z + (1 - t).clamp_min(net.config.t_eps) * v,
                                                               net.patch_size * merge, size, size)))
                z = z + (t_next - t) * v
                x = net.unpatchify(z, net.patch_size * merge, size, size)
        finally:
            for r in readings:
                clear_flash_kv_cache(r.cache["kv"])
        return x, guesses

    def from_picture(self, image: Image.Image, size: int = DRAW_DEFAULTS["size"]) -> torch.Tensor:
        """A picture as SenseNova's drawing state [1, 3, size, size] in [-1, 1] (centre-cropped square; it draws in
        pixels, so there is no encoder)."""
        import numpy as np

        image = image.convert("RGB")
        w, h = image.size
        s = min(w, h)
        image = image.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.BICUBIC)
        return torch.from_numpy(np.asarray(image)).float().div(127.5).sub(1).permute(2, 0, 1)[None].to(self.device)

    def denoise_loss(self, reading: TextReading, x0: torch.Tensor, level: float, noise: torch.Tensor) -> torch.Tensor:
        """SenseNova's training loss for the picture x0 at noise `level` (its time t = 1 - level), with standard
        noise `noise` scaled as SenseNova scales it: the squared error of the predicted velocity, x0 - noise. Gradients
        flow into the reading; the text cache is read without the preallocated buffers draw() uses (same numbers)."""
        size = x0.shape[-1]
        merge, _, _ = self._grid(size)
        noise = self.noise_scale(size) * noise
        t = torch.tensor(1.0 - level, device=self.device)
        x_t = (t * x0 + (1 - t) * noise).to(self.dtype)
        _, v = self._velocity(reading, x_t, t, size)
        target = self.net.patchify(x0 - noise, self.net.patch_size * merge)
        return ((v.float() - target.float()) ** 2).mean()

    @staticmethod
    def _stack(readings: list[TextReading], repeat: int = 1) -> TextReading:
        """One reading whose cache holds the caches of several readings of the same length side by side (batch)."""
        import copy

        kv = copy.deepcopy(readings[0].cache["kv"])
        for i, layer in enumerate(kv.layers):
            layer.keys = torch.cat([r.cache["kv"].layers[i].keys for r in readings] * repeat)
            layer.values = torch.cat([r.cache["kv"].layers[i].values for r in readings] * repeat)
        return TextReading(text="", token_ids=readings[0].token_ids, token_strs=[], hidden=[], content=readings[0].content,
                           cache={"kv": kv, "text_len": readings[0].cache["text_len"]})

    @torch.no_grad()
    def draw_many(self, readings: list[TextReading], size: int = DRAW_DEFAULTS["size"], seed: int = 0,
                  steps: int = DRAW_DEFAULTS["steps"], shift: float = DRAW_DEFAULTS["shift"],
                  guidance: float = DRAW_DEFAULTS["guidance"],
                  guidance_interval: tuple = DRAW_DEFAULTS["guidance_interval"]) -> torch.Tensor:
        """draw() for several drawing prompts of the same length (the same prompt with different edits) in one batch,
        all from the same starting noise. Returns the final pictures [n, 3, size, size]."""
        from sensenova_u1.models.neo_unify.modeling_neo_chat import clear_flash_kv_cache, prepare_flash_kv_cache

        net = self.net
        merge, side, _ = self._grid(size)
        b = len(readings)
        cond = self._stack(readings)
        empty = self._stack([self.read_nothing()], repeat=b)
        for r in (cond, empty):
            prepare_flash_kv_cache(r.cache["kv"], current_len=side * side, batch_size=b)
        x = self.noise(size, seed).expand(b, -1, -1, -1).contiguous()
        ts = net._apply_time_schedule(torch.linspace(0.0, 1.0, steps + 1, device=self.device), side * side, shift)
        lo, hi = guidance_interval
        try:
            for i in range(steps):
                t, t_next = ts[i], ts[i + 1]
                z, v = self._velocity(cond, x, t, size)
                if guidance > 1 and lo <= t <= hi:
                    _, v_u = self._velocity(empty, x, t, size)
                    v = v_u + guidance * (v - v_u)
                z = z + (t_next - t) * v
                x = net.unpatchify(z, net.patch_size * merge, size, size)
        finally:
            for r in (cond, empty):
                clear_flash_kv_cache(r.cache["kv"])
        return x

    @torch.no_grad()
    def to_picture(self, latent: torch.Tensor) -> Image.Image:
        image = (latent[0].float() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).mul(255).byte().cpu().numpy()
        return Image.fromarray(image)
