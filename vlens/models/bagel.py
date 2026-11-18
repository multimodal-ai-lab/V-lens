"""BAGEL-7B-MoT behind the common model interface.

How BAGEL draws (the parts this code relies on):

- The text is read once by the "understanding" weights. Every layer stores keys and values for every text
  token: the cache. Each layer's keys and values are computed from the hidden state *entering* that layer.
- The picture is a grid of image tokens, 64 numbers each (16 VAE channels x a 2x2 patch). A 512 px picture
  is 32 x 32 = 1024 image tokens. They pass through a second copy of every layer (the "generation"
  weights), and at every layer they attend to the cached keys and values of the text. That attention is
  the only way the text reaches the picture.
- Noise level t runs from 1 (pure noise) to 0 (finished). The latent at level t is
  x_t = (1 - t) * finished + t * noise, the model predicts the velocity v = noise - finished, so its guess
  of the finished latent is x_t - t * v.
- Guidance mixes the prediction for the prompt with the prediction for the empty prompt and then rescales
  the result so its overall size does not grow ("global renorm"). The rescale makes guidance nonlinear.

BAGEL's own drawing functions run under @torch.no_grad. This adapter calls the lower-level pieces of the
vendored code directly so that gradients can flow from the picture back into the text's hidden states.
"""

from __future__ import annotations

import contextlib
import math
import os
from typing import Optional

import torch
from PIL import Image

try:  # upstream BAGEL code, fetched and patched by third_party/setup.sh
    from bagel.data.data_utils import add_special_tokens
    from bagel.modeling.autoencoder import load_ae
    from bagel.modeling.bagel import (
        Bagel as BagelNetwork,
        BagelConfig,
        Qwen2Config,
        Qwen2ForCausalLM,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from bagel.modeling.bagel import qwen2_navit
    from bagel.modeling.bagel.qwen2_navit import NaiveCache
    from bagel.modeling.qwen2 import Qwen2Tokenizer
except ImportError as _error:  # the module stays importable (on a CPU, for tests); loading the model needs the code
    _MISSING = _error
else:
    _MISSING = None

from vlens import paths
from vlens.models.base import Edit, TextReading

REPO = "ByteDance-Seed/BAGEL-7B-MoT"

# Drawing settings (close to upstream's defaults).
DRAW_DEFAULTS = dict(steps=28, shift=3.0, guidance=4.0, guidance_interval=(0.4, 1.0), renorm="global")


def schedule(steps: int, shift: float, start_level: float = 1.0) -> torch.Tensor:
    """Noise levels visited by the sampler, from start_level down to 0 (BAGEL's shifted linear schedule).

    Returns a float32 tensor that ends with 0.0, computed exactly as BAGEL's sampler does. With
    start_level < 1, only the levels below start_level are kept and the schedule begins at start_level.
    """
    ts = torch.linspace(1, 0, steps)
    ts = shift * ts / (1 + (shift - 1) * ts)
    if start_level < 1.0:
        ts = torch.cat([torch.tensor([start_level], dtype=ts.dtype), ts[ts < start_level]])
    return ts


class BagelModel:
    name = "bagel-7b-mot"
    patch = 2
    channels = 16
    pixels_per_token = 16  # 8x VAE downsampling times the 2x2 patch

    def __init__(self, path: str | None = None, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        """dtype=torch.float32 runs everything in full precision (slower, about 60 GB of weights), with plain
        attention instead of flash-attn. Used only for precise gradient checks. The setting is global to the
        process (it lives in the vendored module), so do not mix precisions in one process."""
        if _MISSING is not None:
            raise ImportError("BAGEL's code is missing: run third_party/setup.sh and add third_party to PYTHONPATH") \
                from _MISSING
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch

        path = path or str(paths.snapshot(REPO))
        self.path = path
        self.device = torch.device(device)
        self.dtype = dtype
        qwen2_navit.COMPUTE_DTYPE = dtype

        llm_config = Qwen2Config.from_json_file(os.path.join(path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1
        vae, vae_config = load_ae(local_path=os.path.join(path, "ae.safetensors"))
        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
        )
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            net = BagelNetwork(language_model, vit_model, config)
            net.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        tokenizer = Qwen2Tokenizer.from_pretrained(path)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(tokenizer)
        self.text_vocab = len(self.tokenizer)  # computed once: len() rebuilds the whole vocabulary every call

        # Everything on one GPU; no accelerate hooks, so our own hooks see plain module calls.
        net = load_checkpoint_and_dispatch(
            net,
            checkpoint=os.path.join(path, "ema.safetensors"),
            device_map={"": str(self.device)},
            dtype=torch.bfloat16,
        )
        if dtype != torch.bfloat16:
            # Convert after loading, one parameter at a time: loading straight to float32 first puts the whole
            # bf16 checkpoint on the GPU next to its float32 copy (about 87 GB).
            net = net.to(dtype)
        leftover = [n for n, p in list(net.named_parameters()) + list(net.named_buffers()) if p.is_meta]
        if leftover:
            raise RuntimeError(f"BAGEL weights not loaded for: {leftover[:5]} ...")
        self.net = net.eval().requires_grad_(False)
        self.vae = vae.to(device=self.device, dtype=dtype).eval().requires_grad_(False)

        self.layers = self.net.language_model.model.layers
        self.n_layers = len(self.layers)
        self.d_model = llm_config.hidden_size

    # ------------------------------------------------------------------ helpers

    def _autocast(self):
        if self.dtype == torch.float32:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.dtype)

    def _to_device(self, inputs: dict) -> dict:
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    def image_tokens(self, size: int) -> int:
        return (size // self.pixels_per_token) ** 2

    # ------------------------------------------------------------------ reading text

    def read_reply(self, question: str, reply: str) -> TextReading:
        """The question and the model's own reply, as answer() produced them: the question fills the cache, and the reply
        is read as the next message on top of it (BAGEL wraps each message in start and end markers). The reading holds
        the reply's tokens only (all of them content except the markers)."""
        first = self.read(question)
        return self.read(reply, previous=first.cache)

    def read(self, text: str, *, keep_grad: bool = False, edit: Optional[Edit] = None,
             previous: Optional[dict] = None) -> TextReading:
        """Read `text` with the understanding weights and fill a fresh cache (or, with `previous`, continue that cache as
        the next message).

        The hidden state entering each layer is recorded with a pre-hook (and changed by `edit`, if
        given). With keep_grad=True the token embeddings become the root of the computation graph, so
        gradients of anything computed from the cache can be taken with respect to every recorded hidden
        state.
        """
        inputs, kv_lens, ropes = self.net.prepare_prompts(
            curr_kvlens=previous["kv_lens"] if previous else [0], curr_rope=previous["ropes"] if previous else [0],
            prompts=[text], tokenizer=self.tokenizer, new_token_ids=self.new_token_ids,
        )
        inputs = self._to_device(inputs)
        cache = previous["past_key_values"] if previous else NaiveCache(self.n_layers)
        hidden: list[Optional[torch.Tensor]] = [None] * self.n_layers

        def make_hook(layer_index: int):
            def hook(module, args, kwargs):
                h = kwargs["packed_query_sequence"]
                if edit is not None:
                    h = edit(layer_index, h)
                    kwargs = {**kwargs, "packed_query_sequence": h}
                hidden[layer_index] = h
                return args, kwargs
            return hook

        last_output: list[Optional[torch.Tensor]] = [None]

        def keep_last_output(module, args, kwargs, output):
            last_output[0] = output[0]

        handles = [layer.register_forward_pre_hook(make_hook(i), with_kwargs=True)
                   for i, layer in enumerate(self.layers)]
        handles.append(self.layers[-1].register_forward_hook(keep_last_output, with_kwargs=True))
        grad_mode = torch.enable_grad() if keep_grad else torch.no_grad()
        try:
            with grad_mode, self._autocast():
                embedding = self.net.language_model.model.embed_tokens(inputs["packed_text_ids"])
                if keep_grad:
                    embedding = embedding.detach().requires_grad_(True)
                out = self.net.language_model.forward_inference(
                    packed_query_sequence=embedding,
                    query_lens=inputs["text_token_lens"],
                    packed_query_position_ids=inputs["packed_text_position_ids"],
                    packed_query_indexes=inputs["packed_text_indexes"],
                    past_key_values=cache,
                    packed_key_value_indexes=inputs["packed_key_value_indexes"],
                    key_values_lens=inputs["key_values_lens"],
                    update_past_key_values=True,
                    is_causal=True,
                    mode="und",
                )
        finally:
            for handle in handles:
                handle.remove()

        ids = inputs["packed_text_ids"].tolist()
        content = torch.ones(len(ids), dtype=torch.bool)
        content[0] = content[-1] = False  # the start and end markers BAGEL wraps around every message
        return TextReading(
            text=text,
            token_ids=ids,
            token_strs=[self.tokenizer.decode([i]) for i in ids],
            hidden=hidden,
            content=content,
            cache={"past_key_values": out.past_key_values, "kv_lens": kv_lens, "ropes": ropes},
            final=out.packed_query_sequence,
            last_output=last_output[0],
        )

    def read_prompt(self, text: str, *, keep_grad: bool = False, edit: Optional[Edit] = None) -> TextReading:
        """A drawing prompt. The same as read() for this model (SenseNova reads drawing prompts differently)."""
        return self.read(text, keep_grad=keep_grad, edit=edit)

    def read_chat(self, text: str, *, edit: Optional[Edit] = None) -> TextReading:
        """A question as answer() reads it: for BAGEL the same as read()."""
        return self.read(text, edit=edit)

    def read_nothing(self) -> TextReading:
        return TextReading(
            text="", token_ids=[], token_strs=[], hidden=[], content=torch.zeros(0, dtype=torch.bool),
            cache={"past_key_values": NaiveCache(self.n_layers), "kv_lens": [0], "ropes": [0]},
        )

    def next_word_scores(self, normed_hidden: torch.Tensor) -> torch.Tensor:
        """Text head applied to an already-normalized hidden state."""
        return self.net.language_model.lm_head(normed_hidden)

    def logit_lens(self, hidden: torch.Tensor) -> torch.Tensor:
        """Next-word scores read straight off a hidden state (final norm + text head)."""
        with torch.no_grad(), self._autocast():
            return self.net.language_model.lm_head(self.net.language_model.model.norm(hidden)).float()

    @torch.no_grad()
    def read_image(self, image: Image.Image, *, previous: Optional[dict] = None) -> TextReading:
        """Show BAGEL a picture in understanding mode: the picture goes through the ViT into the understanding stream,
        as upstream's inferencer does (ImageTransform(980, 224, 14), wrapped in start/end-of-image markers). Returns a
        reading whose `hidden` holds the state entering every layer at every position of this message (the start
        marker, the ViT tokens, the end marker; `content` marks the ViT tokens) and whose `cache` continues the
        conversation (pass it as `previous` to read() or answer())."""
        from bagel.data.transforms import ImageTransform
        if not hasattr(self, "_vit_transform"):
            self._vit_transform = ImageTransform(980, 224, 14)
        image = image.convert("RGB")
        inputs, kv_lens, ropes = self.net.prepare_vit_images(
            curr_kvlens=previous["kv_lens"] if previous else [0], curr_rope=previous["ropes"] if previous else [0],
            images=[image], transforms=self._vit_transform, new_token_ids=self.new_token_ids,
        )
        inputs = self._to_device(inputs)
        cache = previous["past_key_values"] if previous else NaiveCache(self.n_layers)
        hidden: list[Optional[torch.Tensor]] = [None] * self.n_layers

        def make_hook(layer_index: int):
            def hook(module, args, kwargs):
                hidden[layer_index] = kwargs["packed_query_sequence"]
                return args, kwargs
            return hook

        handles = [layer.register_forward_pre_hook(make_hook(i), with_kwargs=True)
                   for i, layer in enumerate(self.layers)]
        try:
            with self._autocast():
                cache = self.net.forward_cache_update_vit(cache, **inputs)
        finally:
            for handle in handles:
                handle.remove()
        n = int(inputs["packed_seqlens"].sum())
        content = torch.zeros(n, dtype=torch.bool)
        content[inputs["packed_vit_token_indexes"].cpu()] = True
        return TextReading(
            text="<image>", token_ids=[-1] * n, token_strs=["<image>"] * n, hidden=hidden, content=content,
            cache={"past_key_values": cache, "kv_lens": kv_lens, "ropes": ropes},
        )

    @torch.no_grad()
    def answer(self, text: str, max_new_tokens: int = 40, edit: Optional[Edit] = None,
               after: Optional[dict] = None) -> str:
        """Greedy text reply to `text` (BAGEL's understanding mode, no thinking). `edit` changes the hidden states of the
        question while it is read (used for steering); the reply itself is generated without edits. `after`: a cache to
        continue (e.g. read_image(...).cache, so the question follows a picture)."""
        reading = self.read(text, edit=edit, previous=after)
        cache = reading.cache
        start = self._to_device(self.net.prepare_start_tokens(cache["kv_lens"], cache["ropes"], self.new_token_ids))
        with self._autocast():
            ids = self.net.generate_text(
                past_key_values=cache["past_key_values"],
                max_length=max_new_tokens,
                do_sample=False,
                end_token_id=self.new_token_ids["eos_token_id"],
                **start,
            )
        ids = ids[:, 0].tolist()[1:]  # drop the start marker
        return self.tokenizer.decode(ids).replace("<|im_end|>", "").strip()

    @torch.no_grad()
    def answer_image(self, image: Image.Image, question: str, max_new_tokens: int = 16) -> str:
        """Greedy text reply to a question about a picture, as upstream's interleave_inference(understanding_output=True)
        does it (used by scripts/benchmarks.py): the picture first resized by the
        VAE transform's resize (ImageTransform(1024, 512, 16), as upstream does for every input picture), then the ViT
        message (read_image), then the question as the next message, then the reply."""
        from bagel.data.data_utils import pil_img2rgb
        from bagel.data.transforms import ImageTransform
        if not hasattr(self, "_vae_resize"):
            self._vae_resize = ImageTransform(1024, 512, 16).resize_transform
        picture = self.read_image(self._vae_resize(pil_img2rgb(image)))
        return self.answer(question, max_new_tokens=max_new_tokens, after=picture.cache)

    # ------------------------------------------------------------------ the image side

    def noise(self, size: int, seed: int) -> torch.Tensor:
        """Starting noise, [image_tokens, 64]. Same numbers BAGEL draws after torch.manual_seed(seed)."""
        g = torch.Generator().manual_seed(seed)
        return torch.randn(self.image_tokens(size), self.channels * self.patch ** 2, generator=g).to(self.device)

    def velocity(self, reading: TextReading, x: torch.Tensor, level: float) -> torch.Tensor:
        """One pass of the image side for latent x at noise `level`, without guidance (bf16 output)."""
        side = int(math.isqrt(x.shape[0])) * self.pixels_per_token
        cache = reading.cache
        inputs = self.net.prepare_vae_latent(
            curr_kvlens=cache["kv_lens"], curr_rope=cache["ropes"],
            image_sizes=[(side, side)], new_token_ids=self.new_token_ids,
        )
        inputs.pop("packed_init_noises")
        inputs = self._to_device(inputs)
        with self._autocast():
            marker_embedding = self.net.language_model.model.embed_tokens(inputs["packed_text_ids"])
            sequence = marker_embedding.new_zeros((int(inputs["packed_seqlens"].sum()), self.d_model))
            sequence[inputs["packed_text_indexes"]] = marker_embedding
            level_t = torch.full((x.shape[0],), float(level), device=self.device)
            tokens = (self.net.vae2llm(x) + self.net.time_embedder(level_t)
                      + self.net.latent_pos_embed(inputs["packed_vae_position_ids"]))
            sequence[inputs["packed_vae_token_indexes"]] = tokens.to(sequence.dtype)
            out = self.net.language_model.forward_inference(
                packed_query_sequence=sequence,
                query_lens=inputs["packed_seqlens"],
                packed_query_position_ids=inputs["packed_position_ids"],
                packed_query_indexes=inputs["packed_indexes"],
                past_key_values=cache["past_key_values"],
                key_values_lens=inputs["key_values_lens"],
                packed_key_value_indexes=inputs["packed_key_value_indexes"],
                update_past_key_values=False,
                is_causal=False,
                mode="gen",
                packed_vae_token_indexes=inputs["packed_vae_token_indexes"],
                packed_text_indexes=inputs["packed_text_indexes"],
            )
            v = self.net.llm2vae(out.packed_query_sequence)[inputs["packed_vae_token_indexes"]]
        return v

    def denoise_loss(self, reading: TextReading, x0: torch.Tensor, level: float, noise: torch.Tensor) -> torch.Tensor:
        """BAGEL's training loss for the latent x0 at noise `level` with standard noise `noise` (gradients flow into the
        reading): the squared error of the predicted velocity, noise - x0."""
        x_t = (1 - level) * x0 + level * noise
        return ((self.velocity(reading, x_t, level).float() - (noise - x0)) ** 2).mean()

    def guess_finished(self, reading: TextReading, x: torch.Tensor, level: float) -> torch.Tensor:
        """The model's guess of the finished latent (float32), without guidance."""
        v = self.velocity(reading, x, level)
        return x.float() - float(level) * v.float()

    def guided_velocity(self, reading, empty, x, level, guidance, renorm="global", renorm_min=0.0):
        """Velocity with guidance, exactly as BAGEL's sampler combines it (text guidance only)."""
        v = self.velocity(reading, x, level)
        if guidance <= 1.0:
            return v
        v_empty = self.velocity(empty, x, level)
        v_mix = v_empty + guidance * (v - v_empty)
        if renorm == "global":
            scale = (torch.norm(v) / (torch.norm(v_mix) + 1e-8)).clamp(min=renorm_min, max=1.0)
        elif renorm == "channel":
            scale = (torch.norm(v, dim=-1, keepdim=True)
                     / (torch.norm(v_mix, dim=-1, keepdim=True) + 1e-8)).clamp(min=renorm_min, max=1.0)
        elif renorm == "none":
            return v_mix
        else:
            raise ValueError(renorm)
        return v_mix * scale

    @torch.no_grad()
    def draw(
        self,
        reading: TextReading,
        size: int = 512,
        seed: int = 0,
        steps: int = DRAW_DEFAULTS["steps"],
        shift: float = DRAW_DEFAULTS["shift"],
        guidance: float = DRAW_DEFAULTS["guidance"],
        guidance_interval: tuple[float, float] = DRAW_DEFAULTS["guidance_interval"],
        renorm: str = DRAW_DEFAULTS["renorm"],
        start: Optional[tuple[torch.Tensor, float]] = None,
        stop_level: float = 0.0,
    ) -> tuple[torch.Tensor, list[tuple[float, torch.Tensor]]]:
        """Run BAGEL's sampler. Returns the final latent and, for every step, (level, guess of finished).

        `start=(x, level)` begins from a partly noised latent instead of pure noise.
        `stop_level` ends the run early: the returned latent is then the latent at that noise level.
        """
        empty = self.read_nothing() if guidance > 1.0 else None
        if start is None:
            x, levels = self.noise(size, seed).float(), schedule(steps, shift)
        else:
            x, levels = start[0].float(), schedule(steps, shift, start_level=start[1])
        levels = levels.to(self.device)
        steps_down = levels[:-1] - levels[1:]
        guesses = []
        lo, hi = guidance_interval
        for i, t in enumerate(levels[:-1]):
            if float(t) <= stop_level:
                break
            g = guidance if lo < t <= hi else 1.0
            v = self.guided_velocity(reading, empty, x, float(t), g, renorm=renorm)
            guesses.append((float(t), x - float(t) * v.float()))
            x = x - v * steps_down[i]  # same arithmetic as BAGEL: bf16 velocity times a float32 step
        return x, guesses

    @torch.no_grad()
    def from_picture(self, image: Image.Image, size: int = 512) -> torch.Tensor:
        """A picture as BAGEL's latent [image_tokens, 64]: centre-cropped to a square, resized, VAE-encoded (the mean,
        no sampling). The inverse of to_picture."""
        import numpy as np

        image = image.convert("RGB")
        w, h = image.size
        s = min(w, h)
        image = image.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(image)).float().div(127.5).sub(1).permute(2, 0, 1)[None]
        with self._autocast():
            mean = self.vae.encoder(x.to(device=self.device, dtype=self.dtype)).chunk(2, dim=1)[0]
            z = self.vae.scale_factor * (mean - self.vae.shift_factor)
        p, c = self.patch, self.channels
        n = z.shape[-1] // p
        grid = z.float().reshape(1, c, n, p, n, p)
        return torch.einsum("nchpwq->nhwpqc", grid).reshape(n * n, p * p * c)

    @torch.no_grad()
    def to_picture(self, latent: torch.Tensor) -> Image.Image:
        """Decode [image_tokens, 64] (or a flat vector of that size) into a picture."""
        latent = latent.reshape(-1, self.channels * self.patch ** 2)
        n = int(math.isqrt(latent.shape[0]))
        p, c = self.patch, self.channels
        grid = latent.reshape(1, n, n, p, p, c)
        grid = torch.einsum("nhwpqc->nchpwq", grid).reshape(1, c, n * p, n * p)
        with self._autocast():
            image = self.vae.decode(grid.to(device=self.device, dtype=self.dtype))
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        return Image.fromarray(image.to(torch.uint8).cpu().numpy())
