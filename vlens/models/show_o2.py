"""Show-o2 behind the common model interface.

How Show-o2 draws (the parts this code relies on):

- One Qwen2.5 language model reads text and image tokens with the same weights. For a picture the sequence is
      <|im_start|> prompt <|vision_start|> [time] 729 image tokens <|vision_end|> <|endoftext|> [PAD] ...
  padded to 1024 tokens. Text tokens see only earlier tokens; image tokens see the whole text and each other.
  So the text is read exactly as it would be without a picture, and the image tokens read the text's hidden states
  at every layer, through attention.
- A flow head of 10 blocks on top of the last layer turns the image tokens into a velocity. The picture is a
  16-channel Wan2.1 VAE latent of 54 x 54 (432 px), cut into 2 x 2 patches.
- There is no cache: the whole sequence runs again at every denoising step. An edit of a text hidden state is
  therefore applied at every step (only to the prompt's half of the batch, not to the empty prompt used for guidance).
- Show-o2's own time runs from 0 (noise) to 1 (picture): x_t = t * picture + (1 - t) * noise, and the model predicts
  v = picture - noise. The noise level used here is 1 - t, and the guess of the finished latent is x_t + (1 - t) * v.
- Guidance is the plain mix v_empty + g * (v_prompt - v_empty), with no rescale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image

try:  # upstream Show-o2 code, fetched and patched by third_party/setup.sh
    from show_o2 import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
    from show_o2.misc import get_text_tokenizer
except ImportError as _error:  # the module stays importable (on a CPU, for tests); loading the model needs the code
    _MISSING = _error
else:
    _MISSING = None

from vlens import paths
from vlens.models.base import Edit, TextReading

REPOS = {"7B": "showlab/show-o2-7B"}
LLM = {"7B": "Qwen/Qwen2.5-7B-Instruct"}
VAE_PATH = str(paths.MODELS / "Wan2.1_VAE.pth")  # from Wan-AI/Wan2.1-T2V-14B, see README.md

# Upstream's demo settings for 432 px (README: guidance 7.5, 50 steps; config: time shift 3).
DRAW_DEFAULTS = dict(steps=50, shift=3.0, guidance=7.5)
SEQ_LEN = 1024
LATENT = 54  # latent height and width (27 x 27 patches of 2 x 2)


def schedule(steps: int, shift: float) -> torch.Tensor:
    """Show-o2's own times visited by the sampler, from 0 (noise) to 1 (picture), exactly as upstream computes them."""
    t = torch.linspace(0, 1, steps)
    return t / (t + shift - shift * t)


class ShowO2Model:
    name = "show_o2"

    def __init__(self, size: str = "7B", device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        if _MISSING is not None:
            raise ImportError("Show-o2's code is missing: run third_party/setup.sh and add third_party to PYTHONPATH") \
                from _MISSING
        self.name = f"show_o2_{size.lower()}"
        self.device = torch.device(device)
        self.dtype = dtype
        self.tokenizer, ids = get_text_tokenizer(LLM[size], add_showo_tokens=True, return_showo_token_ids=True,
                                                 llm_name="qwen2_5")
        self.bos, self.eos, self.boi, self.eoi = ids["bos_id"], ids["eos_id"], ids["boi_id"], ids["eoi_id"]
        self.img_pad, self.pad = ids["img_pad_id"], self.tokenizer.pad_token_id
        self.net = Showo2Qwen2_5.from_pretrained(str(paths.snapshot(REPOS[size])), use_safetensors=False)
        self.net.to(self.device, dtype).eval()
        self.net.requires_grad_(False)
        self.vae = WanVAE(vae_pth=VAE_PATH, dtype=dtype, device=self.device)
        self.layers = self.net.showo.model.layers
        self.n_layers = len(self.layers)
        self.d_model = self.net.showo.config.hidden_size
        self.image_tokens = (LATENT // 2) ** 2 + 1  # 729 patches plus the time token
        self.max_text = SEQ_LEN - self.image_tokens - 4
        self.text_vocab = len(self.tokenizer)

    # ------------------------------------------------------------------ the text side

    def _ids(self, text: str) -> list[int]:
        return [self.bos] + self.tokenizer(text, add_special_tokens=False)["input_ids"][: self.max_text]

    def _hooks(self, edit: Optional[Edit], record: Optional[list], batch_index: int = 0):
        """Pre-hooks that hand the hidden state entering each layer to `edit` (for one batch row) and record it."""

        def make(layer_index: int):
            def hook(module, args, kwargs):
                h = args[0]
                row = h[batch_index]
                if edit is not None:
                    row = edit(layer_index, row)
                if record is not None:
                    record[layer_index] = row
                if edit is None and record is None:
                    return None
                # rebuild the batch from `row`, so the recorded tensor is the one the layer actually reads
                h = torch.cat([h[:batch_index], row[None], h[batch_index + 1:]]) if h.shape[0] > 1 else row[None]
                return (h, *args[1:]), kwargs
            return hook

        return [layer.register_forward_pre_hook(make(i), with_kwargs=True) for i, layer in enumerate(self.layers)]

    def read(self, text: str, *, keep_grad: bool = False, edit: Optional[Edit] = None) -> TextReading:
        """Read `text` (as <|im_start|> text, the start of a drawing prompt), recording the hidden state entering every
        layer. With keep_grad=True the token embeddings are the root of the graph, so gradients can be taken with
        respect to every recorded hidden state. The edit is kept with the reading: draw() applies it again at every
        denoising step."""
        return self._read_ids(self._ids(text), text, 1, keep_grad, edit)

    def _chat_ids(self, text: str) -> tuple[list[int], int]:
        """The chat format of Show-o2's understanding demo, up to the start of the reply; and where `text` starts."""
        tok = lambda s: self.tokenizer(s, add_special_tokens=False)["input_ids"]
        head = [self.bos] + tok("system\nYou are a helpful assistant.<|im_end|>") + tok("\n<|im_start|>user\n")
        return head + tok(text) + tok("<|im_end|>\n<|im_start|>assistant\n"), len(head)

    def read_chat(self, text: str, *, edit: Optional[Edit] = None) -> TextReading:
        """A question read the way answer() reads it, so edits can name its positions."""
        ids, start = self._chat_ids(text)
        reading = self._read_ids(ids, text, start, False, edit)
        reading.content[start + len(self.tokenizer(text, add_special_tokens=False)["input_ids"]):] = False
        return reading

    def read_reply(self, question: str, reply: str) -> TextReading:
        """The question and the model's own reply, as answer() produced them; the reply's tokens are the content."""
        ids, _ = self._chat_ids(question)
        return self._read_ids(ids + self.tokenizer(reply, add_special_tokens=False)["input_ids"], reply, len(ids),
                              False, None)

    def _read_ids(self, ids: list[int], text: str, start: int, keep_grad: bool, edit: Optional[Edit]) -> TextReading:
        hidden: list[Optional[torch.Tensor]] = [None] * self.n_layers
        last: list[Optional[torch.Tensor]] = [None]

        def keep_last(module, args, kwargs, output):
            last[0] = output[0][0]

        handles = self._hooks(edit, hidden)
        handles.append(self.layers[-1].register_forward_hook(keep_last, with_kwargs=True))
        grad_mode = torch.enable_grad() if keep_grad else torch.no_grad()
        try:
            with grad_mode:
                emb = self.net.showo.model.embed_tokens(torch.tensor([ids], device=self.device))
                if keep_grad:
                    emb = emb.detach().requires_grad_(True)
                out = self.net.showo.model(inputs_embeds=emb, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        content = torch.ones(len(ids), dtype=torch.bool)
        content[:start] = False  # <|im_start|> (and the chat header, for read_chat)
        return TextReading(text=text, token_ids=ids, token_strs=[self.tokenizer.decode([i]) for i in ids],
                           hidden=hidden, content=content, cache={"edit": edit},
                           final=out.last_hidden_state[0], last_output=last[0])

    def read_prompt(self, text: str, *, keep_grad: bool = False, edit: Optional[Edit] = None) -> TextReading:
        """A drawing prompt. The same as read() for this model (SenseNova reads drawing prompts differently)."""
        return self.read(text, keep_grad=keep_grad, edit=edit)

    def read_nothing(self) -> TextReading:
        return self.read("")

    def logit_lens(self, hidden: torch.Tensor) -> torch.Tensor:
        """Next-word scores read straight off a hidden state (final norm + text head)."""
        with torch.no_grad():
            return self.net.showo.lm_head(self.net.showo.model.norm(hidden.to(self.dtype))).float()

    @torch.no_grad()
    def answer(self, text: str, max_new_tokens: int = 40, edit: Optional[Edit] = None) -> str:
        """Greedy text reply to `text`, in the chat format of Show-o2's understanding demo. `edit` changes the hidden
        states of the question (positions as in read_chat); the reply's own tokens are not edited."""
        ids, _ = self._chat_ids(text)
        new = []
        handles = self._hooks(edit, None) if edit is not None else []
        try:
            for _ in range(max_new_tokens):
                logits = self.net.showo(input_ids=torch.tensor([ids + new], device=self.device), use_cache=False).logits
                nxt = int(logits[0, -1].argmax())
                if nxt in (self.tokenizer.convert_tokens_to_ids("<|im_end|>"), self.eos):
                    break
                new.append(nxt)
        finally:
            for handle in handles:
                handle.remove()
        return self.tokenizer.decode(new).strip()

    @torch.no_grad()
    def answer_image(self, image: Image.Image, question: str, max_new_tokens: int = 16) -> str:
        """Greedy text reply to a question about a picture, as upstream's understanding demo builds it (show-o2/
        inference_mmu.py @45a5a2d; used by scripts/benchmarks.py): the picture
        resized (bicubic, short side 432) and centre-cropped to 432 px, scaled to [-1, 1], encoded by the Wan VAE (a
        sample of its posterior as upstream takes it, seeded so every run sees the same latent), passed through both
        image embedders, the semantic layers and the fusion projection; the sequence is
            <|im_start|> system ... <|im_end|> \\n<|im_start|>user\\n <|vision_start|> [time t=1] 729 image tokens
            <|vision_end|> question \\n<|im_start|>assistant\\n
        (upstream has no <|im_end|> after the question), with the omni-attention mask (upstream's image block offset
        and length, kept as they are). Greedy, the whole sequence recomputed for every new token, as upstream does.
        Adapters switched on around this call act on every token, the picture's included."""
        from torchvision import transforms

        net, tok = self.net, lambda s: self.tokenizer(s, add_special_tokens=False)["input_ids"]
        img = transforms.Resize(432, interpolation=transforms.InterpolationMode.BICUBIC)(image.convert("RGB"))
        img = transforms.CenterCrop((432, 432))(img)
        x = transforms.Normalize([0.5] * 3, [0.5] * 3)(transforms.ToTensor()(img)).to(self.device)[None]
        torch.manual_seed(0)
        z = self.vae.sample(x.unsqueeze(2)).squeeze(2).to(self.dtype)
        und = net.image_embedder_und(z) + net.position_embedding(net.image_position_ids)
        und = net.und_trans(und)["last_hidden_state"]
        image_embeds = net.fusion_proj(torch.cat([und, net.image_embedder_gen(z)], dim=-1))
        a = [self.bos] + tok("system\nYou are a helpful assistant.<|im_end|>") + tok("\n<|im_start|>user\n")
        b = [self.boi, self.eoi] + tok(question) + tok("\n<|im_start|>assistant\n")
        emb = net.showo.model.embed_tokens
        ea, eb = emb(torch.tensor([a], device=self.device)), emb(torch.tensor([b], device=self.device))
        t_emb = net.time_embed(torch.Tensor([[1.0]]).to(self.device), ea.dtype)
        if hasattr(net, "time_embed_proj"):
            t_emb = net.time_embed_proj(t_emb)
        prompt = torch.cat([ea, eb[:, :1], t_emb.reshape(1, 1, -1).to(ea.dtype), image_embeds.to(ea.dtype), eb[:, 1:]], 1)
        modality = torch.tensor([len(a) + 2, self.image_tokens], device=self.device)[None, None, :]
        end = {self.tokenizer.convert_tokens_to_ids("<|im_end|>"), self.eos, self.tokenizer.eos_token_id}
        new = []
        for _ in range(max_new_tokens):
            seq = prompt if not new else torch.cat([prompt, emb(torch.tensor([new], device=self.device))], 1)
            mask = omni_attn_mask_naive(1, seq.shape[1], modality, self.device, inverted=True).to(seq.dtype)
            nxt = int(net.showo(inputs_embeds=seq, attention_mask=mask)["logits"][0, -1].argmax())
            if nxt in end:
                break
            new.append(nxt)
        return self.tokenizer.decode(new, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------ the image side

    def noise(self, size: int = 432, seed: int = 0) -> torch.Tensor:
        """Starting noise [1, 16, 54, 54], rounded to bfloat16 as upstream does. Only 432 px is supported."""
        if size != 432:
            raise ValueError("Show-o2-7B draws at 432 px")
        g = torch.Generator().manual_seed(seed)
        return torch.randn(1, 16, LATENT, LATENT, generator=g).to(torch.bfloat16).float().to(self.device)

    def _sequence(self, ids: list[int]) -> tuple[list[int], list[int]]:
        n = len(ids) - 1  # prompt tokens after <|im_start|>
        seq = ids + [self.boi] + [self.img_pad] * self.image_tokens + [self.eoi, self.eos] + [self.pad] * (self.max_text - n)
        return seq, [len(ids) + 1, self.image_tokens]

    @torch.no_grad()
    def _velocities(self, readings: list[TextReading], x: torch.Tensor, t: float) -> torch.Tensor:
        """One pass for a batch of readings (each with its own edit) at Show-o2 time t; returns [B, 16, 54, 54]."""
        return self._velocities_graph(readings, x, t)

    def _velocities_graph(self, readings: list[TextReading], x: torch.Tensor, t: float) -> torch.Tensor:
        """_velocities without torch.no_grad, so gradients reach a tensor an edit inserts (denoise_loss)."""
        seqs, positions = zip(*(self._sequence(r.token_ids) for r in readings))
        tokens = torch.tensor(seqs, device=self.device)
        modality = torch.tensor(positions, device=self.device)[:, None, :]
        mask = omni_attn_mask_naive(len(readings), SEQ_LEN, modality, self.device).to(self.dtype)
        handles = []
        for b, r in enumerate(readings):
            if r.cache.get("edit") is not None:
                handles += self._hooks(r.cache["edit"], None, batch_index=b)
        try:
            x = x if x.shape[0] == len(readings) else x.expand(len(readings), -1, -1, -1)
            _, v = self.net(tokens, image_latents=x.to(self.dtype),
                            t=torch.full((len(readings),), t, device=self.device), attention_mask=mask,
                            modality_positions=modality, output_hidden_states=True, max_seq_len=SEQ_LEN)
        finally:
            for handle in handles:
                handle.remove()
        return v.float()

    def velocity(self, reading: TextReading, x: torch.Tensor, level: float) -> torch.Tensor:
        """Show-o2's velocity (picture - noise) at noise `level` (this code's convention), without guidance."""
        return self._velocities([reading], x, 1.0 - level)

    def denoise_loss(self, reading: TextReading, x0: torch.Tensor, level: float, noise: torch.Tensor) -> torch.Tensor:
        """Training loss for the latent x0 at noise `level` (this code's convention) with standard noise `noise`, gradients flowing into
        the reading's edit (the state-question training, scripts/state_tuning.py): the squared error of Show-o2's
        velocity, which is picture - noise (the opposite sign of BAGEL's), without guidance."""
        x_t = (1 - level) * x0 + level * noise
        with torch.enable_grad():
            v = self._velocities_graph([reading], x_t, 1.0 - level)
        return ((v.float() - (x0 - noise)) ** 2).mean()

    def guess_finished(self, reading: TextReading, x: torch.Tensor, level: float) -> torch.Tensor:
        return x + level * self.velocity(reading, x, level)

    def draw(self, reading: TextReading, size: int = 432, seed: int = 0, steps: int = DRAW_DEFAULTS["steps"],
             shift: float = DRAW_DEFAULTS["shift"], guidance: float = DRAW_DEFAULTS["guidance"],
             ) -> tuple[torch.Tensor, list[tuple[float, torch.Tensor]]]:
        """Upstream's sampler (Euler on the shifted time grid, plain guidance). Returns the final latent and, for every
        step, (noise level, guess of the finished latent)."""
        empty = self.read_nothing()
        x = self.noise(size, seed)
        ts = schedule(steps, shift).tolist()
        guesses = []
        for i in range(steps - 1):
            t = ts[i]
            if guidance > 0:
                v_c, v_u = self._velocities([reading, empty], x, t).chunk(2)
                v = v_u + guidance * (v_c - v_u)
            else:
                v = self._velocities([reading], x, t)
            guesses.append((1.0 - t, x + (1.0 - t) * v))
            x = x + (ts[i + 1] - t) * v
        return x, guesses

    def draw_many(self, readings: list[TextReading], size: int = 432, seed: int = 0, steps: int = DRAW_DEFAULTS["steps"],
                  shift: float = DRAW_DEFAULTS["shift"], guidance: float = DRAW_DEFAULTS["guidance"]) -> torch.Tensor:
        """draw() for several readings at once, all from the same starting noise; returns the final latents [n, ...].
        Each reading keeps its own edit and its own guidance branch, so every picture is the one draw() would give
        (up to bfloat16 rounding from the larger batch)."""
        n = len(readings)
        empty = self.read_nothing()
        x = self.noise(size, seed).expand(n, -1, -1, -1).clone()
        ts = schedule(steps, shift).tolist()
        for i in range(steps - 1):
            t = ts[i]
            v_c, v_u = self._velocities(list(readings) + [empty] * n, torch.cat([x, x]), t).chunk(2)
            x = x + (ts[i + 1] - t) * (v_u + guidance * (v_c - v_u))
        return x

    @torch.no_grad()
    def to_picture(self, latent: torch.Tensor) -> Image.Image:
        image = self.vae.batch_decode(latent.to(self.dtype).unsqueeze(2)).squeeze(2)[0]
        image = ((image + 1) / 2).clamp(0, 1).permute(1, 2, 0).mul(255).byte().cpu().numpy()
        return Image.fromarray(image)
