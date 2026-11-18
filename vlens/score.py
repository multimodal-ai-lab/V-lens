"""Does a picture show a given concept? CLIP ranks (used to choose the insertion layer) and summary statistics.

CLIP ranks the candidate names (the 100 concepts of configs/prompts/concepts.yaml) by their similarity to a decoded
picture; a concept is recovered when it is ranked first.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image

CLIP_ID = "openai/clip-vit-large-patch14"


class ClipJudge:
    """Zero-shot CLIP: which of the concept names fits a picture best."""

    def __init__(self, names: Sequence[str], model_id: str = CLIP_ID, device: str = "cuda",
                 template: str = "a photo of a {}"):
        from transformers import CLIPModel, CLIPProcessor

        self.names = list(names)
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            # pin "cuda" to a concrete GPU now: a model spread over several GPUs (HunyuanImage) can switch the current
            # device while drawing, and the pictures would then go to a different GPU than CLIP
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.dtype = torch.float32 if str(device) == "cpu" else torch.float16  # half precision is very slow on a CPU
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=self.dtype).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        with torch.no_grad():
            tokens = self.processor(text=[template.format(n) for n in self.names], return_tensors="pt",
                                    padding=True).to(device)
            text = self.model.get_text_features(**tokens).float()
        self.text = text / text.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def similarities(self, images: Sequence[Image.Image], batch: int = 32) -> torch.Tensor:
        """Cosine similarity of each picture to each concept name. [n_images, n_names]."""
        out = []
        for i in range(0, len(images), batch):
            pixels = self.processor(images=list(images[i:i + batch]), return_tensors="pt")["pixel_values"]
            feats = self.model.get_image_features(pixel_values=pixels.to(self.device, self.dtype)).float()
            out.append(feats / feats.norm(dim=-1, keepdim=True) @ self.text.T)
        return torch.cat(out).cpu()


def ranks(scores: torch.Tensor, targets: Sequence[int]) -> np.ndarray:
    """Rank of the target concept in each row of `scores` (1 = highest score)."""
    scores = scores.float()
    t = torch.as_tensor(list(targets))
    target_scores = scores[torch.arange(len(t)), t]
    return (1 + (scores > target_scores[:, None]).sum(1)).numpy()
