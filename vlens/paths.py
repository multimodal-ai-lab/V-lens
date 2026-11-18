"""Where the code finds model weights and keeps its data.

Every location comes from an environment variable, with a default inside the repository:

- VLENS_MODELS: checkpoints that are not loaded by Hugging Face repository id (the Wan2.1 VAE used by Show-o2, the
  Chameleon VQGAN used by Liquid, HunyuanImage-3.0-Instruct); see README.md. Default: <repo>/models.
- VLENS_DATA: fitted lens statistics and maps (lenses/) and saved hidden states of the statistics passages (states/).
  Default: <repo>/data.
- VLENS_OUT: experiment outputs, one folder per experiment. Default: <repo>/outputs.
- HF_HOME: the Hugging Face cache (the library's usual default when unset).
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("VLENS_MODELS", REPO / "models"))
DATA = Path(os.environ.get("VLENS_DATA", REPO / "data"))
OUT = Path(os.environ.get("VLENS_OUT", REPO / "outputs"))
HF_HUB = (Path(os.environ["HF_HOME"]) if "HF_HOME" in os.environ else Path.home() / ".cache" / "huggingface") / "hub"
#: fitted statistics and maps, one folder per name
LENS_STORE = DATA / "lenses"


def snapshot(repo: str) -> Path:
    """The local snapshot of a Hugging Face model repository ("owner/name"): the newest complete one in the cache,
    downloaded first when the cache has none."""
    folder = HF_HUB / f"models--{repo.replace('/', '--')}" / "snapshots"
    if folder.is_dir():
        snaps = [p for p in folder.iterdir() if (p / "config.json").exists()] or list(folder.iterdir())
        if snaps:
            return sorted(snaps, key=lambda p: p.stat().st_mtime)[-1]
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo))
