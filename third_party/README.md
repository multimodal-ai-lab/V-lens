# Upstream model code

Nothing is vendored here. `bash third_party/setup.sh` fetches the upstream repositories at the pinned commits (sparse
checkouts, a few MB), copies the needed files into this folder and applies the patches in `patches/`. Add this folder to
`PYTHONPATH` afterwards (see the main README). Each copied folder keeps the upstream LICENSE.

| Folder | Upstream (commit) | Files | Licence | Changes (`patches/`) |
|---|---|---|---|---|
| `bagel/` | [ByteDance-Seed/Bagel](https://github.com/ByteDance-Seed/Bagel) (`056b5fd51a88c1eb4547318609e25d40080fcf87`) | `modeling/`, `data/{__init__,data_utils,transforms}.py` | Apache-2.0 | `bagel.patch`: imports rewritten from `modeling.*`/`data.*` to `bagel.modeling.*`/`bagel.data.*` (one package); a `.clone()` of the generation-mode attention output before its in-place row writes, so gradients pass through the image side (forward result unchanged); a module-level `COMPUTE_DTYPE` (default bfloat16, i.e. upstream behaviour) with a plain-PyTorch attention fallback for float32 gradient checks; `import cv2` made optional in `data/transforms.py` |
| `show_o2/` | [showlab/Show-o](https://github.com/showlab/Show-o) (`45a5a2de01d1ebd10cd5864d29310a76476cdf23`) | `show-o2/models/` (8 model files and `__init__.py`) | Apache-2.0 | `show_o2.patch`: `__init__.py` imports only what the adapter uses; `to_2tuple` defined locally instead of imported from timm; `load_state_dict(model_file)` without `variant` (diffusers 0.37) |
| `sensenova_u1/` | [OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1) (`c35c83b58ef6e238a08b986aa58f3b73303aa42f`) | `src/sensenova_u1/` | Apache-2.0 | none |
| `chameleon/` | [FoundationVision/Liquid](https://github.com/FoundationVision/Liquid) (`b2a7dd53bb02ef9728a848d3aa8aaf1955553b02`), Meta's Chameleon VQGAN code as shipped there | `evaluation/chameleon/inference/{image_tokenizer,vqgan}.py` | Chameleon License (Meta), see [facebookresearch/chameleon](https://github.com/facebookresearch/chameleon) | `chameleon.patch`: one import made relative (`from .vqgan import VQModel`), a float32 cast before `.numpy()`, whitespace |

Janus-Pro's code ([deepseek-ai/Janus](https://github.com/deepseek-ai/Janus), commit
`1daa72fa409002d40931bd7b36a9280362469ead`, MIT code licence, DeepSeek model licence) is used unchanged and installed as a
package in its own environment (`requirements/janus.txt`). HunyuanImage-3.0-Instruct and Liquid run on the code that
ships with their Hugging Face checkpoints (`trust_remote_code` / plain transformers).

The adapters in `vlens/models/` call the lower-level pieces of these codebases (for example BAGEL's text pass and
velocity function, which upstream wraps in `@torch.no_grad`) so that hidden states can be recorded, replaced and
differentiated.
