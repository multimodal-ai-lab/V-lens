#!/usr/bin/env bash
# Fetch the upstream model code at pinned commits and apply the small patches in patches/.
# Creates third_party/{bagel,show_o2,sensenova_u1,chameleon}; add third_party to PYTHONPATH (see README.md).
# Janus-Pro's code is installed as a package instead (requirements/janus.txt).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/_upstream"
mkdir -p "$SRC"

fetch() {  # url, folder, commit, paths to check out (sparse, so only these files are downloaded)
    local url=$1 dir=$2 commit=$3
    shift 3
    if [ ! -d "$SRC/$dir/.git" ]; then
        git clone --quiet --filter=blob:none --no-checkout "$url" "$SRC/$dir"
    fi
    git -C "$SRC/$dir" sparse-checkout set --no-cone LICENSE "$@"
    git -C "$SRC/$dir" checkout --quiet "$commit"
}

fetch https://github.com/ByteDance-Seed/Bagel bagel 056b5fd51a88c1eb4547318609e25d40080fcf87 /modeling/ /data/
fetch https://github.com/showlab/Show-o show-o 45a5a2de01d1ebd10cd5864d29310a76476cdf23 /show-o2/models/
fetch https://github.com/OpenSenseNova/SenseNova-U1 sensenova-u1 c35c83b58ef6e238a08b986aa58f3b73303aa42f /src/sensenova_u1/
fetch https://github.com/FoundationVision/Liquid liquid b2a7dd53bb02ef9728a848d3aa8aaf1955553b02 \
    /evaluation/chameleon/inference/image_tokenizer.py /evaluation/chameleon/inference/vqgan.py

# BAGEL: modeling/ and three data/ files under one package `bagel`; imports made package-relative, a clone that lets
# autograd pass the generation-mode attention, a switchable compute precision, and an optional OpenCV import.
rm -rf "$HERE/bagel"
mkdir -p "$HERE/bagel/data"
cp -r "$SRC/bagel/modeling" "$HERE/bagel/modeling"
cp "$SRC/bagel/data/__init__.py" "$SRC/bagel/data/data_utils.py" "$SRC/bagel/data/transforms.py" "$HERE/bagel/data/"
cp "$SRC/bagel/LICENSE" "$HERE/bagel/LICENSE"
touch "$HERE/bagel/modeling/cache_utils/__init__.py"
patch --quiet -d "$HERE/bagel" -p1 < "$HERE/patches/bagel.patch"

# Show-o2: the model files of show-o2/models; a smaller __init__.py, no timm, diffusers 0.37 compatibility.
rm -rf "$HERE/show_o2"
mkdir -p "$HERE/show_o2"
for f in __init__.py misc.py modeling_showo2_qwen2_5.py modeling_siglip.py modeling_utils.py modules.py \
         omni_attention.py qwen2.py wan21_vae.py; do
    cp "$SRC/show-o/show-o2/models/$f" "$HERE/show_o2/$f"
done
cp "$SRC/show-o/LICENSE" "$HERE/show_o2/LICENSE"
patch --quiet -d "$HERE/show_o2" -p1 < "$HERE/patches/show_o2.patch"

# SenseNova-U1: the package src/sensenova_u1, unchanged.
rm -rf "$HERE/sensenova_u1"
cp -r "$SRC/sensenova-u1/src/sensenova_u1" "$HERE/sensenova_u1"
cp "$SRC/sensenova-u1/LICENSE" "$HERE/sensenova_u1/LICENSE"

# Chameleon VQGAN decoder for Liquid's image codes (Meta's Chameleon code as shipped in Liquid's repository):
# one import made relative, one float32 cast before converting to numpy.
rm -rf "$HERE/chameleon"
mkdir -p "$HERE/chameleon"
cp "$SRC/liquid/evaluation/chameleon/inference/image_tokenizer.py" "$SRC/liquid/evaluation/chameleon/inference/vqgan.py" \
   "$HERE/chameleon/"
touch "$HERE/chameleon/__init__.py"
patch --quiet -d "$HERE/chameleon" -p1 < "$HERE/patches/chameleon.patch"

echo "third_party ready: bagel show_o2 sensenova_u1 chameleon (add $HERE to PYTHONPATH)"
