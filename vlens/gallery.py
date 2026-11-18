"""Picture grids for looking at results: a table of small pictures with row and column labels, e.g. one row per
source and one column per layer (scripts/calibrate_insertion_layer.py).
"""

from __future__ import annotations

from typing import Optional, Sequence

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def grid(cells: Sequence[Sequence[Optional[Image.Image]]], row_labels: Sequence[str] = (),
         col_labels: Sequence[str] = (), cell: int = 96, title: str = "",
         captions: Optional[Sequence[Sequence[str]]] = None) -> Image.Image:
    """Arrange pictures in a labelled table. Empty cells (None) stay blank.

    `captions`, if given, puts a short text under each picture (same shape as `cells`).
    """
    n_rows, n_cols = len(cells), max(len(r) for r in cells)
    font = _font(max(10, cell // 8))
    small = _font(max(9, cell // 10))
    strip = cell // 6 + 4 if captions else 0
    row_h = cell + strip
    left = 8 + max([int(font.getlength(s)) for s in row_labels] + [0])
    top = (cell // 3 if col_labels else 0) + (cell // 3 if title else 0)
    sheet = Image.new("RGB", (left + n_cols * cell, top + n_rows * row_h), "white")
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((left, 2), title, fill="black", font=font)
    for j, label in enumerate(col_labels):
        x = left + j * cell + cell // 2 - int(font.getlength(label)) // 2
        draw.text((x, top - cell // 3 + 2), label, fill="black", font=font)
    for i, row in enumerate(cells):
        y = top + i * row_h
        if i < len(row_labels):
            draw.text((4, y + cell // 2 - 6), row_labels[i], fill="black", font=font)
        for j, picture in enumerate(row):
            if picture is not None:
                sheet.paste(picture.convert("RGB").resize((cell - 2, cell - 2)), (left + j * cell + 1, y + 1))
            if captions and i < len(captions) and j < len(captions[i]):
                text = captions[i][j]
                while text and small.getlength(text) > cell - 4:
                    text = text[:-1]
                draw.text((left + j * cell + 2, y + cell + 1), text, fill="black", font=small)
    return sheet
