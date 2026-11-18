"""Draw cells (a word position read at a layer), in several ways, with several seeds: the drawing step of the lens
variants (Table 1, Supp. Tables S2, S3), the ablation (Table 5), the context tests (Table 2, Supp. Tables S7, S8, I) and
the judge calibration pictures.

Items come from a YAML file in configs/prompts/ (vlens/cells.py): two_hop (positions: the question's last word and the
description's last word), double_meaning (both occurrences of the word, both sentence orders) or reads (explicitly named
words of any text).

Ways to fill the slot ("thing" in the drawing request, at the insertion layer R):
- word:  no slot edit; the request with the word itself in place of "thing" (Table 1: "Word (no state)");
- copy:  the hidden state exactly as it is at its own layer ("Unchanged copy");
- moved: the state minus its layer's average, set to the typical insertion-layer distance, plus the insertion layer's
         average ("Moved copy", D = I);
- copy:<variant>: a copy without a map (vlens/carry.py): copy:diag is the V-lens (Eq. 1), copy:rel and copy:pct the
         relative and percentile sizes; copy:diag-<ablation> the V-lens with one component ablated (Table 5);
- fnorm: the state through the final normalization of the model's drawing pathway (learned gain included, as Diffusion
         Lens normalizes intermediate text-encoder layers), inserted unchanged; fnorm:sR the same normed state placed at
         the typical insertion-layer distance (mu_R + s_R z / |z|) (Supp. Table S3);
- vlens: an average-Jacobian map (vlens/transport.py) with the transport named by --transport (default: the model's
         standard one), size matched; several transports comma separated give the ways vlens:<name>. The
         "Jacobian map with gain" variant is vlens:<model>_transport_to<R>_n64_selfgain (scripts/self_gain.py).

Writes one PNG per picture into $VLENS_OUT/<folder>/<model>/ and one record file per job,
records_<items>_<job>.json (item, text, position, word, layer, way, seed, file, truth): jobs running side by side must
not share a record file. Pictures already drawn are skipped (a picture on disk without a record gets its record without
being drawn again), so a stopped job resumes and jobs can share items. --rebuild writes the records again from the
pictures on disk (no model).

The lens-variant cells of the paper (Test; Val uses --items heldout_items --test-set val and the folders
heldout_cells, or heldout_cells_thing for SenseNova-U1):
    python scripts/draw_cells.py --model bagel --items fresh2_items --kind two_hop --test-set test \\
        --layers 2,4,6,8,10,14,20 --ways word,copy,moved,copy:diag,vlens \\
        --transport bagel_transport_to8_n64,bagel_transport_to8_n64_selfgain --seeds 0,1,2 --folder fresh2_cells
    python scripts/draw_cells.py --model sensenova --request thing --items fresh2_items --kind double_meaning \\
        --layers 8,12,16,20,24,28,36 --ways word,copy,moved,copy:diag --seeds 0,1,2 --folder fresh2_cells
"""

import argparse
import json
import time

from vlens import prompts, results
from vlens.cells import TARGET, all_records, picture_name, texts, word_tokens  # noqa: F401
from vlens.hidden import find_word, replace_at
from vlens.models import load, no_system_message, transport_name
from vlens.transport import Transport


def rebuild(args, out, layers, seeds, only):
    """Records for every picture of these items that exists on disk (token positions are not known without the model)."""
    records = []
    for rec in texts(args.items, args.kind, only):
        for pos_name, word, occurrence, truth in rec["positions"]:
            for png in sorted(out.glob(f"{args.items}_{rec['item']:03d}{rec.get('order') or ''}_{pos_name}_L*_s*.png")):
                stem = png.stem[len(f"{args.items}_{rec['item']:03d}{rec.get('order') or ''}_{pos_name}_"):]
                layer_part, rest = stem.split("_", 1)
                way, seed = rest.rsplit("_s", 1)
                records.append(dict(file=f"{args.model}/{png.name}", model=args.model, items=args.items,
                                    kind=args.kind, item=rec["item"], order=rec.get("order"), text=rec["text"],
                                    position=pos_name, word=word, token=None, layer=int(layer_part[1:]),
                                    way=way.replace("-", ":", 1) if way.startswith(("vlens-", "copy-", "fnorm-")) else way,
                                    seed=int(seed), truth=truth))
    path = out / f"records_{args.items}_{args.only or 'all'}.json"
    path.write_text(json.dumps(records, indent=1))
    print(f"rebuilt {len(records)} records -> {path}")


def main():
    ap = argparse.ArgumentParser(description="Draw chosen cells in several ways and seeds.")
    ap.add_argument("--model", default="bagel")
    ap.add_argument("--items", required=True, help="YAML file in configs/prompts/ (without .yaml)")
    ap.add_argument("--kind", required=True, choices=["two_hop", "double_meaning", "reads"])
    ap.add_argument("--only", default="", help="item indices, comma separated (default: all)")
    ap.add_argument("--test-set", default="", choices=["", "val", "test"],
                    help="two_hop only: draw the questions this model answers correctly, configs/test_sets/<name>.json "
                         "(sets --only)")
    ap.add_argument("--layers", required=True, help="source layers, comma separated")
    ap.add_argument("--ways", default="word,copy,moved,vlens")
    ap.add_argument("--transport", default="", help="transport names for vlens, comma separated (default: standard)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--folder", required=True, help="output folder under $VLENS_OUT")
    ap.add_argument("--rebuild", action="store_true", help="write the records from the pictures on disk and stop")
    ap.add_argument("--positions", default="", help="only these positions, comma separated (default: all)")
    ap.add_argument("--request", default=TARGET, help='the drawing request around the slot "thing" (default: '
                                                       '"a photo of a thing"; "thing" alone works too)')
    ap.add_argument("--no-system", action="store_true",
                    help="SenseNova-U1 only: read the request without its fixed generation system message (not used "
                         "for the paper's numbers)")
    ap.add_argument("--batch", type=int, default=1,
                    help="draw the slot ways of one text position this many at a time with the model's draw_many (same "
                         "seed per batch; pictures equal single draws up to GPU rounding)")
    ap.add_argument("--insert", type=int, default=None,
                    help="another insertion layer for the V-lens (copy:diag only; not used for the paper's numbers)")
    args = ap.parse_args()
    if args.test_set:
        sets = json.loads((prompts.REPO / "configs" / "test_sets" / f"{args.test_set}.json").read_text())
        args.only = ",".join(str(i) for i in sets[args.model])
    out = results.folder(args.folder) / args.model
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    only = [int(x) for x in args.only.split(",")] if args.only else None
    if args.rebuild:
        return rebuild(args, out, layers, seeds, only)
    m = load(args.model)
    if args.no_system:
        no_system_message(m)
    # the Jacobian map is needed only for the ways that use it; models without one (Liquid, Janus-Pro, HunyuanImage) run
    # the word and the V-lens (copy:diag)
    needs_map = any(w in ("vlens", "moved") for w in args.ways.split(","))
    transports = {spec: Transport.load(spec.split("@")[0]) for spec in (args.transport.split(",") if args.transport
                                                                      else [transport_name(args.model)])} if needs_map else {}
    copies = None
    if any(w.startswith(("copy:", "fnorm")) for w in args.ways.split(",")):
        from vlens.carry import CopyTransform
        copies = CopyTransform(args.model)
    norm = None
    if any(w.startswith("fnorm") for w in args.ways.split(",")):  # the final norm of the drawing pathway
        from vlens.carry import FINAL_NORM, final_norm, fnorm
        norm = final_norm(m, args.model)
        print(f"fnorm: {FINAL_NORM[args.model]} = {norm}, gain mean {norm.weight.float().mean():.4f} "
              f"(min {norm.weight.float().min():.4f}, max {norm.weight.float().max():.4f})", flush=True)
    if needs_map:
        standard = next(iter(transports.values()))
        R, spread = standard.R, standard.meta["spread"]
    else:
        R = copies.s["R"]
    if args.insert is not None:
        assert set(args.ways.split(",")) <= {"word", "copy:diag"}, "--insert: only the ways word and copy:diag"
        R = args.insert
    target = m.read_prompt(args.request)
    slot = find_word(target, "thing")[-1]
    ways = []
    for w in args.ways.split(","):
        ways += [f"vlens:{name}" for name in transports] if w == "vlens" else [w]
    job = (f"{args.items}_{args.only or 'all'}_L{args.layers}_{args.ways}_{args.transport or 'standard'}"
           + (f"_{args.positions}" if args.positions else ""))
    records_path = out / f"records_{job.replace(',', '-').replace(':', '-')}.json"  # one file per job, see the docstring
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    done = {r["file"] for r in all_records(out)}
    t0 = time.time()
    batch = args.batch if args.batch > 1 and hasattr(m, "draw_many") else 1
    pending = []  # (file, reading, seed, record): drawn by flush(), and only then recorded

    def flush():
        for seed_ in sorted({p[2] for p in pending}):
            group = [p for p in pending if p[2] == seed_]
            for k in range(0, len(group), batch):
                chunk = group[k:k + batch]
                xs = m.draw_many([p[1] for p in chunk], seed=seed_)
                for i, (f, _, _, r) in enumerate(chunk):
                    m.to_picture(xs[i:i + 1]).save(out / f)
                    records.append(r)
                    done.add(r["file"])
        pending.clear()

    for rec in texts(args.items, args.kind, only):
        src = m.read(rec["text"])
        for pos_name, word, occurrence, truth in rec["positions"]:
            if args.positions and pos_name not in args.positions.split(","):
                continue
            positions = [p for p in range(src.n_tokens) if src.content[p]]
            covers = word_tokens(src, word, occurrence)
            pos = covers[-1]
            assert pos in positions, (rec["text"], word)
            for layer in layers:
                h = src.hidden[layer][pos].float()
                for way in ways:
                    for seed in seeds:
                        file = picture_name(args.items, rec, pos_name, layer, way, seed)
                        if f"{args.model}/{file}" in done or (way == "word" and layer != layers[0]):
                            continue  # the word on its own does not depend on the layer
                        if (out / file).exists():
                            pass  # drawn by a job that stopped before writing its records, or by one running beside
                        elif way == "word":
                            reading = m.read_prompt(args.request.replace("thing", word))
                            m.to_picture(m.draw(reading, seed=seed)[0]).save(out / file)
                        else:
                            if way == "copy":
                                v = h
                            elif way == "moved":
                                d = h - standard.mean[layer].to(h.device).float()
                                v = standard.mean[R].to(h.device).float() + d * (spread / d.norm())
                            elif way.startswith("copy:"):  # a copy without a map, vlens/carry.py
                                v = copies.carry(h, layer, way.split(":", 1)[1], R=R)
                            elif way == "fnorm":  # the final norm of the drawing pathway, vlens/carry.py
                                v = fnorm(h, norm)
                            elif way == "fnorm:sR":
                                v = fnorm(h, norm, copies.s["spread"], copies.s["mean"][R])
                            else:
                                spec = way.split(":", 1)[1]
                                size = spec.split("@")[1] if "@" in spec else "fixed"
                                v = transports[spec].to_reading_layer(h[None].cpu(), layer, match_size=True, size=size)[0]
                            reading = m.read_prompt(args.request, edit=replace_at(R, [slot], v[None].to(m.device)))
                            if batch > 1:
                                pending.append((file, reading, seed, dict(
                                    file=f"{args.model}/{file}", model=args.model, items=args.items, kind=args.kind,
                                    item=rec["item"], order=rec.get("order"), text=rec["text"], position=pos_name,
                                    word=word, token=pos, layer=layer, way=way, seed=seed, truth=truth)))
                                continue
                            m.to_picture(m.draw(reading, seed=seed)[0]).save(out / file)
                        records.append(dict(file=f"{args.model}/{file}", model=args.model, items=args.items,
                                            kind=args.kind, item=rec["item"], order=rec.get("order"), text=rec["text"],
                                            position=pos_name, word=word, token=pos, layer=layer, way=way, seed=seed,
                                            truth=truth))
                        done.add(f"{args.model}/{file}")
            flush()
            records_path.write_text(json.dumps(records, indent=1))
        print(f"[{time.time() - t0:6.0f}s] item {rec['item']} {rec.get('order', '')}: {len(records)} pictures", flush=True)
    print("->", out)


if __name__ == "__main__":
    main()
