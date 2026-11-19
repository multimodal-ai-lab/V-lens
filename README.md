# V-lens: drawing the hidden states of unified models

**[Project page](https://multimodal-ai-lab.github.io/V-lens/)** · **[Paper (arXiv)](https://arxiv.org/abs/2611.24793)**

Code for the paper "The Elephant in the Question: Unified Models Picture the Concepts Behind Their Answers" by Tobias
Braun, Marcus Rohrbach and Lorenzo Cavallaro. The V-lens reads a token's hidden state from the text computation
of a unified text-and-image model and draws it: the state is centred, rescaled coordinate by coordinate to the
statistics of a fixed insertion layer R (Eq. 1), and placed into the slot "thing" of a drawing request such as
"a photo of a thing", after which the model's own image generator draws it. The repository contains the model adapters
(BAGEL, Show-o2, SenseNova-U1, HunyuanImage-3.0, Liquid, Janus-Pro), the transport and its statistics, reading and
inserting hidden states, the judge panel and all scoring, lens tuning (LoRA adapters trained through the frozen
generator), and one entry point per paper result. No pictures, results or weights are included.

## Layout

```
vlens/              the package
  models/           one adapter per model: read a text with the hidden state entering every layer, edit states,
                    draw, answer; BAGEL, Show-o2, SenseNova-U1, HunyuanImage-3.0, Liquid, Janus-Pro
  carry.py          the V-lens transport (Eq. 1), the simpler copies, the final-norm variant and the ablations
  transport.py      average-Jacobian maps (the Jacobian lens variants), layer averages and radius
  hidden.py         edits (replace a state at a layer) and word positions
  cells.py          items, positions, record files and picture names shared by drawing and scoring
  readout.py        the V-lens from any layer, batched drawing, one-word answers (depth, held thoughts, two states)
  judges.py         the three multimodal judges and their questions
  lora.py           switchable LoRA adapters for lens tuning
  text_lens.py, lens_fit.py   logit lens, text J-lens and its fitter (the text readouts of Supp. D)
  score.py, prompts.py, paths.py, results.py, gallery.py
scripts/            one or a few entry points per paper result (table below)
configs/prompts/    the item files of every experiment (texts, words to read, intended objects)
configs/test_sets/  which two-hop questions each model answers correctly (val = Table 1 validation, test = Table 1 test
                    and Table 5)
requirements/       one requirements file per model environment
third_party/        setup.sh fetches the upstream model code at pinned commits and applies small patches
tests/              CPU tests (tiny model in float64)
```

Item files (`configs/prompts/`): `concepts.yaml` and `ladder.yaml` (calibration names and one-hop sources),
`vlens_items.yaml` (calibration answers), `text_lens_passages.txt` (300 plain-text passages for all statistics and text
lenses), `heldout_items.yaml` and `fresh2_items.yaml` (Val and Test of Table 1 and Table 5), `subject_calibration.yaml`
(Table S5), `followup_matched.yaml`, `followup_reference.yaml`, `followup_meanings.yaml` (Table 2, S7, S8, S24),
`bridges.yaml` and `bridge_confirm.yaml` (Fig. 4, Supp. D), `held_hunyuan.yaml` (Sec. 4.4), `held_search.yaml`
(Supp. J), `two_slots.yaml`
(Supp. E), `vsup_nouns.yaml`, `vsup_items.yaml`, `vsup_targets.yaml` (Sec. 4.6, Supp. F, Table S23),
`state_questions.yaml`, `state_questions_v2.yaml` and `state_questions_v5.yaml` (Sec. 4.7, Table 4, Supp. F.4,
F.5; the last holds the 80 stories of Table 4, the second set's 20 scenarios and 20 added ones).

## Setup

### 1. Environments

The six models need different library versions, so there is one environment per group (Python 3.11, CUDA 12 GPUs).
Each file in `requirements/` says how to build its environment; install this package into each one with
`pip install -e . --no-deps`.

| Environment | File | Used for |
|---|---|---|
| main | `requirements/main.txt` | BAGEL, Show-o2, Liquid, CLIP (calibration), every CPU scoring script, tests |
| sensenova | `requirements/sensenova.txt` | SenseNova-U1; the Qwen3-VL and Gemma-3 judges |
| judges | `requirements/judges.txt` | the LLaVA-OneVision-2 judge (transformers 5) |
| hunyuan | `requirements/hunyuan.txt` | HunyuanImage-3.0-Instruct |
| janus | `requirements/janus.txt` | Janus-Pro (upstream package installed from its pinned commit) |

### 2. Upstream model code

```bash
bash third_party/setup.sh                       # BAGEL, Show-o2, SenseNova-U1 and the Chameleon VQGAN code
export PYTHONPATH=$PWD:$PWD/third_party
```

`third_party/README.md` lists the pinned commits and what each patch changes.

### 3. Weights and data

Models load from the Hugging Face Hub by repository id (downloaded on first use into the usual cache, or set
`HF_HOME`). A few files are not loaded by id and go into `$VLENS_MODELS`:

| Model | Hugging Face repository | Extra files in `$VLENS_MODELS` |
|---|---|---|
| BAGEL | `ByteDance-Seed/BAGEL-7B-MoT` | |
| Show-o2 | `showlab/show-o2-7B` (also pulls `Qwen/Qwen2.5-7B-Instruct`, `google/siglip-so400m-patch14-384`) | `Wan2.1_VAE.pth` from `Wan-AI/Wan2.1-T2V-14B` |
| SenseNova-U1 | `sensenova/SenseNova-U1-8B-MoT` | |
| HunyuanImage-3.0 | `tencent/HunyuanImage-3.0-Instruct` | the whole checkpoint as `HunyuanImage3Instruct/` (its remote code fails on a folder name with dots) |
| Liquid | `Junfeng5/Liquid_V1_7B` | `chameleon_vqgan/vqgan.ckpt` and `vqgan.yaml` from the Space `Junfeng5/Liquid_demo` (folder `chameleon/`) |
| Janus-Pro | `deepseek-ai/Janus-Pro-7B` | |
| CLIP (calibration) | `openai/clip-vit-large-patch14` | |
| Judges | `Qwen/Qwen3-VL-30B-A3B-Instruct`, `google/gemma-3-27b-it` (gated: accept its licence), `lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct` | |

```bash
export VLENS_MODELS=/path/to/models    # extra checkpoints (table above)
export VLENS_DATA=/path/to/data        # lens statistics, Jacobian maps, saved passage states (a few GB per model)
export VLENS_OUT=/path/to/outputs      # pictures, records, judge answers, scores (one folder per experiment)
huggingface-cli download Wan-AI/Wan2.1-T2V-14B Wan2.1_VAE.pth --local-dir $VLENS_MODELS
huggingface-cli download tencent/HunyuanImage-3.0-Instruct --local-dir $VLENS_MODELS/HunyuanImage3Instruct
huggingface-cli download Junfeng5/Liquid_demo chameleon/vqgan.ckpt chameleon/vqgan.yaml --repo-type space --local-dir $VLENS_MODELS/_liquid
mkdir -p $VLENS_MODELS/chameleon_vqgan && mv $VLENS_MODELS/_liquid/chameleon/vqgan.* $VLENS_MODELS/chameleon_vqgan/
```

Defaults without these variables: `models/`, `data/` and `outputs/` inside the repository.

Datasets of the capability check (Supp. F.3) are fetched by the scripts and checked by checksum: MMLU-Pro
(`TIGER-Lab/MMLU-Pro`, pinned revision, `scripts/benchmarks.py prep`) and the GenEval prompt file (repository
`djghosh13/geneval`, pinned commit, `scripts/capability.py prep`).

## How the pieces fit

1. **Lens statistics, once per model.** The V-lens needs the layer means mu_l, per-coordinate spreads sigma_l and the
   radius s_R, all from plain-text passages (`configs/prompts/text_lens_passages.txt`, documents of the Pile, unrelated
   to the test items), never from pictures.
   - BAGEL, Show-o2, SenseNova-U1 (insertion layers 8, 27, 24 are set in `vlens/models/__init__.py`):
     ```bash
     python scripts/fit_transport.py --model bagel --shard $i --of 32       # i = 0..31 (GPU); also fits the Jacobian maps
     python scripts/fit_transport.py --model bagel --of 32 --merge          # CPU: layer means
     python scripts/fit_transport.py --model bagel --validate               # GPU: radius s_R on 20 held-out passages
     python scripts/collect_states.py --model bagel --splits fit,heldout,calib
     python scripts/build_lens_stats.py --model bagel                       # CPU: $VLENS_DATA/lenses/bagel_copy_stats
     python scripts/self_gain.py --model bagel --save                       # only for the "Jacobian + gain" variant
     ```
   - HunyuanImage-3.0, Liquid, Janus-Pro: calibrate the insertion layer first (it is read back from the calibration),
     then `python scripts/lens_stats_own_tokens.py --model hunyuan` (or `liquid`, `janus_pro`).
2. **Drawing.** `scripts/draw_cells.py` (and the experiment scripts with their own `draw` steps) read a text, take the
   hidden state of a word at an extraction layer, carry it to R and draw it with seeds 0, 1, 2 (depth study: 3, 4, 5).
   Every picture gets a record (text, word, layer, way, seed, intended object) next to it in `$VLENS_OUT/<folder>/<model>/`.
3. **Judging.** `scripts/run_judge.py --judge {qwen3vl,gemma3,llavaov2} --set cells:<folder>` asks each judge about every
   picture of a folder. `--subject` is the main-subject question of Sec. 4.1 (intended object, alternative, both,
   neither); without it, the choice question used for the lens-variant and ablation tables. Some experiments run the
   judges through their own `judge` step with the same `vlens.judges` code. Qwen3-VL and Gemma-3 run in the sensenova
   environment, LLaVA-OneVision-2 in the judges environment.
4. **Scoring** (CPU): a picture is correct when at least two of three judges choose the intended object; a state when two
   of three seeds are correct; pairs, words and questions as described in the paper. Each `score*` script prints the
   table and writes JSON into the experiment's output folder.

Most drawing steps accept `--shard i --of n` to split the work over GPUs, skip pictures that already exist, and can be
stopped and restarted.

## Paper results and commands

`M` stands for each model of the result (bagel, show_o2, sensenova; hunyuan, liquid, janus_pro where listed), `J` for
each judge (qwen3vl, gemma3, llavaov2). All scripts are in `scripts/`; `--help` lists every option. Output paths are
relative to `$VLENS_OUT`.

| Paper result | Command(s) | Expected output |
|---|---|---|
| Insertion layers R (Sec. 3, Supp. A.5, Table S4; Supp. I for Liquid and Janus-Pro) | `calibrate_insertion_layer.py --model M --shard i --of n`; BAGEL with `--layers 0,2,4,8,12,16,20,24,27`, Show-o2 with `--layers 0,2,4,6,8,10,12,14,16,20,24,27`, the others with the default layers | `check_<model>/numbers_*.json`: CLIP rank of each of the 12 words per layer; R = latest layer with the most words ranked first (8, 27, 24, 9; Liquid 6, Janus-Pro 8) |
| V-lens transport, Eq. 1 (Sec. 3, Supp. A.2) | step 1 above (`fit_transport.py`, `collect_states.py`, `build_lens_stats.py`; `lens_stats_own_tokens.py`) | `$VLENS_DATA/lenses/<model>_copy_stats/stats.pt` (mu, sigma, s_R, R); the transport itself is `vlens/carry.py` (`CopyTransform.carry(h, layer, "diag")`) |
| Which two-hop questions each model answers (used by Table 1, Table 5, Supp. Table S2) | `check_answers.py --model M --items heldout_items` (and `fresh2_items`) | `heldout_answers/`, `fresh2_answers/`; the selections after a check by eye are shipped as `configs/test_sets/{val,test}.json` |
| Lens variants: Table 1, Supp. Tables S2 and S3, Supp. A.3 and A.4 (BAGEL, Show-o2, SenseNova-U1) | `draw_cells.py --model M --items heldout_items --kind two_hop --test-set val --layers <L_M> --ways word,copy,moved,copy:rel,copy:pct,copy:diag,vlens --transport <T_M>,<T_M>_selfgain --folder heldout_cells` and the same with `--kind double_meaning` (no `--test-set`); Test: `--items fresh2_items --test-set test --ways word,copy,moved,copy:diag,vlens ... --folder fresh2_cells`; final norm: `--ways fnorm,fnorm:sR --folder fnorm_val_cells` / `fnorm_test_cells`; SenseNova-U1 adds `--request thing` (Val folder `heldout_cells_thing`); then `run_judge.py --judge J --set cells:<folder>`, `score_cells.py --folder <folder>`, `score_variant_choice.py`, `score_variants.py scale`, `score_variants.py table` | `fnorm/table.json` and printed tables: Val 0.16 / 0.17 / 0.48 / 0.49 / 0.51 / 0.54 and Test 0.03 / 0.11 / 0.25 / 0.27 / 0.25 / 0.33 (Table 1), the per-model counts of Table S2, final norm 0.140 and 0.183 against 0.467 (Table S3); `score_variant_choice.py` prints the Val choice of the coordinate scaling |
| Lens ablation, Table 5 | `lens_ablation.py stats`; `draw_cells.py --model M --items fresh2_items --kind two_hop --test-set test --layers <L_M> --ways copy:diag-nocenter,copy:diag-nomuR,copy:diag-srconly,copy:diag-dstonly --folder lens_ablation_test_cells` (SenseNova-U1 `--request thing`); `run_judge.py --judge J --set cells:lens_ablation_test_cells`; `score_cells.py --folder lens_ablation_test_cells`; `score_ablation.py` (`lens_ablation.py check` and `table` give the CPU checks and both test sets) | `ablation/slim_table.{txt,json}`: BAGEL 10 / 9 / 0 / 0 / 8 / 0 of 35, Show-o2 15 / 13 / 12 / 11 / 11 / 9 of 30, SenseNova-U1 8 / 7 / 1 / 1 / 4 / 1 of 39, mean 0.33 / 0.29 / 0.14 / 0.13 / 0.23 / 0.11 |
| Judge calibration: Sec. 4.1, Supp. B, Table S5 | `draw_judge_calibration.py --model M`; `run_judge.py --judge J --set cells:subject_calibration --subject`; `score_judge_calibration.py` | `subject_calibration/scores.json`: panel accuracy per picture type, 146/156 overall |
| Context tests: Table 2, Supp. Tables S7 and S8 | for `--items followup_matched --kind reads`, `--items followup_reference --kind reads` and `--items followup_meanings --kind double_meaning`: `draw_cells.py --model M --items ... --kind ... --layers <C_M> --ways word,copy:diag,copy,vlens --transport <T_M> --seeds 0,1,2 --folder followup_cells` (SenseNova-U1 `--request thing`; HunyuanImage `--layers 9,12,15 --ways word,copy:diag`); `run_judge.py --judge J --set cells:followup_cells --model M --subject`; `score_context.py`; `score_context.py --models hunyuan --out scores_hunyuan.json` | `followup_cells/scores.json`, `scores_hunyuan.json`: pairs or words right at R, label-swap chance, one-sided p, 95% interval, other layers, pronoun subtypes and controls |
| Autoregressive models: Sec. 4.5, Supp. I, Table S24 | as the context tests with `--model liquid --layers 6,10,14` and `--model janus_pro --layers 8,12,16`, `--ways word,copy:diag`; `score_context.py --models liquid,janus_pro --out scores_autoregressive.json` | `followup_cells/scores_autoregressive.json`: two meanings 12/20 (Liquid) and 11.5/20 (Janus-Pro) |
| Text readouts: Supp. D (Table S12), Sec. 4.3 | `fit_text_jlens.py --model M --n-passages 200 --shard i --of 25`, then `--of 25 --merge` (not HunyuanImage); `fit_tuned_lens.py --model M`; context tests: `text_lens_ranks.py --model M --set followup` (also hunyuan), `score_text_lenses.py`; depth study (after the depth row below): `text_readout_depth.py read --model M` (also hunyuan), `text_readout_depth.py score` | `followup_cells/text_lens_scores.json` (the table's context columns: the keys ending in `top10`, at the insertion layer); `text_readout_depth/scores.json` (the depth columns: `top10` at the description's last word at R and the best later token at the best layer) |
| Depth over tokens and layers: Fig. 4, Sec. 4.3, Supp. D (Fig. S1, Tables S9, S10, S11) | fixed positions: `depth_fixed_positions.py draw --model M --shard i --of n`, judges `run_judge.py --judge J --set cells:bridge_confirm --subject --model M`, `depth_fixed_positions.py score`; every question token: `depth_all_tokens.py draw ...`, judges on `cells:bridge_alltokens`, `depth_all_tokens.py score`; through the first answer token: `depth_through_answer.py plan`, `draw --seeds 3,4 --tag a`, judges on `cells:alltokens_answer`, `need`, `draw --need <need.json> --seeds 5 --tag b`, judges, `score`; `depth_token_report.py` (HunyuanImage steps in each script's `--help`); the autoregressive models (Fig. 4's Liquid panel): `depth_autoregressive.py answers --model M`, `draw --model M --shard i --of n`, judges on `cells:depth_autoregressive`, `depth_autoregressive.py score` | `bridge_confirm/scores.json` (Tables S9, S10, Fig. S1, Show-o2 layer 24 minus 27: +0.73 [0.53, 0.93]); `alltokens_answer/scores.json` (Fig. 4 best later token per layer, Table S11 token types); `depth_autoregressive/scores.json` (Fig. 4 Liquid panel: description word 11/13 at R; Supp. I, Table S25 for both models) |
| Held thoughts: Sec. 4.4 and its figure; the systematic search of Supp. J | Show-o2: `held_thoughts.py reply --model show_o2`, `cells --model show_o2 --spec progress_short:the:1,progress_short:.:1 --layers 22,23,24,25,26 --cells cells_confirm.json`, `draw --model show_o2 --cells cells_confirm.json --conditions think,mention,none --seeds 0,1,2`, `judge --model show_o2 --judge J`, `score --model show_o2`; HunyuanImage: `reply`, `register`, `draw --cells cells_register.json --seeds 0`, judge, score, then `cells --model hunyuan --spec bus:.:1 --layers 23,24,25,26,27 --cells cells_confirm_bus.json` and the same draw, judge, score (needs `fit_tuned_lens.py` for both models). Supp. J: `held_search.py reply --model M`, `plan`, `draw --shard i --of n`, `judge --judge J`, for HunyuanImage then `adapt`, `draw`, `judge`; `score --model show_o2,hunyuan` | `held_register_show_o2/` and `held_hunyuan/`: copy similarity per trial (kept at >= 0.85), judged pictures per condition and layer (bicycle at "the", layer 25; bread roll at the final period, layer 25); `held_search/scores_<model>.json`: kept copies, successful token and layer combinations by layer and token kind (Show-o2 97 of 105 at "the"; HunyuanImage 12 of 13 at the final period), sanity checks |
| Two states: Supp. E (Fig. two states, Table S13) | `two_states.py draw --model M --shard i --of n`, `judge --model M --judge J`, `score`; `two_states_layers.py draw/judge/score` (a layer sweep not shown in the paper); `two_states_presence.py judge --judge J --model M --set main` and `--set layers`, `two_states_presence.py score`; HunyuanImage: `score --models hunyuan --out scores_hunyuan.json` in each | `two_slots/`, `two_slots_layers/`, `object_presence/` scores: pairs right per condition (Table S13) and per extraction layer and route (layer sweep) |
| Lens tuning on pronouns: Sec. 4.6, Table 3, Supp. F.1, F.2 (Tables S14 to S18), items Table S23 | `make_tuning_items.py` (reproduces `vsup_items.yaml`); per model: `lens_tuning.py targets`, `text --arm 0 --groups train,dev,test1,test2,test3`, judges `run_judge.py --judge J --set cells:vsup_targets --subject`, `latents`, `states`, `train --arm A --seed s` (A in T, V, G, R, S; s in 0, 1, 2; HunyuanImage `--updates 200 --batch 16`, arms T, V, G), `draw --arm A --seed s`, `text --arm A --seed s`, `generate --arm A --seed s`, judges on `cells:vsup_cells` and `cells:vsup_gen` with `--subject`; `score_lens_tuning.py --model M` | `vsup_pilot/scores[_<model>].json`: lens success per test set (test1 = Phrases, test2 = Verbs, test3 = Nouns), text answers, preservation, ordinary drawing and the paired intervals (Table 3: BAGEL 0.70 / 0.93 / 0.96 after lens tuning) |
| Capability check: Supp. F.3, Table S19 | `benchmarks.py prep`; `capability.py prep`; `capability.py run --model M --arm A` (A in base, T, G, V; training seed 0; M in bagel, sensenova, hunyuan); `capability.py judge --judge J`; `capability.py score` | `capability/table.txt`: GenEval pass rate (120 prompts, three-judge scoring) and MMLU-Pro accuracy (300 questions) with changes and 95% paired intervals |
| State stories: Sec. 4.7, Table 4, Fig. states, Supp. F.4 (Table S20) | `make_state_items.py` (the second set, whose 20 scenarios open `state_questions_v5.yaml`); per model M: `state_tuning.py targets --model M`, then `targets --model M --set v5` (the added states; HunyuanImage: both with `--target-seeds 2`, the first with `--set v2`); `eval` and `replies` with `--set v5 --arm 0`; `train --set v5 --arm V --tag 600 --updates 600 --seed s` and the same with `--arm S` (s in 0, 1, 2; both models); `eval` and `replies` with `--set v5` per arm and seed (HunyuanImage `eval --picture-seeds 0`); `judge --model M --judge J`; then `state_table.py --model M` | `state_train/table_v5_<model>.json`: of the 80 stories, right as written, reversed, in both orders, right pictures and other-state pictures per arm and seed (Table S20); Table 4 in % (text / pictures): Show-o2 untrained 36.3 / 52.5, right state 49.6 / 73.3, wrong state 34.6 / 27.1; HunyuanImage 40.0 / 37.5, 50.8 / 64.6, 47.5 / 45.8; the figure's probabilities are `1/(1+exp(-m))` of `margins.json` |
| State interchange: Sec. 4.7, Supp. F.5, Table S21 | `state_tuning.py train --model show_o2 --arm V --tag 600 --updates 600 --seed s`; `swap --model show_o2 --arm V --tag 600 --seed s --swap-layers all+27`; `swap_table --model show_o2 --arm V --tag 600` | answers 47 / 47 / 49 of 56, mean margins, carried share of the margin gain, layer-27-only rows |

Placeholders: `<L_M>` the lens-variant layers (BAGEL 2,4,6,8,10,14,20; Show-o2 6,10,14,20,24,27; SenseNova-U1
8,12,16,20,24,28,36); `<C_M>` the context-test layers (BAGEL 6,8,14; Show-o2 20,24,27; SenseNova-U1 20,24,32);
`<T_M>` the transport name `<model>_transport_to<R>_n64` (only for the Jacobian ways `vlens`). The supplement's
figures of pictures (Figs. S2 to S4 and the galleries of the main paper) show pictures drawn by these same scripts.

## Hardware and cost

Inference uses bfloat16 weights. BAGEL, Show-o2, SenseNova-U1, Liquid and Janus-Pro run on one 80 GB GPU per job
(drawing alone also fits on 48 GB GPUs). HunyuanImage-3.0 needs four 80 GB GPUs (`device_map="auto"`). Qwen3-VL (about
61 GB) and Gemma-3 (about 54 GB) fit on one 80 GB GPU or spread over two smaller ones; LLaVA-OneVision-2 needs one GPU.

| Step | Approximate cost |
|---|---|
| One picture, full denoising | BAGEL 9 s, Show-o2 12.6 s, SenseNova-U1 9.4 s, Liquid about 1 min, HunyuanImage-3.0 110 s on four GPUs |
| Layer means and Jacobian maps (`fit_transport.py`) | 130 to 270 s per passage, 64 passages: 2 to 5 GPU-hours per model |
| Text J-lens (`fit_text_jlens.py`) | about 150 s per passage, 200 passages: about 8 GPU-hours per model |
| Context tests (Table 2) | 15 to 20 GPU-hours per main model; about 60 hours of four GPUs for HunyuanImage |
| Lens variants (Table 1, S2, S3) | about 30,000 pictures per variant (Val and Test, three models): several hundred GPU-hours |
| Ablation (Table 5, both test sets) | about 180,000 pictures: about 480 GPU-hours of drawing and 110 of judging |
| Depth through the first answer token (Fig. 4) | about 16,600 pictures for three models |
| HunyuanImage lens tuning (Table 3) | about 210 GPU-hours per training seed (targets, training, test pictures) |
| HunyuanImage state training (Table 4) | about 12 hours on four GPUs per run of 600 updates |
| Scoring scripts | CPU, seconds to minutes |

## Tests

```bash
PYTHONPATH=. python -m pytest tests        # CPU, a few minutes; the BAGEL attention test runs only after setup.sh
```

The tests check the Jacobian fitter and the transport maps against brute-force derivatives on a tiny float64 model,
the final-norm variant, every lens ablation (nothing ablated reproduces the V-lens bit for bit), the token positions of
the depth study, and the plain attention fallback of the BAGEL patch.

## Not included

- Pictures, judge answers, fitted statistics and all other outputs; model weights.
- The human rating of Supp. B (Table S6): the rating pages and the raters' answers.
- The scripts that lay out the paper's figures; the numbers behind Fig. 4 and the states figure come from the scoring
  scripts above.

## Licence

The code is released under the MIT License (see `LICENSE`). The upstream model code fetched by
`third_party/setup.sh` keeps its own licence
(`third_party/README.md`); model weights and datasets are subject to their providers' licences.
