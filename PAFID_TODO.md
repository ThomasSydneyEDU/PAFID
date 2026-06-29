# PAFID To-Do List (compiled 2026-06-12, updated 2026-06-29)

Outstanding items for the PAFID repo and the companion stimulus paper (Stella et al.,
in preparation), following the session that integrated PAFID as the single source of
truth for `Foodpictures_information_dynamic.csv`.

Status legend: `[x]` done · `[~]` resolved by decision (documented, not changed) · `[ ]` open.

## Companion paper (Stella et al.)

*These are manuscript-writing tasks and cannot be closed in the repo, but the repo now
holds the source material for all of them (prompts in the README appendix with version
stamps, models-per-stage and HOG-PC caveat in the README, NOVA provenance noted, udon
`manual_correction_note` present in `stimuli_master.json`).*

- [ ] **Document the CURRENT classification prompt (v3)**, not the original. The
      Intuitive 7 definitions were rewritten on 2026-06-12 (folk categories,
      base-food rule, sweetness+treat requirement for Dessert, Dish = multi-ingredient
      meals only, snacks by base ingredient, dairy → Animal protein). The live prompt
      is in `src/classify_food.py` (`CLASSIFY_PROMPT_TEMPLATE`, version
      `v3-2026-06-culinary-folk`); the main manuscript Methods now
      points to the companion paper for it.
- [ ] **Document the NOVA scheme** as the fifth label: definitions ported from the
      original manual batch protocol (Monteiro et al., 2016), now automated per food
      (`NOVA_CLASSIFY_PROMPT_TEMPLATE`, version `v2-2026-06-processing-focus` inside `src/classify_food.py`).
      Provenance of the 350 existing labels = manual batch classification
      (FoodTriplet-Analysis: `manuscript/nova_classification_prompt.md`).
- [ ] **Document the udon manual correction** (name-based classification said plain
      noodles → Grain/NOVA 1; image shows a multi-ingredient noodle soup → Dish /
      Prepared foods / Composite Meals / NOVA 3). Recorded in
      `stimuli_master.json` under `manual_correction_note`. *(Note confirmed present in repo.)*
- [ ] **State the models used per step**: classification = `gemini-2.5-flash`
      (temperature 0); aware ratings/QC = `gemini-2.5-pro`; blind ratings =
      `gemini-2.5-pro`. Image generation model per stimuli_master metadata.
      *(Now stated in the README; still needs to land in the paper.)*
- [ ] **State the HOG-PC caveat** for visual features: PCA components are fit on the
      image set processed in a given run and are not in a shared basis across runs;
      scalar features are directly comparable, HOG PCs are not.
      *(Now stated in the README; still needs to land in the paper.)*

## README / documentation fixes — DONE

- [x] README model name corrected to `gemini-2.5-flash` (temperature 0).
- [x] Duplicate section numbering fixed; Usage Pipeline now reads 1–6 aligned to the
      six stages (QC folded into Stage 2, Editorial Review added as Stage 3), plus
      utilities 7–8.
- [x] `Foodpictures_information_reference.csv` filename reconciled and the "351-item"
      references corrected to **350** (verified: 350 rows, no duplicate filenames/foods;
      the spring-rolls "duplicate" is two distinct items, shrimp spring rolls vs spring rolls).
- [x] Added a **Pipeline Overview** section (authoritative stage table + Mermaid diagram
      + design rationale); `PROJECT_SPEC.md` now links to it instead of duplicating;
      `PIPELINE_REFACTOR_NOTES.md` removed with its smoke-test cycle folded into the README.

## Data hygiene

- [x] Backup files: `*.bak` is now gitignored; `_old.csv` and join backups removed.
      One older `rendered_images/stimuli_master.json.pre_nova_backfill.bak` remains on
      disk (ignored) — delete once the master run is verified.
- [~] **"Borrito bowl" spelling** — decision: **document, do not change.** The slugified
      filename `borrito-bowl.png` is the immutable join key for the collected human
      ratings, the feature table, and the survey archive (Qualtrics image
      `IM_z9SvLwAR07gVxR9`), so renaming would desync the DB from its own ratings'
      provenance. Documented in the README "Known label quirks" section.
- [ ] **Commit everything** — a large change set from the 2026-06-29 session is staged
      in the working tree but not yet committed.

## Validation / audits

- [x] **Diff WHO_10 / Culinary_9 / NvT / Transformation_score against pre-v2 values.**
      Pre-v2 values recovered from git (`d43899c:…pre_blind_join.bak`). Report:
      `data/QC/classification_pre_v2_diff.csv`. Summary: WHO_10 11 flips, Culinary_9 12
      flips, Natural-vs-transformed 3 flips, Transformation_score 127 changed (mean
      |Δ|≈8, 20 with |Δ|≥15). Changes are consistent with borderline items settling
      under temperature 0 and look like net improvements (e.g. avocado/tomato → Fruits,
      udon → Prepared foods). No regressions requiring rollback identified.
- [x] **Caption-based name-vs-image mismatch scan** (the udon pattern). Report:
      `data/QC/caption_mismatch_scan.csv`. 28 of 266 single-source items match composite
      cues; on review most are ordinary garnish/seasoning. Clearest genuine composite
      plate: "Roast lamb". Surfaced for transparency, not corrected (per project policy).
- [ ] Revisit "sweet breakfast cereal" → Grain (borderline under the v2 sweetness
      rule; behaviourally nearest Dessert). Currently accepted as classified — decision pending.
- [ ] **End-to-end test of the new-stimulus path** once convenient: add one test food
      to the seed list → generate → classify (5 schemes incl. NOVA) → QC/aware →
      blind → visual features → confirm the new row in the dynamic CSV is complete,
      then remove the test item (or keep a dedicated test seed list).
      *(Smoke-test commands now documented in the README "Testing on a subset" section;
      not yet executed.)*

## Extension support (PROJECT_SPEC planned changes) — DONE

- [x] `--food-list`, `--output-dir`, `--stimulus-set` flags across `classify_food.py`,
      `generate_images.py`, `run_qc.py`, and `run_pipeline.sh`.
- [x] `stimulus_set` provenance column added to both CSVs and `stimuli_master.json`;
      canonical 350 tagged `pafid_v1`; `run_qc.py` export and `generate_images.py`
      default (`pafid_v1`) propagate it so `reset_pipeline.py --stimulus-set` works.
- [x] `reset_pipeline.py` source-aware reset implemented.

## Pipeline behaviour reminders (already implemented, verify on next use)

- Classification is resumable: per-entry version stamps + checkpoint every item;
  failed/quota-aborted items retry on rerun; 1.5 s pacing between calls.
- `run_qc.py` export: emits `Category_NOVA_4`, drops legacy `food_classification`,
  dedup guard on merge, preserves blind/ll_ columns.
- `rate_images.py`: only rates rows missing blind values; `--overwrite` to re-rate.
- `extract_visual_features.py --merge-canonical`: incremental (fills missing rows
  only); `--overwrite` to recompute all.
