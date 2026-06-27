# PAFID To-Do List (compiled 2026-06-12)

Outstanding items for the PAFID repo and the companion stimulus paper (Stella et al.,
in preparation), following the session that integrated PAFID as the single source of
truth for `Foodpictures_information_dynamic.csv`.

## Companion paper (Stella et al.)

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
      `stimuli_master.json` under `manual_correction_note`.
- [ ] **State the models used per step**: classification = `gemini-2.5-flash`
      (temperature 0); aware ratings/QC = `gemini-2.5-pro`; blind ratings =
      `gemini-2.5-pro`. Image generation model per stimuli_master metadata.
- [ ] **State the HOG-PC caveat** for visual features: PCA components are fit on the
      image set processed in a given run and are not in a shared basis across runs;
      scalar features are directly comparable, HOG PCs are not.

## README / documentation fixes

- [ ] README still says classification uses `gemini-2.0-flash` — actual default is
      `gemini-2.5-flash` (temperature 0). Update.
- [ ] Duplicate section numbering: "6. Prepare for Experiments" and "6. Reset
      Pipeline" — renumber.
- [ ] Data Management section references `Foodpictures_information_reference.csv`,
      but the file on disk is `Foodpictures_information_reference_old.csv` —
      reconcile (rename file or fix README). Note the static 351-item copy still
      contains the duplicate spring-rolls row.

## Data hygiene

- [ ] Decide fate of backup files: `data/*.pre_blind_join.bak`,
      `data/*.pre_ll_join.bak`, `rendered_images/stimuli_master.json.bak`,
      `stimuli_master.json.pre_nova_backfill.bak`, `data/Foodpictures_information_dynamic_old.csv`.
      Keep (gitignored), archive, or delete once the master run is verified.
- [ ] Check seed list spelling: "Borrito bowl" (likely "Burrito bowl") — note that
      fixing the name changes the food↔master matching key, so coordinate with
      stimuli_master if corrected.
- [ ] **Commit everything** (code + data) — none of this session's PAFID changes are
      committed yet.

## Validation / audits

- [ ] **Diff WHO_10 / Culinary_9 / NvT / Transformation_score against their
      pre-v2-rerun values.** Those schemes' definitions were unchanged, but the
      2026-06-12 reclassification regenerated them (now at temperature 0) — borderline
      items may have flipped. The pre-rerun values are in
      `data/Foodpictures_information_dynamic.csv.pre_blind_join.bak` (and git).
- [ ] **Caption-based name-vs-image mismatch scan** (the udon pattern): check all 350
      QC captions for composite-dish cues ("soup", "topped with", "with ... and ...")
      on items classified as single-source foods.
- [ ] Revisit "sweet breakfast cereal" → Grain (borderline under the v2 sweetness
      rule; behaviourally nearest Dessert). Currently accepted as classified.
- [ ] **End-to-end test of the new-stimulus path** once convenient: add one test food
      to the seed list → generate → classify (5 schemes incl. NOVA) → QC/aware →
      blind → visual features → confirm the new row in the dynamic CSV is complete,
      then remove the test item (or keep a dedicated test seed list).

## Pipeline behaviour reminders (already implemented, verify on next use)

- Classification is resumable: per-entry version stamps + checkpoint every item;
  failed/quota-aborted items retry on rerun; 1.5 s pacing between calls.
- `run_qc.py` export: emits `Category_NOVA_4`, drops legacy `food_classification`,
  dedup guard on merge, preserves blind/ll_ columns.
- `rate_images.py`: only rates rows missing blind values; `--overwrite` to re-rate.
- `extract_visual_features.py --merge-canonical`: incremental (fills missing rows
  only); `--overwrite` to recompute all.
