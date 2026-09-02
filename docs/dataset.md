# Dataset provenance

## Smoke set (committed)
`data/smoke/smoke.jsonl` - 40 hand-written, fabricated, tame examples (balanced
safe/unsafe). Not a real dataset and not a benchmark. Exists ONLY to prove the
training loop runs and that a LoRA adapter moves held-out behavior. Numbers from
it measure the harness, not moderation quality, and are never reported as a result.

## Real dataset - Civil Comments (NOT committed)
- Source:            Civil Comments, distributed by Jigsaw/Google.
                     HuggingFace: https://huggingface.co/datasets/google/civil_comments
                     (the bare `civil_comments` id was retired - HF moved it under
                     the `google/` namespace; the bare URL now 404s.)
                     Origin: the ~2M public comments from the Civil Comments
                     platform (2015-2017, 50 English-language news sites), released
                     when it shut down and later annotated for toxicity by Jigsaw
                     for the Kaggle competition "Jigsaw Unintended Bias in Toxicity
                     Classification" (2019).
- License:           CC0 1.0 (public-domain dedication). The HuggingFace dataset
                     card states it directly: "released under CC0 1.0, as is the
                     underlying comment text." CC0 waives copyright entirely, so it
                     permits both research use AND redistribution - including the
                     raw text itself. We still do NOT republish raw text here (see
                     handling rules); CC0 removes the LICENSE barrier to publishing
                     derived labels/metrics, which is all this repo publishes.
- Retrieved:         (filled in by scripts/download-dataset.sh on first pull)
- Download:          `scripts/download-dataset.sh` -> `data/real/` (git-ignored),
                     via HuggingFace `datasets`
                     (`load_dataset("google/civil_comments")`). No login/token needed.
- What we publish:   derived labels + accuracy/calibration metrics only, never
                     raw comment text.
- Label mapping:     the source `toxicity` field is a CONTINUOUS score in [0,1] =
                     the fraction of human annotators who rated the comment toxic.
                     We BINARIZE it to the taxonomy's single label:
                       unsafe  if toxicity >= TAU
                       safe    otherwise
                     TAU is a chosen threshold (a conceptual call - start 0.5).
                     The binarize is applied at download time; only the resulting
                     safe/unsafe label flows through data prep, exactly like smoke.

Handling rules (public repo, toxic domain): raw text stays local under
`data/real/` (git-ignored), cited before first use. CC0 would permit
redistributing the corpus, but this repo deliberately does not - it publishes only
derived labels and metrics, so no toxic comment text ever lands in git history.

### License claim to sanity-check (human gate)
The load-bearing claim is: **Civil Comments is CC0, and CC0 permits publishing
derived labels + metrics computed from it.** Verify against the HuggingFace
dataset card (the "License: cc0-1.0" field) and the CC0 deed
(https://creativecommons.org/publicdomain/zero/1.0/). CC0 places the work in the
public domain with no conditions, so derived-metric publishing is unambiguously
permitted; the repo's no-raw-text rule is a self-imposed handling choice, not a
license requirement.
