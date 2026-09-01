# Dataset provenance

## Smoke set (committed)
`data/smoke/smoke.jsonl` - 40 hand-written, fabricated, tame examples (balanced
safe/unsafe). Not a real dataset and not a benchmark. Exists ONLY to prove the
training loop runs and that a LoRA adapter moves held-out behavior. Numbers from
it measure the harness, not moderation quality, and are never reported as a result.

## Real dataset (NOT committed - fill in when chosen)
- Source:            <name + URL>
- License:           <license; must permit publishing derived labels/metrics>
- Retrieved:         <date>
- Download:          `scripts/download-dataset.sh` -> `data/real/` (git-ignored)
- What we publish:   derived labels + accuracy/calibration metrics only, never
                     raw text.
- Label mapping:     <how the source labels map to safe/unsafe or the taxonomy>

Handling rules (public repo, toxic domain): raw text stays local, cite before
first use, prefer datasets whose license allows redistribution of derived
results rather than the corpus itself.
