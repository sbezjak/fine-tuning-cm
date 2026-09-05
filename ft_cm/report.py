"""Render a before/after eval into one self-contained HTML report.

The analog of llm-benchmark's report step, kept deliberately simpler: this task has
no cost, latency, or judge, and 24 hand-seeded held-out rows cannot support a 95%
interval or a paired significance test (the instinct is right, the scale is toy - see
notes.md). So the three benchmark tables collapse to what this study actually measures:

  1. Summary      - before vs after: overall + per-label accuracy, non-answers, hedged.
  2. Confusion    - the before/after 2x2 grid. "Read the grid, not the delta" is the
                    whole finding, so the safety-critical unsafe->safe cell is flagged.
  3. Full trace   - one row per held-out item, the model's RAW completion verbatim for
                    base and tuned. The raw output is the ground truth, so it is shown
                    inline, never hidden behind a toggle; flips and the unsafe->safe
                    misses are marked so a wrong answer is scannable.

Reads a receipts JSON written by ft_cm.eval (it already embeds every completion), so
this regenerates for free whenever the framing changes, never a re-run:

    uv run python -m ft_cm.report evidence/hard-1.5b-before-after.json

Self-contained and dependency-free on purpose: string-built HTML with inline CSS.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
from pathlib import Path

_CSS = """
:root { color-scheme: light dark; }
body { font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 2rem auto;
       max-width: 1000px; padding: 0 1rem; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.15rem; margin-top: 2rem; }
.meta { color: #666; margin-bottom: 1.5rem; }
.caveat { color: #666; font-size: 0.9rem; border-left: 3px solid #ccc; padding-left: 0.8rem; }
table { border-collapse: collapse; width: 100%; margin: 0.5rem 0 1rem;
        font-variant-numeric: tabular-nums; }
th, td { padding: 0.35rem 0.6rem; border-bottom: 1px solid #ccc; text-align: right; }
th:first-child, td:first-child { text-align: left; }
thead th { border-bottom: 2px solid #888; }
.grids { display: flex; gap: 1.5rem; flex-wrap: wrap; }
.grids > div { flex: 1; min-width: 260px; }
.ok { color: #1a7f37; } .bad { color: #b3261e; }
.crit { background: rgba(179,38,30,0.14); font-weight: 600; }
.mono { font-variant-numeric: tabular-nums; }
.txt { text-align: left; color: #444; }
.flip { color: #8250df; font-weight: 600; }
"""


def _esc(x: object) -> str:
    return html.escape(str(x))


def _pct(x: float) -> str:
    return f"{x:.0%}"


def _file_link(p: Path | str) -> str:
    # Link the path AS GIVEN (repo-relative), never a resolved absolute file:// URI:
    # this is a public repo, and an absolute href would bake the local username and
    # machine path into a committed report. Relative opens correctly from the repo
    # root; the path text stays the useful part, and both name the source.
    return f"<a href='{_esc(str(p))}'>{_esc(str(p))}</a>"


def _summary_table(rep: dict) -> str:
    b, a = rep["before"], rep["after"]
    head = (
        "<tr><th>run</th><th>overall</th><th>safe acc</th><th>unsafe acc</th>"
        "<th>non-answers</th><th>hedged</th></tr>"
    )

    def row(name: str, r: dict) -> str:
        pl = r["per_label_accuracy"]
        return (
            f"<tr><td>{_esc(name)}</td>"
            f"<td>{r['accuracy']:.3f} ({r['correct']}/{r['n']})</td>"
            f"<td>{_pct(pl['safe'])}</td><td>{_pct(pl['unsafe'])}</td>"
            f"<td>{r['n_none']}</td><td>{r['n_ambiguous']}</td></tr>"
        )

    delta = rep["delta_accuracy"]
    foot = (
        f"<tr><td>delta</td><td class='{'ok' if delta >= 0 else 'bad'}'>{delta:+.3f}</td>"
        "<td colspan='4' class='txt'>overall delta hides the per-class move - read the grid</td></tr>"
    )
    return f"<table><thead>{head}</thead><tbody>{row('before (base)', b)}{row('after (tuned)', a)}{foot}</tbody></table>"


def _confusion(name: str, r: dict) -> str:
    c = r["confusion"]
    # outer key = gold, inner = pred. The inner dict is sparse (only labels that
    # actually occurred, plus possibly "none"), so a zero cell is simply absent -
    # a perfect unsafe column has no unsafe->safe key at all. Default missing to 0.
    def cell(gold: str, pred: str) -> str:
        v = c.get(gold, {}).get(pred, 0)
        cls = "ok" if gold == pred else "bad"
        if gold == "unsafe" and pred == "safe":
            cls = "crit"
        return f"<td class='{cls}'>{v}</td>"

    return (
        f"<div><h3 style='font-size:1rem;margin:0 0 .3rem'>{_esc(name)}</h3>"
        "<table><thead><tr><th>gold \\ pred</th><th>safe</th><th>unsafe</th></tr></thead>"
        f"<tbody><tr><td>safe</td>{cell('safe', 'safe')}{cell('safe', 'unsafe')}</tr>"
        f"<tr><td>unsafe</td>{cell('unsafe', 'safe')}{cell('unsafe', 'unsafe')}</tr>"
        "</tbody></table></div>"
    )


def _trace_table(rep: dict) -> str:
    before, after = rep["before"]["rows"], rep["after"]["rows"]
    head = (
        "<tr><th>#</th><th>gold</th><th>base</th><th>tuned</th><th>note</th>"
        "<th class='txt'>text</th></tr>"
    )
    rows = []
    for i, (b, a) in enumerate(zip(before, after, strict=True), start=1):
        # raw completion verbatim = the ground truth (usually one word, e.g. 'safe').
        base_raw, tuned_raw = b["raw"], a["raw"]
        base_cls = "ok" if b["ok"] else "bad"
        tuned_cls = "ok" if a["ok"] else "bad"
        notes = []
        if b["pred"] != a["pred"]:
            notes.append("<span class='flip'>flip</span>")
        if a["gold"] == "unsafe" and a["pred"] == "safe":
            notes.append("<span class='bad'>unsafe&rarr;safe MISS</span>")
        rows.append(
            f"<tr><td>{i}</td><td>{_esc(a['gold'])}</td>"
            f"<td class='{base_cls} mono'>{_esc(base_raw)}</td>"
            f"<td class='{tuned_cls} mono'>{_esc(tuned_raw)}</td>"
            f"<td>{' '.join(notes)}</td>"
            f"<td class='txt'>{_esc(a['text'])}</td></tr>"
        )
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _provenance(receipts_path: Path | str | None, raw_log: Path | str | None) -> str:
    # The evidence chain: raw run log (terse, verbatim completions) -> receipts JSON
    # (rows + tallies) -> this HTML. Cite both so any number here traces back to the
    # untruncated log that produced it, never taken on faith.
    if not receipts_path and not raw_log:
        return ""
    items = []
    if receipts_path:
        items.append(f"receipts JSON (rows + tallies): {_file_link(receipts_path)}")
    if raw_log:
        items.append(f"raw run log (full stdout, verbatim completions): {_file_link(raw_log)}")
    lis = "".join(f"<li>{it}</li>" for it in items)
    return (
        "<h2>Raw logs</h2>"
        "<p class='meta'>This report is rendered off the receipts alone; the receipts are "
        "tallied from the raw log. Same computation, three views.</p>"
        f"<ul class='meta'>{lis}</ul>"
    )


def render_report(
    rep: dict,
    out_path: Path | str,
    *,
    title: str | None = None,
    receipts_path: Path | str | None = None,
    raw_log: Path | str | None = None,
) -> Path:
    """Build the self-contained HTML report from a receipts dict and write it.

    Everything comes off the receipts alone (no model call), so it regenerates for
    free whenever the framing changes. `receipts_path`/`raw_log` are cited in a
    provenance footer so every number traces back to the untruncated source log.
    Returns the path written."""
    title = title or f"Before/after eval - {rep['model']}"
    generated = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%SZ")

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title><style>{_CSS}</style></head><body>
<h1>{_esc(title)}</h1>
<p class="meta">base <b>{_esc(rep['model'])}</b> &middot; adapter {_esc(rep['adapter_path'])}
 &middot; held-out {_esc(rep['holdout'])} &middot; {rep['n']} rows &middot; generated {generated}
 &middot; computed off the receipts (no re-run)</p>
<p class="caveat">Hand-seeded smoke rows: this shows the MECHANISM and the calibration
behavior, not a real-classifier number. {rep['n']} rows cannot support a 95% interval or a
paired significance test, so those benchmark tables are deliberately omitted.</p>

<h2>Summary - before (base) vs after (tuned)</h2>
{_summary_table(rep)}

<h2>Confusion grid - read this, not the delta</h2>
<p class="meta">Rows are the gold label, columns the prediction. The diagonal is correct.
The highlighted <span class="crit">unsafe&rarr;safe</span> cell is the safety-critical
error: a real threat let through. A rise in overall accuracy that comes with a rise in
this cell is a regression the headline number hides.</p>
<div class="grids">{_confusion('before (base)', rep['before'])}{_confusion('after (tuned)', rep['after'])}</div>

<h2>Full trace - every row, raw completion verbatim</h2>
<p class="meta">One row per held-out item; base and tuned columns are the model's RAW
completion (green correct, red wrong). <span class="flip">flip</span> marks rows where the
adapter changed the answer; the tables above are computed from these.</p>
{_trace_table(rep)}

{_provenance(receipts_path, raw_log)}
</body></html>"""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a before/after eval receipts JSON to HTML.")
    ap.add_argument("receipts", help="path to a receipts JSON written by ft_cm.eval")
    ap.add_argument("--out", default=None, help="output HTML path (default: reports/<stem>.html)")
    ap.add_argument("--title", default=None)
    ap.add_argument(
        "--raw-log",
        default=None,
        help="path to the run-evidence raw log to cite in the provenance footer",
    )
    args = ap.parse_args()

    rep = json.loads(Path(args.receipts).read_text())
    out = args.out or f"reports/{Path(args.receipts).stem}.html"
    path = render_report(
        rep, out, title=args.title, receipts_path=args.receipts, raw_log=args.raw_log
    )
    print(f"[report] wrote {path}")


if __name__ == "__main__":
    main()
