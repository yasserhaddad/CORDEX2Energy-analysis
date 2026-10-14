"""Artwork rules for every publication PDF, checked against the PDF itself.

* page width <= 181.5 mm (180 mm double column, 1.5 mm tolerance for the save padding)
* page height within what Nature allows for the caption's length (CAPTION_HEIGHT_MM)
* fonts: DejaVu only (subset-embedded TrueType)
* no two words overlapping (pdftotext -bbox), and no merged comparison labels
* the panel letters `save()` stamped into the PDF Keywords match the letters the caption cites
* no PDF that no `save()` in its notebook writes (stale from a renumber)
* panel letters aligned: letters within 4 mm of each other in x (a column) share one x, and in y
  (a row) one y, to 0.3 mm -- the rule `panel_letters` implements, read back from the PDF

    python scripts/figure_checks/check_artwork.py
"""
import json
import re
import subprocess
import sys
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MM_PER_PT = 25.4 / 72
MAX_WIDTH_MM = 181.5
#: Nature: caption length buys figure height. (max words, max height mm), tried in order; a
#: caption of 300 words or more leaves no permissible height at all.
CAPTION_HEIGHT_MM = {180: [(50, 225), (150, 210), (300, 185)],
                     88: [(50, 225), (150, 210), (300, 185)]}
SETS = [
    ("paper_figures_publication", ROOT / "figs/paper_publication", r"fig(\d+)$",
     r"\*\*Figure {n} \|"),
    ("si_note1_pipeline", ROOT / "figs/si_note1_pipeline", r"S(\d+)_", r"\*\*Supplementary Figure {n} \|"),
    ("si_note2_system", ROOT / "figs/si_note2_system", r"S(\d+)_", r"\*\*Supplementary Figure {n} \|"),
    ("si_note3_results", ROOT / "figs/si_note3_results", r"S(\d+)_", r"\*\*Supplementary Figure {n} \|"),
]
#: Assembled outside the notebooks: caption only, no PDF to measure.
EXTERNAL_FIGURES = {("paper_figures_publication", "1")}      # the pipeline schematic
NEXT_CAPTION = re.compile(r"\n(?:##|> \*\*(?:Supplementary )?Figure \d+ \|)")
#: Two duration labels rendered so close they extract as ONE word: "<= 24 h" beside "> 24 h" came
#: through as "h>". The signature is a unit letter (h, d) running straight into an operator; labels
#: that legitimately contain one ("cooling>1TWh", "CDD>50", ">p100") do not match it.
MERGED = re.compile(r"(?:^|[^A-Za-z])[hd][<>\u2264\u2265]")
failures = []


def words(pdf):
    html = subprocess.run(["pdftotext", "-bbox", str(pdf), "-"], capture_output=True, text=True).stdout
    return [(float(a), float(b), float(c), float(d), unescape(w)) for a, b, c, d, w in re.findall(
        r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>', html)]


def letter_misalignment(page, drawn, near_mm=4.0, tol_mm=0.3):
    """Panel letters (single 8 pt words among the stamped letters) that nearly but not exactly line up."""
    letters = [w for w in page if w[4] in drawn and (w[3] - w[1]) > 8.8]   # 8 pt bold box ~9.3 pt; 7 pt ~8.1
    bad = []
    for axis, (lo, hi) in (("x", (0, 0)), ("y", (1, 1))):
        for i, a in enumerate(letters):
            for b in letters[i + 1:]:
                gap = abs(a[lo] - b[hi]) * MM_PER_PT
                if tol_mm < gap <= near_mm:
                    bad.append(f"{a[4]}/{b[4]} {axis} off by {gap:.1f} mm")
    return bad


def collisions(page, tolerance=0.15):
    hits = []
    for i, a in enumerate(page):
        for b in page[i + 1:]:
            if a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]:
                continue
            overlap = (min(a[2], b[2]) - max(a[0], b[0])) * (min(a[3], b[3]) - max(a[1], b[1]))
            area = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            if area > 0 and overlap / area > tolerance:
                hits.append(f"{a[4]!r} x {b[4]!r} ({overlap / area:.0%})")
    return hits


for notebook, figs_dir, stem_pattern, caption_pattern in SETS:
    cells = json.loads((ROOT / "notebooks" / f"{notebook}.ipynb").read_text())["cells"]
    markdown = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    print(f"\n{notebook}")
    for nb_name, external in EXTERNAL_FIGURES:
        if nb_name == notebook:
            ok = re.search(caption_pattern.format(n=external), markdown)
            print(f"  {'S' + external + ' (external, caption only)':<44}{'' if ok else 'NO CAPTION'}")
            if not ok:
                failures.append(f"{notebook}: external S{external} has no caption")
    written = set(re.findall(r'save\((?:fig|_fig), "([^"]+)"', code))
    # Stems held in a mapping and passed by name (S4-S7: one figure per season).
    written |= set(re.findall(r'"((?:S\d+_\w+|fig\d+))"', code))
    written |= {f"{prefix}{v}" for prefix in re.findall(r'save\(fig, f"([^"{]+)\{', code)
                for v in ("DJF", "MAM", "JJA", "SON")}
    written |= set(re.findall(r'\("(S\d+_[a-z_0-9]+)"', code))
    for orphan in sorted({p.stem for p in figs_dir.glob("*.pdf")} - written):
        failures.append(f"{orphan}.pdf: no save() writes this stem -- stale from a renumber?")

    for path in sorted(figs_dir.glob("*.pdf")):
        match = re.match(stem_pattern, path.stem)
        if not match:
            failures.append(f"{path.name}: stem does not match {stem_pattern}")
            continue
        number = match.group(1)
        info = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True).stdout
        w_pt, h_pt = map(float, re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info).groups())
        w_mm, h_mm = w_pt * MM_PER_PT, h_pt * MM_PER_PT
        stamped = re.search(r"^Keywords:\s*panels=(\S*)", info, re.M)
        drawn = sorted(stamped.group(1)) if stamped else None
        fonts = subprocess.run(["pdffonts", str(path)], capture_output=True, text=True).stdout
        families = sorted({line.split()[0].split("+")[-1] for line in fonts.splitlines()[2:] if line.strip()})
        page = words(path)
        hits = collisions(page) + [f"merged label {w[4]!r}" for w in page if MERGED.search(w[4])]

        caption_words, cited = None, []
        m = re.search(caption_pattern.format(n=number), markdown)
        if m:
            body = markdown[m.start():]
            nxt = NEXT_CAPTION.search(body, 5)
            body = body[:nxt.start()] if nxt else body
            caption_words = len(re.sub(r"^>\s?", "", body, flags=re.M).split())
            cited = sorted(set(re.findall(r"\*\*([a-l])\*\*", body)))
        else:
            failures.append(f"{path.name}: no caption matching {caption_pattern.format(n=number)!r}")

        if w_mm > MAX_WIDTH_MM:
            failures.append(f"{path.name}: page {w_mm:.2f} mm wide")
        if caption_words is not None:
            band = CAPTION_HEIGHT_MM[180 if w_mm > 120 else 88]
            allowed = next((mm for limit, mm in band if caption_words < limit), None)
            if allowed is None:
                failures.append(f"{path.name}: caption {caption_words} words: 300+ leaves no legal height")
            elif h_mm > allowed:
                failures.append(f"{path.name}: height {h_mm:.1f} mm > {allowed} mm allowed for a "
                                f"{caption_words}-word caption")
        bad_fonts = [f for f in families if not f.startswith("DejaVu")]
        if bad_fonts:
            failures.append(f"{path.name}: non-DejaVu fonts {bad_fonts}")
        if drawn is not None and caption_words is not None and cited:
            if set(cited) - set(drawn):
                failures.append(f"{path.name}: caption cites {sorted(set(cited) - set(drawn))}, not drawn")
            if set(drawn) - set(cited):
                failures.append(f"{path.name}: drew {sorted(set(drawn) - set(cited))}, caption silent")
        if hits:
            failures.append(f"{path.name}: {len(hits)} collisions: {hits[:4]}")
        ragged = letter_misalignment(page, set(drawn or []))
        if ragged:
            failures.append(f"{path.name}: panel letters not aligned: {ragged[:4]}")
        print(f"  {path.name:<46}{w_mm:5.1f}x{h_mm:<6.1f}mm  panels={''.join(drawn) if drawn else '-':<8}"
              f"{str(caption_words) + 'w':<6}{len(hits)} collisions")

print("\nFAILURES:" if failures else "\nall artwork rules pass")
for f in failures:
    print("  -", f)
sys.exit(1 if failures else 0)
