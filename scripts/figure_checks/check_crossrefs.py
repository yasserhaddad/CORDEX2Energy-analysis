"""Every figure reference resolves, and each SI section's heading and figure stems agree.

* main figures: `figs/paper_publication/figN.pdf`; SI: `figs/si_note*/SN_*.pdf` (+ EXTERNAL_SI)
* both numberings contiguous from 1
* every "Figure N" / "Fig. N" / "Supplementary Fig. SN" / "Supplementary Figure N" in the notebooks'
  markdown and code and in doc/paper_summary.md points at a rendered figure
* inside each note, the section heading (`## SN - ...`) and the figure stems `SN_...` written under it
  carry the same N

Passing says every number EXISTS; it cannot say a citation means what its sentence claims.
Read citations by meaning after any renumber.

    python scripts/figure_checks/check_crossrefs.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "figs"
SI_DIRS = [FIGS / n for n in ("si_note1_pipeline", "si_note2_system", "si_note3_results")]
NOTEBOOKS = ["paper_figures_publication", "si_note1_pipeline", "si_note2_system", "si_note3_results"]
EXTERNAL_SI = set()
EXTERNAL_MAIN = {1}          # Figure 1, the pipeline schematic, is assembled outside the notebooks
problems = []

main = {int(m.group(1)) for p in (FIGS / "paper_publication").glob("fig*.pdf")
        if (m := re.match(r"fig(\d+)$", p.stem))}
si = {int(m.group(1)) for d in SI_DIRS for p in d.glob("S*.pdf") if (m := re.match(r"S(\d+)_", p.stem))}
si |= EXTERNAL_SI
main |= EXTERNAL_MAIN
print("rendered: main", sorted(main))
print("rendered: supplementary", sorted(si))
for label, numbers in (("main", main), ("supplementary", si)):
    if numbers and sorted(numbers) != list(range(1, max(numbers) + 1)):
        problems.append(f"{label} numbering has gaps: missing {sorted(set(range(1, max(numbers) + 1)) - numbers)}")

texts = {name: "\n".join("".join(c["source"]) for c in json.loads(
    (ROOT / "notebooks" / f"{name}.ipynb").read_text())["cells"]) for name in NOTEBOOKS}
texts["paper_summary.md"] = (ROOT / "doc" / "paper_summary.md").read_text()
# \s+ not a space: captions wrap, and "Supplementary\nFig. 6" is still a supplementary reference.
SUPP = re.compile(r"Supplementary\s+(?:Fig\.|Figs\.|Figure|Figures)\s*S?(\d+)")
MAIN = re.compile(r"\b(?:Figure|Fig\.)\s+(\d+)(?![\d.])")
unresolved = []
for name, text in texts.items():
    for n in map(int, SUPP.findall(text)):
        if n not in si:
            unresolved.append(f"{name}: Supplementary Fig. {n}")
    # Main-text references are read from the text with the supplementary ones removed.
    for n in map(int, MAIN.findall(SUPP.sub(" ", text))):
        if n not in main:
            unresolved.append(f"{name}: Figure {n}")
print("\nunresolved references:")
print("  none" if not unresolved else "\n".join("  " + u for u in sorted(set(unresolved))))
problems += sorted(set(unresolved))

print("\nsection heading / stem / caption:")
for name in NOTEBOOKS[1:]:
    cells = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())["cells"]
    current, bad = None, []                      # the S numbers the current section covers
    for c in cells:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            for line in re.findall(r"^##+ .*$", src, re.M):
                numbers = {int(n) for n in re.findall(r"\bS(\d+)\b", line)}
                if numbers:            # "S4 and S5", "S9-S14": a section can hold a range
                    for lo, hi in re.findall(r"\bS(\d+)\s*[-\u2013]\s*S(\d+)\b", line):
                        numbers |= set(range(int(lo), int(hi) + 1))
                    current = numbers
            for cap in re.findall(r"\*\*Supplementary Figure (\d+) \|", src):
                if current is not None and int(cap) not in current:
                    bad.append(f"caption {cap} under heading S{sorted(current)}")
        else:
            for stem in re.findall(r'"S(\d+)_\w+"', src):
                if current is not None and int(stem) not in current:
                    bad.append(f"stem S{stem} under heading S{sorted(current)}")
    print(f"  {name:<32} {'checked' if not bad else bad}")
    problems += [f"{name}: {b}" for b in bad]

print("\n" + ("Every figure reference resolves and every stem matches its section."
              if not problems else "PROBLEMS:\n" + "\n".join("  - " + p for p in problems)))
sys.exit(1 if problems else 0)
