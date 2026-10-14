"""Execute a publication notebook's code cells in one namespace and write the outputs back.

Used instead of nbconvert, which is not installed in the c2e-recipe env. Each code cell's stdout
becomes a stream output and every open figure a PNG display output, so the notebook stays a
faithful record of the run. Magics (`%...`) are skipped. Prints one line per cell and a final
"=== N code cells, M failures [...] ===" line that a watcher can key on.

    cd notebooks && python ../scripts/figure_checks/run_notebook.py si_note3_results.ipynb [--cells 2,3,30]
"""
import argparse
import base64
import io
import json
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("notebook")
parser.add_argument("--cells", help="comma-separated cell indices to run (default: all)")
args = parser.parse_args()

path = Path(args.notebook).resolve()
nb = json.loads(path.read_text())
wanted = None if args.cells is None else {int(x) for x in args.cells.split(",")}
ns = {"__name__": "__main__", "__file__": str(path)}
failures, n_code = [], 0

for index, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code" or (wanted is not None and index not in wanted):
        continue
    n_code += 1
    source = "".join(cell["source"])
    if source.lstrip().startswith("%"):
        continue
    buf = io.StringIO()
    stdout, sys.stdout = sys.stdout, buf
    ok = True
    try:
        exec(compile(source, f"<cell {index}>", "exec"), ns)
    except Exception:
        ok = False
        traceback.print_exc(file=buf)
    finally:
        sys.stdout = stdout
    outputs = []
    text = buf.getvalue()
    if text:
        outputs.append({"output_type": "stream", "name": "stdout",
                        "text": text.splitlines(keepends=True)})
    for number in plt.get_fignums():
        figure = plt.figure(number)
        png = io.BytesIO()
        figure.savefig(png, format="png", dpi=90, bbox_inches="tight")
        outputs.append({"output_type": "display_data", "metadata": {},
                        "data": {"image/png": base64.b64encode(png.getvalue()).decode()}})
        plt.close(figure)
    cell["outputs"], cell["execution_count"] = outputs, index
    print(f"[cell {index}] {'ok' if ok else 'FAILED'} ({len(outputs)} outputs)", flush=True)
    if not ok:
        print("\n".join("    " + l for l in text.rstrip().split("\n")[-25:]), flush=True)
        failures.append(index)

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"\n=== {n_code} code cells, {len(failures)} failures {failures} ===", flush=True)
