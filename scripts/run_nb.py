"""Execute all code cells of a notebook via nbclient (correct IOPub ordering).

Kernel: "Energy Audit (Python 3.11)" (kernelspec name: energy-audit).

Usage:  python3 scripts/run_nb.py 02_Parser_Evaluation
        python3 scripts/run_nb.py notebooks/*.ipynb

Designed to be run from repo root in the foreground, or one notebook at a
time under nohup when the run may be long.
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
KERNEL = "energy-audit"  # display name: "Energy Audit (Python 3.11)"


def run_one(path: Path) -> bool:
    """Execute one notebook. Returns True on clean completion, False on error."""
    nb = nbformat.read(str(path), as_version=4)
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"[nb] {path.name}: {n_code} code cells; kernel={KERNEL}", flush=True)

    ok = True
    try:
        client = NotebookClient(
            nb,
            kernel_name=KERNEL,
            cwd=str(ROOT),
            timeout=None,
            capture_output=True,   # buffered, flushed per cell
            iopub_timeout=600,    # generous per-message wait
            startup_timeout=60,
        )
        client.execute()
    except Exception:
        ok = False
    # after execute(), each code cell has outputs already attached
    for i, c in enumerate(nb.cells, 1):
        if c.cell_type != "code":
            continue
        for o in c.get("outputs", []):
            ot = o.get("output_type") or o.get("msg_type")
            if ot == "stream":
                sys.stdout.write(o.get("text", ""))
            elif ot == "execute_result":
                d = o.get("data", {})
                if "text/plain" in d:
                    sys.stdout.write(d["text/plain"] + "\n")
            elif ot == "error":
                ok = False
                print(f"\n[ERROR] in cell {i}", flush=True)
                print("\n".join(o.get("traceback", [])), flush=True)
    # persist executed notebook (outputs + metadata), same format as before
    nbformat.write(nb, str(path))
    print(f"[nb] {path.name} {'OK' if ok else 'FAILED'}", flush=True)
    return ok


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    results = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.suffix:
            p = p.with_suffix(".ipynb")
        results.append((p.name, run_one(p)))
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}", flush=True)
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n  {len(results) - n_fail}/{len(results)} passed, ALL OK" if n_fail == 0
          else f"\n  {n_fail}/{len(results)} FAILED", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
