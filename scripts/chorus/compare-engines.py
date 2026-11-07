"""Score speech engines on the test sermon, line by line.

Input: files from `detect-lines` (<clip>\t<seconds>\t<verses>), one per
engine and sound condition, named <engine>_<condition>.tsv. Clips are the
per-line files from make-sermon.py (lineNN.wav / lineNN_noisy.wav).

Prints a table: verses right, missed, wrong verses shown (with the lines
they came from), and the average
and slowest transcription time per line.
Usage: python3 compare-engines.py <sermon_dir> <result.tsv>...
"""

import importlib.util
import os
import re
import sys

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("make_sermon", os.path.join(here, "make-sermon.py"))
make_sermon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_sermon)
SERMON = make_sermon.SERMON


def score(path):
    expected_total = sum(len(refs) for _, refs in SERMON)
    right = wrong = 0
    times = []
    details = []
    for row in open(path, encoding="utf-8"):
        parts = row.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        clip, secs = parts[0], float(parts[1])
        found = [v for v in (parts[2].split("; ") if len(parts) > 2 and parts[2] else [])]
        n = int(re.search(r"line(\d+)", clip).group(1))
        want = SERMON[n][1]
        times.append(secs)
        right += sum(1 for w in want if w in found)
        bad = [f for f in found if f not in want]
        wrong += len(bad)
        miss = [w for w in want if w not in found]
        if miss or bad:
            details.append(f"    line {n}: missed {miss} wrong {bad}")
    avg = sum(times) / max(len(times), 1)
    return expected_total, right, wrong, avg, max(times or [0]), details


def main():
    print("\n==== ENGINE COMPARISON ====")
    print(f"{'engine / sound':28} {'right':>7} {'wrong':>6} {'avg s/line':>11} {'slowest':>8}")
    for path in sys.argv[2:]:
        label = os.path.basename(path).removesuffix(".tsv")
        total, right, wrong, avg, worst, details = score(path)
        print(f"{label:28} {right:>3}/{total:<3} {wrong:>6} {avg:>11.2f} {worst:>8.2f}")
        for d in details:
            print(d)


if __name__ == "__main__":
    main()
