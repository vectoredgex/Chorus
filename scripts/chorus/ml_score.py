"""Score Malayalam speech engines on the test sermon.

Input: files from ml_transcribe.py, named <engine>_<voice>_<sound>.tsv.
For each file prints:
  key words  book names and chapter/verse number words heard exactly
  close      heard nearly right (one or two letters off), which a fuzzy
             matcher in the reference finder can still rescue
  CER        character error rate over whole lines (lower is better)
  s/line     average transcription time
  verses     references the Chorus reference finder found in the text
             (right / wrong), from <name>.refs (detect_tsv output) if present
Then the transcript of every reference line, so mistakes can be read.

Usage: python3 ml_score.py <result.tsv>...
"""

import importlib.util
import os
import re
import sys
import unicodedata

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ml_sermon", os.path.join(here, "ml_sermon.py"))
ml_sermon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml_sermon)
SERMON = ml_sermon.SERMON

# Old-style chillu letters (consonant + virama + ZWJ) -> atomic chillu.
CHILLU = {"ണ്‍": "ൺ", "ന്‍": "ൻ", "ര്‍": "ർ", "ല്‍": "ൽ", "ള്‍": "ൾ"}


def norm(text):
    text = unicodedata.normalize("NFC", text)
    for old, new in CHILLU.items():
        text = text.replace(old, new)
    text = text.replace("‌", "").replace("‍", "")
    text = text.replace("ൌ", "ൗ")
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(text.split())


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# Number words in the test sermon and their digits: an engine that writes
# "16" for പതിനാറാം heard it right.
DIGITS = {
    "ഒന്നാം": "1", "രണ്ടാം": "2", "മൂന്നാം": "3", "നാലാം": "4", "നാലു": "4",
    "അഞ്ചാം": "5", "ആറാം": "6", "ഏഴു": "7", "എട്ടാം": "8", "എട്ടും": "8",
    "ഒമ്പതാം": "9", "ഒമ്പതും": "9", "പതിനൊന്നാം": "11", "പതിമൂന്നാം": "13",
    "പതിനഞ്ചാം": "15", "പതിനാറാം": "16", "ഇരുപത്തൊന്നാം": "21",
    "ഇരുപത്തിമൂന്നാം": "23", "ഇരുപത്തെട്ടാം": "28", "ഇരുപത്തൊമ്പതാം": "29",
    "മുപ്പത്തൊന്നാം": "31", "മുപ്പത്തിമൂന്നാം": "33", "നാല്പതാം": "40",
}


def key_hits(keys, heard):
    """Count key words heard exactly, and heard within ~20% of letters."""
    words = heard.split()
    exact = close = 0
    pool = list(words)
    for k in keys:
        k = norm(k)
        digit = DIGITS.get(k)
        if digit in pool:
            k = digit
        if k in pool:
            exact += 1
            close += 1
            pool.remove(k)
            continue
        best = min(pool, key=lambda w: edit(k, w), default=None)
        if best is not None and edit(k, best) <= max(1, round(len(k) * 0.2)):
            close += 1
            pool.remove(best)
    return exact, close


def score(path):
    keys_total = exact = close = 0
    err = chars = 0
    times = []
    lines = []
    for row in open(path, encoding="utf-8"):
        parts = row.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        clip, secs = parts[0], float(parts[1])
        heard = norm(parts[2] if len(parts) > 2 else "")
        n = int(re.search(r"line(\d+)", clip).group(1))
        said, refs, keys = SERMON[n]
        said = norm(said)
        times.append(secs)
        err += edit(said, heard)
        chars += len(said)
        e, c = key_hits(keys, heard)
        keys_total += len(keys)
        exact += e
        close += c
        if refs:
            lines.append(f"    {n:02d} {refs[0]:<22} {e}/{len(keys)} | {heard}")
    avg = sum(times) / max(len(times), 1)
    verses = None
    refs_path = path.removesuffix(".tsv") + ".refs"
    if os.path.exists(refs_path):
        right = wrong = 0
        for row in open(refs_path, encoding="utf-8"):
            parts = row.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            n = int(re.search(r"line(\d+)", parts[0]).group(1))
            found = parts[2].split("; ") if len(parts) > 2 and parts[2] else []
            want = SERMON[n][1]
            right += sum(1 for w in want if w in found)
            bad = [f for f in found if f not in want]
            wrong += len(bad)
            if bad or any(w not in found for w in want):
                lines.append(f"    {n:02d} finder: wanted {want} got {found}")
        verses = (right, wrong)
    return keys_total, exact, close, err / max(chars, 1), avg, lines, verses


def main():
    rows = []
    print("\n==== MALAYALAM ENGINE COMPARISON ====")
    total_refs = sum(len(r) for _, r, _ in SERMON)
    print(f"{'engine / voice / sound':40} {'verses':>7} {'wrong':>6} {'key words':>10} {'close':>8} {'CER':>6} {'s/line':>7}")
    for path in sys.argv[1:]:
        label = os.path.basename(path).removesuffix(".tsv")
        total, exact, close, cer, avg, lines, verses = score(path)
        rows.append((label, lines))
        v = f"{verses[0]:>3}/{total_refs:<3} {verses[1]:>6}" if verses else f"{'-':>7} {'-':>6}"
        print(f"{label:40} {v} {exact:>4}/{total:<5} {close:>3}/{total:<4} {cer:>6.1%} {avg:>7.2f}")
    print("\n==== WHAT EACH ENGINE HEARD (reference lines; n/m = key words exact) ====")
    for label, lines in rows:
        print(f"\n-- {label}")
        for l in lines:
            print(l)


if __name__ == "__main__":
    main()
