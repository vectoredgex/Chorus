"""Build a test sermon for the Chorus replay test.

Speaks each line with a text-to-speech voice, joins the lines with natural
pauses, and writes:
  <out>/sermon_clean.wav   16 kHz mono, as spoken
  <out>/sermon_noisy.wav   same, with room echo and background hiss
  <out>/expected.txt       the verses Chorus should show, in order

Voices: Piper (natural neural voices) when installed, else espeak-ng.
Usage: python3 make-sermon.py <out_dir> [piper_voice.onnx]
"""

import os
import random
import subprocess
import sys

# (spoken line, verses it should produce)
SERMON = [
    ("Good morning church, it is wonderful to be together today.", []),
    ("Please open your Bibles with me to John chapter three, verse sixteen.", ["John 3:16"]),
    ("For God so loved the world, that he gave his only Son.", []),
    ("Now look at the very next verse.", ["John 3:17"]),
    ("Paul says the same thing in Romans chapter eight, verse twenty eight.", ["Romans 8:28"]),
    ("And we know that all things work together for good.", []),
    ("Let us turn to First Corinthians thirteen, verses four to seven.", ["1 Corinthians 13:4-7"]),
    ("Love is patient, love is kind.", []),
    ("My job this morning is simply to encourage you.", []),
    ("Mark my words, God is faithful.", []),
    ("Psalm twenty three, verse one says the Lord is my shepherd.", ["Psalms 23:1"]),
    ("Philippians four thirteen. I can do all things through Christ.", ["Philippians 4:13"]),
    ("Turn with me to Isaiah forty, verse thirty one.", ["Isaiah 40:31"]),
    ("Jeremiah twenty nine eleven tells us that God has plans for us.", ["Jeremiah 29:11"]),
    ("Ephesians two, verses eight and nine.", ["Ephesians 2:8-9"]),
    ("In Hebrews eleven one, faith is the substance of things hoped for.", ["Hebrews 11:1"]),
    ("Second Timothy three sixteen, all scripture is inspired by God.", ["2 Timothy 3:16"]),
    ("Joshua one nine. Be strong and courageous.", ["Joshua 1:9"]),
    ("First John one nine. If we confess our sins, he is faithful.", ["1 John 1:9"]),
    ("Finally, Revelation chapter twenty one, verse four.", ["Revelation 21:4"]),
    ("Let us pray.", []),
]


def run(*cmd):
    subprocess.run(cmd, check=True)


def speak(text, path, piper_voice):
    raw = path + ".raw.wav"
    if piper_voice:
        subprocess.run(
            ["piper", "--model", piper_voice, "--output_file", raw],
            input=text.encode(),
            check=True,
        )
    else:
        run("espeak-ng", "-s", "145", "-w", raw, text)
    run("ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", raw,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", path)
    os.remove(raw)


def main():
    out = sys.argv[1]
    piper_voice = sys.argv[2] if len(sys.argv) > 2 else None
    os.makedirs(out, exist_ok=True)
    random.seed(7)
    parts = []
    for n, (line, _) in enumerate(SERMON):
        wav = os.path.join(out, f"line{n:02d}.wav")
        speak(line, wav, piper_voice)
        parts.append(wav)
        gap = os.path.join(out, f"gap{n:02d}.wav")
        # Preachers pause between sentences for roughly 0.6 to 1.6 s.
        secs = round(random.uniform(0.6, 1.6), 2)
        run("ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi",
            "-i", "anullsrc=r=16000:cl=mono", "-t", str(secs),
            "-sample_fmt", "s16", gap)
        parts.append(gap)

    listing = os.path.join(out, "parts.txt")
    with open(listing, "w") as f:
        for p in parts:
            f.write(f"file '{os.path.abspath(p)}'\n")
    clean = os.path.join(out, "sermon_clean.wav")
    run("ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
        "-i", listing, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", clean)

    noisy = os.path.join(out, "sermon_noisy.wav")
    run("ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", clean,
        "-f", "lavfi", "-i", "anoisesrc=color=pink:amplitude=0.03:sample_rate=16000:seed=11",
        "-filter_complex",
        "[0:a]aecho=0.8:0.5:60:0.35,lowpass=3800[v];[v][1:a]amix=inputs=2:duration=first:weights=1 0.6",
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", noisy)

    with open(os.path.join(out, "expected.txt"), "w") as f:
        for _, refs in SERMON:
            for r in refs:
                f.write(r + "\n")
    print("voice:", piper_voice or "espeak-ng")


if __name__ == "__main__":
    main()
