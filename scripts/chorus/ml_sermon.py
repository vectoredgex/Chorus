"""Malayalam test sermon for comparing speech engines.

Each line is spoken by a text-to-speech voice and saved as its own clip:
  <out>/lineNN.wav        16 kHz mono, as spoken
  <out>/lineNN_noisy.wav  same, with room echo and background hiss

Voices:
  natural  Meta MMS Malayalam voice (facebook/mms-tts-mal, via transformers)
  robot    espeak-ng Malayalam

Usage: python3 ml_sermon.py <out_dir> natural|robot
"""

import os
import subprocess
import sys

# (spoken line, verses it should produce, key words the engine must hear)
# Key words are the book name and the chapter / verse number words. They are
# what the reference finder needs; the rest of the sentence matters less.
SERMON = [
    ("പ്രിയ ദൈവജനമേ, ഇന്ന് നമ്മൾ ഒരുമിച്ച് ആരാധിക്കുന്നതിൽ വളരെ സന്തോഷം.", [], []),
    ("യോഹന്നാൻ മൂന്നാം അദ്ധ്യായം പതിനാറാം വാക്യം നമുക്ക് വായിക്കാം.",
     ["John 3:16"], ["യോഹന്നാൻ", "മൂന്നാം", "പതിനാറാം"]),
    ("ദൈവം തന്റെ ഏകജാതനായ പുത്രനെ നല്കുവാൻ തക്കവണ്ണം ലോകത്തെ സ്നേഹിച്ചു.", [], []),
    ("റോമർ എട്ടാം അദ്ധ്യായം ഇരുപത്തെട്ടാം വാക്യം ശ്രദ്ധിക്കുക.",
     ["Romans 8:28"], ["റോമർ", "എട്ടാം", "ഇരുപത്തെട്ടാം"]),
    ("ഒന്നാം കൊരിന്ത്യർ പതിമൂന്നാം അദ്ധ്യായം നാലു മുതൽ ഏഴു വരെയുള്ള വാക്യങ്ങൾ.",
     ["1 Corinthians 13:4-7"], ["ഒന്നാം", "കൊരിന്ത്യർ", "പതിമൂന്നാം", "നാലു", "ഏഴു"]),
    ("സ്നേഹം ദീർഘമായി ക്ഷമിക്കുന്നു, സ്നേഹം ദയ കാണിക്കുന്നു.", [], []),
    ("സങ്കീർത്തനം ഇരുപത്തിമൂന്നാം അദ്ധ്യായം ഒന്നാം വാക്യം.",
     ["Psalms 23:1"], ["സങ്കീർത്തനം", "ഇരുപത്തിമൂന്നാം", "ഒന്നാം"]),
    ("ഫിലിപ്പിയർ നാലാം അദ്ധ്യായം പതിമൂന്നാം വാക്യം ഇങ്ങനെ പറയുന്നു.",
     ["Philippians 4:13"], ["ഫിലിപ്പിയർ", "നാലാം", "പതിമൂന്നാം"]),
    ("യെശയ്യാവ് നാല്പതാം അദ്ധ്യായം മുപ്പത്തൊന്നാം വാക്യം.",
     ["Isaiah 40:31"], ["യെശയ്യാവ്", "നാല്പതാം", "മുപ്പത്തൊന്നാം"]),
    ("യിരെമ്യാവ് ഇരുപത്തൊമ്പതാം അദ്ധ്യായം പതിനൊന്നാം വാക്യം.",
     ["Jeremiah 29:11"], ["യിരെമ്യാവ്", "ഇരുപത്തൊമ്പതാം", "പതിനൊന്നാം"]),
    ("എഫെസ്യർ രണ്ടാം അദ്ധ്യായം എട്ടും ഒമ്പതും വാക്യങ്ങൾ.",
     ["Ephesians 2:8-9"], ["എഫെസ്യർ", "രണ്ടാം", "എട്ടും", "ഒമ്പതും"]),
    ("എബ്രായർ പതിനൊന്നാം അദ്ധ്യായം ഒന്നാം വാക്യം.",
     ["Hebrews 11:1"], ["എബ്രായർ", "പതിനൊന്നാം", "ഒന്നാം"]),
    ("രണ്ടാം തിമൊഥെയൊസ് മൂന്നാം അദ്ധ്യായം പതിനാറാം വാക്യം.",
     ["2 Timothy 3:16"], ["രണ്ടാം", "തിമൊഥെയൊസ്", "മൂന്നാം", "പതിനാറാം"]),
    ("യോശുവ ഒന്നാം അദ്ധ്യായം ഒമ്പതാം വാക്യം, ബലവും ധൈര്യവുമുള്ളവനായിരിക്ക.",
     ["Joshua 1:9"], ["യോശുവ", "ഒന്നാം", "ഒമ്പതാം"]),
    ("ഒന്നാം യോഹന്നാൻ ഒന്നാം അദ്ധ്യായം ഒമ്പതാം വാക്യം.",
     ["1 John 1:9"], ["ഒന്നാം", "യോഹന്നാൻ", "ഒന്നാം", "ഒമ്പതാം"]),
    ("മത്തായി ആറാം അദ്ധ്യായം മുപ്പത്തിമൂന്നാം വാക്യം.",
     ["Matthew 6:33"], ["മത്തായി", "ആറാം", "മുപ്പത്തിമൂന്നാം"]),
    ("ഉല്പത്തി ഒന്നാം അദ്ധ്യായം ഒന്നാം വാക്യം, ആദിയിൽ ദൈവം ആകാശവും ഭൂമിയും സൃഷ്ടിച്ചു.",
     ["Genesis 1:1"], ["ഉല്പത്തി", "ഒന്നാം", "ഒന്നാം"]),
    ("സദൃശവാക്യങ്ങൾ മൂന്നാം അദ്ധ്യായം അഞ്ചാം വാക്യം.",
     ["Proverbs 3:5"], ["സദൃശവാക്യങ്ങൾ", "മൂന്നാം", "അഞ്ചാം"]),
    ("ലൂക്കൊസ് പതിനഞ്ചാം അദ്ധ്യായം പതിനൊന്നാം വാക്യം മുതൽ നാം വായിക്കുന്നു.",
     ["Luke 15:11"], ["ലൂക്കൊസ്", "പതിനഞ്ചാം", "പതിനൊന്നാം"]),
    ("ഒടുവിൽ, വെളിപ്പാട് ഇരുപത്തൊന്നാം അദ്ധ്യായം നാലാം വാക്യം.",
     ["Revelation 21:4"], ["വെളിപ്പാട്", "ഇരുപത്തൊന്നാം", "നാലാം"]),
    ("നമുക്ക് പ്രാർത്ഥിക്കാം.", [], []),
]


def run(*cmd):
    subprocess.run(cmd, check=True)


def to16k(src, dst):
    run("ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", dst)


def add_noise(src, dst):
    run("ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src,
        "-f", "lavfi", "-i", "anoisesrc=color=pink:amplitude=0.03:sample_rate=16000:seed=11",
        "-filter_complex",
        "[0:a]aecho=0.8:0.5:60:0.35,lowpass=3800[v];[v][1:a]amix=inputs=2:duration=first:weights=1 0.6",
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", dst)


class MmsVoice:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, VitsModel
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained("facebook/mms-tts-mal")
        self.model = VitsModel.from_pretrained("facebook/mms-tts-mal")
        self.model.eval()
        torch.manual_seed(7)

    def say(self, text, path):
        import numpy as np
        import soundfile as sf
        inputs = self.tok(text, return_tensors="pt")
        with self.torch.no_grad():
            wav = self.model(**inputs).waveform[0].numpy()
        raw = path + ".raw.wav"
        sf.write(raw, np.asarray(wav, dtype="float32"), self.model.config.sampling_rate)
        to16k(raw, path)
        os.remove(raw)


def espeak_say(text, path):
    raw = path + ".raw.wav"
    run("espeak-ng", "-v", "ml", "-s", "140", "-w", raw, text)
    to16k(raw, path)
    os.remove(raw)


def main():
    out, voice = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    say = MmsVoice().say if voice == "natural" else espeak_say
    for n, (line, _, _) in enumerate(SERMON):
        wav = os.path.join(out, f"line{n:02d}.wav")
        say(line, wav)
        add_noise(wav, os.path.join(out, f"line{n:02d}_noisy.wav"))
    print(f"voice: {voice}, {len(SERMON)} lines")


if __name__ == "__main__":
    main()
