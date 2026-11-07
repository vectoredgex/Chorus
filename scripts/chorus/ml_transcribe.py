"""Transcribe Malayalam test clips with one speech engine.

Writes one row per clip to stdout: <clip>\t<seconds>\t<text>
Models are downloaded from Hugging Face on first use (run this in CI; the
cloud container cannot reach Hugging Face).

Engines:
  indicconformer      AI4Bharat IndicConformer 600M multilingual, Malayalam
                      head, int8 sherpa-onnx (Docglasses-AI conversion)
  indicconformer-alt  community Malayalam-only sherpa-onnx export
                      (jeswinjestin), for a second opinion
  whisper-ml          Whisper medium fine-tuned for Malayalam (thennal)
  whisper-turbo       Whisper large-v3-turbo, language set to Malayalam
                      (the Whisper Chorus ships today)
  omnilingual-300m    Meta Omnilingual ASR CTC 300M int8 (1,600 languages)
  omnilingual-1b      Meta Omnilingual ASR CTC 1B int8
                      (both sherpa-onnx, from the sherpa-onnx GitHub release)
  indic-transcribe    Bodhan AI / AI4Bharat Indic-Transcribe core (1.2B,
                      Canary-based, Sept 2026). Gated: needs HF_TOKEN from an
                      account that accepted its licence.

Usage: python3 ml_transcribe.py <engine> <threads> <clip.wav>...
"""

import sys
import time

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download


def load(path):
    wav, sr = sf.read(path, dtype="float32")
    assert sr == 16000, f"{path}: {sr} Hz"
    return wav


def sherpa(model_repo, model_file, tokens_file, threads):
    import sherpa_onnx
    model = hf_hub_download(model_repo, model_file)
    tokens = hf_hub_download(model_repo, tokens_file)
    rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=model, tokens=tokens, num_threads=threads,
        decoding_method="greedy_search", provider="cpu")

    def run(wav):
        s = rec.create_stream()
        s.accept_waveform(16000, wav)
        rec.decode_stream(s)
        return s.result.text
    return run


def omnilingual(size, threads):
    import os
    import subprocess
    import sherpa_onnx
    name = f"sherpa-onnx-omnilingual-asr-1600-languages-{size}-ctc-int8-2025-11-12"
    root = os.path.expanduser("~/.cache/omnilingual")
    d = os.path.join(root, name)
    if not os.path.exists(os.path.join(d, "tokens.txt")):
        os.makedirs(root, exist_ok=True)
        url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{name}.tar.bz2"
        subprocess.run(["curl", "-sSL", "--fail", "--retry", "3", "-o", d + ".tar.bz2", url], check=True)
        subprocess.run(["tar", "xjf", d + ".tar.bz2", "-C", root], check=True)
        os.remove(d + ".tar.bz2")
    model = os.path.join(d, "model.int8.onnx")
    tokens = os.path.join(d, "tokens.txt")
    if hasattr(sherpa_onnx.OfflineRecognizer, "from_omnilingual_asr_ctc"):
        rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
            model=model, tokens=tokens, num_threads=threads)
    else:
        sys.exit("sherpa_onnx has no from_omnilingual_asr_ctc: " +
                 ", ".join(a for a in dir(sherpa_onnx.OfflineRecognizer) if a.startswith("from_")))

    def run(wav):
        s = rec.create_stream()
        s.accept_waveform(16000, wav)
        rec.decode_stream(s)
        return s.result.text
    return run


def whisper(repo, threads):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    torch.set_num_threads(threads)
    proc = WhisperProcessor.from_pretrained(repo)
    model = WhisperForConditionalGeneration.from_pretrained(repo, torch_dtype=torch.float32)
    model.eval()

    def run(wav):
        feats = proc(wav, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(feats, language="ml", task="transcribe",
                                 num_beams=1, max_new_tokens=200)
        return proc.batch_decode(ids, skip_special_tokens=True)[0]
    return run


def indic_transcribe(threads):
    import os
    import torch
    from huggingface_hub import snapshot_download
    torch.set_num_threads(threads)
    model_dir = snapshot_download("bodhan-ai/indic-transcribe-core",
                                  token=os.environ.get("HF_TOKEN"),
                                  ignore_patterns=["nemo/*", "*.png"])
    req = os.path.join(model_dir, "requirements.txt")
    if os.path.exists(req):
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-r", req], check=False)
    sys.path.insert(0, model_dir)  # the model code ships inside the download
    from indic_transcribe import IndicTranscribe
    asr = IndicTranscribe.from_pretrained(model_dir)

    def run(wav):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, wav, 16000)
            out = asr(f.name, lang="ml")
        if isinstance(out, dict):
            out = out.get("text", "")
        elif isinstance(out, (list, tuple)):
            out = " ".join(o if isinstance(o, str) else o.get("text", "") for o in out)
        return str(out)
    return run


def main():
    engine, threads, clips = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    if engine == "indicconformer":
        run = sherpa("Docglasses-AI/indic-conformer-600m-sherpa-onnx-int8",
                     "indicconformer-ml.int8.onnx", "indicconformer-tokens.txt", threads)
    elif engine == "indicconformer-alt":
        run = sherpa("jeswinjestin/sherpa-onnx-nemo-ctc-indicconformer-malayalam",
                     "model.onnx", "tokens.txt", threads)
    elif engine == "whisper-ml":
        run = whisper("thennal/whisper-medium-ml", threads)
    elif engine == "omnilingual-300m":
        run = omnilingual("300M", threads)
    elif engine == "omnilingual-1b":
        run = omnilingual("1B", threads)
    elif engine == "indic-transcribe":
        run = indic_transcribe(threads)
    elif engine == "whisper-turbo":
        run = whisper("openai/whisper-large-v3-turbo", threads)
    else:
        sys.exit(f"unknown engine {engine}")

    run(load(clips[0]))  # warm-up, not timed
    for clip in clips:
        wav = load(clip)
        t = time.perf_counter()
        text = run(wav).replace("\t", " ").replace("\n", " ").strip()
        print(f"{clip}\t{time.perf_counter() - t:.3f}\t{text}", flush=True)


if __name__ == "__main__":
    main()
