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


def whisper(repo, threads):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    torch.set_num_threads(threads)
    proc = WhisperProcessor.from_pretrained(repo)
    model = WhisperForConditionalGeneration.from_pretrained(repo)
    model.eval()

    def run(wav):
        feats = proc(wav, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(feats, language="ml", task="transcribe",
                                 num_beams=1, max_new_tokens=200)
        return proc.batch_decode(ids, skip_special_tokens=True)[0]
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
