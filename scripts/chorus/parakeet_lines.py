"""Transcribe clips with NVIDIA Parakeet (sherpa-onnx, CPU) and print
<clip>\t<seconds>\t<text> per clip, the same format as whisper_lines.

Usage: python3 parakeet_lines.py <model_dir> <threads> <clip.wav>...
"""

import sys
import time
import wave

import numpy as np
import sherpa_onnx


def read_wav(path):
    with wave.open(path) as w:
        data = w.readframes(w.getnframes())
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0, w.getframerate()


def main():
    model, threads, clips = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{model}/encoder.int8.onnx",
        decoder=f"{model}/decoder.int8.onnx",
        joiner=f"{model}/joiner.int8.onnx",
        tokens=f"{model}/tokens.txt",
        num_threads=threads,
        model_type="nemo_transducer",
        decoding_method="greedy_search",
    )
    # Warm up once so the first clip isn't charged for start-up.
    s = rec.create_stream()
    s.accept_waveform(16000, np.zeros(16000, dtype=np.float32))
    rec.decode_stream(s)
    for clip in clips:
        audio, rate = read_wav(clip)
        t = time.perf_counter()
        s = rec.create_stream()
        s.accept_waveform(rate, audio)
        rec.decode_stream(s)
        secs = time.perf_counter() - t
        text = s.result.text.replace("\t", " ").replace("\n", " ").strip()
        print(f"{clip}\t{secs:.3f}\t{text}", flush=True)


if __name__ == "__main__":
    main()
