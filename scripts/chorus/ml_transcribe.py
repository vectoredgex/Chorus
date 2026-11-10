"""Transcribe Malayalam test clips with one speech engine.

Writes one row per clip to stdout: <clip>\t<seconds>\t<text>
Models are downloaded from Hugging Face on first use (gated ones need HF_TOKEN) (run this in CI; the
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
  koyal               Adalat AI Koyal Malayalam 120M (Conformer TDT-CTC,
                      NeMo, Sept 2026). Gated: needs HF_TOKEN from an
                      account that accepted its terms.
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


def koyal(threads):
    import glob
    import os
    import torch
    from huggingface_hub import snapshot_download
    import nemo.collections.asr as nemo_asr
    torch.set_num_threads(threads)
    d = snapshot_download("adalat-ai/koyal-ml-120m-1.0", token=os.environ.get("HF_TOKEN"))
    nemo_files = glob.glob(os.path.join(d, "**", "*.nemo"), recursive=True)
    if not nemo_files:
        sys.exit("no .nemo file in koyal download: " + ", ".join(os.listdir(d)))
    model = nemo_asr.models.ASRModel.restore_from(nemo_files[0], map_location="cpu")
    model.eval()

    def run(wav):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, wav, 16000)
            out = model.transcribe([f.name], batch_size=1, verbose=False)
        if isinstance(out, tuple):
            out = out[0]
        h = out[0]
        return h if isinstance(h, str) else getattr(h, "text", str(h))
    return run


def indicconformer_official(decoder, threads, int8=""):
    """AI4Bharat's own IndicConformer 600M (gated on Hugging Face), with
    its CTC or RNNT decoder."""
    import os
    import numpy as np
    import torch
    from transformers import AutoModel
    torch.set_num_threads(threads)
    model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual",
                                      trust_remote_code=True, token=os.environ.get("HF_TOKEN"))

    if int8:
        # Shrink the 2.4 GB encoder to int8 weights (MatMul/Gemm only), the
        # way the app would ship it.
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from onnxruntime.quantization import QuantType, quantize_dynamic
        src = os.path.join(snapshot_download("ai4bharat/indic-conformer-600m-multilingual",
                                             token=os.environ.get("HF_TOKEN")), "assets", "encoder.onnx")
        # int8 = "plain" | "pc" (per-channel scales) | "pcx" (per-channel,
        # and the first and last two conformer layers kept full precision).
        dst = os.path.expanduser(f"~/.cache/omnilingual/ic-encoder-int8-{int8}.onnx")
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # onnx refuses external weights behind the HF cache's symlinks:
            # copy the real files first.
            import shutil
            real = os.path.expanduser("~/ic-assets")
            shutil.rmtree(real, ignore_errors=True)
            shutil.copytree(os.path.dirname(src), real, symlinks=False)
            src = os.path.join(real, "encoder.onnx")
            exclude = []
            if int8 == "pcx":
                import re
                import onnx
                g = onnx.load(src, load_external_data=False).graph
                def layer(n):
                    m = re.search(r"layers\.(\d+)[./]", " ".join([n.name, *n.input]))
                    return int(m.group(1)) if m else None
                last = max((layer(n) for n in g.node if layer(n) is not None), default=0)
                keep = {0, 1, last - 1, last}
                exclude = [n.name for n in g.node
                           if n.op_type in ("MatMul", "Gemm") and layer(n) in keep]
                print(f"layers 0..{last}; keeping {len(exclude)} MatMuls full precision", file=sys.stderr)
            quantize_dynamic(src, dst, weight_type=QuantType.QInt8,
                             op_types_to_quantize=["MatMul", "Gemm"],
                             per_channel=int8 in ("pc", "pcx"),
                             nodes_to_exclude=exclude)
        print(f"int8 encoder: {os.path.getsize(dst)/1e6:.0f} MB", file=sys.stderr)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        model.models["encoder"] = ort.InferenceSession(dst, opts, providers=["CPUExecutionProvider"])

    def run(wav):
        x = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
        out = model(x, "ml", decoder)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return str(out)
    return run


def indic_transcribe_onnx(threads):
    """Indic-Transcribe as int8 ONNX for CPU (third-party export,
    Mvkd2k06/indic-transcribe-core-onnx): encoder + cached decoder,
    greedy decoding."""
    import glob
    import json
    import os
    import tarfile
    import numpy as np
    import onnxruntime as ort
    tar = hf_hub_download("Mvkd2k06/indic-transcribe-core-onnx",
                          "variants/int8/indic-transcribe-core-int8.tar")
    root = os.path.join(os.path.dirname(tar), "int8-extracted")
    if not os.path.isdir(root):
        with tarfile.open(tar) as t:
            t.extractall(root, filter="data")
    find = lambda pat: sorted(glob.glob(os.path.join(root, "**", pat), recursive=True))[0]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    enc = ort.InferenceSession(find("encoder*.onnx"), opts, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(find("decoder*.onnx"), opts, providers=["CPUExecutionProvider"])
    tok = json.load(open(find("tokenizer.json"), encoding="utf-8"))
    for sess in (enc, dec):
        print("io:", [(i.name, i.type, i.shape) for i in sess.get_inputs()],
              [(o.name, o.type) for o in sess.get_outputs()], file=sys.stderr)
    prompts = tok["prompts"]
    prompt = list(prompts["ml"] if isinstance(prompts, dict) else prompts)
    print("ml prompt:", prompt, file=sys.stderr)
    pieces, special = tok["pieces"], tok["special_count"]
    stop = {tok["eos_id"], tok["pad_id"]}
    np_type = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
               "tensor(int64)": np.int64, "tensor(int32)": np.int32}
    dtypes = {i.name: np_type.get(i.type, np.float32) for i in dec.get_inputs()}

    def run(wav):
        wav = np.asarray(wav, dtype=np.float32)
        if len(wav) < 16000:
            pad = 16000 - len(wav)
            wav = np.pad(wav, (pad // 2, pad - pad // 2))
        ck, cv = enc.run(None, {"audio": wav[None, :]})[:2]
        empty = np.zeros((0, 24, 8, 128), dtype=dtypes.get("past_k", np.float32))
        pk, pv, ids, out = empty, empty, prompt, []
        for _ in range(256):
            logits, nk, nv = dec.run(None, {
                "input_ids": np.array([ids], dtype=dtypes.get("input_ids", np.int64)),
                "past_k": pk, "past_v": pv, "cross_k": ck, "cross_v": cv})
            pk, pv = np.concatenate([pk, nk]), np.concatenate([pv, nv])
            t = int(np.argmax(logits[0]))
            if t in stop:
                break
            out.append(t)
            ids = [t]
        return "".join(pieces[t] for t in out if t >= special).replace("\u2581", " ").strip()
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
    elif engine == "indicconformer-rnnt":
        run = indicconformer_official("rnnt", threads)
    elif engine.startswith("indicconformer-rnnt-int8"):
        run = indicconformer_official("rnnt", threads, int8=engine.split("int8")[1].strip("-") or "plain")
    elif engine == "indicconformer-ctc":
        run = indicconformer_official("ctc", threads)
    elif engine == "indic-transcribe-int8":
        run = indic_transcribe_onnx(threads)
    elif engine == "koyal":
        run = koyal(threads)
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
