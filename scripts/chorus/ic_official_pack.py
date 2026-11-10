"""Package AI4Bharat's official IndicConformer (600M multilingual, RNNT)
for Chorus: Malayalam only, int8 encoder, plain ONNX files the app's Rust
code can run with onnxruntime (no PyTorch).

    python3 ic_official_pack.py <out_dir> [plain|pc|pcx]

Needs HF_TOKEN (the model is gated), torch, onnx, onnxruntime.

Output folder:
  preprocessor.onnx   audio (1, samples) f32 16 kHz, length (1,) i64
                      -> features (1, 80, frames), feat_len (1,)
  encoder.onnx        int8; audio_signal, length -> outputs, encoded_lengths
  rnnt_decoder.onnx, joint_enc.onnx, joint_pred.onnx, joint_pre_net.onnx,
  joint_post_net_ml.onnx   (unchanged from the original)
  tokens.json         Malayalam pieces, blank, sos, LSTM sizes
"""
import json
import os
import re
import shutil
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import snapshot_download
from onnxruntime.quantization import QuantType, quantize_dynamic

REPO = "ai4bharat/indic-conformer-600m-multilingual"


def io(path):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return ([(i.name, i.type, i.shape) for i in s.get_inputs()],
            [(o.name, o.type, o.shape) for o in s.get_outputs()])


def main():
    out, mode = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "pc")
    if mode.startswith("ic-official-"):
        mode = mode.rsplit("-", 1)[-1]
        mode = "plain" if mode in ("pack", "rust") else mode
    snap = snapshot_download(REPO, token=os.environ.get("HF_TOKEN"))
    # onnx refuses external weights behind the HF cache's symlinks.
    real = os.path.expanduser("~/ic-assets")
    if not os.path.isdir(real):
        shutil.copytree(os.path.join(snap, "assets"), real, symlinks=False)
    os.makedirs(out, exist_ok=True)

    # Preprocessor: TorchScript -> ONNX, checked against the original.
    pre = torch.jit.load(os.path.join(real, "preprocessor.ts"), map_location="cpu")
    print("=== preprocessor code\n", pre.code[:6000])
    for name, sub in pre.named_modules():
        attrs = {k: v for k, v in vars(sub).items() if not k.startswith("_")} if hasattr(sub, "__dict__") else {}
        print("module", name, type(sub).__name__, str(attrs)[:300])

    class Pre(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, audio, length):
            return self.m(input_signal=audio, length=length)

    rng = np.random.default_rng(0)
    wav = torch.from_numpy((rng.standard_normal(16000 * 3) * 0.05).astype(np.float32)).unsqueeze(0)
    ln = torch.tensor([wav.shape[-1]])
    ref, ref_len = pre(input_signal=wav, length=ln)
    print("features", tuple(ref.shape), ref.dtype, "len", ref_len)
    pre_path = os.path.join(out, "preprocessor.onnx")
    torch.onnx.export(Pre(pre), (wav, ln), pre_path, input_names=["audio", "length"],
                      output_names=["features", "feat_len"], opset_version=17,
                      dynamic_axes={"audio": {1: "samples"}, "features": {2: "frames"}},
                      dynamo=False)
    s = ort.InferenceSession(pre_path, providers=["CPUExecutionProvider"])
    for secs in (1.3, 3.0, 7.7):
        w = (rng.standard_normal(int(16000 * secs)) * 0.05).astype(np.float32)[None]
        a, al = pre(input_signal=torch.from_numpy(w), length=torch.tensor([w.shape[1]]))
        b, bl = s.run(None, {"audio": w, "length": np.array([w.shape[1]], dtype=np.int64)})
        print(f"preprocessor {secs}s: max diff {np.abs(a.numpy() - b).max():.2e}, len {al.item()} vs {bl}")

    # Encoder: int8.
    src = os.path.join(real, "encoder.onnx")
    exclude = []
    if mode in ("pcx", "px", "px6"):
        g = onnx.load(src, load_external_data=False).graph

        def layer(n):
            m = re.search(r"layers\.(\d+)[./]", " ".join([n.name, *n.input]))
            return int(m.group(1)) if m else None
        last = max((layer(n) for n in g.node if layer(n) is not None), default=0)
        keep = {0, 1, 2, last - 2, last - 1, last} if mode == "px6" else {0, 1, last - 1, last}
        exclude = [n.name for n in g.node if n.op_type in ("MatMul", "Gemm") and layer(n) in keep]
    enc = os.path.join(out, "encoder.onnx")
    quantize_dynamic(src, enc, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"],
                     per_channel=mode in ("pc", "pcx"), nodes_to_exclude=exclude)

    for n in ["rnnt_decoder", "joint_enc", "joint_pred", "joint_pre_net", "joint_post_net_ml"]:
        shutil.copy(os.path.join(real, f"{n}.onnx"), os.path.join(out, f"{n}.onnx"))
    cfg = json.load(open(os.path.join(snap, "config.json")))
    vocab = json.load(open(os.path.join(real, "vocab.json"), encoding="utf-8"))["ml"]
    json.dump({"pieces": vocab, "blank": cfg.get("BLANK_ID", 256), "sos": cfg.get("SOS", 5632),
               "pred_layers": cfg.get("PRED_RNN_LAYERS", 2), "pred_hidden": cfg.get("PRED_RNN_HIDDEN_DIM", 640),
               "max_symbols": cfg.get("RNNT_MAX_SYMBOLS", 10)},
              open(os.path.join(out, "tokens.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("config", cfg)
    for f in sorted(os.listdir(out)):
        p = os.path.join(out, f)
        print(f"{os.path.getsize(p)/1e6:9.1f} MB {f}", io(p) if f.endswith(".onnx") else "")


if __name__ == "__main__":
    main()
