//! AI4Bharat's official IndicConformer (600M multilingual, RNNT decoding),
//! Malayalam only, with an int8 encoder: the quick engine of the
//! Malayalam combo.
//!
//! Measured on the Chorus Malayalam test sermon (384 lines; the Python
//! reference of this code): 71% of verses found at 0.35 s per line on a
//! 4-core cloud CPU, against 65% at 0.53 s for the sherpa-onnx
//! IndicConformer conversion it replaces.
//!
//! The folder is made by `scripts/chorus/ic_official_pack.py`:
//! - `preprocessor.onnx`: `audio` (1, samples) f32 16 kHz, `length` (1) i64
//!   -> `features` (1, 80, frames), `feat_len`
//! - `encoder.onnx`: `audio_signal`, `length` -> `outputs` (1, D, T)
//! - `joint_enc.onnx`: (1, T, D) -> (1, T, H)
//! - `rnnt_decoder.onnx`: `targets` (1, 1) i32, `target_length` (1) i32,
//!   LSTM state `states.1` / `onnx::Slice_3` -> `outputs` (1, P, 1),
//!   new state `states` / `162`
//! - `joint_pred.onnx`: (1, 1, P) -> (1, 1, H)
//! - `joint_pre_net.onnx`, `joint_post_net_ml.onnx`: (1, 1, H) -> logits
//! - `tokens.json`: Malayalam pieces, blank and start ids, LSTM sizes.
//!
//! Decoding is greedy RNNT, as in AI4Bharat's own `model_onnx.py`.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

/// Folder name of the model inside `models/malayalam/`.
pub const DIR_NAME: &str = "indicconformer-official-ml";

const FILES: &[&str] = &[
    "preprocessor",
    "encoder",
    "rnnt_decoder",
    "joint_enc",
    "joint_pred",
    "joint_pre_net",
    "joint_post_net_ml",
];

/// True when `dir` holds every part of the model.
pub fn model_present(dir: &Path) -> bool {
    FILES.iter().all(|f| dir.join(format!("{f}.onnx")).is_file()) && dir.join("tokens.json").is_file()
}

#[derive(serde::Deserialize)]
struct TokensFile {
    pieces: Vec<String>,
    blank: i64,
    sos: i64,
    pred_layers: usize,
    pred_hidden: usize,
    max_symbols: usize,
}

pub struct IcOfficial {
    pre: Session,
    enc: Session,
    dec: Session,
    jenc: Session,
    jpred: Session,
    jpre: Session,
    jpost: Session,
    tok: TokensFile,
}

impl std::fmt::Debug for IcOfficial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcOfficial").finish_non_exhaustive()
    }
}

fn dims(s: &ort::value::Shape) -> Vec<usize> {
    s.iter().map(|&d| usize::try_from(d).unwrap_or(0)).collect()
}

impl IcOfficial {
    pub fn load(dir: &Path, n_threads: usize) -> Result<Self, String> {
        let session = |name: &str| -> Result<Session, String> {
            let path = dir.join(format!("{name}.onnx"));
            let fail = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
            // The small joint/decoder graphs run hundreds of times per
            // phrase: one thread each avoids thread start-up per call.
            let threads = if matches!(name, "preprocessor" | "encoder") { n_threads.max(1) } else { 1 };
            Session::builder()
                .map_err(|e| fail(&e))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| fail(&e))?
                .with_intra_threads(threads)
                .map_err(|e| fail(&e))?
                .commit_from_file(&path)
                .map_err(|e| fail(&e))
        };
        let text = std::fs::read_to_string(dir.join("tokens.json")).map_err(|e| format!("tokens.json: {e}"))?;
        let tok: TokensFile = serde_json::from_str(&text).map_err(|e| format!("tokens.json: {e}"))?;
        Ok(Self {
            pre: session("preprocessor")?,
            enc: session("encoder")?,
            dec: session("rnnt_decoder")?,
            jenc: session("joint_enc")?,
            jpred: session("joint_pred")?,
            jpre: session("joint_pre_net")?,
            jpost: session("joint_post_net_ml")?,
            tok,
        })
    }

    /// Transcribe one phrase of 16 kHz mono audio.
    pub fn transcribe(&mut self, audio: &[f32]) -> Result<String, String> {
        let err = |e: ort::Error| e.to_string();
        if audio.len() < 1_600 {
            return Ok(String::new());
        }
        let n = audio.len();

        // Features.
        let out = self
            .pre
            .run(ort::inputs![
                "audio" => Tensor::from_array(([1usize, n], audio.to_vec())).map_err(err)?,
                "length" => Tensor::from_array(([1usize], vec![n as i64])).map_err(err)?,
            ])
            .map_err(err)?;
        let (fs, feats) = out["features"].try_extract_tensor::<f32>().map_err(err)?;
        let (fs, feats) = (dims(fs), feats.to_vec());
        let flen = fs.get(2).copied().unwrap_or(0) as i64;
        drop(out);

        // Encoder: (1, D, T), turned to (1, T, D) for the joint.
        let out = self
            .enc
            .run(ort::inputs![
                "audio_signal" => Tensor::from_array((fs, feats)).map_err(err)?,
                "length" => Tensor::from_array(([1usize], vec![flen])).map_err(err)?,
            ])
            .map_err(err)?;
        let (es, e) = out["outputs"].try_extract_tensor::<f32>().map_err(err)?;
        let es = dims(es);
        let (d, t) = (es[1], es[2]);
        let mut et = vec![0f32; d * t];
        for di in 0..d {
            for ti in 0..t {
                et[ti * d + di] = e[di * t + ti];
            }
        }
        drop(out);
        let out = self
            .jenc
            .run(ort::inputs!["input" => Tensor::from_array(([1usize, t, d], et)).map_err(err)?])
            .map_err(err)?;
        let (_, f_all) = out["output"].try_extract_tensor::<f32>().map_err(err)?;
        let f_all = f_all.to_vec();
        drop(out);
        let h = if t == 0 { 0 } else { f_all.len() / t };

        // Greedy RNNT.
        let state_len = self.tok.pred_layers * self.tok.pred_hidden;
        let state_shape = [self.tok.pred_layers, 1usize, self.tok.pred_hidden];
        let (mut hs, mut cs) = (vec![0f32; state_len], vec![0f32; state_len]);
        let mut last = self.tok.sos;
        let mut hyp: Vec<i64> = Vec::new();
        // Prediction for `last` and the state it leads to, made once per token.
        let mut pred: Option<(Vec<f32>, Vec<f32>, Vec<f32>)> = None;
        for ti in 0..t {
            let f = &f_all[ti * h..(ti + 1) * h];
            for _ in 0..self.tok.max_symbols {
                if pred.is_none() {
                    let out = self
                        .dec
                        .run(ort::inputs![
                            "targets" => Tensor::from_array(([1usize, 1], vec![last as i32])).map_err(err)?,
                            "target_length" => Tensor::from_array(([1usize], vec![1i32])).map_err(err)?,
                            "states.1" => Tensor::from_array((state_shape, hs.clone())).map_err(err)?,
                            "onnx::Slice_3" => Tensor::from_array((state_shape, cs.clone())).map_err(err)?,
                        ])
                        .map_err(err)?;
                    let (gs, g) = out["outputs"].try_extract_tensor::<f32>().map_err(err)?;
                    let p = dims(gs).get(1).copied().unwrap_or(g.len());
                    let g = g.to_vec();
                    let (_, nh) = out["states"].try_extract_tensor::<f32>().map_err(err)?;
                    let (_, nc) = out["162"].try_extract_tensor::<f32>().map_err(err)?;
                    let (nh, nc) = (nh.to_vec(), nc.to_vec());
                    drop(out);
                    // (1, P, 1) is already (1, 1, P) in memory.
                    let out = self
                        .jpred
                        .run(ort::inputs!["input" => Tensor::from_array(([1usize, 1, p], g)).map_err(err)?])
                        .map_err(err)?;
                    let (_, gp) = out["output"].try_extract_tensor::<f32>().map_err(err)?;
                    pred = Some((gp.to_vec(), nh, nc));
                }
                let (g, nh, nc) = pred.as_ref().expect("set above");
                let x: Vec<f32> = f.iter().zip(g).map(|(a, b)| a + b).collect();
                let out = self
                    .jpre
                    .run(ort::inputs!["input" => Tensor::from_array(([1usize, 1, h], x)).map_err(err)?])
                    .map_err(err)?;
                let (xs, x) = out["output"].try_extract_tensor::<f32>().map_err(err)?;
                let (xs, x) = (dims(xs), x.to_vec());
                drop(out);
                let out = self
                    .jpost
                    .run(ort::inputs!["input" => Tensor::from_array((xs, x)).map_err(err)?])
                    .map_err(err)?;
                let (_, logits) = out["output"].try_extract_tensor::<f32>().map_err(err)?;
                let k = logits
                    .iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
                    .0 as i64;
                drop(out);
                if k == self.tok.blank {
                    break;
                }
                hyp.push(k);
                last = k;
                hs = nh.clone();
                cs = nc.clone();
                pred = None;
            }
        }
        Ok(self.text_of(&hyp))
    }

    fn text_of(&self, hyp: &[i64]) -> String {
        hyp.iter()
            .filter_map(|&k| usize::try_from(k).ok().and_then(|k| self.tok.pieces.get(k)))
            .map(String::as_str)
            .collect::<String>()
            .replace('\u{2581}', " ")
            .trim()
            .to_string()
    }
}

/// The model kept loaded between Start/Stop, as for Indic-Transcribe.
static LOADED: Mutex<Option<(PathBuf, usize, IcOfficial)>> = Mutex::new(None);

pub(crate) fn take_or_load(dir: &Path, n_threads: usize) -> Result<IcOfficial, String> {
    if let Ok(mut g) = LOADED.lock() {
        if let Some((d, n, m)) = g.take() {
            if d == dir && n == n_threads {
                return Ok(m);
            }
        }
    }
    IcOfficial::load(dir, n_threads)
}

pub(crate) fn keep_loaded(dir: PathBuf, n_threads: usize, model: IcOfficial) {
    if let Ok(mut g) = LOADED.lock() {
        *g = Some((dir, n_threads, model));
    }
}
