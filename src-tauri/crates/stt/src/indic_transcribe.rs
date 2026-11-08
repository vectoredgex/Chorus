//! Bodhan AI Indic-Transcribe (Canary-style encoder + decoder, 1.2B), int8
//! ONNX export for the CPU, used for Malayalam.
//!
//! Measured on the Chorus Malayalam test sermon (3 voices; clear, fast,
//! noisy, echoey hall; 384 reference lines): 83% of verses found against
//! 61% for `IndicConformer`, at about 1.5 s per sentence on a 4-core cloud
//! CPU (`IndicConformer`: 0.5 s). It hears far better in echo and noise.
//!
//! It runs on the onnxruntime built into the app by the `ort` crate (the
//! one the verse search embeddings use), not sherpa-onnx's.
//!
//! Graphs (from the export, `Mvkd2k06/indic-transcribe-core-onnx`):
//! - encoder: `audio` (1, samples) f32 16 kHz -> `cross_k`, `cross_v`
//! - decoder: `input_ids` (1, n) i64, `past_k`/`past_v` (past, 24, 8, 128),
//!   `cross_k`/`cross_v` -> `logits` (1, vocab) for the last token, and
//!   `new_k`/`new_v` (n, 24, 8, 128) to append to the cache.
//!
//! Decoding is greedy: the 10-token Malayalam prompt first, then one token
//! per step until end-of-text.

use std::path::{Path, PathBuf};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{Tensor, TensorRef};

/// Folder name of the model inside `models/malayalam/`.
pub const DIR_NAME: &str = "indic-transcribe-core-int8";

const LAYERS: usize = 24;
const HEADS: usize = 8;
const HEAD_DIM: usize = 128;
/// Longest transcript of one phrase, in tokens.
const MAX_TOKENS: usize = 200;

/// The model file whose name starts with `prefix` and ends in `.onnx`.
fn find_onnx(dir: &Path, prefix: &str) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.starts_with(prefix) && name.ends_with(".onnx")
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

/// True when `dir` holds the model, its tokenizer and its runtime.
pub fn model_present(dir: &Path) -> bool {
    find_onnx(dir, "encoder").is_some()
        && find_onnx(dir, "decoder").is_some()
        && dir.join("tokenizer.json").is_file()
}

#[derive(serde::Deserialize)]
struct TokenizerFile {
    pieces: Vec<String>,
    special_count: usize,
    eos_id: i64,
    pad_id: i64,
    prompts: serde_json::Value,
}

/// Steers decoding toward Bible book names: among the model's top
/// candidates for the next token, one that starts (or continues) a book
/// name gets a bonus in log-probability. Only re-ranks what the model
/// already thinks likely, so ordinary speech is left alone.
#[derive(Debug, Clone)]
pub struct BookBias {
    /// Every prefix of every book spelling.
    prefixes: std::collections::HashSet<String>,
    /// Bonus for a token that begins a book name.
    start: f32,
    /// Bonus for a token that continues one already begun.
    cont: f32,
}

/// Candidates looked at per step.
const BIAS_TOP_K: usize = 8;
/// Default bonuses (log-probability), measured on the test sermon.
pub const DEFAULT_BIAS: (f32, f32) = (0.0, 0.0);

impl BookBias {
    pub fn new(start: f32, cont: f32) -> Option<Self> {
        if start <= 0.0 && cont <= 0.0 {
            return None;
        }
        let mut prefixes = std::collections::HashSet::new();
        for name in chorus_refparse::malayalam::book_spellings() {
            let mut p = String::new();
            for c in name.chars() {
                p.push(c);
                prefixes.insert(p.clone());
            }
        }
        Some(Self {
            prefixes,
            start,
            cont,
        })
    }

    /// `CHORUS_IT_BIAS="start,cont"` overrides the default.
    pub fn from_env_or_default() -> Option<Self> {
        let (start, cont) = std::env::var("CHORUS_IT_BIAS")
            .ok()
            .and_then(|v| {
                let (a, b) = v.split_once(',')?;
                Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
            })
            .unwrap_or(DEFAULT_BIAS);
        Self::new(start, cont)
    }
}

/// A loaded Indic-Transcribe model.
pub struct IndicTranscribe {
    encoder: Session,
    decoder: Session,
    pieces: Vec<String>,
    special_count: usize,
    stop: [i64; 2],
    prompt: Vec<i64>,
    bias: Option<BookBias>,
}

impl std::fmt::Debug for IndicTranscribe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndicTranscribe")
            .field("prompt", &self.prompt)
            .finish_non_exhaustive()
    }
}

impl IndicTranscribe {
    /// Load the model from `dir` for language `lang` ("ml").
    pub fn load(dir: &Path, lang: &str, n_threads: usize) -> Result<Self, String> {
        let session = |prefix: &str| -> Result<Session, String> {
            let path = find_onnx(dir, prefix)
                .ok_or_else(|| format!("no {prefix}*.onnx in {}", dir.display()))?;
            let fail = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
            Session::builder()
                .map_err(|e| fail(&e))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| fail(&e))?
                .with_intra_threads(n_threads.max(1))
                .map_err(|e| fail(&e))?
                .commit_from_file(&path)
                .map_err(|e| fail(&e))
        };
        let encoder = session("encoder")?;
        let decoder = session("decoder")?;
        let text = std::fs::read_to_string(dir.join("tokenizer.json"))
            .map_err(|e| format!("tokenizer.json: {e}"))?;
        let tok: TokenizerFile =
            serde_json::from_str(&text).map_err(|e| format!("tokenizer.json: {e}"))?;
        let prompt: Vec<i64> = match &tok.prompts {
            serde_json::Value::Object(m) => m.get(lang).cloned(),
            other => Some(other.clone()),
        }
        .and_then(|v| serde_json::from_value(v).ok())
        .ok_or_else(|| format!("tokenizer.json has no prompt for \"{lang}\""))?;
        Ok(Self {
            encoder,
            decoder,
            pieces: tok.pieces,
            special_count: tok.special_count,
            stop: [tok.eos_id, tok.pad_id],
            prompt,
            bias: BookBias::from_env_or_default(),
        })
    }

    /// Transcribe one phrase of 16 kHz mono audio.
    pub fn transcribe(&mut self, audio: &[f32]) -> Result<String, String> {
        self.transcribe_with(audio, &mut |_| {})
    }

    /// Like [`Self::transcribe`], calling `on_words` with the text so far
    /// each time a word is complete, so a reference can be shown before
    /// the rest of a long sentence is written.
    pub fn transcribe_with(&mut self, audio: &[f32], on_words: &mut dyn FnMut(&str)) -> Result<String, String> {
        let err = |e: ort::Error| e.to_string();
        // Under one second: centred in a second of silence, as trained.
        let mut wav = audio.to_vec();
        if wav.len() < 16_000 {
            let pad = 16_000 - wav.len();
            let mut padded = vec![0.0; pad / 2];
            padded.extend_from_slice(&wav);
            padded.resize(16_000, 0.0);
            wav = padded;
        }
        let n = wav.len();
        let enc_out = self
            .encoder
            .run(ort::inputs!["audio" => Tensor::from_array(([1usize, n], wav)).map_err(err)?])
            .map_err(err)?;
        let (k_shape, k) = enc_out["cross_k"]
            .try_extract_tensor::<f32>()
            .map_err(err)?;
        let (v_shape, v) = enc_out["cross_v"]
            .try_extract_tensor::<f32>()
            .map_err(err)?;
        let to_dims = |s: &ort::value::Shape| {
            s.iter()
                .map(|&d| usize::try_from(d).unwrap_or(0))
                .collect::<Vec<_>>()
        };
        // Made once, passed by reference to every decoder step.
        let cross_k = Tensor::from_array((to_dims(k_shape), k.to_vec())).map_err(err)?;
        let cross_v = Tensor::from_array((to_dims(v_shape), v.to_vec())).map_err(err)?;
        drop(enc_out);

        let step = LAYERS * HEADS * HEAD_DIM;
        let (mut past_k, mut past_v): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
        let mut ids = self.prompt.clone();
        let mut out: Vec<i64> = Vec::new();
        // The word being written, for the book-name bias.
        let mut word = String::new();
        for _ in 0..MAX_TOKENS {
            let past = past_k.len() / step;
            let shape = [past, LAYERS, HEADS, HEAD_DIM];
            let inputs = ort::inputs![
                "input_ids" => Tensor::from_array(([1usize, ids.len()], ids.clone())).map_err(err)?,
                "past_k" => TensorRef::from_array_view((shape, past_k.as_slice())).map_err(err)?,
                "past_v" => TensorRef::from_array_view((shape, past_v.as_slice())).map_err(err)?,
                "cross_k" => &cross_k,
                "cross_v" => &cross_v,
            ];
            let dec_out = self.decoder.run(inputs).map_err(err)?;
            let (_, logits) = dec_out["logits"].try_extract_tensor::<f32>().map_err(err)?;
            let next = pick(
                &self.pieces,
                self.special_count,
                self.bias.as_ref(),
                logits,
                &mut word,
            );
            let (_, nk) = dec_out["new_k"].try_extract_tensor::<f32>().map_err(err)?;
            let (_, nv) = dec_out["new_v"].try_extract_tensor::<f32>().map_err(err)?;
            past_k.extend_from_slice(nk);
            past_v.extend_from_slice(nv);
            if self.stop.contains(&next) {
                break;
            }
            let starts_word = usize::try_from(next)
                .ok()
                .filter(|&t| t >= self.special_count)
                .and_then(|t| self.pieces.get(t))
                .is_some_and(|p| p.starts_with('\u{2581}'));
            if starts_word && !out.is_empty() {
                on_words(&text_of(&self.pieces, self.special_count, &out));
            }
            out.push(next);
            ids = vec![next];
        }
        Ok(text_of(&self.pieces, self.special_count, &out))
    }

}

/// The text of decoded tokens.
fn text_of(pieces: &[String], special_count: usize, tokens: &[i64]) -> String {
    let text: String = tokens
        .iter()
        .filter_map(|&t| usize::try_from(t).ok())
        .filter(|&t| t >= special_count)
        .filter_map(|t| pieces.get(t).map(String::as_str))
        .collect();
    text.replace('\u{2581}', " ").trim().to_string()
}

/// The next token: the most likely one, or with the book-name bias the
/// best of the top candidates after bonuses. Tracks the current word.
fn pick(
    pieces: &[String],
    special_count: usize,
    bias: Option<&BookBias>,
    logits: &[f32],
    word: &mut String,
) -> i64 {
    let best = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i);
    let piece = |t: usize| {
        (t >= special_count)
            .then(|| pieces.get(t).map(String::as_str))
            .flatten()
    };
    let chosen = match bias {
        None => best,
        Some(bias) => {
            let mut top: Vec<usize> = (0..logits.len()).collect();
            let k = BIAS_TOP_K.min(top.len());
            top.select_nth_unstable_by(k - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
            top.truncate(k);
            let lse = {
                let m = logits[best];
                m + logits.iter().map(|&x| (x - m).exp()).sum::<f32>().ln()
            };
            top.into_iter()
                .map(|t| {
                    let bonus = match piece(t) {
                        Some(p) if p.starts_with('\u{2581}') => {
                            let w = &p['\u{2581}'.len_utf8()..];
                            if !w.is_empty() && bias.prefixes.contains(w) {
                                bias.start
                            } else {
                                0.0
                            }
                        }
                        Some(p) if !word.is_empty() => {
                            if bias.prefixes.contains(&format!("{word}{p}")) {
                                bias.cont
                            } else {
                                0.0
                            }
                        }
                        _ => 0.0,
                    };
                    (t, logits[t] - lse + bonus)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map_or(best, |(t, _)| t)
        }
    };
    match piece(chosen) {
        Some(p) if p.starts_with('\u{2581}') => *word = p['\u{2581}'.len_utf8()..].to_string(),
        Some(p) => word.push_str(p),
        None => word.clear(),
    }
    i64::try_from(chosen).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Live provider
// ---------------------------------------------------------------------------

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crossbeam_channel::Receiver;
use tokio::sync::mpsc;

use crate::error::SttError;
use crate::provider::SttProvider;
use crate::types::{TranscriptEvent, Word};
use crate::whisper::{i16_to_f32, level_volume, next_job, run_segmenter, Job, PartialTiming};

/// No live previews: each transcription takes about a second on a laptop
/// CPU, and a preview still running would hold up the final of the phrase.
/// References are read from finals (pauses inside a Malayalam reference are
/// joined by the detector). Quiet speech counts, as for `IndicConformer`.
const TIMING: PartialTiming = PartialTiming {
    partial_min: 16_000 * 3600,
    partial_every: 16_000 * 3600,
    voice_rms: 0.005,
};

/// Live Malayalam transcription with Indic-Transcribe.
pub struct IndicTranscribeProvider {
    model_dir: PathBuf,
    n_threads: usize,
    cancelled: Arc<AtomicBool>,
}

impl IndicTranscribeProvider {
    pub fn new(model_dir: PathBuf, n_threads: usize) -> Self {
        Self {
            model_dir,
            n_threads: n_threads.max(1),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl std::fmt::Debug for IndicTranscribeProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndicTranscribeProvider")
            .field("model_dir", &self.model_dir)
            .field("n_threads", &self.n_threads)
            .finish_non_exhaustive()
    }
}

/// Words with even timing over the clip.
#[expect(
    clippy::cast_precision_loss,
    reason = "clip lengths and word counts are small"
)]
fn words_of(text: &str, duration_s: f64) -> Vec<Word> {
    let n = text.split_whitespace().count();
    if n == 0 {
        return Vec::new();
    }
    let step = duration_s / n as f64;
    text.split_whitespace()
        .enumerate()
        .map(|(i, w)| Word {
            text: w
                .trim_matches(|c: char| {
                    !(c.is_alphanumeric() || ('\u{0D00}'..='\u{0D7F}').contains(&c))
                })
                .to_lowercase(),
            start: i as f64 * step,
            end: (i + 1) as f64 * step,
            confidence: 0.9,
            punctuated_word: Some(w.to_string()),
        })
        .collect()
}

#[async_trait::async_trait]
impl SttProvider for IndicTranscribeProvider {
    async fn start(
        &self,
        audio_rx: Receiver<Vec<i16>>,
        event_tx: mpsc::Sender<TranscriptEvent>,
    ) -> Result<(), SttError> {
        if !model_present(&self.model_dir) {
            return Err(SttError::ModelNotFound(format!(
                "Indic-Transcribe model not found in {} (run: bun run download:malayalam-it)",
                self.model_dir.display()
            )));
        }
        let _ = event_tx.send(TranscriptEvent::Connected).await;

        let model_dir = self.model_dir.clone();
        let n_threads = self.n_threads;
        let cancelled = self.cancelled.clone();
        let (inference_tx, mut inference_rx) = mpsc::channel::<Job>(4);

        let vad_cancelled = cancelled.clone();
        let vad_event_tx = event_tx.clone();
        let vad_handle = tokio::task::spawn_blocking(move || {
            run_segmenter(audio_rx, inference_tx, vad_event_tx, vad_cancelled, TIMING);
        });

        let inf_event_tx = event_tx.clone();
        let inf_handle = tokio::task::spawn_blocking(move || {
            let load_start = std::time::Instant::now();
            let mut model = match IndicTranscribe::load(&model_dir, "ml", n_threads) {
                Ok(m) => m,
                Err(e) => {
                    log::error!("[IndicTranscribe] Failed to load model: {e}");
                    let _ = inf_event_tx.blocking_send(TranscriptEvent::Error(format!(
                        "Failed to load the Indic-Transcribe model: {e}"
                    )));
                    return;
                }
            };
            log::info!(
                "[IndicTranscribe] Model loaded in {:.1?}, ready for inference (threads={n_threads})",
                load_start.elapsed()
            );

            let mut pending_job: Option<Job> = None;
            while let Some((audio_i16, is_partial)) = next_job(&mut inference_rx, &mut pending_job)
            {
                if cancelled.load(Ordering::SeqCst) {
                    break;
                }
                let mut audio = i16_to_f32(&audio_i16);
                level_volume(&mut audio);
                let start = std::time::Instant::now();
                #[expect(clippy::cast_precision_loss, reason = "audio sample count fits in f64")]
                let clip_s = audio_i16.len() as f64 / 16_000.0;
                // Words so far go out as previews, so a reference at the
                // start of a long sentence shows before the rest is written.
                let preview_tx = inf_event_tx.clone();
                let mut on_words = |so_far: &str| {
                    let so_far = crate::cleanup::remove_repeats(so_far);
                    if so_far.split_whitespace().count() >= 2 {
                        let words = words_of(&so_far, clip_s);
                        let _ = preview_tx.blocking_send(TranscriptEvent::Partial { transcript: so_far, words });
                    }
                };
                let text = match model.transcribe_with(&audio, &mut on_words) {
                    Ok(t) => crate::cleanup::remove_repeats(&t),
                    Err(e) => {
                        log::warn!("[IndicTranscribe] Transcription failed: {e}");
                        String::new()
                    }
                };
                #[expect(clippy::cast_precision_loss, reason = "audio sample count fits in f64")]
                let duration_s = audio_i16.len() as f64 / 16_000.0;
                log::info!(
                    "[IndicTranscribe] Transcribed {duration_s:.1}s audio in {:.1?} (partial={is_partial}): \"{text}\"",
                    start.elapsed()
                );
                let words = words_of(&text, duration_s);
                if is_partial {
                    if !text.is_empty() {
                        let _ = inf_event_tx.blocking_send(TranscriptEvent::Partial {
                            transcript: text,
                            words,
                        });
                    }
                    continue;
                }
                if !text.is_empty() {
                    let _ = inf_event_tx.blocking_send(TranscriptEvent::Final {
                        transcript: text,
                        words,
                        confidence: 0.9,
                        speech_final: true,
                    });
                }
                let _ = inf_event_tx.blocking_send(TranscriptEvent::UtteranceEnd);
            }
            log::info!("[IndicTranscribe] Inference task exiting");
        });

        let _ = tokio::join!(vad_handle, inf_handle);
        let _ = event_tx.send(TranscriptEvent::Disconnected).await;
        Ok(())
    }

    fn stop(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    fn name(&self) -> &'static str {
        "malayalam-it"
    }
}
