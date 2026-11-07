//! Local NVIDIA Parakeet STT provider (Parakeet TDT 0.6B v2, English) run
//! through sherpa-onnx on the CPU.
//!
//! Measured on the Chorus test sermon (4 versions, 56 verses): 55 verses
//! right and none wrong, against 51 right and 1 wrong for Whisper
//! large-v3-turbo, at about 0.3 s per sentence instead of 4-7 s on the same
//! CPU. It needs no graphics card.
//!
//! Speech detection and phrase cutting are shared with the Whisper provider
//! ([`crate::whisper::run_segmenter`]); only the recogniser differs.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crossbeam_channel::Receiver;
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig};
use tokio::sync::mpsc;

use crate::error::SttError;
use crate::provider::SttProvider;
use crate::types::{TranscriptEvent, Word};
use crate::whisper::{i16_to_f32, level_volume, next_job, run_segmenter, Job, PartialTiming};

/// Folder name of the model inside `models/parakeet/`.
pub const MODEL_DIR_NAME: &str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8";

/// Files the model folder must contain.
pub const MODEL_FILES: [&str; 4] = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"];

/// Parakeet transcribes a phrase in a fraction of a second, so live partials
/// can start after 1 s of speech and repeat every 0.6 s: a reference reaches
/// the screen while the preacher is still talking.
const PARAKEET_PARTIALS: PartialTiming = PartialTiming {
    partial_min: 16_000,
    partial_every: 16_000 * 6 / 10,
};

/// True when `dir` holds every model file.
pub fn model_present(dir: &Path) -> bool {
    MODEL_FILES.iter().all(|f| dir.join(f).is_file())
}

/// Local Parakeet STT provider.
pub struct ParakeetProvider {
    model_dir: PathBuf,
    n_threads: i32,
    cancelled: Arc<AtomicBool>,
}

impl ParakeetProvider {
    /// - `model_dir`: folder with the four model files ([`MODEL_FILES`])
    /// - `n_threads`: CPU threads for inference
    pub fn new(model_dir: PathBuf, n_threads: i32) -> Self {
        Self { model_dir, n_threads: n_threads.max(1), cancelled: Arc::new(AtomicBool::new(false)) }
    }
}

impl std::fmt::Debug for ParakeetProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParakeetProvider")
            .field("model_dir", &self.model_dir)
            .field("n_threads", &self.n_threads)
            .finish_non_exhaustive()
    }
}

fn path_string(dir: &Path, file: &str) -> Option<String> {
    Some(dir.join(file).to_string_lossy().into_owned())
}

/// Build the recogniser. `None` if the model could not be loaded.
pub fn load_recognizer(model_dir: &Path, n_threads: i32) -> Option<OfflineRecognizer> {
    let mut config = OfflineRecognizerConfig::default();
    config.model_config.transducer = OfflineTransducerModelConfig {
        encoder: path_string(model_dir, MODEL_FILES[0]),
        decoder: path_string(model_dir, MODEL_FILES[1]),
        joiner: path_string(model_dir, MODEL_FILES[2]),
    };
    config.model_config.tokens = path_string(model_dir, MODEL_FILES[3]);
    config.model_config.model_type = Some("nemo_transducer".into());
    config.model_config.num_threads = n_threads;
    config.model_config.provider = Some("cpu".into());
    config.decoding_method = Some("greedy_search".into());
    OfflineRecognizer::create(&config)
}

/// Transcribe one clip of 16 kHz mono audio. Returns the cleaned text.
pub fn transcribe(recognizer: &OfflineRecognizer, audio: &[f32]) -> String {
    let stream = recognizer.create_stream();
    stream.accept_waveform(16_000, audio);
    recognizer.decode(&stream);
    let text = stream.get_result().map(|r| r.text).unwrap_or_default();
    crate::cleanup::remove_repeats(text.trim())
}

/// Words with even timing over the clip; Parakeet's text is already
/// punctuated and capitalised.
#[expect(clippy::cast_precision_loss, reason = "clip lengths and word counts are small")]
fn words_of(text: &str, duration_s: f64) -> Vec<Word> {
    let n = text.split_whitespace().count();
    if n == 0 {
        return Vec::new();
    }
    let step = duration_s / n as f64;
    text.split_whitespace()
        .enumerate()
        .map(|(i, w)| Word {
            text: w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase(),
            start: i as f64 * step,
            end: (i + 1) as f64 * step,
            confidence: 0.9,
            punctuated_word: Some(w.to_string()),
        })
        .collect()
}

#[async_trait::async_trait]
impl SttProvider for ParakeetProvider {
    async fn start(
        &self,
        audio_rx: Receiver<Vec<i16>>,
        event_tx: mpsc::Sender<TranscriptEvent>,
    ) -> Result<(), SttError> {
        if !model_present(&self.model_dir) {
            return Err(SttError::ModelNotFound(format!(
                "Parakeet model not found in {} (run: bun run download:parakeet)",
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
            run_segmenter(audio_rx, inference_tx, vad_event_tx, vad_cancelled, PARAKEET_PARTIALS);
        });

        let inf_event_tx = event_tx.clone();
        let inf_handle = tokio::task::spawn_blocking(move || {
            let Some(recognizer) = load_recognizer(&model_dir, n_threads) else {
                log::error!("[Parakeet] Failed to load model from {}", model_dir.display());
                let _ = inf_event_tx.blocking_send(TranscriptEvent::Error(
                    "Failed to load the Parakeet speech model".into(),
                ));
                return;
            };
            log::info!("[Parakeet] Model loaded, ready for inference (threads={n_threads})");

            let mut pending_job: Option<Job> = None;
            while let Some((audio_i16, is_partial)) = next_job(&mut inference_rx, &mut pending_job) {
                if cancelled.load(Ordering::SeqCst) {
                    break;
                }
                let mut audio = i16_to_f32(&audio_i16);
                level_volume(&mut audio);

                let start = std::time::Instant::now();
                let text = transcribe(&recognizer, &audio);
                #[expect(clippy::cast_precision_loss, reason = "audio sample count fits in f64")]
                let duration_s = audio_i16.len() as f64 / 16_000.0;
                log::info!(
                    "[Parakeet] Transcribed {duration_s:.1}s audio in {:.1?} (partial={is_partial}): \"{text}\"",
                    start.elapsed()
                );
                let words = words_of(&text, duration_s);

                if is_partial {
                    if !text.is_empty() {
                        let _ = inf_event_tx.blocking_send(TranscriptEvent::Partial { transcript: text, words });
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
            log::info!("[Parakeet] Inference task exiting");
        });

        let _ = tokio::join!(vad_handle, inf_handle);
        let _ = event_tx.send(TranscriptEvent::Disconnected).await;
        Ok(())
    }

    fn stop(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    fn name(&self) -> &'static str {
        "parakeet"
    }
}
