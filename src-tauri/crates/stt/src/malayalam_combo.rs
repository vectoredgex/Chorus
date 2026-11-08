//! Malayalam with two engines: fast and accurate.
//!
//! While someone speaks, `IndicConformer` (about 0.3 s per look) writes live
//! previews, so a reference can reach the screen before the pause. At the
//! pause, Indic-Transcribe (slower, far more accurate) writes the final
//! text, and its reading of the reference is the one that counts: the
//! detector re-reads the utterance from the final.
//!
//! The two run one after the other on one thread, never at the same time,
//! so each gets the whole CPU budget.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crossbeam_channel::Receiver;
use tokio::sync::mpsc;

use crate::error::SttError;
use crate::indic_transcribe::{keep_loaded, take_or_load, words_of};
use crate::parakeet::{load_recognizer_for, model_present_for, transcribe, SherpaModel, MALAYALAM_PARTIALS};
use crate::provider::SttProvider;
use crate::types::TranscriptEvent;
use crate::whisper::{i16_to_f32, level_volume, next_job, run_segmenter, Job};

/// Live Malayalam transcription: `IndicConformer` previews, Indic-Transcribe finals.
pub struct MalayalamComboProvider {
    fast_dir: PathBuf,
    accurate_dir: PathBuf,
    fast_threads: i32,
    accurate_threads: usize,
    cancelled: Arc<AtomicBool>,
}

impl MalayalamComboProvider {
    pub fn new(fast_dir: PathBuf, accurate_dir: PathBuf, fast_threads: i32, accurate_threads: usize) -> Self {
        Self {
            fast_dir,
            accurate_dir,
            fast_threads: fast_threads.max(1),
            accurate_threads: accurate_threads.max(1),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl std::fmt::Debug for MalayalamComboProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MalayalamComboProvider")
            .field("fast_dir", &self.fast_dir)
            .field("accurate_dir", &self.accurate_dir)
            .finish_non_exhaustive()
    }
}

#[async_trait::async_trait]
impl SttProvider for MalayalamComboProvider {
    async fn start(
        &self,
        audio_rx: Receiver<Vec<i16>>,
        event_tx: mpsc::Sender<TranscriptEvent>,
    ) -> Result<(), SttError> {
        if !model_present_for(SherpaModel::IndicConformerMalayalam, &self.fast_dir) {
            return Err(SttError::ModelNotFound(format!(
                "Malayalam model not found in {} (run: bun run download:malayalam)",
                self.fast_dir.display()
            )));
        }
        if !crate::indic_transcribe::model_present(&self.accurate_dir) {
            return Err(SttError::ModelNotFound(format!(
                "Indic-Transcribe model not found in {} (run: bun run download:malayalam-it)",
                self.accurate_dir.display()
            )));
        }
        let _ = event_tx.send(TranscriptEvent::Connected).await;

        let (fast_dir, accurate_dir) = (self.fast_dir.clone(), self.accurate_dir.clone());
        let (fast_threads, accurate_threads) = (self.fast_threads, self.accurate_threads);
        let cancelled = self.cancelled.clone();
        let (inference_tx, mut inference_rx) = mpsc::channel::<Job>(4);

        let vad_cancelled = cancelled.clone();
        let vad_event_tx = event_tx.clone();
        let vad_handle = tokio::task::spawn_blocking(move || {
            run_segmenter(audio_rx, inference_tx, vad_event_tx, vad_cancelled, MALAYALAM_PARTIALS);
        });

        let inf_event_tx = event_tx.clone();
        let inf_handle = tokio::task::spawn_blocking(move || {
            let load_start = std::time::Instant::now();
            // Both load at once.
            let (fast, accurate) = std::thread::scope(|s| {
                let acc = s.spawn(|| take_or_load(&accurate_dir, accurate_threads));
                let fast = load_recognizer_for(SherpaModel::IndicConformerMalayalam, &fast_dir, fast_threads);
                (fast, acc.join().unwrap_or_else(|_| Err("loading panicked".into())))
            });
            let (Some(fast), Ok(mut accurate)) = (fast, accurate) else {
                log::error!("[ML-Combo] Failed to load the Malayalam models");
                let _ = inf_event_tx
                    .blocking_send(TranscriptEvent::Error("Failed to load the Malayalam speech models".into()));
                return;
            };
            log::info!(
                "[ML-Combo] Models loaded in {:.1?} (fast threads={fast_threads}, accurate threads={accurate_threads})",
                load_start.elapsed()
            );

            let mut pending_job: Option<Job> = None;
            while let Some((audio_i16, is_partial)) = next_job(&mut inference_rx, &mut pending_job) {
                if cancelled.load(Ordering::SeqCst) {
                    break;
                }
                let mut audio = i16_to_f32(&audio_i16);
                level_volume(&mut audio);
                #[expect(clippy::cast_precision_loss, reason = "audio sample count fits in f64")]
                let duration_s = audio_i16.len() as f64 / 16_000.0;
                let start = std::time::Instant::now();

                if is_partial {
                    let text = transcribe(&fast, &audio);
                    log::info!(
                        "[ML-Combo] fast {duration_s:.1}s audio in {:.1?}: \"{text}\"",
                        start.elapsed()
                    );
                    if !text.is_empty() {
                        let words = words_of(&text, duration_s);
                        let _ = inf_event_tx.blocking_send(TranscriptEvent::Partial { transcript: text, words });
                    }
                    continue;
                }

                // Words of the accurate text go out as previews while it is
                // still being written.
                let preview_tx = inf_event_tx.clone();
                let mut on_words = |so_far: &str| {
                    let so_far = crate::cleanup::remove_repeats(so_far);
                    if so_far.split_whitespace().count() >= 2 {
                        let words = words_of(&so_far, duration_s);
                        let _ = preview_tx.blocking_send(TranscriptEvent::Partial { transcript: so_far, words });
                    }
                };
                let text = match accurate.transcribe_with(&audio, &mut on_words) {
                    Ok(t) => crate::cleanup::remove_repeats(&t),
                    Err(e) => {
                        // The fast engine stands in rather than losing the phrase.
                        log::warn!("[ML-Combo] accurate engine failed ({e}); using the fast one");
                        transcribe(&fast, &audio)
                    }
                };
                log::info!(
                    "[ML-Combo] accurate {duration_s:.1}s audio in {:.1?}: \"{text}\"",
                    start.elapsed()
                );
                if !text.is_empty() {
                    let words = words_of(&text, duration_s);
                    let _ = inf_event_tx.blocking_send(TranscriptEvent::Final {
                        transcript: text,
                        words,
                        confidence: 0.9,
                        speech_final: true,
                    });
                }
                let _ = inf_event_tx.blocking_send(TranscriptEvent::UtteranceEnd);
            }
            log::info!("[ML-Combo] Inference task exiting");
            keep_loaded(accurate_dir, accurate_threads, accurate);
        });

        let _ = tokio::join!(vad_handle, inf_handle);
        let _ = event_tx.send(TranscriptEvent::Disconnected).await;
        Ok(())
    }

    fn stop(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    fn name(&self) -> &'static str {
        "malayalam-combo"
    }
}
