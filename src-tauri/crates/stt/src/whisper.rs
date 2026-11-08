//! Local Whisper STT provider using whisper.cpp via whisper-rs.
//!
//! Processes audio through the app's VAD to detect speech boundaries,
//! then runs Whisper inference on each speech segment. Emits the same
//! [`TranscriptEvent`] types as the Deepgram provider.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::Receiver;
use tokio::sync::mpsc;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::error::SttError;
use crate::provider::SttProvider;
use crate::types::{TranscriptEvent, Word};

/// Maximum audio buffer before force-flushing to inference (10 seconds at 16 kHz).
const MAX_BUFFER_SAMPLES: usize = 16_000 * 10;

/// Minimum audio buffer for inference (1.0 seconds).
/// Whisper warns "input is too short" below 1s.
const MIN_BUFFER_SAMPLES: usize = 16_000;

/// While someone is speaking, transcribe what has been said so far after
/// this much new audio (1.5 seconds), instead of waiting for a pause.
/// Preachers often talk for many seconds without pausing, and a reference
/// should reach the screen while they are still mid-sentence.
const PARTIAL_EVERY_SAMPLES: usize = 16_000 * 3 / 2;

/// No live partial until the utterance is this long (3 seconds). Most
/// references ("Luke chapter five verse two") are said in under 3 s; a
/// partial on such a short phrase only delays its final transcript, because
/// the final has to wait for the partial to finish (measured on the owner's
/// laptop: ~6 s instead of ~1 s from end of speech to verse on screen).
const PARTIAL_MIN_SAMPLES: usize = 16_000 * 3;

/// When Whisper falls behind and several finished phrases are waiting, they
/// are joined and transcribed in one pass, up to this much audio (20 s).
/// One pass over 6 s costs far less than three passes over 2 s each.
const MERGE_MAX_SAMPLES: usize = 16_000 * 20;

/// Silence put between joined phrases (0.3 s) so words don't run together.
const MERGE_GAP_SAMPLES: usize = 16_000 * 3 / 10;

/// A speech piece shorter than `MIN_BUFFER_SAMPLES` (a single word such as
/// "John" before a short pause) is kept and joined to the next piece, unless
/// the next one starts more than this long after it.
const SHORT_PIECE_JOIN_WINDOW: Duration = Duration::from_millis(2_500);

/// Pieces shorter than this (0.5 s) are clicks and breaths, not words, and
/// are dropped as before.
const MIN_WORD_SAMPLES: usize = 16_000 / 2;

/// Whisper normally encodes a fixed 30 s window (1500 frames of 20 ms) no
/// matter how short the clip is. Encoding only as many frames as the clip
/// needs, plus a margin, cuts the work for a 2 s phrase several times over.
pub fn audio_ctx_for(samples: usize) -> i32 {
    const FRAMES_PER_SECOND: usize = 50;
    // Measured: a 256-frame minimum made Whisper invent words on real
    // voices through a laptop mic; 512 frames kept transcripts clean.
    const MARGIN_FRAMES: usize = 256;
    const MIN_FRAMES: usize = 512;
    const MAX_FRAMES: usize = 1500;
    let needed = samples * FRAMES_PER_SECOND / 16_000 + MARGIN_FRAMES;
    // At most 1500, so the cast cannot truncate.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "clamped to 1500"
    )]
    let frames = needed.clamp(MIN_FRAMES, MAX_FRAMES) as i32;
    frames
}

/// True when Whisper was built for a graphics card (`cuda` or `vulkan`
/// feature): the model runs on the GPU with flash attention.
pub const GPU_BUILD: bool = cfg!(any(feature = "cuda", feature = "vulkan"));

/// Encoder window for a clip.
pub fn encoder_window(samples: usize) -> i32 {
    // Measured on a GeForce MX550: the full window costs ~2.5 s per phrase
    // even on the GPU, and the sized window (512-frame minimum) was as
    // accurate in the benchmark, so the GPU uses it too.
    audio_ctx_for(samples)
}

/// Primes Whisper with how Bible references are written, so spoken book
/// names and chapter/verse numbers come out as references ("John 3:16")
/// instead of sound-alike words.
pub const BIBLE_PROMPT: &str =
    "Sermon with Bible references such as John 3:16, Romans chapter 8 verse 28, \
1 Corinthians 13:4, Psalm 23, Genesis 1:1, Philippians 4:13, Ephesians 2:8-9 and Revelation 21:4.";

/// What the sermon is currently on ("Romans 8:28"), set by the app after
/// each detected reference. Added to the prompt so Whisper expects that
/// book and chapter: a follow-up "verse twenty nine" is then heard as a
/// verse number, not a sound-alike phrase.
static CONTEXT_HINT: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

/// Tell Whisper which passage the sermon is on.
pub fn set_context_hint(reference: &str) {
    if let Ok(mut h) = CONTEXT_HINT.lock() {
        h.clear();
        h.push_str(reference);
    }
}

/// The prompt for the next transcription: the fixed Bible hint, the
/// passage the sermon is currently on, and the end of what was just said.
/// Whisper reads the prompt as the text that came before the audio, so the
/// previous sentence helps it continue a reference split across a pause
/// ("Let us turn to" ... "John chapter 9").
pub fn current_prompt(previous: &str) -> String {
    let hint = CONTEXT_HINT.lock().map(|h| h.clone()).unwrap_or_default();
    let mut prompt = BIBLE_PROMPT.to_string();
    if !hint.is_empty() {
        prompt.push_str(" Now reading ");
        prompt.push_str(&hint);
        prompt.push('.');
    }
    let tail = last_words(previous, PREVIOUS_WORDS);
    if !tail.is_empty() {
        prompt.push(' ');
        prompt.push_str(&tail);
    }
    prompt
}

/// How many words of the previous transcript go into the prompt. Kept short:
/// a long carried-over context is what makes Whisper start repeating itself.
const PREVIOUS_WORDS: usize = 20;

/// Previous text older than this is not carried over (a new thought).
const PREVIOUS_TEXT_MAX_AGE: Duration = Duration::from_secs(20);

fn last_words(text: &str, n: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    words[words.len().saturating_sub(n)..].join(" ")
}

/// Audio is re-cut into frames of this many samples (20 ms at 16 kHz)
/// before speech detection. Microphones deliver differently sized chunks
/// (about 10 ms on Windows), and the detector counts frames, so fixed-size
/// frames make its pause and onset settings mean the same time on every PC.
const VAD_FRAME_SAMPLES: usize = 320;

/// Speech-detector settings in time, converted to 20 ms frames.
/// A phrase ends after 0.7 s of silence: long enough not to cut at a
/// breath between words, short enough to send a reference promptly.
const VAD_SILENCE_END_FRAMES: usize = 700 / 20;
/// Voice must last 120 ms to count as speech (ignores clicks).
const VAD_MIN_VOICE_FRAMES: usize = 120 / 20;
/// Keep 300 ms of audio from before speech starts (catches the first word).
const VAD_PRE_BUFFER_FRAMES: usize = 300 / 20;
/// The detector's own force-flush limit, above our 10 s buffer cap.
const VAD_MAX_UTTERANCE_FRAMES: usize = 15_000 / 20;

/// Audio sent to the inference task.
/// Joins a finished phrase with finished phrases already waiting behind it,
/// so a backlog is cleared in one Whisper pass instead of one pass each.
/// Waiting partials are stale and dropped. The first batch stops before it
/// would pass `MERGE_MAX_SAMPLES`; the phrases that did not fit are joined
/// into a second batch (keeping only its newest `MERGE_MAX_SAMPLES` if
/// Whisper is that far behind) which is transcribed next.
fn merge_queued_finals(first: Vec<i16>, queued: Vec<Job>) -> (Vec<i16>, Option<Job>) {
    fn join(into: &mut Vec<i16>, audio: &[i16]) {
        if !into.is_empty() {
            into.resize(into.len() + MERGE_GAP_SAMPLES, 0);
        }
        into.extend_from_slice(audio);
    }
    let mut merged = first;
    let mut rest: Vec<i16> = Vec::new();
    for job in queued {
        let Job::Final(audio) = job else { continue };
        if rest.is_empty() && merged.len() + MERGE_GAP_SAMPLES + audio.len() <= MERGE_MAX_SAMPLES {
            join(&mut merged, &audio);
        } else {
            join(&mut rest, &audio);
        }
    }
    if rest.len() > MERGE_MAX_SAMPLES {
        rest.drain(..rest.len() - MERGE_MAX_SAMPLES);
    }
    (merged, (!rest.is_empty()).then_some(Job::Final(rest)))
}

pub(crate) enum Job {
    /// Speech so far in an ongoing utterance. Skipped if a newer job is
    /// already waiting, so a slow CPU never falls behind.
    Partial(Vec<i16>),
    /// A finished utterance. Never skipped.
    Final(Vec<i16>),
}

/// Even out the volume of a phrase before Whisper hears it: a quiet voice
/// (pastor turned away, laptop mic across the room) is raised so its loud
/// parts reach about half of full scale. Gain is capped at 8x so silence and
/// hiss are not blown up, and loud audio is never turned down.
pub(crate) fn level_volume(samples: &mut [f32]) {
    const TARGET_PEAK: f32 = 0.5;
    const MAX_GAIN: f32 = 8.0;
    if samples.is_empty() {
        return;
    }
    // 99.5th-percentile level, so a single click doesn't decide the gain.
    let mut mags: Vec<f32> = samples.iter().map(|s| s.abs()).collect();
    let idx = (mags.len() - 1) * 995 / 1000;
    let (_, peak, _) = mags.select_nth_unstable_by(idx, f32::total_cmp);
    let peak = *peak;
    if peak <= 1e-4 || peak >= TARGET_PEAK {
        return;
    }
    let gain = (TARGET_PEAK / peak).min(MAX_GAIN);
    for s in samples.iter_mut() {
        *s = (*s * gain).clamp(-1.0, 1.0);
    }
}

/// Convert i16 PCM samples to f32 in [-1.0, 1.0] range.
pub(crate) fn i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|&s| f32::from(s) / 32768.0).collect()
}

/// Extract transcript text, words, and average confidence from Whisper state.
#[expect(
    clippy::cast_precision_loss,
    reason = "timestamps and word counts are small enough"
)]
fn extract_segments(state: &WhisperState) -> (String, Vec<Word>, f64) {
    let n_segments = state.full_n_segments();
    let mut full_text = String::new();
    let mut words = Vec::new();
    let mut total_confidence: f64 = 0.0;

    for i in 0..n_segments {
        let Some(segment) = state.get_segment(i) else {
            continue;
        };
        let text = segment.to_str_lossy().unwrap_or_default().to_string();
        let start_ts = segment.start_timestamp();
        let end_ts = segment.end_timestamp();

        let start_sec = start_ts as f64 / 100.0;
        let end_sec = end_ts as f64 / 100.0;
        let confidence = 0.9;

        let n_words = text.split_whitespace().count();
        if n_words > 0 {
            let duration_per_word = (end_sec - start_sec) / n_words as f64;
            for (j, word_text) in text.split_whitespace().enumerate() {
                let w_start = start_sec + (j as f64 * duration_per_word);
                let w_end = w_start + duration_per_word;
                words.push(Word {
                    text: word_text.to_lowercase(),
                    start: w_start,
                    end: w_end,
                    confidence,
                    punctuated_word: Some(word_text.to_string()),
                });
            }
        }

        full_text.push_str(&text);
        total_confidence += confidence;
    }

    let avg_confidence = if n_segments > 0 {
        total_confidence / f64::from(n_segments)
    } else {
        0.0
    };

    (full_text.trim().to_string(), words, avg_confidence)
}

/// When live partial transcripts are made while someone is speaking.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PartialTiming {
    /// No partial until the utterance is at least this long (samples).
    pub partial_min: usize,
    /// Then one partial per this much new audio (samples).
    pub partial_every: usize,
    /// Quietest 20 ms frame (RMS) that counts as speech. Whisper needs it
    /// high (it invents words from near-silence); the CTC engines don't.
    pub voice_rms: f32,
}

/// Whisper's timing: see `PARTIAL_MIN_SAMPLES` and `PARTIAL_EVERY_SAMPLES`.
pub(crate) const WHISPER_PARTIALS: PartialTiming = PartialTiming {
    partial_min: PARTIAL_MIN_SAMPLES,
    partial_every: PARTIAL_EVERY_SAMPLES,
    voice_rms: 0.01,
};

/// Speech detection and phrase cutting, shared by every local engine: reads
/// microphone audio, finds where phrases start and end, and queues
/// [`Job`]s (finished phrases, and partials while speaking) for the engine.
pub(crate) fn run_segmenter(
    audio_rx: Receiver<Vec<i16>>,
    inference_tx: mpsc::Sender<Job>,
    event_tx: mpsc::Sender<TranscriptEvent>,
    cancelled: Arc<AtomicBool>,
    cfg: PartialTiming,
) {
    use rhema_audio::{AudioFrame, Vad, VadConfig, VadTransition};

    // Higher thresholds than default to avoid sending near-silence
    // to Whisper (which causes hallucinations).
    let vad_config = VadConfig {
        silence_threshold: cfg.voice_rms,
        frame_threshold: cfg.voice_rms / 2.0,
        min_voice_frames: VAD_MIN_VOICE_FRAMES,
        silence_frame_count: VAD_SILENCE_END_FRAMES,
        pre_buffer_frames: VAD_PRE_BUFFER_FRAMES,
        max_utterance_frames: VAD_MAX_UTTERANCE_FRAMES,
        ..VadConfig::default()
    };
    // Microphone samples waiting to fill a 20 ms frame.
    let mut pending: Vec<i16> = Vec::with_capacity(VAD_FRAME_SAMPLES * 4);
    let mut vad = Vad::new(vad_config);
    let mut audio_buffer: Vec<i16> = Vec::new();
    let mut frames_seen: u64 = 0;
    // Buffer length when the last partial was sent.
    let mut partial_mark: usize = 0;
    // When the last too-short piece ended (it is still in the buffer).
    let mut short_piece_at: Option<std::time::Instant> = None;

    loop {
        if cancelled.load(Ordering::SeqCst) {
            if audio_buffer.len() >= MIN_BUFFER_SAMPLES {
                let _ = inference_tx.blocking_send(Job::Final(std::mem::take(&mut audio_buffer)));
            }
            break;
        }

        match audio_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(samples) => {
                pending.extend_from_slice(&samples);
                while pending.len() >= VAD_FRAME_SAMPLES {
                    let chunk: Vec<i16> = pending.drain(..VAD_FRAME_SAMPLES).collect();
                    let frame = AudioFrame {
                        samples: chunk,
                        timestamp_ms: 0,
                    };
                    let result = vad.process(&frame);

                    if let Some(transition) = result.transition {
                        match transition {
                            VadTransition::SpeechStarted => {
                                // Drop a leftover short piece that is too old
                                // to belong to this utterance.
                                if short_piece_at
                                    .is_some_and(|t| t.elapsed() > SHORT_PIECE_JOIN_WINDOW)
                                {
                                    audio_buffer.clear();
                                    partial_mark = 0;
                                }
                                short_piece_at = None;
                                let _ = event_tx.blocking_send(TranscriptEvent::SpeechStarted);
                            }
                            VadTransition::SpeechEnded => {
                                log::info!(
                                "[WHISPER] flush on SpeechEnded: audio_buffer={} samples ({:.1}s)",
                                audio_buffer.len(),
                                audio_buffer.len() as f64 / 16_000.0,
                            );
                                if audio_buffer.len() >= MIN_BUFFER_SAMPLES {
                                    let _ = inference_tx.blocking_send(Job::Final(std::mem::take(
                                        &mut audio_buffer,
                                    )));
                                    partial_mark = 0;
                                } else if audio_buffer.len() >= MIN_WORD_SAMPLES {
                                    // Too short alone ("John"); keep it so it
                                    // is transcribed with what follows.
                                    short_piece_at = Some(std::time::Instant::now());
                                } else if short_piece_at.is_none() {
                                    // A click or breath on its own: drop it.
                                    audio_buffer.clear();
                                    partial_mark = 0;
                                }
                            }
                        }
                    }

                    for frame in result.frames {
                        audio_buffer.extend_from_slice(&frame.samples);
                        frames_seen += 1;
                    }

                    if frames_seen.is_multiple_of(50) && !audio_buffer.is_empty() {
                        log::debug!(
                            "[WHISPER] audio_buffer={} samples ({:.1}s) frames_seen={frames_seen}",
                            audio_buffer.len(),
                            audio_buffer.len() as f64 / 16_000.0,
                        );
                    }

                    if audio_buffer.len() >= MAX_BUFFER_SAMPLES {
                        log::warn!(
                        "[WHISPER] flush on MAX_BUFFER ({} samples / {:.1}s) — VAD never closed",
                        audio_buffer.len(),
                        audio_buffer.len() as f64 / 16_000.0,
                    );
                        let _ = inference_tx
                            .blocking_send(Job::Final(std::mem::take(&mut audio_buffer)));
                        partial_mark = 0;
                    } else if audio_buffer.len() >= cfg.partial_min
                        && audio_buffer.len() >= partial_mark + cfg.partial_every
                    {
                        // Never block the audio loop for a partial; if the
                        // inference queue is full, the next one will do.
                        partial_mark = audio_buffer.len();
                        let _ = inference_tx.try_send(Job::Partial(audio_buffer.clone()));
                    }
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                if audio_buffer.len() >= MIN_BUFFER_SAMPLES {
                    let _ =
                        inference_tx.blocking_send(Job::Final(std::mem::take(&mut audio_buffer)));
                }
                break;
            }
        }
    }
}

/// Takes the next piece of audio to transcribe: a stale partial is skipped
/// when newer audio is waiting, and finished phrases waiting in line are
/// joined into one (see [`merge_queued_finals`]). Returns the audio and
/// whether it is a partial; `None` when the segmenter has stopped.
pub(crate) fn next_job(
    inference_rx: &mut mpsc::Receiver<Job>,
    pending_job: &mut Option<Job>,
) -> Option<(Vec<i16>, bool)> {
    let mut job = match pending_job.take() {
        Some(j) => j,
        None => inference_rx.blocking_recv()?,
    };
    // A partial is stale once newer audio is queued: skip ahead.
    while matches!(job, Job::Partial(_)) {
        match inference_rx.try_recv() {
            Ok(next) => job = next,
            Err(_) => break,
        }
    }
    Some(match job {
        Job::Partial(a) => (a, true),
        Job::Final(a) => {
            let mut queued = Vec::new();
            while let Ok(next) = inference_rx.try_recv() {
                queued.push(next);
            }
            let (merged, left) = merge_queued_finals(a, queued);
            // A phrase too long to join goes back to the front of the line.
            *pending_job = left;
            (merged, false)
        }
    })
}

/// Local Whisper STT provider.
pub struct WhisperProvider {
    model_path: PathBuf,
    language: Option<String>,
    n_threads: i32,
    cancelled: Arc<AtomicBool>,
}

impl WhisperProvider {
    /// Create a new Whisper provider.
    ///
    /// - `model_path`: path to a GGML model file
    /// - `language`: ISO language code (e.g. "en") or `None` for auto-detect
    /// - `n_threads`: number of CPU threads for inference
    pub fn new(model_path: PathBuf, language: Option<String>, n_threads: i32) -> Self {
        Self {
            model_path,
            language,
            n_threads: n_threads.max(1),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl std::fmt::Debug for WhisperProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhisperProvider")
            .field("model_path", &self.model_path)
            .field("language", &self.language)
            .field("n_threads", &self.n_threads)
            .finish_non_exhaustive()
    }
}

#[async_trait::async_trait]
impl SttProvider for WhisperProvider {
    #[expect(
        clippy::too_many_lines,
        reason = "spawns two blocking tasks with setup; splitting would obscure the pipeline flow"
    )]
    async fn start(
        &self,
        audio_rx: Receiver<Vec<i16>>,
        event_tx: mpsc::Sender<TranscriptEvent>,
    ) -> Result<(), SttError> {
        if !self.model_path.exists() {
            return Err(SttError::ModelNotFound(format!(
                "Whisper model not found: {}",
                self.model_path.display()
            )));
        }

        let _ = event_tx.send(TranscriptEvent::Connected).await;

        let model_path = self.model_path.clone();
        let language = self.language.clone();
        let n_threads = self.n_threads;
        let cancelled = self.cancelled.clone();

        let (inference_tx, mut inference_rx) = mpsc::channel::<Job>(4);

        // ── Task 1: VAD + audio accumulation ─────────────────────────────
        let vad_cancelled = cancelled.clone();
        let vad_event_tx = event_tx.clone();
        let vad_handle = tokio::task::spawn_blocking(move || {
            run_segmenter(
                audio_rx,
                inference_tx,
                vad_event_tx,
                vad_cancelled,
                WHISPER_PARTIALS,
            );
        });

        // ── Task 2: Whisper inference ────────────────────────────────────
        let inf_cancelled = cancelled.clone();
        let inf_event_tx = event_tx.clone();
        let inf_handle = tokio::task::spawn_blocking(move || {
            let mut ctx_params = WhisperContextParameters::default();
            ctx_params.use_gpu(GPU_BUILD);
            // Flash attention speeds up the GPU path; it is not used on CPU.
            ctx_params.flash_attn(GPU_BUILD);
            log::info!("[Whisper] GPU build: {GPU_BUILD}");
            let ctx = match WhisperContext::new_with_params(&model_path, ctx_params) {
                Ok(ctx) => ctx,
                Err(e) => {
                    log::error!("[Whisper] Failed to load model: {e}");
                    let _ = inf_event_tx.blocking_send(TranscriptEvent::Error(format!(
                        "Failed to load Whisper model: {e}"
                    )));
                    return;
                }
            };

            let mut state = match ctx.create_state() {
                Ok(s) => s,
                Err(e) => {
                    log::error!("[Whisper] Failed to create state: {e}");
                    let _ = inf_event_tx.blocking_send(TranscriptEvent::Error(format!(
                        "Failed to create Whisper state: {e}"
                    )));
                    return;
                }
            };

            log::info!("[Whisper] Model loaded, ready for inference");

            // The last final transcript, carried into the next prompt.
            let mut previous_text = String::new();
            let mut previous_at: Option<std::time::Instant> = None;

            // A finished phrase taken off the queue but not yet transcribed.
            let mut pending_job: Option<Job> = None;

            while let Some((audio_i16, is_partial)) = next_job(&mut inference_rx, &mut pending_job)
            {
                if inf_cancelled.load(Ordering::SeqCst) {
                    break;
                }

                let mut audio_f32 = i16_to_f32(&audio_i16);
                level_volume(&mut audio_f32);

                let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
                params.set_language(Some(language.as_deref().unwrap_or("en")));
                params.set_n_threads(n_threads);
                params.set_print_progress(false);
                params.set_print_special(false);
                params.set_print_realtime(false);
                params.set_single_segment(false);
                // Partials only need the words, not timings.
                params.set_token_timestamps(!is_partial);
                params.set_audio_ctx(encoder_window(audio_i16.len()));
                let fresh = previous_at.is_some_and(|t| t.elapsed() < PREVIOUS_TEXT_MAX_AGE);
                let prompt = current_prompt(if fresh { &previous_text } else { "" });
                params.set_initial_prompt(&prompt);
                params.set_no_speech_thold(0.6);
                params.set_suppress_blank(true);
                params.set_suppress_nst(true);

                let start = std::time::Instant::now();
                if let Err(e) = state.full(params, &audio_f32) {
                    log::error!("[Whisper] Inference error: {e}");
                    let _ = inf_event_tx.blocking_send(TranscriptEvent::Error(format!(
                        "Whisper inference error: {e}"
                    )));
                    continue;
                }
                let elapsed = start.elapsed();

                let (text, words, confidence) = extract_segments(&state);
                // Whisper sometimes loops ("Romans 8:28, Romans 8:28, ...").
                let text = crate::cleanup::remove_repeats(&text);

                #[expect(clippy::cast_precision_loss, reason = "audio sample count fits in f64")]
                let audio_duration_s = audio_i16.len() as f64 / 16_000.0;
                log::info!(
                    "[Whisper] Transcribed {audio_duration_s:.1}s audio in {elapsed:.1?} (partial={is_partial}, audio_ctx={}): \"{text}\"",
                    encoder_window(audio_i16.len())
                );

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
                    previous_text.clone_from(&text);
                    previous_at = Some(std::time::Instant::now());
                    let _ = inf_event_tx.blocking_send(TranscriptEvent::Final {
                        transcript: text,
                        words,
                        confidence,
                        speech_final: true,
                    });
                }

                let _ = inf_event_tx.blocking_send(TranscriptEvent::UtteranceEnd);
            }

            log::info!("[Whisper] Inference task exiting");
        });

        let _ = tokio::join!(vad_handle, inf_handle);
        let _ = event_tx.send(TranscriptEvent::Disconnected).await;

        Ok(())
    }

    fn stop(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    fn name(&self) -> &'static str {
        "whisper"
    }
}

#[cfg(test)]
mod chorus_tests {
    use super::{audio_ctx_for, current_prompt, set_context_hint, BIBLE_PROMPT};
    use super::{merge_queued_finals, Job, MERGE_GAP_SAMPLES, MERGE_MAX_SAMPLES};

    #[test]
    fn queued_finals_are_joined_and_partials_dropped() {
        let (merged, rest) = merge_queued_finals(
            vec![1; 100],
            vec![
                Job::Partial(vec![9; 50]),
                Job::Final(vec![2; 200]),
                Job::Final(vec![3; 300]),
            ],
        );
        assert_eq!(
            merged.len(),
            100 + MERGE_GAP_SAMPLES + 200 + MERGE_GAP_SAMPLES + 300
        );
        assert!(!merged.contains(&9));
        assert_eq!(merged[0], 1);
        assert_eq!(*merged.last().unwrap(), 3);
        assert!(rest.is_none());
    }

    #[test]
    fn backlog_over_the_limit_goes_to_a_second_batch() {
        let big = MERGE_MAX_SAMPLES - 1000;
        let (merged, rest) = merge_queued_finals(
            vec![1; big],
            vec![Job::Final(vec![2; 5000]), Job::Final(vec![3; 100])],
        );
        assert_eq!(merged.len(), big);
        let Some(Job::Final(rest)) = rest else {
            panic!("expected a second batch")
        };
        assert_eq!(rest.len(), 5000 + MERGE_GAP_SAMPLES + 100);
        assert_eq!(rest[0], 2);
    }

    #[test]
    fn quiet_audio_is_raised_loud_audio_untouched() {
        let mut quiet: Vec<f32> = (0..1000)
            .map(|i| if i % 2 == 0 { 0.05 } else { -0.05 })
            .collect();
        super::level_volume(&mut quiet);
        assert!((quiet[0] - 0.4).abs() < 1e-6, "capped at 8x: {}", quiet[0]);
        let mut loud: Vec<f32> = (0..1000)
            .map(|i| if i % 2 == 0 { 0.7 } else { -0.7 })
            .collect();
        super::level_volume(&mut loud);
        assert!((loud[0] - 0.7).abs() < 1e-6);
        let mut medium: Vec<f32> = (0..1000)
            .map(|i| if i % 2 == 0 { 0.25 } else { -0.25 })
            .collect();
        super::level_volume(&mut medium);
        assert!((medium[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn prompt_includes_current_passage() {
        set_context_hint("");
        assert_eq!(current_prompt(""), BIBLE_PROMPT);
        set_context_hint("Romans 8:28");
        assert!(current_prompt("").ends_with("Now reading Romans 8:28."));
        assert!(current_prompt("so let us turn to").ends_with("Romans 8:28. so let us turn to"));
        set_context_hint("");
    }

    #[test]
    fn audio_ctx_scales_with_clip_length() {
        assert_eq!(audio_ctx_for(16_000), 512); // 1 s: minimum window
        assert_eq!(audio_ctx_for(16_000 * 5), 512); // 5 s
        assert_eq!(audio_ctx_for(16_000 * 10), 756); // 10 s
        assert_eq!(audio_ctx_for(16_000 * 60), 1500); // capped at 30 s
    }
}
