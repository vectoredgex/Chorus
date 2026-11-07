//! Transcribes each WAV clip once with the settings the app uses (sized
//! encoder window + Bible prompt) and prints one tab-separated line per clip:
//! `<clip>\t<seconds>\t<text>`. Used to compare speech engines.
//!
//! Usage:
//!   cargo run -p rhema-stt --features whisper --example whisper_lines -- \
//!       <model.bin> <threads> <clip1.wav> [clip2.wav ...]
//!
//! Clips must be 16 kHz, mono, 16-bit PCM WAV.

use std::time::Instant;

use rhema_stt::cleanup::remove_repeats;
use rhema_stt::whisper::{audio_ctx_for, BIBLE_PROMPT};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Minimal reader for 16-bit PCM WAV: returns samples as f32 in [-1, 1].
fn read_wav(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut i = 12;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let size = u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]) as usize;
        if id == b"data" {
            let end = (i + 8 + size).min(bytes.len());
            return bytes[i + 8..end]
                .chunks_exact(2)
                .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
                .collect();
        }
        i += 8 + size + (size & 1);
    }
    panic!("{path}: no data chunk");
}

fn transcribe(state: &mut whisper_rs::WhisperState, audio: &[f32], threads: i32, ctx: i32, prompt: bool) -> (String, f64) {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("en"));
    params.set_n_threads(threads);
    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_single_segment(false);
    params.set_token_timestamps(true);
    params.set_no_speech_thold(0.6);
    params.set_suppress_blank(true);
    params.set_suppress_nst(true);
    params.set_audio_ctx(ctx);
    if prompt {
        params.set_initial_prompt(BIBLE_PROMPT);
    }

    let t = Instant::now();
    state.full(params, audio).expect("whisper inference");
    let secs = t.elapsed().as_secs_f64();

    let mut text = String::new();
    for i in 0..state.full_n_segments() {
        if let Some(seg) = state.get_segment(i) {
            text.push_str(&seg.to_str_lossy().unwrap_or_default());
        }
    }
    (remove_repeats(text.trim()), secs)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(args.len() >= 4, "usage: whisper_lines <model> <threads> <clip.wav>...");
    let model = &args[1];
    let threads: i32 = args[2].parse().expect("threads");
    let ctx = WhisperContext::new_with_params(model, WhisperContextParameters::default())
        .expect("load model");
    let mut state = ctx.create_state().expect("state");
    let warm = vec![0.0f32; 16_000];
    let _ = transcribe(&mut state, &warm, threads, 512, false);
    for path in &args[3..] {
        let audio = read_wav(path);
        let (text, secs) = transcribe(&mut state, &audio, threads, audio_ctx_for(audio.len()), true);
        println!("{path}\t{secs:.3}\t{}", text.replace(['\t', '\n'], " "));
    }
}
