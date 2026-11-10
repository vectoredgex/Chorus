//! Transcribes each WAV clip with the official IndicConformer (Malayalam,
//! the combo's quick engine) exactly as the app does and prints one tab-separated line per clip:
//! `<clip>\t<seconds>\t<text>` (the same format as `parakeet_lines`).
//!
//! Usage:
//!   cargo run -p rhema-stt --features indic-transcribe --example ic_official_lines -- \
//!       <model_dir> <threads> <clip1.wav> [clip2.wav ...]

use std::path::Path;
use std::time::Instant;

/// Minimal reader for 16-bit PCM WAV: returns samples as f32 in [-1, 1].
fn read_wav(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut i = 12;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let size =
            u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]) as usize;
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() >= 4,
        "usage: ic_official_lines <model_dir> <threads> <clip.wav>..."
    );
    let threads: usize = args[2].parse().expect("threads");
    let t = Instant::now();
    let mut model =
        rhema_stt::ic_official::IcOfficial::load(Path::new(&args[1]), threads)
            .expect("load model");
    eprintln!("loaded in {:.1} s", t.elapsed().as_secs_f64());
    let _ = model.transcribe(&vec![0.0f32; 16_000]);
    for path in &args[3..] {
        let audio = read_wav(path);
        let t = Instant::now();
        let text = model
            .transcribe(&audio)
            .unwrap_or_else(|e| format!("ERROR {e}"));
        println!(
            "{path}\t{:.3}\t{}",
            t.elapsed().as_secs_f64(),
            text.replace(['\t', '\n'], " ")
        );
    }
}
