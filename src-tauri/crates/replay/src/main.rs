//! Chorus replay test.
//!
//! Plays a WAV recording into the real speech pipeline (speech detection,
//! Whisper with partials, prompts and clean-up) at real-time speed, runs every
//! transcript through the real detector exactly as the app does (partial
//! confirmation, passage hint), and compares the verses found with the
//! expected list.
//!
//! Usage:
//!   chorus-replay <model.bin | parakeet_model_dir> <threads> <recording.wav> <expected.txt>
//!
//! A Whisper model file runs Whisper; a Parakeet model folder runs Parakeet.
//!
//! The recording must be 16 kHz, mono, 16-bit PCM WAV. `expected.txt` lists
//! one reference per line, in order ("Romans 8:28", "Psalms 23"); blank
//! lines and lines starting with '#' are ignored.
//!
//! Prints every transcript and detection with its time, then a score.
//! Exit code is non-zero when CHORUS_REPLAY_MIN is set and the share of
//! expected verses found is below it.

use std::time::{Duration, Instant};

use rhema_detection::{DirectDetector, PartialConfirmer};
use rhema_stt::{SttProvider, TranscriptEvent, WhisperProvider};

const BOOKS: [&str; 66] = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth",
    "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra",
    "Nehemiah", "Esther", "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon",
    "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
    "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah",
    "Malachi", "Matthew", "Mark", "Luke", "John", "Acts", "Romans", "1 Corinthians",
    "2 Corinthians", "Galatians", "Ephesians", "Philippians", "Colossians",
    "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon",
    "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude",
    "Revelation",
];

/// Minimal reader for 16-bit PCM mono WAV.
fn read_wav(path: &str) -> Vec<i16> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut i = 12;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let size = u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]) as usize;
        if id == b"data" {
            let end = (i + 8 + size).min(bytes.len());
            return bytes[i + 8..end]
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();
        }
        i += 8 + size + (size & 1);
    }
    panic!("{path}: no data chunk");
}

/// "John 3:16", "John 3:16-18", or "Psalms 23" for a chapter-only reference.
fn show(d: &rhema_detection::Detection) -> String {
    let r = &d.verse_ref;
    let name = usize::try_from(r.book_number - 1)
        .ok()
        .and_then(|i| BOOKS.get(i))
        .copied()
        .unwrap_or("?");
    if d.is_chapter_only {
        return format!("{name} {}", r.chapter);
    }
    match r.verse_end {
        Some(e) if e != r.verse_start => format!("{name} {}:{}-{e}", r.chapter, r.verse_start),
        _ => format!("{name} {}:{}", r.chapter, r.verse_start),
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(args.len() == 5, "usage: chorus-replay <model.bin> <threads> <recording.wav> <expected.txt>");
    let threads: i32 = args[2].parse().expect("threads");
    let audio = read_wav(&args[3]);
    let expected: Vec<String> = std::fs::read_to_string(&args[4])
        .expect("expected file")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(String::from)
        .collect();

    // A model folder means Parakeet; a model file means Whisper.
    let model = std::path::PathBuf::from(&args[1]);
    let provider: Box<dyn SttProvider> = if model.is_dir() {
        println!("engine: parakeet ({})", model.display());
        Box::new(rhema_stt::ParakeetProvider::new(model, threads))
    } else {
        println!("engine: whisper ({})", model.display());
        Box::new(WhisperProvider::new(model, Some("en".into()), threads))
    };
    let (audio_tx, audio_rx) = crossbeam_channel::bounded::<Vec<i16>>(1000);
    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<TranscriptEvent>(64);

    let started = Instant::now();
    let runner = tokio::spawn(async move {
        let _ = provider.start(audio_rx, event_tx).await;
    });

    // Feed the recording in 10 ms chunks at real-time speed, like a
    // Windows microphone, then 2 s of silence so the last phrase ends.
    #[expect(clippy::cast_precision_loss, reason = "recording lengths are small")]
    let audio_secs = audio.len() as f64 / 16_000.0;
    std::thread::spawn(move || {
        let t0 = Instant::now();
        let silence = vec![0i16; 32_000];
        for (n, chunk) in audio.chunks(160).chain(silence.chunks(160)).enumerate() {
            let due = Duration::from_millis(10 * n as u64);
            if let Some(wait) = due.checked_sub(t0.elapsed()) {
                std::thread::sleep(wait);
            }
            if audio_tx.send(chunk.to_vec()).is_err() {
                return;
            }
        }
        // Dropping the sender tells the provider the audio has ended.
    });

    let mut detector = DirectDetector::new();
    let mut confirmer = PartialConfirmer::new();
    let mut found: Vec<(f64, String)> = Vec::new();

    while let Some(event) = event_rx.recv().await {
        let t = started.elapsed().as_secs_f64();
        let (text, is_final) = match event {
            TranscriptEvent::Partial { transcript, .. } => (transcript, false),
            TranscriptEvent::Final { transcript, .. } => (transcript, true),
            TranscriptEvent::Disconnected => break,
            _ => continue,
        };
        println!("[{t:6.1}s] {} {text:?}", if is_final { "FINAL  " } else { "partial" });
        let detected = detector.detect_utterance(&text, is_final);
        let results = confirmer.filter_settled(detected, is_final, detector.settled());
        for d in results {
            let r = show(&d);
            rhema_stt::whisper::set_context_hint(&r);
            if found.last().map(|(_, x)| x) != Some(&r) {
                println!("[{t:6.1}s]   -> VERSE {r} ({:.0}%)", d.confidence * 100.0);
                found.push((t, r));
            }
        }
    }
    let _ = runner.await;

    // Score: each expected reference must appear, in order.
    let mut hits = 0usize;
    let mut next = 0usize;
    let mut matched = vec![false; found.len()];
    for e in &expected {
        if let Some(pos) = found[next..].iter().position(|(_, r)| r == e) {
            matched[next + pos] = true;
            next += pos + 1;
            hits += 1;
            println!("  OK      {e}");
        } else {
            println!("  MISSED  {e}");
        }
    }
    let wrong: Vec<&String> = found
        .iter()
        .zip(&matched)
        .filter(|(_, m)| !**m)
        .map(|((_, r), _)| r)
        .filter(|r| !expected.contains(r))
        .collect();
    for w in &wrong {
        println!("  WRONG   {w}");
    }
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    let rate = if expected.is_empty() { 1.0 } else { hits as f64 / expected.len() as f64 };
    println!(
        "\n==== CHORUS REPLAY: {hits}/{} verses found ({:.0}%), {} wrong, {:.0}s of audio processed in {:.0}s ====",
        expected.len(),
        rate * 100.0,
        wrong.len(),
        audio_secs,
        started.elapsed().as_secs_f64()
    );
    if let Ok(min) = std::env::var("CHORUS_REPLAY_MIN") {
        let min: f64 = min.parse().expect("CHORUS_REPLAY_MIN must be a number");
        assert!(rate >= min, "found {rate:.2} of expected verses, below {min}");
    }
}
