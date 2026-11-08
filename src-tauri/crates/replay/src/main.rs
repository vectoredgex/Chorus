//! Chorus replay test.
//!
//! Plays a WAV recording into the real speech pipeline (speech detection,
//! Whisper with partials, prompts and clean-up) at real-time speed, runs every
//! transcript through the real detector exactly as the app does (partial
//! confirmation, passage hint), and compares the verses found with the
//! expected list.
//!
//! Usage:
//!   chorus-replay <model.bin | model_dir> <threads> <recording> [expected.txt | -]
//!
//! A Whisper model file runs Whisper. A model folder runs whichever engine
//! it holds: Parakeet (English), IndicConformer or Indic-Transcribe
//! (Malayalam).
//!
//! The recording can be WAV, MP3, M4A/AAC, OGG or FLAC at any sample rate;
//! it is mixed to mono and resampled to 16 kHz. `expected.txt` lists
//! one reference per line, in order ("Romans 8:28", "Psalms 23"); blank
//! lines and lines starting with '#' are ignored. Without it (or with "-")
//! the verses found are only listed, not scored.
//!
//! `CHORUS_REPLAY_MINUTES` limits how much of the recording is played.
//! `CHORUS_REPLAY_FAST=<IndicConformer folder>` with an Indic-Transcribe
//! folder runs both Malayalam engines (fast previews, accurate finals).
//!
//! Prints every transcript and detection with its time, then a score.
//! Exit code is non-zero when CHORUS_REPLAY_MIN is set and the share of
//! expected verses found is below it.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
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

/// Decodes any common audio file to 16 kHz mono 16-bit samples.
fn read_audio(path: &str) -> Vec<i16> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path).extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .unwrap_or_else(|e| panic!("{path}: unsupported audio ({e})"));
    let mut format = probed.format;
    let track = format.default_track().expect("no audio track").clone();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .unwrap_or_else(|e| panic!("{path}: unsupported codec ({e})"));
    let mut rate = track.codec_params.sample_rate.unwrap_or(16_000);
    let mut mono: Vec<f32> = Vec::new();
    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track.id {
            continue;
        }
        let Ok(decoded) = decoder.decode(&packet) else { continue };
        let spec = *decoded.spec();
        rate = spec.rate;
        let channels = spec.channels.count().max(1);
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buf.copy_interleaved_ref(decoded);
        #[expect(clippy::cast_precision_loss, reason = "channel count is tiny")]
        mono.extend(buf.samples().chunks(channels).map(|f| f.iter().sum::<f32>() / channels as f32));
    }
    resample_to_16k(&mono, rate)
}

/// Low-pass (box filter) then linear interpolation to 16 kHz.
#[expect(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "audio maths")]
fn resample_to_16k(input: &[f32], rate: u32) -> Vec<i16> {
    let to_i16 = |x: f32| (x.clamp(-1.0, 1.0) * 32767.0) as i16;
    if rate == 16_000 || input.is_empty() {
        return input.iter().map(|&x| to_i16(x)).collect();
    }
    let ratio = f64::from(rate) / 16_000.0;
    let width = (ratio.round() as usize).max(1);
    let filtered: Vec<f32> = if width > 1 {
        let mut acc = 0.0f32;
        let mut out = Vec::with_capacity(input.len());
        for (i, &x) in input.iter().enumerate() {
            acc += x;
            if i >= width {
                acc -= input[i - width];
            }
            out.push(acc / width.min(i + 1) as f32);
        }
        out
    } else {
        input.to_vec()
    };
    let n_out = (input.len() as f64 / ratio) as usize;
    (0..n_out)
        .map(|k| {
            let pos = k as f64 * ratio;
            let i = pos as usize;
            let frac = (pos - i as f64) as f32;
            let a = filtered[i.min(filtered.len() - 1)];
            let b = filtered[(i + 1).min(filtered.len() - 1)];
            to_i16(a + (b - a) * frac)
        })
        .collect()
}

/// Prints the speech engines' log lines (e.g. how long each phrase took).
struct PrintLog;
impl log::Log for PrintLog {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Info && !m.target().starts_with("ort")
    }
    fn log(&self, r: &log::Record) {
        if self.enabled(r.metadata()) {
            println!("         log: {}", r.args());
        }
    }
    fn flush(&self) {}
}

fn mmss(samples: usize) -> String {
    let s = samples / 16_000;
    format!("{:02}:{:02}", s / 60, s % 60)
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
    assert!(
        args.len() == 4 || args.len() == 5,
        "usage: chorus-replay <model.bin | model_dir> <threads> <recording> [expected.txt | -]"
    );
    let _ = log::set_logger(&PrintLog).map(|()| log::set_max_level(log::LevelFilter::Info));
    let threads: i32 = args[2].parse().expect("threads");
    let mut audio = read_audio(&args[3]);
    if let Some(min) = std::env::var("CHORUS_REPLAY_MINUTES").ok().and_then(|m| m.parse::<usize>().ok()) {
        audio.truncate(min * 60 * 16_000);
    }
    let scored = args.len() == 5 && args[4] != "-";
    let expected: Vec<String> = if scored {
        std::fs::read_to_string(&args[4])
            .expect("expected file")
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(String::from)
            .collect()
    } else {
        Vec::new()
    };

    // A model folder means Parakeet, IndicConformer or Indic-Transcribe
    // (recognised by its files); a model file means Whisper.
    let model = std::path::PathBuf::from(&args[1]);
    // CHORUS_REPLAY_FAST=<IndicConformer folder> with an Indic-Transcribe
    // folder runs Malayalam fast + accurate (both engines).
    let fast = std::env::var("CHORUS_REPLAY_FAST").ok().map(std::path::PathBuf::from);
    let provider: Box<dyn SttProvider> = if let (Some(fast), true) = (fast, rhema_stt::indic_transcribe::model_present(&model)) {
        println!("engine: malayalam fast + accurate ({} + {})", fast.display(), model.display());
        let n = usize::try_from(threads).unwrap_or(4);
        Box::new(rhema_stt::MalayalamComboProvider::new(fast, model, threads, n))
    } else if rhema_stt::indic_transcribe::model_present(&model) {
        println!("engine: indic-transcribe ({})", model.display());
        Box::new(rhema_stt::IndicTranscribeProvider::new(model, usize::try_from(threads).unwrap_or(4)))
    } else if rhema_stt::parakeet::model_present_for(rhema_stt::SherpaModel::IndicConformerMalayalam, &model) {
        println!("engine: indicconformer ({})", model.display());
        Box::new(rhema_stt::ParakeetProvider::with_model(rhema_stt::SherpaModel::IndicConformerMalayalam, model, threads))
    } else if model.is_dir() {
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
        if let Err(e) = provider.start(audio_rx, event_tx).await {
            println!("ENGINE ERROR: {e}");
        }
    });

    // Feed the recording in 10 ms chunks at real-time speed, like a
    // Windows microphone, then 2 s of silence so the last phrase ends.
    #[expect(clippy::cast_precision_loss, reason = "recording lengths are small")]
    let audio_secs = audio.len() as f64 / 16_000.0;
    let fed = Arc::new(AtomicUsize::new(0));
    let fed_by_feeder = fed.clone();
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
            fed_by_feeder.fetch_add(chunk.len(), Ordering::Relaxed);
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
        let at = mmss(fed.load(Ordering::Relaxed));
        println!("[{at}] {} {text:?}", if is_final { "FINAL  " } else { "partial" });
        let detected = detector.detect_utterance(&text, is_final);
        let results = confirmer.filter_settled(detected, is_final, detector.settled());
        for d in results {
            let r = show(&d);
            rhema_stt::whisper::set_context_hint(&r);
            if found.last().map(|(_, x)| x) != Some(&r) {
                println!("[{at}]   -> VERSE {r} ({:.0}%)", d.confidence * 100.0);
                found.push((t, r));
            }
        }
    }
    let _ = runner.await;

    if !scored {
        println!("\n==== VERSES SHOWN ({}) ====", found.len());
        for (_, r) in &found {
            println!("  {r}");
        }
        println!("{audio_secs:.0}s of audio played in {:.0}s", started.elapsed().as_secs_f64());
        return;
    }

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
