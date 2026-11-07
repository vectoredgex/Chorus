//! Chorus accuracy benchmark for the direct (spoken-reference) detector.
//!
//! Runs every phrase in `chorus_cases.txt` through `DirectDetector` and
//! prints a pass/fail line for each, plus a score. Blocks separated by a
//! blank line share one detector, like a single sermon.
//!
//! Scoring rules:
//! - Confidence tags in the file ([H]/[M]/[L]) are ignored here; Rhema's
//!   direct detections are all auto-queue confidence.
//! - "NONE" passes only when nothing is detected.
//! - A chapter-only expectation ("Psalms 23") also passes when nothing is
//!   emitted, because the detector holds chapter-only references and waits
//!   for the verse. Follow-up lines check that the held reference resolves.
//!
//! Run with:  cargo test -p rhema-detection --test chorus_benchmark -- --nocapture
//! Set CHORUS_BENCH_MIN (e.g. 0.95) to fail below a pass rate.

use rhema_detection::DirectDetector;

const NAMES: [&str; 66] = [
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

fn strip_tag(s: &str) -> &str {
    s.split_once(" [").map_or(s, |(r, _)| r).trim()
}

#[test]
fn chorus_benchmark() {
    let data = include_str!("chorus_cases.txt");
    let mut det = DirectDetector::new();
    let (mut total, mut passed) = (0usize, 0usize);
    let mut section = String::new();
    let mut fails = Vec::new();

    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() {
            det = DirectDetector::new();
            continue;
        }
        if let Some(h) = line.strip_prefix("# ---") {
            section = h.trim_matches(|c| c == '-' || c == ' ').to_string();
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        let Some((input, expected)) = line.split_once("=>") else { continue };
        let (input, expected) = (input.trim(), expected.trim());
        total += 1;

        let mut got: Vec<String> = Vec::new();
        for d in det.detect(input) {
            let r = &d.verse_ref;
            let name = usize::try_from(r.book_number - 1)
                .ok()
                .and_then(|i| NAMES.get(i))
                .copied()
                .unwrap_or("?");
            let s = if d.is_chapter_only {
                format!("{name} {}", r.chapter)
            } else {
                match r.verse_end {
                    Some(e) if e != r.verse_start => {
                        format!("{name} {}:{}-{e}", r.chapter, r.verse_start)
                    }
                    _ => format!("{name} {}:{}", r.chapter, r.verse_start),
                }
            };
            if got.last() != Some(&s) {
                got.push(s);
            }
        }

        let ok = if expected == "NONE" {
            got.is_empty()
        } else {
            let exp: Vec<&str> = expected.split(" | ").map(strip_tag).collect();
            let exact = exp.len() == got.len() && exp.iter().zip(&got).all(|(e, g)| e == g);
            // chapter-only expectation held back by the detector
            let held = exp.iter().all(|e| !e.contains(':')) && got.is_empty();
            exact || held
        };
        if ok {
            passed += 1;
        } else {
            fails.push(format!(
                "[{section}] {input:?}\n      expected: {expected}\n      got:      {}",
                if got.is_empty() { "nothing".to_string() } else { got.join(" | ") }
            ));
        }
    }

    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    let rate = passed as f64 / total as f64;
    println!("\n==== CHORUS BENCHMARK: {passed}/{total} passed ({:.1}%) ====", rate * 100.0);
    for f in &fails {
        println!("FAIL {f}");
    }
    if let Ok(min) = std::env::var("CHORUS_BENCH_MIN") {
        let min: f64 = min.parse().expect("CHORUS_BENCH_MIN must be a number");
        assert!(rate >= min, "pass rate {rate:.3} below required {min}");
    }
}
