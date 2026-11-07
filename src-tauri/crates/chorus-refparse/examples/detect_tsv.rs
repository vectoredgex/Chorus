//! Runs the reference finder over speech-engine transcripts.
//!
//! Reads `<clip>\t<seconds>\t<text>` lines (in sermon order) and prints
//! `<clip>\t<seconds>\t<verses>` with the verses (medium confidence or
//! higher, chapter and verse) joined by `; `.
//!
//! Usage: cargo run --example detect_tsv -- <transcripts.tsv>

use chorus_refparse::{Confidence, Detector};

fn main() {
    let path = std::env::args().nth(1).expect("usage: detect_tsv <transcripts.tsv>");
    let input = std::fs::read_to_string(&path).expect("read transcripts");
    let mut det = Detector::new();
    for line in input.lines() {
        let mut parts = line.splitn(3, '\t');
        let (Some(clip), Some(secs)) = (parts.next(), parts.next()) else { continue };
        let text = parts.next().unwrap_or("");
        let found: Vec<String> = det
            .detect(text)
            .iter()
            .filter(|d| d.confidence >= Confidence::Medium && d.reference.verse_start.is_some())
            .map(|d| d.reference.to_string())
            .collect();
        println!("{clip}\t{secs}\t{}", found.join("; "));
    }
}
