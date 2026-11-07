//! Runs the Chorus detector over transcripts from any speech engine.
//!
//! Reads tab-separated lines `<clip>\t<seconds>\t<text>` (in sermon order)
//! from the file given and prints `<clip>\t<seconds>\t<verses>` with the
//! verses joined by `; `. Used to compare speech engines on equal terms.
//!
//! Usage: detect-lines <transcripts.tsv>

use rhema_detection::DirectDetector;

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

fn main() {
    let path = std::env::args().nth(1).expect("usage: detect-lines <transcripts.tsv>");
    let input = std::fs::read_to_string(&path).expect("read transcripts");
    let mut detector = DirectDetector::new();
    for line in input.lines() {
        let mut parts = line.splitn(3, '\t');
        let (Some(clip), Some(secs)) = (parts.next(), parts.next()) else { continue };
        let text = parts.next().unwrap_or("");
        let found: Vec<String> = detector
            .detect(text)
            .iter()
            .filter(|d| !d.is_chapter_only)
            .map(show)
            .collect();
        println!("{clip}\t{secs}\t{}", found.join("; "));
    }
}
