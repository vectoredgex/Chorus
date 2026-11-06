//! Chorus reference engine: finds Bible references in live sermon
//! transcripts.
//!
//! It handles how preachers actually speak, not just written citations:
//! "John 3:16", "John three sixteen", "Romans chapter eight verses twenty
//! eight through thirty", "first Corinthians thirteen", "the twenty third
//! psalm", "the third chapter of John", "verse sixteen of John three",
//! and follow-ups like "look at verse 17" or "the next verse".
//!
//! Every detection carries a confidence level so the operator screen can
//! auto-show only what is certain and ask about the rest.

pub mod books;
pub mod numbers;

use books::{Book, Target, ALIASES, WEAK_ALIASES};
use std::fmt;
use std::sync::OnceLock;

/// Highest verse number in any chapter (Psalm 119:176).
pub const MAX_VERSE: u16 = 176;
/// Ranges wider than this are treated as mishearings.
const MAX_RANGE: u16 = 60;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reference {
    pub book: u8,
    pub chapter: u16,
    pub verse_start: Option<u16>,
    pub verse_end: Option<u16>,
}

impl Reference {
    pub fn book(&self) -> &'static Book {
        books::book(self.book)
    }
}

impl fmt::Display for Reference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.book().name, self.chapter)?;
        if let Some(v) = self.verse_start {
            write!(f, ":{v}")?;
            if let Some(e) = self.verse_end {
                write!(f, "-{e}")?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Confidence {
    /// Probably not a reference; show to the operator only.
    Low,
    /// Likely a reference; suggest, let the operator confirm.
    Medium,
    /// Certain enough to put on screen automatically.
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The book was named in this phrase.
    Explicit,
    /// Filled in from the previous reference ("verse 17", "next verse").
    Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Detection {
    pub reference: Reference,
    pub confidence: Confidence,
    pub source: Source,
    /// The normalized words that produced this detection.
    pub matched: String,
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// Lowercase, split into tokens, keep ":" and "-" as their own tokens
/// when they sit between numbers.
pub fn normalize(text: &str) -> Vec<String> {
    let lower = text.to_lowercase().replace(['\u{2019}', '\u{2018}', '`'], "'");
    let chars: Vec<char> = lower.chars().collect();
    let mut out = String::with_capacity(chars.len() + 16);
    for (idx, &c) in chars.iter().enumerate() {
        let prev = if idx > 0 { chars[idx - 1] } else { ' ' };
        let next = chars.get(idx + 1).copied().unwrap_or(' ');
        match c {
            c if c.is_alphanumeric() => out.push(c),
            '\'' => {
                // drop possessive "'s" ("John's gospel" -> "john gospel")
                if next == 's' && !chars.get(idx + 2).is_some_and(|c| c.is_alphanumeric()) {
                    // skip the apostrophe; the 's' is removed below
                    out.push('\u{1}');
                }
            }
            ':' => out.push_str(" : "),
            '.' if prev.is_ascii_digit() && next.is_ascii_digit() => out.push_str(" : "),
            '-' | '\u{2013}' | '\u{2014}' => {
                if prev.is_ascii_digit() || next.is_ascii_digit() || prev == ' ' || next == ' ' {
                    out.push_str(" - ");
                } else {
                    out.push(' '); // "twenty-three" -> "twenty three"
                }
            }
            _ => out.push(' '),
        }
    }
    out.split_whitespace()
        .map(|w| {
            let w = w.strip_suffix("\u{1}s").unwrap_or(w);
            w.replace('\u{1}', "")
        })
        .filter(|w| !w.is_empty())
        .collect()
}

// ---------------------------------------------------------------------------
// Book matching
// ---------------------------------------------------------------------------

struct Alias {
    words: Vec<&'static str>,
    target: Target,
    weak: bool,
}

fn aliases() -> &'static [Alias] {
    static A: OnceLock<Vec<Alias>> = OnceLock::new();
    A.get_or_init(|| {
        let mut v: Vec<Alias> = ALIASES
            .iter()
            .map(|(s, t)| Alias {
                words: s.split(' ').collect(),
                target: *t,
                weak: WEAK_ALIASES.contains(s),
            })
            .collect();
        v.sort_by(|a, b| b.words.len().cmp(&a.words.len()));
        v
    })
}

fn match_alias(t: &[String], i: usize) -> Option<&'static Alias> {
    aliases().iter().find(|a| {
        a.words.len() <= t.len().saturating_sub(i)
            && a.words.iter().zip(&t[i..]).all(|(w, tok)| *w == tok)
    })
}

#[derive(Debug, Clone, Copy)]
struct BookHit {
    id: u8,
    end: usize,
    weak: bool,
    /// A numbered family resolved without "first/second" being spoken.
    guessed_number: bool,
}

fn match_book(t: &[String], i: usize) -> Option<BookHit> {
    // "first Corinthians", "2 Kings", "third John"
    if let Some(n) = t.get(i).and_then(|s| books::ordinal_prefix(s)) {
        if let Some(a) = match_alias(t, i + 1) {
            if let Target::Family { ids, .. } = a.target {
                let id = ids[n - 1];
                if id != 0 {
                    return Some(BookHit { id, end: i + 1 + a.words.len(), weak: false, guessed_number: false });
                }
            }
        }
    }
    let a = match_alias(t, i)?;
    let end = i + a.words.len();
    match a.target {
        Target::One(id) => Some(BookHit { id, end, weak: a.weak, guessed_number: false }),
        Target::Family { default_id: Some(id), ids } => {
            // "John" alone is the gospel, a real book, not a guess.
            let guessed = id != 43;
            let _ = ids;
            Some(BookHit { id, end, weak: a.weak, guessed_number: guessed })
        }
        Target::Family { default_id: None, .. } => None,
    }
}

// ---------------------------------------------------------------------------
// Chapter / verse grammar
// ---------------------------------------------------------------------------

fn tok(t: &[String], i: usize) -> &str {
    t.get(i).map(|s| s.as_str()).unwrap_or("")
}

fn num(t: &[String], i: usize) -> Option<numbers::Num> {
    numbers::parse(t, i)
}

const CHAPTER_WORDS: &[&str] = &["chapter", "chapters", "chap", "ch", "chpt"];
const VERSE_WORDS: &[&str] = &["verse", "verses", "vs", "v", "vv", "vers", "versus"];
const RANGE_WORDS: &[&str] = &["-", "to", "through", "thru", "till", "until", "and"];
/// Words allowed between a book name and its chapter number.
const BOOK_FILLERS: &[&str] = &[
    "chapter", "chapters", "chap", "ch", "gospel", "book", "letter", "epistle", "in", "at",
    "the", "of", "number",
];

/// A verse number, optionally a range: "16", "verse 16", ": 16",
/// "verses 28 through 30", "1 and 2".
struct VersePart {
    start: u16,
    end: Option<u16>,
    explicit: bool,
    next: usize,
}

fn parse_verse(t: &[String], mut k: usize, allow_bare: bool) -> Option<VersePart> {
    let mut explicit = false;
    // "and verse 5", "at verse 28", "in verse 3"
    if matches!(tok(t, k), "and" | "at" | "in" | ",") && VERSE_WORDS.contains(&tok(t, k + 1)) {
        k += 1;
    }
    if tok(t, k) == ":" || VERSE_WORDS.contains(&tok(t, k)) {
        explicit = true;
        k += 1;
        if tok(t, k) == "number" {
            k += 1;
        }
    } else if !allow_bare {
        return None;
    }
    let n = num(t, k)?;
    if n.value == 0 || n.value > MAX_VERSE as u32 {
        return None;
    }
    let start = n.value as u16;
    k += n.len;
    let mut end = None;
    if RANGE_WORDS.contains(&tok(t, k)) {
        let word = tok(t, k);
        let mut j = k + 1;
        if VERSE_WORDS.contains(&tok(t, j)) {
            j += 1;
        }
        if let Some(e) = num(t, j) {
            let e16 = e.value as u16;
            let ok = if word == "and" {
                e16 == start + 1
            } else {
                e16 > start && e16 - start <= MAX_RANGE && e.value <= MAX_VERSE as u32
            };
            if ok {
                end = Some(e16);
                k = j + e.len;
            }
        }
    }
    Some(VersePart { start, end, explicit, next: k })
}

struct Tail {
    chapter: u16,
    verse: Option<VersePart>,
    chapter_word: bool,
    next: usize,
}

/// Parse what follows a book name.
fn parse_tail(t: &[String], start: usize, book: &Book) -> Option<Tail> {
    let mut k = start;
    let mut chapter_word = false;
    let mut fillers = 0;
    while BOOK_FILLERS.contains(&tok(t, k)) && fillers < 3 {
        if CHAPTER_WORDS.contains(&tok(t, k)) {
            chapter_word = true;
        }
        k += 1;
        fillers += 1;
    }
    // One-chapter books are cited by verse: "Jude verse 3", "Philemon 6".
    if book.chapters == 1 && !chapter_word {
        if let Some(v) = parse_verse(t, k, false) {
            let next = v.next;
            return Some(Tail { chapter: 1, verse: Some(v), chapter_word, next });
        }
    }
    let c = num(t, k)?;
    if book.chapters == 1 && !chapter_word && c.value > 1 {
        // "Jude 24" means verse 24; "Jude 2:1" is not a real reference.
        if tok(t, k + c.len) == ":" || c.value > MAX_VERSE as u32 {
            return None;
        }
        let v = VersePart { start: c.value as u16, end: None, explicit: false, next: k + c.len };
        return Some(Tail { chapter: 1, verse: Some(v), chapter_word, next: k + c.len });
    }
    if c.value == 0 || c.value > book.chapters as u32 {
        return None;
    }
    k += c.len;
    let verse = parse_verse(t, k, true);
    let next = verse.as_ref().map(|v| v.next).unwrap_or(k);
    Some(Tail { chapter: c.value as u16, verse, chapter_word, next })
}

fn make_ref(book: u8, chapter: u16, v: &Option<VersePart>) -> Reference {
    Reference {
        book,
        chapter,
        verse_start: v.as_ref().map(|v| v.start),
        verse_end: v.as_ref().and_then(|v| v.end),
    }
}

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

/// Stateful detector: remembers the last reference so follow-ups like
/// "verse 17" or "the next verse" resolve correctly. Use one per service.
#[derive(Debug, Default, Clone)]
pub struct Detector {
    last: Option<Reference>,
}

impl Detector {
    pub fn new() -> Self {
        Self::default()
    }

    /// The reference follow-ups are resolved against.
    pub fn last(&self) -> Option<Reference> {
        self.last
    }

    pub fn set_last(&mut self, r: Option<Reference>) {
        self.last = r;
    }

    /// Find every reference in a piece of transcript.
    pub fn detect(&mut self, text: &str) -> Vec<Detection> {
        let t = normalize(text);
        let mut out = Vec::new();
        let mut i = 0;
        while i < t.len() {
            let found = self
                .try_of_form(&t, i)
                .or_else(|| self.try_ordinal_psalm(&t, i))
                .or_else(|| self.try_book(&t, i))
                .or_else(|| self.try_context(&t, i));
            match found {
                Some((d, end)) => {
                    if d.confidence >= Confidence::Medium {
                        self.last = Some(d.reference);
                    }
                    out.push(d);
                    i = end.max(i + 1);
                }
                None => i += 1,
            }
        }
        out
    }

    fn emit(t: &[String], i: usize, end: usize, r: Reference, c: Confidence, s: Source) -> (Detection, usize) {
        let matched = t[i..end.min(t.len())].join(" ");
        (Detection { reference: r, confidence: c, source: s, matched }, end)
    }

    /// "Romans 8:28", "first Corinthians chapter 13", "John three sixteen".
    fn try_book(&self, t: &[String], i: usize) -> Option<(Detection, usize)> {
        let hit = match_book(t, i)?;
        let book = books::book(hit.id);
        // Short or everyday-word aliases need a number right after.
        let next = tok(t, hit.end);
        let number_follows = num(t, hit.end).is_some()
            || next == ":"
            || (CHAPTER_WORDS.contains(&next) && num(t, hit.end + 1).is_some());
        if hit.weak && !number_follows {
            return None;
        }
        let tail = parse_tail(t, hit.end, book)?;
        let r = make_ref(hit.id, tail.chapter, &tail.verse);
        let conf = match &tail.verse {
            Some(v) => {
                if hit.weak && !v.explicit && !tail.chapter_word {
                    Confidence::Medium
                } else if hit.guessed_number && !v.explicit {
                    Confidence::Medium
                } else {
                    Confidence::High
                }
            }
            None => {
                if tail.chapter_word || !(book.ambiguous || hit.weak) {
                    Confidence::Medium
                } else {
                    Confidence::Low
                }
            }
        };
        Some(Self::emit(t, i, tail.next, r, conf, Source::Explicit))
    }

    /// "chapter 3 of John", "the third chapter of John verse 16",
    /// "verse 16 of John 3", "verse 4 of Psalm 23".
    fn try_of_form(&self, t: &[String], i: usize) -> Option<(Detection, usize)> {
        let mut k = i;
        if tok(t, k) == "the" {
            k += 1;
        }
        enum Kind { Chapter(u16), Verse(VersePart) }
        let kind;
        if CHAPTER_WORDS.contains(&tok(t, k)) {
            let n = num(t, k + 1)?;
            kind = Kind::Chapter(n.value as u16);
            k += 1 + n.len;
        } else if VERSE_WORDS.contains(&tok(t, k)) {
            let v = parse_verse(t, k, false)?;
            k = v.next;
            kind = Kind::Verse(v);
        } else {
            let n = num(t, k)?;
            if !CHAPTER_WORDS.contains(&tok(t, k + n.len)) {
                return None;
            }
            kind = Kind::Chapter(n.value as u16);
            k += n.len + 1;
        }
        if tok(t, k) != "of" {
            return None;
        }
        k += 1;
        let mut fillers = 0;
        while matches!(tok(t, k), "the" | "book" | "gospel" | "letter" | "epistle" | "of" | "according" | "to" | "saint" | "st")
            && fillers < 4
        {
            k += 1;
            fillers += 1;
        }
        let hit = match_book(t, k)?;
        let book = books::book(hit.id);
        match kind {
            Kind::Chapter(c) => {
                if c == 0 || c > book.chapters {
                    return None;
                }
                let verse = parse_verse(t, hit.end, false);
                let end = verse.as_ref().map(|v| v.next).unwrap_or(hit.end);
                let conf = if verse.is_some() { Confidence::High } else { Confidence::Medium };
                let r = make_ref(hit.id, c, &verse);
                Some(Self::emit(t, i, end, r, conf, Source::Explicit))
            }
            Kind::Verse(v) => {
                // "verse 16 of John 3" / "verse 16 of John chapter 3"
                let mut j = hit.end;
                if CHAPTER_WORDS.contains(&tok(t, j)) {
                    j += 1;
                }
                let (chapter, end) = match num(t, j) {
                    Some(c) if c.value >= 1 && c.value <= book.chapters as u32 => (c.value as u16, j + c.len),
                    _ if book.chapters == 1 => (1, hit.end),
                    _ => return None,
                };
                let r = make_ref(hit.id, chapter, &Some(v));
                Some(Self::emit(t, i, end, r, Confidence::High, Source::Explicit))
            }
        }
    }

    /// "the twenty third psalm", "the 23rd psalm".
    fn try_ordinal_psalm(&self, t: &[String], i: usize) -> Option<(Detection, usize)> {
        let mut k = i;
        if tok(t, k) == "the" {
            k += 1;
        }
        let n = num(t, k)?;
        if !n.ordinal || !matches!(tok(t, k + n.len), "psalm" | "psalms") {
            return None;
        }
        if n.value == 0 || n.value > 150 {
            return None;
        }
        let end = k + n.len + 1;
        let r = Reference { book: 19, chapter: n.value as u16, verse_start: None, verse_end: None };
        Some(Self::emit(t, i, end, r, Confidence::High, Source::Explicit))
    }

    /// Follow-ups that rely on the previous reference.
    fn try_context(&self, t: &[String], i: usize) -> Option<(Detection, usize)> {
        let last = self.last?;
        // "the next verse" / "next verse"
        let mut k = i;
        if tok(t, k) == "the" {
            k += 1;
        }
        if tok(t, k) == "next" && VERSE_WORDS.contains(&tok(t, k + 1)) {
            let v = last.verse_end.or(last.verse_start)? + 1;
            let r = Reference { verse_start: Some(v), verse_end: None, ..last };
            return Some(Self::emit(t, i, k + 2, r, Confidence::Medium, Source::Context));
        }
        // "the first verse", "the sixteenth verse"
        if let Some(n) = num(t, k) {
            if n.ordinal && VERSE_WORDS.contains(&tok(t, k + n.len)) && n.value >= 1 && n.value <= MAX_VERSE as u32 {
                let r = Reference { verse_start: Some(n.value as u16), verse_end: None, ..last };
                return Some(Self::emit(t, i, k + n.len + 1, r, Confidence::Medium, Source::Context));
            }
        }
        // "chapter 4 verse 5", "chapter 5"
        if CHAPTER_WORDS.contains(&tok(t, i)) {
            let c = num(t, i + 1)?;
            let book = last.book();
            if c.value == 0 || c.value > book.chapters as u32 {
                return None;
            }
            let k = i + 1 + c.len;
            let verse = parse_verse(t, k, false);
            let end = verse.as_ref().map(|v| v.next).unwrap_or(k);
            let conf = if verse.is_some() { Confidence::Medium } else { Confidence::Low };
            let r = make_ref(last.book, c.value as u16, &verse);
            return Some(Self::emit(t, i, end, r, conf, Source::Context));
        }
        // "verse 17", "verses 3 to 5"
        if VERSE_WORDS.contains(&tok(t, i)) {
            let v = parse_verse(t, i, false)?;
            let end = v.next;
            let r = make_ref(last.book, last.chapter, &Some(v));
            return Some(Self::emit(t, i, end, r, Confidence::Medium, Source::Context));
        }
        None
    }
}

/// One-shot helper for a single phrase with no earlier context.
pub fn detect(text: &str) -> Vec<Detection> {
    Detector::new().detect(text)
}
