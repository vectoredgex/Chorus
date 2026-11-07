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
    /// The book name was misheard ("Geddes chapter 2 verse 3") and matched
    /// to the closest book because the words around it form a reference.
    Fuzzy,
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

/// A word that sounds like "verse"/"verses" as speech-to-text mishears it in
/// noise: a short start, then an "er/ur/or" sound, then an "s" sound
/// ("mercer", "worse", "purse", "hers", "mers"). Only used between a
/// chapter number and a verse number, where nothing else fits.
fn sounds_like_verse(w: &str) -> bool {
    const ENDINGS: &[&str] = &["se", "s", "ce", "cer", "ser", "ses", "ces", "sus", "sis"];
    if !(3..=7).contains(&w.len()) || !w.bytes().all(|b| b.is_ascii_lowercase()) {
        return false;
    }
    ["er", "ur", "or"].iter().any(|v| {
        w.match_indices(v).any(|(i, _)| {
            i <= 3 && ENDINGS.contains(&&w[i + 2..])
        })
    })
}

/// True when the words at `k` are a numbered book name ("2 Corinthians",
/// "first John"), so the number must not be read as a chapter or verse.
fn starts_numbered_book(t: &[String], k: usize) -> bool {
    let Some(n) = num(t, k) else { return false };
    if n.len != 1 || !(1..=3).contains(&n.value) {
        return false;
    }
    matches!(match_alias(t, k + 1).map(|a| a.target), Some(Target::Family { ids, .. }) if ids[(n.value - 1) as usize] != 0)
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
    } else if sounds_like_verse(tok(t, k)) && num(t, k + 1).is_some() {
        // "Ephesians 2, Mercer 8 and 9": a misheard "verses" between the
        // chapter and the verse number.
        k += 1;
    }
    let n = num(t, k)?;
    if n.value == 0 || n.value > MAX_VERSE as u32 {
        return None;
    }
    // In a list, "Revelation 21, 2 Corinthians 13": the "2" belongs to the
    // next book, it is not a verse.
    if !explicit && starts_numbered_book(t, k) {
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
    // Speech engines sometimes split one spoken chapter into digit groups:
    // "Psalm one thirty-six verse 1" arrives as "Psalm 1 36 verse 1". When a
    // run of digit tokens is followed by an explicit verse marker, the run is
    // one chapter number.
    let run = t[k.min(t.len())..]
        .iter()
        .take_while(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
        .count();
    let marker = tok(t, k + run);
    if run >= 2 && (marker == ":" || VERSE_WORDS.contains(&marker)) {
        let joined: String = t[k..k + run].concat();
        if let Ok(ch) = joined.parse::<u16>() {
            if ch >= 1 && ch <= book.chapters {
                if let Some(v) = parse_verse(t, k + run, false) {
                    let next = v.next;
                    return Some(Tail { chapter: ch, verse: Some(v), chapter_word, next });
                }
            }
        }
    }
    // Speech engines write a spoken "four" as "for" right after a book
    // name: "Philippians for 13" is Philippians 4:13. Only when a number
    // follows, so "John 3 for God so loved" is unaffected.
    if starts_numbered_book(t, k) {
        return None;
    }
    let c = if tok(t, k) == "for" && num(t, k + 1).is_some() && book.chapters >= 4 {
        numbers::Num { value: 4, len: 1, ordinal: false }
    } else {
        num(t, k)?
    };
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
    // Parakeet writes a spoken "Jeremiah twenty-nine eleven" as
    // "Jeremiah 29-11" and "Joshua one nine" as "Joshua 1-9" (measured on
    // the test sermon). Right after a book and chapter, with no "chapters"
    // word, a dash is the chapter-verse break, not a chapter range: spoken
    // chapter ranges are rare and come with "to" ("Psalms 1 to 3").
    if !chapter_word && tok(t, k) == "-" && tok(t, k + 2) != ":" {
        if let Some(v) = parse_verse(t, k + 1, true) {
            let next = v.next;
            return Some(Tail { chapter: c.value as u16, verse: Some(v), chapter_word, next });
        }
    }
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
// Misheard book names
// ---------------------------------------------------------------------------

/// Jaro-Winkler similarity in [0, 1].
fn jaro_winkler(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a == b {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let window = (a.len().max(b.len()) / 2).saturating_sub(1);
    let mut a_hit = vec![false; a.len()];
    let mut b_hit = vec![false; b.len()];
    let mut matches = 0usize;
    for (i, ca) in a.iter().enumerate() {
        let lo = i.saturating_sub(window);
        let hi = (i + window + 1).min(b.len());
        for j in lo..hi {
            if !b_hit[j] && b[j] == *ca {
                a_hit[i] = true;
                b_hit[j] = true;
                matches += 1;
                break;
            }
        }
    }
    if matches == 0 {
        return 0.0;
    }
    let (mut k, mut transpositions) = (0usize, 0usize);
    for i in 0..a.len() {
        if a_hit[i] {
            while !b_hit[k] {
                k += 1;
            }
            if a[i] != b[k] {
                transpositions += 1;
            }
            k += 1;
        }
    }
    let m = matches as f64;
    let t = transpositions as f64 / 2.0;
    let jaro = (m / a.len() as f64 + m / b.len() as f64 + (m - t) / m) / 3.0;
    let prefix = a.iter().zip(&b).take(4).take_while(|(x, y)| x == y).count();
    jaro + prefix as f64 * 0.1 * (1.0 - jaro)
}

/// Everyday words that must never be read as a misheard book name.
const NOT_BOOKS: &[&str] = &[
    "the", "this", "that", "there", "these", "those", "then", "than", "them", "they", "turn",
    "open", "read", "look", "says", "said", "verse", "verses", "chapter", "chapters", "jesus",
    "christ", "lord", "father", "grace", "faith", "people", "romantic", "romance", "matters",
    "genius", "nations", "heaven", "going", "morning", "seven", "eleven", "number", "numbers",
    "page", "pages", "about", "again", "where", "which", "while", "with", "from", "into", "unto",
];

/// Closest book to a word that is not itself a known name, or `None`.
/// `min_score` depends on how strongly the following words look like a
/// reference. The runner-up (a different book) must score clearly lower.
fn fuzzy_book(word: &str, chapter: u16, min_score: f64) -> Option<u8> {
    if word.chars().count() < 4
        || !word.chars().all(char::is_alphabetic)
        || NOT_BOOKS.contains(&word)
    {
        return None;
    }
    let mut best: Option<(f64, u8)> = None;
    let mut second = 0.0f64;
    for (name, target) in ALIASES {
        if name.contains(' ') || name.len() < 5 {
            continue;
        }
        let id = match target {
            Target::One(id) => *id,
            Target::Family { default_id: Some(id), .. } => *id,
            Target::Family { default_id: None, .. } => continue,
        };
        let score = jaro_winkler(word, name);
        match best {
            Some((b, bid)) if score > b => {
                if bid != id {
                    second = second.max(b);
                }
                best = Some((score, id));
            }
            Some((_, bid)) if bid != id => second = second.max(score),
            None => best = Some((score, id)),
            _ => {}
        }
    }
    let (score, id) = best?;
    let valid_chapter = chapter >= 1 && chapter <= books::book(id).chapters;
    (score >= min_score && score - second >= 0.05 && valid_chapter).then_some(id)
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
        // A book named in this utterance without a usable chapter
        // ("1 Corinthians 30", misheard thirteen). A follow-up such as
        // "verses 4 to 7" after it belongs to that book, so it must not be
        // resolved against an older, different book (measured: it showed
        // Romans 8:4-7).
        let mut unresolved_book: Option<u8> = None;
        while i < t.len() {
            let found = self
                .try_of_form(&t, i)
                .or_else(|| self.try_ordinal_psalm(&t, i))
                .or_else(|| self.try_book(&t, i))
                .or_else(|| self.try_fuzzy(&t, i))
                .or_else(|| {
                    if let Some(hit) = match_book(&t, i) {
                        if !hit.weak {
                            unresolved_book = Some(hit.id);
                        }
                    }
                    let other_book = unresolved_book
                        .is_some_and(|b| self.last.is_none_or(|l| l.book != b));
                    if other_book { None } else { self.try_context(&t, i) }
                });
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

    /// A misheard book name followed by an unmistakable chapter-and-verse
    /// pattern: "Geddes chapter 2 verse 3", "Galations 5:22".
    /// Only fires when the verse is spoken too, never on a bare number.
    fn try_fuzzy(&self, t: &[String], i: usize) -> Option<(Detection, usize)> {
        let word = t.get(i)?;
        if match_alias(t, i).is_some() {
            return None;
        }
        let mut k = i + 1;
        let chapter_word = CHAPTER_WORDS.contains(&tok(t, k));
        if chapter_word {
            k += 1;
        }
        let c = num(t, k)?;
        let chapter = u16::try_from(c.value).ok()?;
        // Needs ":" or "verse", or Parakeet's "N-M" style ("Doshu 1-9").
        let dash = !chapter_word && tok(t, k + c.len) == "-" && tok(t, k + c.len + 2) != ":";
        let verse = if dash {
            let mut v = parse_verse(t, k + c.len + 1, true)?;
            v.explicit = true;
            v
        } else {
            parse_verse(t, k + c.len, false)?
        };
        // "chapter N verse M" is the strongest pattern; "N:M" is weaker.
        // "N-M" right after an unknown word is almost always a reference
        // whose book name was misheard.
        let min_score = if chapter_word { 0.78 } else if dash { 0.80 } else { 0.85 };
        let id = fuzzy_book(word, chapter, min_score)?;
        let end = verse.next;
        let r = make_ref(id, chapter, &Some(verse));
        Some(Self::emit(t, i, end, r, Confidence::Medium, Source::Fuzzy))
    }

    /// Follow-ups that rely on the previous reference.
    fn try_context(&self, t: &[String], i: usize) -> Option<(Detection, usize)> {
        let last = self.last?;
        // "the next verse" / "next verse"
        let mut k = i;
        if tok(t, k) == "the" {
            k += 1;
        }
        if tok(t, k) == "very" {
            k += 1;
        }
        // "next verse", also misheard as "next merse".
        // A short command on its own ("Next was", "next shows") is also
        // "next verse" heard badly; inside a longer sentence it is not.
        let short_s_word = |w: &str| t.len() <= 4 && w.len() >= 3 && w.ends_with('s') && w.bytes().all(|b| b.is_ascii_lowercase());
        let w = tok(t, k + 1);
        if tok(t, k) == "next" && (VERSE_WORDS.contains(&w) || sounds_like_verse(w) || short_s_word(w)) {
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
        // "the book of <misheard word> chapter 4 verse 3": the speaker named a
        // new book that the speech engine lost. Borrowing the last book here
        // shows a confidently wrong verse, so wait for a clear repeat instead.
        if named_lost_book(t, i) {
            return None;
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
        // "verse 17", "verses 3 to 5". If a "chapter N" just before it was
        // rejected above, this verse belongs to that unknown chapter too.
        let after_rejected_chapter = (2..=3).any(|b| i >= b && CHAPTER_WORDS.contains(&tok(t, i - b)));
        if VERSE_WORDS.contains(&tok(t, i)) && !after_rejected_chapter {
            let v = parse_verse(t, i, false)?;
            let end = v.next;
            let r = make_ref(last.book, last.chapter, &Some(v));
            return Some(Self::emit(t, i, end, r, Confidence::Medium, Source::Context));
        }
        None
    }
}

/// True when the text talks about a verse of a song, not of Scripture:
/// "we sang verse two of the hymn", "the second verse of this song".
/// Callers should not resolve any reference from such text.
/// "the book of <word that is not a book> chapter ...": a book was named but
/// the speech engine lost it, so the last book must not be borrowed.
fn named_lost_book(t: &[String], i: usize) -> bool {
    (1..=3).any(|b| {
        i > b
            && matches!(tok(t, i - b - 1), "book" | "gospel" | "epistle" | "letter")
            && tok(t, i - b) == "of"
    })
}

/// True when the text names a book that was not heard ("turn to the book of
/// this chapter 4 verse 3"). Other detectors use it to avoid filling in the
/// last book themselves.
pub fn names_unheard_book(text: &str) -> bool {
    let t = normalize(text);
    (0..t.len()).any(|i| {
        CHAPTER_WORDS.contains(&tok(&t, i)) && named_lost_book(&t, i) && match_book(&t, i - 1).is_none()
    })
}

pub fn is_song_verse(text: &str) -> bool {
    const MUSIC: &[&str] = &["hymn", "hymns", "song", "songs", "chorus", "carol", "anthem", "poem", "hymnal"];
    let t = normalize(text);
    (0..t.len()).any(|i| {
        if !VERSE_WORDS.contains(&t[i].as_str()) {
            return false;
        }
        let window = (i + 1)..(i + 7).min(t.len());
        let has_music = window.clone().any(|j| MUSIC.contains(&t[j].as_str()));
        let has_book = window.into_iter().any(|j| match_book(&t, j).is_some());
        has_music && !has_book
    })
}

/// One-shot helper for a single phrase with no earlier context.
pub fn detect(text: &str) -> Vec<Detection> {
    Detector::new().detect(text)
}

#[cfg(test)]
mod verse_sound_tests {
    use super::sounds_like_verse;
    #[test]
    fn misheard_verse_words() {
        for w in ["mercer", "worse", "purse", "nurse", "hers", "mers", "verse", "versus", "horse", "verses"] {
            assert!(sounds_like_verse(w), "{w}");
        }
        for w in ["were", "where", "years", "first", "says", "and", "through", "most", "is", "mercy", "herself", "person"] {
            assert!(!sounds_like_verse(w), "{w}");
        }
    }
}

#[cfg(test)]
mod song_tests {
    use super::is_song_verse;
    #[test]
    fn song_verses() {
        assert!(is_song_verse("we sang verse two of the hymn"));
        assert!(is_song_verse("let's sing the second verse of this song"));
        assert!(is_song_verse("verse 3 of the chorus again"));
        assert!(!is_song_verse("John 3 verse 16"));
        assert!(!is_song_verse("verse 16 of John 3 is like a song"));
        assert!(!is_song_verse("the song of Moses in Exodus 15"));
    }
}
