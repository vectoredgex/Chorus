//! The 66 books of the Protestant canon, with chapter counts and every
//! spoken or written form we expect to see in a live transcript.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Book {
    /// 1 = Genesis … 66 = Revelation.
    pub id: u8,
    pub name: &'static str,
    pub chapters: u16,
    /// True when the name is also an everyday English word or a common
    /// first name ("Job", "Acts", "Mark", "James"…). A bare number after
    /// such a word is weak evidence, so detections get lower confidence.
    pub ambiguous: bool,
}

const fn b(id: u8, name: &'static str, chapters: u16, ambiguous: bool) -> Book {
    Book { id, name, chapters, ambiguous }
}

pub const BOOKS: [Book; 66] = [
    b(1, "Genesis", 50, false),
    b(2, "Exodus", 40, false),
    b(3, "Leviticus", 27, false),
    b(4, "Numbers", 36, true),
    b(5, "Deuteronomy", 34, false),
    b(6, "Joshua", 24, true),
    b(7, "Judges", 21, true),
    b(8, "Ruth", 4, true),
    b(9, "1 Samuel", 31, false),
    b(10, "2 Samuel", 24, false),
    b(11, "1 Kings", 22, false),
    b(12, "2 Kings", 25, false),
    b(13, "1 Chronicles", 29, false),
    b(14, "2 Chronicles", 36, false),
    b(15, "Ezra", 10, false),
    b(16, "Nehemiah", 13, false),
    b(17, "Esther", 10, true),
    b(18, "Job", 42, true),
    b(19, "Psalms", 150, false),
    b(20, "Proverbs", 31, false),
    b(21, "Ecclesiastes", 12, false),
    b(22, "Song of Solomon", 8, false),
    b(23, "Isaiah", 66, false),
    b(24, "Jeremiah", 52, false),
    b(25, "Lamentations", 5, false),
    b(26, "Ezekiel", 48, false),
    b(27, "Daniel", 12, true),
    b(28, "Hosea", 14, false),
    b(29, "Joel", 3, true),
    b(30, "Amos", 9, true),
    b(31, "Obadiah", 1, false),
    b(32, "Jonah", 4, true),
    b(33, "Micah", 7, true),
    b(34, "Nahum", 3, false),
    b(35, "Habakkuk", 3, false),
    b(36, "Zephaniah", 3, false),
    b(37, "Haggai", 2, false),
    b(38, "Zechariah", 14, false),
    b(39, "Malachi", 4, false),
    b(40, "Matthew", 28, true),
    b(41, "Mark", 16, true),
    b(42, "Luke", 24, true),
    b(43, "John", 21, true),
    b(44, "Acts", 28, true),
    b(45, "Romans", 16, false),
    b(46, "1 Corinthians", 16, false),
    b(47, "2 Corinthians", 13, false),
    b(48, "Galatians", 6, false),
    b(49, "Ephesians", 6, false),
    b(50, "Philippians", 4, false),
    b(51, "Colossians", 4, false),
    b(52, "1 Thessalonians", 5, false),
    b(53, "2 Thessalonians", 3, false),
    b(54, "1 Timothy", 6, true),
    b(55, "2 Timothy", 4, true),
    b(56, "Titus", 3, true),
    b(57, "Philemon", 1, false),
    b(58, "Hebrews", 13, false),
    b(59, "James", 5, true),
    b(60, "1 Peter", 5, true),
    b(61, "2 Peter", 3, true),
    b(62, "1 John", 5, true),
    b(63, "2 John", 1, true),
    b(64, "3 John", 1, true),
    b(65, "Jude", 1, true),
    b(66, "Revelation", 22, false),
];

pub fn book(id: u8) -> &'static Book {
    &BOOKS[(id - 1) as usize]
}

/// What a spoken name refers to. Numbered families ("Corinthians") need a
/// "first/second/third" prefix to pick the exact book.
#[derive(Debug, Clone, Copy)]
pub enum Target {
    One(u8),
    /// ids for 1st, 2nd, 3rd (0 = none). `default_id` is used when no
    /// prefix was spoken; `None` means the name alone is not a book.
    Family { ids: [u8; 3], default_id: Option<u8> },
}

/// Alias table. Each alias is written lowercase with single spaces, the
/// same way the normalizer produces tokens. Includes abbreviations and
/// common speech-to-text mishearings.
pub const ALIASES: &[(&str, Target)] = &[
    // --- Law ---
    ("genesis", Target::One(1)), ("gen", Target::One(1)), ("genisis", Target::One(1)),
    ("exodus", Target::One(2)), ("exod", Target::One(2)), ("exodis", Target::One(2)),
    ("leviticus", Target::One(3)), ("lev", Target::One(3)), ("leviticous", Target::One(3)),
    ("numbers", Target::One(4)), ("num", Target::One(4)),
    ("deuteronomy", Target::One(5)), ("deut", Target::One(5)), ("dueteronomy", Target::One(5)),
    ("deuteronomie", Target::One(5)),
    // --- History ---
    ("joshua", Target::One(6)), ("josh", Target::One(6)),
    ("judges", Target::One(7)), ("judg", Target::One(7)),
    ("ruth", Target::One(8)),
    ("samuel", Target::Family { ids: [9, 10, 0], default_id: Some(9) }),
    ("sam", Target::Family { ids: [9, 10, 0], default_id: None }),
    ("kings", Target::Family { ids: [11, 12, 0], default_id: Some(11) }),
    ("kgs", Target::Family { ids: [11, 12, 0], default_id: None }),
    ("chronicles", Target::Family { ids: [13, 14, 0], default_id: Some(13) }),
    ("chronicle", Target::Family { ids: [13, 14, 0], default_id: Some(13) }),
    ("chron", Target::Family { ids: [13, 14, 0], default_id: None }),
    ("ezra", Target::One(15)),
    ("nehemiah", Target::One(16)), ("neh", Target::One(16)), ("nehemia", Target::One(16)),
    ("esther", Target::One(17)), ("esth", Target::One(17)), ("ester", Target::One(17)),
    // --- Poetry / wisdom ---
    ("job", Target::One(18)),
    ("psalms", Target::One(19)), ("psalm", Target::One(19)), ("psa", Target::One(19)),
    ("ps", Target::One(19)), ("psalter", Target::One(19)), ("salm", Target::One(19)),
    ("salms", Target::One(19)), ("palms", Target::One(19)),
    ("proverbs", Target::One(20)), ("proverb", Target::One(20)), ("prov", Target::One(20)),
    ("ecclesiastes", Target::One(21)), ("eccl", Target::One(21)), ("eccles", Target::One(21)),
    ("ecclesiastics", Target::One(21)),
    ("song of solomon", Target::One(22)), ("song of songs", Target::One(22)),
    ("songs of solomon", Target::One(22)), ("song of sol", Target::One(22)),
    ("canticles", Target::One(22)), ("song of salomon", Target::One(22)),
    // --- Major prophets ---
    ("isaiah", Target::One(23)), ("isa", Target::One(23)), ("isiah", Target::One(23)),
    ("isaia", Target::One(23)),
    ("jeremiah", Target::One(24)), ("jer", Target::One(24)), ("jeremia", Target::One(24)),
    ("lamentations", Target::One(25)), ("lam", Target::One(25)), ("lamentation", Target::One(25)),
    ("ezekiel", Target::One(26)), ("ezek", Target::One(26)), ("ezekial", Target::One(26)),
    ("ezekiel's", Target::One(26)),
    ("daniel", Target::One(27)), ("dan", Target::One(27)),
    // --- Minor prophets ---
    ("hosea", Target::One(28)), ("hos", Target::One(28)), ("hosiah", Target::One(28)),
    ("joel", Target::One(29)),
    ("amos", Target::One(30)),
    ("obadiah", Target::One(31)), ("obad", Target::One(31)), ("obadia", Target::One(31)),
    ("jonah", Target::One(32)), ("jona", Target::One(32)),
    ("micah", Target::One(33)), ("mic", Target::One(33)),
    ("nahum", Target::One(34)), ("nah", Target::One(34)),
    ("habakkuk", Target::One(35)), ("hab", Target::One(35)), ("habakuk", Target::One(35)),
    ("habbakuk", Target::One(35)), ("habakkuk's", Target::One(35)),
    ("zephaniah", Target::One(36)), ("zeph", Target::One(36)), ("zephania", Target::One(36)),
    ("haggai", Target::One(37)), ("hag", Target::One(37)), ("hagai", Target::One(37)),
    ("zechariah", Target::One(38)), ("zech", Target::One(38)), ("zachariah", Target::One(38)),
    ("zecharia", Target::One(38)),
    ("malachi", Target::One(39)), ("mal", Target::One(39)), ("malaki", Target::One(39)),
    // --- Gospels & Acts ---
    ("matthew", Target::One(40)), ("matt", Target::One(40)), ("mt", Target::One(40)),
    ("mathew", Target::One(40)),
    ("mark", Target::One(41)), ("mk", Target::One(41)), ("marc", Target::One(41)),
    ("luke", Target::One(42)), ("lk", Target::One(42)),
    // "John" alone is the gospel; "first/second/third John" are the letters.
    ("john", Target::Family { ids: [62, 63, 64], default_id: Some(43) }),
    ("jn", Target::Family { ids: [62, 63, 64], default_id: Some(43) }),
    ("acts", Target::One(44)), ("acts of the apostles", Target::One(44)),
    // --- Paul ---
    ("romans", Target::One(45)), ("rom", Target::One(45)), ("roman", Target::One(45)),
    ("corinthians", Target::Family { ids: [46, 47, 0], default_id: Some(46) }),
    ("corinthian", Target::Family { ids: [46, 47, 0], default_id: Some(46) }),
    ("cor", Target::Family { ids: [46, 47, 0], default_id: None }),
    ("galatians", Target::One(48)), ("gal", Target::One(48)), ("galatian", Target::One(48)),
    ("ephesians", Target::One(49)), ("eph", Target::One(49)), ("ephesian", Target::One(49)),
    ("ephemians", Target::One(49)), ("efesians", Target::One(49)),
    ("philippians", Target::One(50)), ("phil", Target::One(50)), ("phillipians", Target::One(50)),
    ("philipians", Target::One(50)), ("philippines", Target::One(50)),
    ("colossians", Target::One(51)), ("col", Target::One(51)), ("colosians", Target::One(51)),
    ("thessalonians", Target::Family { ids: [52, 53, 0], default_id: Some(52) }),
    ("thessalonian", Target::Family { ids: [52, 53, 0], default_id: Some(52) }),
    ("thess", Target::Family { ids: [52, 53, 0], default_id: None }),
    ("timothy", Target::Family { ids: [54, 55, 0], default_id: Some(54) }),
    ("tim", Target::Family { ids: [54, 55, 0], default_id: None }),
    ("titus", Target::One(56)),
    ("philemon", Target::One(57)), ("phlm", Target::One(57)), ("philemen", Target::One(57)),
    // --- General letters ---
    ("hebrews", Target::One(58)), ("heb", Target::One(58)), ("hebrew", Target::One(58)),
    ("james", Target::One(59)), ("jas", Target::One(59)),
    ("peter", Target::Family { ids: [60, 61, 0], default_id: Some(60) }),
    ("pet", Target::Family { ids: [60, 61, 0], default_id: None }),
    ("jude", Target::One(65)),
    ("revelation", Target::One(66)), ("revelations", Target::One(66)), ("rev", Target::One(66)),
    ("the revelation", Target::One(66)),
];

/// Short aliases that are also ordinary words; they only count when a
/// number follows directly.
pub const WEAK_ALIASES: &[&str] = &[
    "gen", "num", "josh", "sam", "ester", "ps", "isa", "lam", "dan", "mic", "nah", "hab",
    "hag", "mal", "matt", "mt", "mark", "mk", "marc", "lk", "jn", "rom", "roman", "cor", "gal",
    "phil", "col", "tim", "heb", "hebrew", "jas", "pet", "rev", "jona", "palms", "lev", "deut",
    "hos", "esth", "prov", "neh", "proverb", "chronicle",
];

/// Words spoken before a numbered book: "first Corinthians", "2 Kings".
pub fn ordinal_prefix(tok: &str) -> Option<usize> {
    match tok {
        "1" | "1st" | "first" | "one" => Some(1),
        "2" | "2nd" | "second" | "two" | "ii" => Some(2),
        "3" | "3rd" | "third" | "three" | "iii" => Some(3),
        _ => None,
    }
}
