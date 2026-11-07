//! Malayalam support: rewrites a Malayalam transcript into the English
//! word order the reference grammar already understands.
//!
//! "യോഹന്നാൻ മൂന്നാം അദ്ധ്യായം പതിനാറാം വാക്യം" (John, third chapter,
//! sixteenth verse) becomes "john chapter 3 verse 16". Malayalam puts the
//! chapter/verse word after its number, so those words are moved in front.
//!
//! Handled:
//! - book names as in the 1910 Sathyavedapusthakam, common modern spellings,
//!   and case endings ("റോമരിൽ", "യോഹന്നാന്റെ", "സങ്കീർത്തനത്തിൽ")
//! - number words 1-176, cardinal and ordinal ("പതിനാറ്", "പതിനാറാം",
//!   "പതിനാറാമത്തെ", "ഇരുപത്തിമൂന്ന്", "നൂറ്റിപ്പത്തൊമ്പത്"), digits,
//!   Malayalam digits, and "16-ാം"
//! - അദ്ധ്യായം / അധ്യായം (chapter), വാക്യം / വചനം (verse)
//! - ranges: "നാലു മുതൽ ഏഴു വരെ" (4 to 7), "എട്ടും ഒമ്പതും" (8 and 9)
//! - numbered books: "ഒന്നാം കൊരിന്ത്യർ", "1 യോഹന്നാൻ",
//!   "കൊരിന്ത്യർക്ക് എഴുതിയ ഒന്നാം ലേഖനം"
//! - English words pass through untouched, so mixed speech still works.

use std::collections::HashMap;
use std::sync::OnceLock;

/// True when the text contains Malayalam script.
pub fn has_malayalam(text: &str) -> bool {
    text.chars().any(is_ml)
}

fn is_ml(c: char) -> bool {
    ('\u{0D00}'..='\u{0D7F}').contains(&c)
}

// ---------------------------------------------------------------------------
// Spelling normalization
// ---------------------------------------------------------------------------

const VIRAMA: char = '\u{0D4D}';

/// One canonical spelling: composed vowel signs, atomic chillu letters
/// written as consonant + virama, no joiners, ASCII digits.
pub fn norm(word: &str) -> String {
    let mut s = word
        .replace("\u{0D46}\u{0D3E}", "\u{0D4A}") // െ + ാ -> ൊ
        .replace("\u{0D47}\u{0D3E}", "\u{0D4B}") // േ + ാ -> ോ
        .replace("\u{0D46}\u{0D57}", "\u{0D57}") // െ + ൗ -> ൗ
        .replace('\u{0D4C}', "\u{0D57}"); // ൌ -> ൗ
    s = s.replace(['\u{200C}', '\u{200D}'], "");
    // Speech engines sometimes double a vowel sign ("നാാല").
    for sign in ['\u{0D3E}', '\u{0D3F}', '\u{0D40}', '\u{0D41}', '\u{0D46}', '\u{0D47}'] {
        let two: String = [sign, sign].iter().collect();
        while s.contains(&two) {
            s = s.replace(&two, &sign.to_string());
        }
    }
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '\u{0D7A}' => out.push_str("ണ്"),
            '\u{0D7B}' => out.push_str("ന്"),
            '\u{0D7C}' | '\u{0D4E}' => out.push_str("ര്"),
            '\u{0D7D}' => out.push_str("ല്"),
            '\u{0D7E}' => out.push_str("ള്"),
            '\u{0D7F}' => out.push_str("ക്"),
            '\u{0D66}'..='\u{0D6F}' => {
                out.push(char::from(b'0' + (c as u32 - 0x0D66) as u8));
            }
            _ => out.push(c),
        }
    }
    out
}

/// The part of a word that stays the same when endings are added: drop a
/// final virama, "ു" (half-u) or "ം".
fn stem(word: &str) -> String {
    let w = norm(word);
    w.strip_suffix(VIRAMA)
        .or_else(|| w.strip_suffix('\u{0D41}'))
        .or_else(|| w.strip_suffix('\u{0D02}'))
        .unwrap_or(&w)
        .to_string()
}

// ---------------------------------------------------------------------------
// Books
// ---------------------------------------------------------------------------

/// Malayalam book names -> an English alias the grammar knows. Numbered
/// books map to the family name; "ഒന്നാം"/"1" before them picks the book.
const BOOKS: &[(&str, &[&str])] = &[
    ("genesis", &["ഉല്പത്തി", "ഉൽപ്പത്തി", "ഉത്പത്തി"]),
    ("exodus", &["പുറപ്പാട്"]),
    ("leviticus", &["ലേവ്യപുസ്തകം", "ലേവ്യർ", "ലേവ്യ"]),
    ("numbers", &["സംഖ്യാപുസ്തകം", "സംഖ്യ"]),
    ("deuteronomy", &["ആവർത്തനപുസ്തകം", "ആവർത്തനം"]),
    ("joshua", &["യോശുവ", "യോശുവാ"]),
    ("judges", &["ന്യായാധിപന്മാർ"]),
    ("ruth", &["രൂത്ത്"]),
    ("samuel", &["ശമൂവേൽ", "ശമുവേൽ", "സാമുവേൽ"]),
    ("kings", &["രാജാക്കന്മാർ"]),
    ("chronicles", &["ദിനവൃത്താന്തം"]),
    ("ezra", &["എസ്രാ", "എസ്ര"]),
    ("nehemiah", &["നെഹെമ്യാവ്", "നെഹമ്യാവ്", "നെഹെമ്യാ"]),
    ("esther", &["എസ്ഥേർ", "എസ്തേർ"]),
    ("job", &["ഇയ്യോബ്"]),
    ("psalms", &["സങ്കീർത്തനം", "സങ്കീർത്തനങ്ങൾ"]),
    ("proverbs", &["സദൃശവാക്യങ്ങൾ", "സദൃശവാക്യം"]),
    ("ecclesiastes", &["സഭാപ്രസംഗി"]),
    ("song of solomon", &["ഉത്തമഗീതം", "ഉത്തമഗീതങ്ങൾ"]),
    ("isaiah", &["യെശയ്യാവ്", "യെശയ്യാ", "ഏശയ്യാ", "യെശായാ"]),
    ("jeremiah", &["യിരെമ്യാവ്", "യിരമ്യാവ്", "യിരെമ്യാ", "ജെറമിയ"]),
    ("lamentations", &["വിലാപങ്ങൾ", "വിലാപം"]),
    ("ezekiel", &["യെഹെസ്കേൽ", "യെഹസ്കേൽ", "എസെക്കിയേൽ"]),
    ("daniel", &["ദാനീയേൽ", "ദാനിയേൽ"]),
    ("hosea", &["ഹോശേയ", "ഹോശേയാ"]),
    ("joel", &["യോവേൽ"]),
    ("amos", &["ആമോസ്"]),
    ("obadiah", &["ഓബദ്യാവ്", "ഓബദ്യാ", "ഒബദ്യാവ്"]),
    ("jonah", &["യോനാ", "യോന"]),
    ("micah", &["മീഖാ", "മിഖാ"]),
    ("nahum", &["നഹൂം"]),
    ("habakkuk", &["ഹബക്കൂക്ക്", "ഹബക്കൂക്"]),
    ("zephaniah", &["സെഫന്യാവ്", "സെഫന്യാ"]),
    ("haggai", &["ഹഗ്ഗായി", "ഹഗ്ഗായ്"]),
    ("zechariah", &["സെഖര്യാവ്", "സെഖര്യാ", "സഖറിയാ"]),
    ("malachi", &["മലാഖി"]),
    ("matthew", &["മത്തായി"]),
    ("mark", &["മർക്കൊസ്", "മർക്കോസ്"]),
    ("luke", &["ലൂക്കൊസ്", "ലൂക്കോസ്"]),
    ("john", &["യോഹന്നാൻ"]),
    ("acts", &["പ്രവൃത്തികൾ", "അപ്പൊസ്തലപ്രവൃത്തികൾ", "അപ്പോസ്തലപ്രവൃത്തികൾ", "പ്രവർത്തികൾ"]),
    ("romans", &["റോമർ", "റോമാ"]),
    ("corinthians", &["കൊരിന്ത്യർ", "കൊരിന്ത്യ"]),
    ("galatians", &["ഗലാത്യർ", "ഗലാത്യ"]),
    ("ephesians", &["എഫെസ്യർ", "എഫേസ്യർ", "എഫെസ്യ"]),
    ("philippians", &["ഫിലിപ്പിയർ", "ഫിലിപ്പ്യർ", "ഫിലിപ്പിയ"]),
    ("colossians", &["കൊലൊസ്സ്യർ", "കൊലോസ്യർ", "കൊലോസ്സ്യർ"]),
    ("thessalonians", &["തെസ്സലൊനീക്യർ", "തെസ്സലോനിക്യർ", "തെസലോനിക്യർ"]),
    ("timothy", &["തിമൊഥെയൊസ്", "തിമൊഥെയോസ്", "തിമോത്തിയോസ്"]),
    ("titus", &["തീത്തൊസ്", "തീത്തോസ്"]),
    ("philemon", &["ഫിലേമോൻ"]),
    ("hebrews", &["എബ്രായർ", "എബ്രായ"]),
    ("james", &["യാക്കോബ്"]),
    ("peter", &["പത്രൊസ്", "പത്രോസ്"]),
    ("jude", &["യൂദാ"]),
    ("revelation", &["വെളിപ്പാട്", "വെളിപാട്"]),
];

/// English book names as a Malayalam speech engine writes them when a
/// preacher says them in English ("ജോൺ ത്രീ സിക്സ്റ്റീൻ").
const ENGLISH_BOOKS: &[(&str, &[&str])] = &[
    ("genesis", &["ജെനസിസ്", "ജനസിസ്", "ജെനിസിസ്"]),
    ("exodus", &["എക്സോഡസ്", "എക്സൊഡസ്"]),
    ("leviticus", &["ലെവിറ്റിക്കസ്", "ലേവിറ്റിക്കസ്"]),
    ("numbers", &["നമ്പേഴ്സ്", "നമ്പർസ്"]),
    ("deuteronomy", &["ഡ്യൂട്ടറോണമി", "ഡ്യൂറ്ററോണമി", "ഡിയൂട്ടറോണമി"]),
    ("joshua", &["ജോഷ്വ", "ജോഷ്വാ", "ജോഷുവ"]),
    ("judges", &["ജഡ്ജസ്"]),
    ("ruth", &["റൂത്ത്"]),
    ("samuel", &["സാമുവൽ", "സാമുവേൽ", "സാമുവെൽ"]),
    ("kings", &["കിങ്സ്", "കിംഗ്സ്"]),
    ("chronicles", &["ക്രോണിക്കിൾസ്", "ക്രോണിക്കിൾ"]),
    ("nehemiah", &["നെഹമിയ", "നെഹെമിയ"]),
    ("esther", &["എസ്തർ", "എസ്ഥർ"]),
    ("psalms", &["സാംസ്", "സാം", "സാംസ്‌", "സോം"]),
    ("proverbs", &["പ്രോവെർബ്സ്", "പ്രോവർബ്സ്", "പ്രൊവെർബ്സ്"]),
    ("ecclesiastes", &["എക്ലീസിയാസ്റ്റസ്", "എക്ലീസിയസ്റ്റസ്"]),
    ("isaiah", &["ഐസയ", "ഐസയ്യ", "ഐസൈയ", "ഐസായ", "ഇസയ്യ"]),
    ("jeremiah", &["ജെറമിയ", "ജെറമിയാ", "ജെറെമിയ"]),
    ("lamentations", &["ലാമെന്റേഷൻസ്"]),
    ("ezekiel", &["എസീക്കിയൽ", "എസെക്കിയൽ", "ഇസീക്കിയൽ"]),
    ("daniel", &["ഡാനിയേൽ", "ഡാനിയൽ"]),
    ("hosea", &["ഹോസിയ", "ഹോസിയാ"]),
    ("joel", &["ജോയൽ", "ജോവൽ"]),
    ("obadiah", &["ഒബദിയ", "ഒബദിയാ"]),
    ("jonah", &["ജോണ", "ജോനാ"]),
    ("micah", &["മൈക്ക", "മൈക്കാ"]),
    ("habakkuk", &["ഹബക്കുക്ക്", "ഹബക്കുക്"]),
    ("zephaniah", &["സെഫനിയ", "സെഫനിയാ"]),
    ("haggai", &["ഹഗായ്", "ഹഗായി"]),
    ("zechariah", &["സെക്കറിയ", "സക്കറിയ"]),
    ("malachi", &["മലാക്കി", "മലാഖൈ"]),
    ("matthew", &["മാത്യു", "മാത്യൂ", "മാത്ത്യു"]),
    ("mark", &["മാർക്ക്", "മാർക്"]),
    ("luke", &["ലൂക്ക്", "ലൂക്"]),
    ("john", &["ജോൺ", "ജോണ്"]),
    ("acts", &["ആക്ട്സ്", "ആക്ട്സ്‌"]),
    ("romans", &["റോമൻസ്", "റോമൻസ്‌"]),
    ("corinthians", &["കൊരിന്ത്യൻസ്", "കൊറിന്ത്യൻസ്", "കൊരിന്തിയൻസ്", "കൊറിന്തിയൻസ്"]),
    ("galatians", &["ഗലേഷ്യൻസ്", "ഗലാത്യൻസ്", "ഗലേഷ്യൻസ്"]),
    ("ephesians", &["എഫേഷ്യൻസ്", "എഫീഷ്യൻസ്", "എഫെഷ്യൻസ്"]),
    ("philippians", &["ഫിലിപ്പിയൻസ്", "ഫിലിപ്പിയൻസ്‌", "ഫിലിപ്പിൻസ്"]),
    ("colossians", &["കൊളോസിയൻസ്", "കൊലോസിയൻസ്", "കൊളൊസ്സിയൻസ്"]),
    ("thessalonians", &["തെസലോനിയൻസ്", "തെസ്സലോനിയൻസ്", "തെസ്സലൊനിയൻസ്"]),
    ("timothy", &["തിമോത്തി", "ടിമോത്തി", "തിമൊത്തി"]),
    ("titus", &["ടൈറ്റസ്"]),
    ("philemon", &["ഫിലെമോൻ", "ഫിലേമൻ"]),
    ("hebrews", &["ഹീബ്രൂസ്", "ഹീബ്രൂ"]),
    ("james", &["ജെയിംസ്", "ജെയിംസ്‌", "ജേംസ്"]),
    ("peter", &["പീറ്റർ"]),
    ("jude", &["ജൂഡ്"]),
    ("revelation", &["റെവലേഷൻ", "റവലേഷൻ", "റെവലേഷൻസ്", "റിവലേഷൻ"]),
];

/// English number and reference words in Malayalam letters.
const ENGLISH_WORDS: &[(&str, &[&str])] = &[
    ("one", &["വൺ"]), ("two", &["ടു", "ടൂ"]), ("three", &["ത്രീ"]), ("four", &["ഫോർ"]),
    ("five", &["ഫൈവ്"]), ("six", &["സിക്സ്"]), ("seven", &["സെവൻ"]),
    ("eight", &["എയ്റ്റ്", "എയ്ട്ട്", "എയിറ്റ്", "എട്ട്‌"]), ("nine", &["നൈൻ"]),
    ("ten", &["ടെൻ"]), ("eleven", &["ഇലവൻ"]), ("twelve", &["ട്വൽവ്", "ട്വെൽവ്"]),
    ("thirteen", &["തെർട്ടീൻ", "തേർട്ടീൻ", "തർട്ടീൻ"]), ("fourteen", &["ഫോർട്ടീൻ"]),
    ("fifteen", &["ഫിഫ്റ്റീൻ"]), ("sixteen", &["സിക്സ്റ്റീൻ"]), ("seventeen", &["സെവന്റീൻ"]),
    ("eighteen", &["എയ്റ്റീൻ", "എയിറ്റീൻ"]), ("nineteen", &["നൈന്റീൻ"]),
    ("twenty", &["ട്വന്റി", "ട്വെന്റി"]), ("thirty", &["തെർട്ടി", "തേർട്ടി", "തർട്ടി"]),
    ("forty", &["ഫോർട്ടി"]), ("fifty", &["ഫിഫ്റ്റി"]), ("sixty", &["സിക്സ്റ്റി"]),
    ("seventy", &["സെവന്റി"]), ("eighty", &["എയ്റ്റി", "എയിറ്റി"]), ("ninety", &["നൈന്റി"]),
    ("hundred", &["ഹൺഡ്രഡ്", "ഹണ്ട്രഡ്"]),
    ("first", &["ഫസ്റ്റ്"]), ("second", &["സെക്കൻഡ്", "സെക്കന്റ്"]), ("third", &["തേർഡ്", "തേഡ്"]),
    ("verse", &["വേഴ്സ്", "വെർസ്", "വേർസ്", "വേഴ്‌സ്"]), ("verses", &["വേഴ്സസ്", "വെർസസ്"]),
    ("to", &["ടു"]), ("and", &["ആൻഡ്", "ആന്റ്"]), ("of", &["ഓഫ്"]), ("the", &["ദ"]),
    ("song", &["സോങ്", "സോംഗ്"]), ("solomon", &["സോളമൻ", "സോളമന്റെ"]),
];

fn english_word(word: &str) -> Option<&'static str> {
    static M: OnceLock<HashMap<String, &'static str>> = OnceLock::new();
    let m = M.get_or_init(|| {
        let mut m = HashMap::new();
        for (en, forms) in ENGLISH_WORDS {
            for f in *forms {
                // "ടു" is "two" unless it sits between numbers; see to_english.
                m.entry(norm(f)).or_insert(*en);
            }
        }
        m
    });
    m.get(word).copied()
}

/// Case endings and particles that can follow a name or number stem
/// (written in normalized spelling: chillu = consonant + virama).
const ENDINGS: &[&str] = &[
    "", "്", "ു", "ം", "ും", "ില്", "ിലെ", "ിന്", "ിന്റെ", "്റെ", "ുടെ", "ക്ക്", "്ക്ക്", "ക്കു",
    "്ക്കു", "ക്കുള്ള", "്ക്കുള്ള", "ത്തില്", "ത്തിലെ", "ത്തിന്റെ", "ത്തിലും", "യില്", "യിലെ",
    "യുടെ", "യും", "ങ്ങള്", "ങ്ങളില്", "ങ്ങളിലെ", "ങ്ങളുടെ", "ങ്ങളും", "ില്നിന്ന്",
    "ത്തില്നിന്ന്", "യില്നിന്ന്",
];

fn book_table() -> &'static Vec<(String, &'static str)> {
    static T: OnceLock<Vec<(String, &'static str)>> = OnceLock::new();
    T.get_or_init(|| {
        let mut v: Vec<(String, &'static str)> = BOOKS
            .iter()
            .chain(ENGLISH_BOOKS)
            .flat_map(|(en, forms)| forms.iter().map(move |f| (stem(f), *en)))
            // Very short stems ("സാ" from "സാം") would match too much.
            .filter(|(st, _)| st.chars().count() >= 3)
            .collect();
        // Longest stem first, so "ലേവ്യപുസ്തക" wins over "ലേവ്യ".
        v.sort_by(|a, b| b.0.chars().count().cmp(&a.0.chars().count()));
        v
    })
}

fn match_book(word: &str) -> Option<&'static str> {
    // Short English names match only as written: "സാം" (Psalm).
    for (en, forms) in ENGLISH_BOOKS {
        if forms.iter().any(|f| norm(f) == word) {
            return Some(en);
        }
    }
    book_table().iter().find_map(|(s, en)| {
        let rest = word.strip_prefix(s.as_str())?;
        ENDINGS.contains(&rest).then_some(*en)
    })
}

// ---------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------

/// Stems of 1-19 (normalized, without the final virama / "ു").
const SMALL: &[(u32, &[&str])] = &[
    (1, &["ഒന്ന"]), (2, &["രണ്ട"]), (3, &["മൂന്ന"]), (4, &["നാല"]), (5, &["അഞ്ച"]),
    (6, &["ആറ"]), (7, &["ഏഴ"]), (8, &["എട്ട"]), (9, &["ഒമ്പത", "ഒന്പത"]),
    (10, &["പത്ത"]), (11, &["പതിനൊന്ന"]), (12, &["പന്ത്രണ്ട"]), (13, &["പതിമൂന്ന"]),
    (14, &["പതിനാല"]), (15, &["പതിനഞ്ച"]), (16, &["പതിനാറ"]), (17, &["പതിനേഴ"]),
    (18, &["പതിനെട്ട"]), (19, &["പത്തൊമ്പത", "പത്തൊന്പത"]),
];

/// Tens: (value, stem of the round number, prefixes used in compounds).
const TENS: &[(u32, &[&str], &[&str])] = &[
    (20, &["ഇരുപത"], &["ഇരുപത്ത"]),
    (30, &["മുപ്പത"], &["മുപ്പത്ത"]),
    (40, &["നാല്പത", "നാല്പ്പത"], &["നാല്പത്ത", "നാല്പ്പത്ത"]),
    (50, &["അമ്പത", "അന്പത"], &["അമ്പത്ത", "അന്പത്ത"]),
    (60, &["അറുപത"], &["അറുപത്ത"]),
    (70, &["എഴുപത"], &["എഴുപത്ത"]),
    (80, &["എണ്പത"], &["എണ്പത്ത"]),
    (90, &["തൊണ്ണൂറ"], &["തൊണ്ണൂറ്റ"]),
];

/// Vowel letter -> the sign it becomes after a consonant.
fn vowel_sign(c: char) -> Option<&'static str> {
    Some(match c {
        'അ' => "",
        'ആ' => "\u{0D3E}",
        'ഇ' => "\u{0D3F}",
        'ഈ' => "\u{0D40}",
        'ഉ' => "\u{0D41}",
        'എ' => "\u{0D46}",
        'ഏ' => "\u{0D47}",
        'ഒ' => "\u{0D4A}",
        'ഓ' => "\u{0D4B}",
        _ => return None,
    })
}

/// Join a compound prefix ("ഇരുപത്ത") and a number stem ("ഒന്ന"), in
/// every way it is written: ഇരുപത്തൊന്ന, ഇരുപത്തിയൊന്ന, ഇരുപത്തിഒന്ന,
/// ഇരുപത്തിമൂന്ന, and with a doubled consonant (നൂറ്റിപ്പത്ത).
fn join(prefix: &str, s: &str) -> Vec<String> {
    let first = s.chars().next().unwrap();
    let rest: String = s.chars().skip(1).collect();
    let mut v = vec![format!("{prefix}\u{0D3F}{s}")];
    if let Some(sign) = vowel_sign(first) {
        v.push(format!("{prefix}{sign}{rest}"));
        v.push(format!("{prefix}\u{0D3F}യ{sign}{rest}"));
        // The vowel swallowed by a speech engine: "ഇരുപത്തിട്ട്" (28).
        if !rest.is_empty() {
            v.push(format!("{prefix}\u{0D3F}{rest}"));
        }
    } else if matches!(first, 'ക' | 'ച' | 'ട' | 'ത' | 'പ') && !rest.starts_with(VIRAMA) {
        v.push(format!("{prefix}\u{0D3F}{first}{VIRAMA}{s}"));
    }
    v
}

fn number_stems() -> &'static HashMap<String, u32> {
    static M: OnceLock<HashMap<String, u32>> = OnceLock::new();
    M.get_or_init(|| {
        let mut m: HashMap<String, u32> = HashMap::new();
        for (n, stems) in SMALL {
            for s in *stems {
                m.insert(s.to_string(), *n);
            }
        }
        for (t, round, prefixes) in TENS {
            for s in *round {
                m.insert(s.to_string(), *t);
            }
            for p in *prefixes {
                for (u, stems) in SMALL.iter().take(9) {
                    for s in *stems {
                        for j in join(p, s) {
                            m.insert(j, t + u);
                        }
                    }
                }
            }
        }
        m.insert("നൂറ".into(), 100);
        let below_100: Vec<(String, u32)> =
            m.iter().filter(|(_, v)| **v < 100).map(|(k, v)| (k.clone(), *v)).collect();
        for (s, n) in below_100 {
            if n + 100 > 176 {
                continue;
            }
            for j in join("നൂറ്റ", &s) {
                m.insert(j, 100 + n);
            }
        }
        m
    })
}

const CARDINAL_ENDINGS: &[&str] = &[
    "", "്", "ു", "ും", "ില്", "ിലെ", "ിന്റെ", "ിലും", "ിലാണ്", "ില്നിന്ന്",
];
const ORDINAL_ENDINGS: &[&str] = &[
    "ാം", "ാമ", "ാമത്തെ", "ാമത്തേ", "ാമത്", "ാമതു", "ാമത്തില്", "ാമത്തിലെ", "ാമതായി", "ാമത്തേത്",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MlNum {
    value: u32,
    ordinal: bool,
    /// Ends in "-ഉം" ("and"): "എട്ടും ഒമ്പതും" = 8 and 9.
    and: bool,
}

fn match_number(word: &str) -> Option<MlNum> {
    // "16", "16-ാം", "16ാം", "3ൽ"
    let digits: String = word.chars().take_while(|c| c.is_ascii_digit()).collect();
    if !digits.is_empty() {
        let rest = word[digits.len()..].trim_start_matches('-');
        if !rest.is_empty() && !rest.chars().all(is_ml) {
            return None;
        }
        let value = digits.parse().ok()?;
        let ordinal = rest.starts_with('\u{0D3E}') || rest.starts_with('ആ');
        return Some(MlNum { value, ordinal, and: rest.ends_with("ും") });
    }
    let stems = number_stems();
    for (endings, ordinal) in [(ORDINAL_ENDINGS, true), (CARDINAL_ENDINGS, false)] {
        for e in endings {
            if let Some(s) = word.strip_suffix(e) {
                if let Some(&value) = stems.get(s) {
                    return Some(MlNum { value, ordinal, and: *e == "ും" });
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Other words
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(MlNum),
    /// A Malayalam chapter/verse word; it follows its number.
    Marker(&'static str),
    /// Words that become English as they are ("john", "to", "gospel"...).
    Word(String),
    /// A Malayalam word with no meaning for references.
    Other,
}

fn match_word(word: &str) -> Option<Tok> {
    const CHAPTER: &[&str] =
        &["അദ്ധ്യായ", "അധ്യായ", "അദ്ധ്യയ", "അദ്യായ", "അത്യായ", "അദ്ദ്യായ", "അധ്യയ", "ചാപ്റ്റര"];
    const VERSE: &[&str] = &["വാക്യ", "വചന", "വേഴ്സ"];
    let stem_end = |stems: &[&str]| {
        stems.iter().any(|s| word.strip_prefix(s).is_some_and(|r| ENDINGS.contains(&r)))
    };
    if stem_end(CHAPTER) {
        return Some(Tok::Marker("chapter"));
    }
    if stem_end(VERSE) {
        let plural = word.contains("ങ്ങള്");
        return Some(Tok::Marker(if plural { "verses" } else { "verse" }));
    }
    let w = match word {
        "മുതല്" | "തൊട്ട്" => "to",
        "വരെ" | "വരെയുള്ള" | "വരെയും" | "എഴുതിയ" | "എഴുതിയിരിക്കുന്ന" | "എഴുതപ്പെട്ട" => "",
        "അടുത്ത" => "next",
        "സുവിശേഷം" | "സുവിശേഷത്തില്" | "സുവിശേഷത്തിലെ" => "gospel",
        "ലേഖനം" | "ലേഖനത്തില്" | "ലേഖനത്തിലെ" => "letter",
        "പുസ്തകം" | "പുസ്തകത്തില്" | "പുസ്തകത്തിലെ" => "book",
        "പ്രവാചകന്" | "പ്രവാചകന്റെ" => "",
        _ => return None,
    };
    Some(Tok::Word(w.to_string()))
}

fn classify(word: &str) -> Vec<(String, Tok)> {
    if !word.chars().any(is_ml) {
        return vec![(word.to_string(), Tok::Word(word.to_string()))];
    }
    let w = norm(word);
    if let Some(n) = match_number(&w) {
        return vec![(w, Tok::Num(n))];
    }
    if let Some(t) = match_word(&w) {
        return vec![(w, t)];
    }
    if let Some(en) = match_book(&w) {
        return vec![(w, Tok::Word(en.to_string()))];
    }
    if let Some(en) = english_word(&w) {
        return vec![(w, Tok::Word(en.to_string()))];
    }
    // A book name run into its number: "റോമരെട്ടാം" = റോമർ എട്ടാം.
    for (stem, en) in book_table() {
        let Some(rest) = w.strip_prefix(stem.as_str()) else { continue };
        let mut chars = rest.chars();
        let Some(first) = chars.next() else { continue };
        let rest = match independent_vowel(first) {
            Some(v) => format!("{v}{}", chars.as_str()),
            None => rest.to_string(),
        };
        if let Some(n) = match_number(&rest) {
            return vec![(stem.clone(), Tok::Word(en.to_string())), (rest, Tok::Num(n))];
        }
    }
    vec![(w, Tok::Other)]
}

fn independent_vowel(sign: char) -> Option<char> {
    Some(match sign {
        '\u{0D3E}' => 'ആ',
        '\u{0D3F}' => 'ഇ',
        '\u{0D40}' => 'ഈ',
        '\u{0D41}' => 'ഉ',
        '\u{0D46}' => 'എ',
        '\u{0D47}' => 'ഏ',
        '\u{0D4A}' => 'ഒ',
        '\u{0D4B}' => 'ഓ',
        _ => return None,
    })
}

/// The sound outline of a word, for matching misheard names: vowels
/// dropped, similar consonants merged (ത/ഥ/ദ/ധ, ശ/ഷ/സ...), doubled
/// consonants written once. "യോഹന്നാൻ" and "യോഹൻാൻ" both give "യഹന".
fn skeleton(word: &str) -> Vec<char> {
    let mut out: Vec<char> = Vec::new();
    for (i, c) in norm(word).chars().enumerate() {
        let k = match c {
            'ക' | 'ഖ' | 'ഗ' | 'ഘ' => 'ക',
            'ച' | 'ഛ' | 'ജ' | 'ഝ' => 'ച',
            'ട' | 'ഠ' | 'ഡ' | 'ഢ' => 'ട',
            'ത' | 'ഥ' | 'ദ' | 'ധ' => 'ത',
            // ബ/വ are often confused ("ബാക്യം" for വാക്യം).
            'പ' | 'ഫ' | 'ബ' | 'ഭ' | 'വ' => 'പ',
            'ം' => 'മ',
            'ശ' | 'ഷ' | 'സ' => 'സ',
            'ള' | 'ഴ' | 'ല' => 'ല',
            'ണ' | 'ന' => 'ന',
            'റ' | 'ര' => 'ര',
            'ങ' | 'ഞ' | 'മ' | 'യ' | 'ഹ' => c,
            // A vowel letter only counts at the start of a word.
            'അ'..='ഔ' if i == 0 => 'അ',
            _ => continue,
        };
        if out.last() != Some(&k) {
            out.push(k);
        }
    }
    out
}

fn char_distance(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            cur.push((prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + usize::from(ca != cb)));
        }
        prev = cur;
    }
    prev[b.len()]
}

/// The book a misheard word most likely is, when one book is clearly
/// closest by sound. Only called where a chapter/verse pattern follows.
fn fuzzy_book(word: &str) -> Option<&'static str> {
    // The word as heard, and without any case ending it may carry.
    let w = norm(word);
    let mut heard: Vec<Vec<char>> = ENDINGS
        .iter()
        .filter(|e| !e.is_empty())
        .filter_map(|e| w.strip_suffix(e))
        .map(skeleton)
        .collect();
    heard.push(skeleton(&w));
    if heard.iter().all(|h| h.len() < 3) {
        return None;
    }
    let sk = heard.last().unwrap().clone();
    let letters: Vec<char> = w.chars().collect();
    // Closest by sound; ties broken by spelling.
    let mut scored: Vec<((usize, usize), &'static str)> = BOOKS
        .iter()
        .map(|(en, forms)| {
            let d = forms
                .iter()
                .map(|f| {
                    let st = stem(f);
                    let fs = skeleton(&st);
                    let sd = heard.iter().map(|h| char_distance(h, &fs)).min().unwrap_or(usize::MAX);
                    let fd = char_distance(&letters, &st.chars().collect::<Vec<_>>());
                    (sd, fd)
                })
                .min()
                .unwrap_or((usize::MAX, usize::MAX));
            (d, *en)
        })
        .collect();
    scored.sort();
    let ((d, _), en) = scored[0];
    let allowed = if sk.len() >= 6 { 2 } else { 1 };
    (d <= allowed && scored[1].0 > scored[0].0).then_some(en)
}

/// A misheard chapter or verse word ("അത്യായം", "ബാക്കിയം"), judged by
/// sound. Only called right after a number.
fn fuzzy_marker(word: &str) -> Option<Tok> {
    let w = norm(word);
    let mut heard: Vec<Vec<char>> = ENDINGS
        .iter()
        .filter(|e| !e.is_empty())
        .filter_map(|e| w.strip_suffix(e))
        .map(skeleton)
        .collect();
    heard.push(skeleton(&w));
    let near = |target: &str| {
        let t = skeleton(target);
        heard.iter().any(|h| h.len() >= 2 && char_distance(h, &t) <= 1)
    };
    if near("അദ്ധ്യായ") {
        return Some(Tok::Marker("chapter"));
    }
    if near("വാക്യ") {
        let plural = w.contains("ങ്ങള്");
        return Some(Tok::Marker(if plural { "verses" } else { "verse" }));
    }
    None
}

/// Split into words; punctuation other than ":", "-", "." stays with the
/// English normalizer, which treats it as a break.
fn split(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        if c.is_whitespace() || matches!(c, ',' | ';' | '!' | '?' | '(' | ')' | '"' | '\u{0964}') {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            continue;
        }
        if c == ':' {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            out.push(":".into());
            continue;
        }
        cur.push(c);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    // "3:16" stays one token for the English grammar; a trailing "." ends a
    // sentence.
    out.into_iter()
        .map(|w| w.trim_end_matches('.').to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Rewrite a Malayalam (or mixed) transcript in the English word order the
/// reference grammar reads.
pub fn to_english(text: &str) -> String {
    let (words, mut toks): (Vec<String>, Vec<Tok>) =
        split(text).iter().flat_map(|w| classify(w)).unzip();

    // A word nobody knows, right before "<number> അദ്ധ്യായം/വാക്യം", is
    // probably a misheard book name: "കുരിന്ന്ത്യ പതിമൂന്നാം അധ്യായം".
    // First, misheard chapter/verse words right after a number.
    for i in 1..toks.len() {
        if toks[i] == Tok::Other && matches!(toks[i - 1], Tok::Num(_)) {
            if let Some(m) = fuzzy_marker(&words[i]) {
                toks[i] = m;
            }
        }
    }
    let mut words = words;
    for i in 0..toks.len() {
        if toks[i] != Tok::Other {
            continue;
        }
        let pattern = |j: usize| {
            matches!(toks.get(j), Some(Tok::Num(_))) && matches!(toks.get(j + 1), Some(Tok::Marker(_)))
        };
        if pattern(i + 1) {
            if let Some(en) = fuzzy_book(&words[i]) {
                toks[i] = Tok::Word(en.to_string());
            }
        } else if matches!(toks.get(i + 1), Some(Tok::Marker(_))) && pattern(i + 2) {
            // A name split in two: "സദരിച്ച വാക്യങ്ങൾ" for സദൃശവാക്യങ്ങൾ.
            let joined = format!("{}{}", words[i], words[i + 1]);
            if let Some(en) = fuzzy_book(&joined) {
                toks[i] = Tok::Word(en.to_string());
                toks[i + 1] = Tok::Word(String::new());
                words[i + 1].clear();
            }
        }
    }

    // An unknown word right before "<number> അദ്ധ്യായം" is a book name we
    // could not make out. The numbers after it belong to that book, so they
    // must not complete the previous reference ("Jeremiah 2:8-9" for a
    // misheard Ephesians 2:8-9). Drop them.
    for i in 0..toks.len() {
        // "X 2 അധ്യായം ..." or, with the chapter word garbled too,
        // "X 1 Y 9 വാക്യം".
        let unknown_book = toks[i] == Tok::Other
            && matches!(toks.get(i + 1), Some(Tok::Num(_)))
            && (matches!(toks.get(i + 2), Some(Tok::Marker("chapter")))
                || (matches!(toks.get(i + 2), Some(Tok::Other))
                    && matches!(toks.get(i + 3), Some(Tok::Num(_)))
                    && matches!(toks.get(i + 4), Some(Tok::Marker(m)) if m.starts_with("verse"))));
        if unknown_book {
            let mut j = i + 1;
            while j < toks.len()
                && (matches!(toks[j], Tok::Num(_) | Tok::Marker(_) | Tok::Other)
                    || matches!(&toks[j], Tok::Word(w) if w == "to" || w == "and" || w.is_empty()))
            {
                toks[j] = Tok::Other;
                j += 1;
            }
        }
    }

    // Join split compounds: "ഇരുപത്തി മൂന്നാം", "ഇരുപത് മൂന്ന്",
    // "നൂറ്റി പത്തൊമ്പത്".
    let mut merged: Vec<Tok> = Vec::with_capacity(toks.len());
    let mut i = 0;
    while i < toks.len() {
        if let (Some(prefix), Some(Tok::Num(b))) = (compound_prefix(&words[i], &toks[i]), toks.get(i + 1)) {
            let (base, limit) = prefix;
            if b.value < limit && !(base == 100 && b.value == 0) {
                merged.push(Tok::Num(MlNum { value: base + b.value, ..*b }));
                i += 2;
                continue;
            }
        }
        merged.push(toks[i].clone());
        i += 1;
    }

    // "കൊരിന്ത്യർക്ക് എഴുതിയ ഒന്നാം ലേഖനം" -> "1 corinthians"
    let mut toks: Vec<Tok> = Vec::with_capacity(merged.len());
    let mut i = 0;
    while i < merged.len() {
        if let Tok::Word(w) = &merged[i] {
            if is_family(w) {
                let mut j = i + 1;
                while matches!(&merged.get(j), Some(Tok::Word(x)) if x.is_empty()) {
                    j += 1;
                }
                if let (Some(Tok::Num(n)), Some(Tok::Word(l))) = (merged.get(j), merged.get(j + 1)) {
                    if n.ordinal && (1..=3).contains(&n.value) && l == "letter" {
                        toks.push(Tok::Word(n.value.to_string()));
                        toks.push(Tok::Word(w.clone()));
                        i = j + 2;
                        continue;
                    }
                }
            }
        }
        toks.push(merged[i].clone());
        i += 1;
    }

    // Insert "and" between "എട്ടും ഒമ്പതും", then move each Malayalam
    // chapter/verse word in front of its number group.
    toks.retain(|t| !matches!(t, Tok::Word(w) if w.is_empty()));
    // "ടു" is both "two" and "to": between two numbers it is "to"
    // ("ഫോർ ടു സെവൻ" = 4 to 7).
    for k in 1..toks.len().saturating_sub(1) {
        let numberish = |t: &Tok| match t {
            Tok::Num(_) => true,
            Tok::Word(w) => crate::numbers::parse(&[w.clone()], 0).is_some(),
            _ => false,
        };
        if toks[k] == Tok::Word("two".into()) && numberish(&toks[k - 1]) && numberish(&toks[k + 1]) {
            toks[k] = Tok::Word("to".into());
        }
    }
    let mut out: Vec<Tok> = Vec::with_capacity(toks.len() + 4);
    for (k, t) in toks.iter().enumerate() {
        out.push(t.clone());
        if let Tok::Num(n) = t {
            if n.and && matches!(toks.get(k + 1), Some(Tok::Num(_))) {
                out.push(Tok::Word("and".into()));
            }
        }
    }
    // "ഇരുപത്തിരണ്ട് ഇരുപത്തിമൂന്ന് വാക്യങ്ങൾ" (22 23 verses) is 22-23.
    let mut k = 2;
    while k < out.len() {
        if let (Tok::Num(a), Tok::Num(b), Tok::Marker("verses")) = (&out[k - 2], &out[k - 1], &out[k]) {
            if b.value == a.value + 1 {
                out.insert(k - 1, Tok::Word("and".into()));
                k += 1;
            }
        }
        k += 1;
    }
    let mut k = 0;
    while k < out.len() {
        if let Tok::Marker(m) = out[k] {
            if let Some(start) = group_start(&out, k) {
                let marker = out.remove(k);
                out.insert(start, marker);
                let _ = m;
            }
        }
        k += 1;
    }

    let mut s = String::with_capacity(text.len());
    for t in &out {
        let w = match t {
            Tok::Num(n) => n.value.to_string(),
            Tok::Marker(m) => m.to_string(),
            Tok::Word(w) if w.is_empty() => continue,
            Tok::Word(w) => w.clone(),
            Tok::Other => "x".into(),
        };
        if !s.is_empty() {
            s.push(' ');
        }
        s.push_str(&w);
    }
    s
}

fn is_family(w: &str) -> bool {
    matches!(w, "samuel" | "kings" | "chronicles" | "corinthians" | "thessalonians" | "timothy" | "peter" | "john")
}

/// A tens word ("ഇരുപത്", "ഇരുപത്തി") or hundred word ("നൂറ്റി") that
/// combines with the next number: (value, the next number must be below).
fn compound_prefix(word: &str, tok: &Tok) -> Option<(u32, u32)> {
    if let Tok::Num(n) = tok {
        if !n.ordinal && !n.and && n.value % 10 == 0 && (20..=90).contains(&n.value) {
            return Some((n.value, 10));
        }
        if !n.ordinal && n.value == 100 {
            return Some((100, 77));
        }
        return None;
    }
    let w = norm(word);
    let w = w.strip_suffix('\u{0D3F}').or_else(|| w.strip_suffix("\u{0D3F}യ"))?;
    if w == "നൂറ്റ" {
        return Some((100, 77));
    }
    TENS.iter().find(|(_, _, p)| p.contains(&w)).map(|(t, _, _)| (*t, 10))
}

/// Start of the number group that ends right before `marker`: one number,
/// or numbers joined by "to" / "and" / "-". `None` if no number precedes.
fn group_start(t: &[Tok], marker: usize) -> Option<usize> {
    let is_num = |i: usize| matches!(t.get(i), Some(Tok::Num(_)));
    let is_join = |i: usize| matches!(t.get(i), Some(Tok::Word(w)) if w == "to" || w == "and" || w == "-");
    let mut start = marker.checked_sub(1)?;
    if !is_num(start) {
        return None;
    }
    while start >= 2 && is_join(start - 1) && is_num(start - 2) {
        start -= 2;
    }
    Some(start)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(w: &str) -> Option<u32> {
        match_number(&norm(w)).map(|n| n.value)
    }

    #[test]
    fn numbers() {
        let cases = [
            ("ഒന്ന്", 1), ("ഒന്നാം", 1), ("രണ്ടാം", 2), ("മൂന്നാം", 3), ("മൂന്ന്", 3), ("നാലു", 4),
            ("നാലാം", 4), ("അഞ്ചാം", 5), ("ആറാം", 6), ("ഏഴു", 7), ("ഏഴാം", 7), ("എട്ടാം", 8),
            ("എട്ടും", 8), ("ഒമ്പതാം", 9), ("ഒൻപതാം", 9), ("പത്താം", 10), ("പത്ത്", 10),
            ("പതിനൊന്നാം", 11), ("പന്ത്രണ്ടാം", 12), ("പതിമൂന്നാം", 13), ("പതിനാലാം", 14),
            ("പതിനഞ്ചാം", 15), ("പതിനാറാം", 16), ("പതിനാറ്", 16), ("പതിനാറാമത്തെ", 16),
            ("പതിനേഴാം", 17), ("പതിനെട്ടാം", 18), ("പത്തൊമ്പതാം", 19), ("ഇരുപതാം", 20),
            ("ഇരുപത്", 20), ("ഇരുപത്തൊന്നാം", 21), ("ഇരുപത്തിയൊന്നാം", 21), ("ഇരുപത്തിരണ്ടാം", 22),
            ("ഇരുപത്തിമൂന്നാം", 23), ("ഇരുപത്തിമൂന്ന്", 23), ("ഇരുപത്തിനാലാം", 24),
            ("ഇരുപത്തഞ്ചാം", 25), ("ഇരുപത്തിയഞ്ചാം", 25), ("ഇരുപത്താറാം", 26), ("ഇരുപത്തിയാറാം", 26),
            ("ഇരുപത്തേഴാം", 27), ("ഇരുപത്തെട്ടാം", 28), ("ഇരുപത്തിയെട്ട്", 28),
            ("ഇരുപത്തൊമ്പതാം", 29), ("മുപ്പതാം", 30), ("മുപ്പത്തൊന്നാം", 31),
            ("മുപ്പത്തിമൂന്നാം", 33), ("നാല്പതാം", 40), ("നാൽപതാം", 40), ("നാല്പത്തൊന്ന്", 41),
            ("അമ്പതാം", 50), ("അൻപത്", 50), ("അറുപത്താറ്", 66), ("എഴുപതാം", 70),
            ("എൺപതാം", 80), ("തൊണ്ണൂറാം", 90), ("തൊണ്ണൂറ്റിയൊന്ന്", 91), ("നൂറാം", 100),
            ("നൂറ്റിയൊന്നാം", 101), ("നൂറ്റിപ്പത്തൊമ്പതാം", 119), ("നൂറ്റിപത്തൊമ്പത്", 119),
            ("നൂറ്റിയഞ്ചാം", 105), ("നൂറ്റമ്പതാം", 150), ("നൂറ്റിയമ്പത്", 150),
            ("16", 16), ("16-ാം", 16), ("൧൬", 16), ("3ൽ", 3), ("പതിനാറിൽ", 16),
        ];
        let mut bad = Vec::new();
        for (w, v) in cases {
            if n(w) != Some(v) {
                bad.push(format!("{w}: {:?} (want {v})", n(w)));
            }
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
        assert_eq!(n("സ്നേഹം"), None);
        assert_eq!(n("വാക്യം"), None);
    }

    #[test]
    fn books() {
        let cases = [
            ("യോഹന്നാൻ", "john"), ("യോഹന്നാന്റെ", "john"), ("യോഹന്നാനിൽ", "john"),
            ("റോമർ", "romans"), ("റോമരിൽ", "romans"), ("റോമർക്ക്", "romans"),
            ("സങ്കീർത്തനം", "psalms"), ("സങ്കീർത്തനങ്ങൾ", "psalms"), ("സങ്കീർത്തനത്തിൽ", "psalms"),
            ("മത്തായി", "matthew"), ("മത്തായിയിൽ", "matthew"), ("ലൂക്കോസ്", "luke"),
            ("ലേവ്യപുസ്തകം", "leviticus"), ("സദൃശവാക്യങ്ങൾ", "proverbs"),
            ("യെശയ്യാവ്", "isaiah"), ("യെശയ്യാവു", "isaiah"), ("വെളിപ്പാട്", "revelation"),
            ("കൊരിന്ത്യർ", "corinthians"), ("ഫിലിപ്പിയർക്ക്", "philippians"),
        ];
        for (w, en) in cases {
            assert_eq!(match_book(&norm(w)), Some(en), "{w}");
        }
        assert_eq!(match_book(&norm("സ്നേഹം")), None);
        assert_eq!(match_book(&norm("വാക്യം")), None);
    }

    #[test]
    fn rewrite() {
        let cases = [
            ("യോഹന്നാൻ മൂന്നാം അദ്ധ്യായം പതിനാറാം വാക്യം നമുക്ക് വായിക്കാം.",
             "john chapter 3 verse 16 x x"),
            ("ഒന്നാം കൊരിന്ത്യർ പതിമൂന്നാം അദ്ധ്യായം നാലു മുതൽ ഏഴു വരെയുള്ള വാക്യങ്ങൾ.",
             "1 corinthians chapter 13 verses 4 to 7"),
            ("എഫെസ്യർ രണ്ടാം അദ്ധ്യായം എട്ടും ഒമ്പതും വാക്യങ്ങൾ.",
             "ephesians chapter 2 verses 8 and 9"),
            ("കൊരിന്ത്യർക്ക് എഴുതിയ ഒന്നാം ലേഖനം പതിമൂന്നാം അദ്ധ്യായം",
             "1 corinthians chapter 13"),
            ("സങ്കീർത്തനം ഇരുപത്തി മൂന്നാം അദ്ധ്യായം", "psalms chapter 23"),
            ("യോഹന്നാൻ 3:16", "john 3 : 16"),
            // English said inside Malayalam speech, written in Malayalam letters.
            ("ജോൺ ത്രീ സിക്സ്റ്റീൻ", "john three sixteen"),
            ("നമുക്ക് റോമൻസ് ചാപ്റ്റർ എയ്റ്റ് വേഴ്സ് ട്വന്റി എയ്റ്റ് വായിക്കാം",
             "x romans chapter eight verse twenty eight x"),
            ("ഫസ്റ്റ് കൊരിന്ത്യൻസ് തേർട്ടീൻ", "first corinthians thirteen"),
            ("സാം ട്വന്റി ത്രീ", "psalms twenty three"),
            // What IndicConformer actually wrote for the test sermon.
            ("യോഹൻാൻ മൂന്നാം അധ്യായം പതിനാറാം വാക്യം", "john chapter 3 verse 16"),
            ("റോമരെട്ടാം അത്യായം ഇരുപത്തെട്ടാം വാക്യം", "romans chapter 8 verse 28"),
            ("ഒന്നാം കുരിന്ന്ത്യ പതിമൂന്നാം അധ്യായം നാാല മുതൽ ഏഴ് വരെയുള്ള വാക്യങ്ങൾ",
             "1 corinthians chapter 13 verses 4 to 7"),
            ("യഏശിയാവ് നാൽപ്പതാം അദ്ധ്യായ മുപ്പത്തൊന്നാം വാക്യം", "isaiah chapter 40 verse 31"),
            ("രണ്ടാം തിമധേോസ് മൂന്നാം അധ്യായം പതിനാറാം വാക്യം", "2 timothy chapter 3 verse 16"),
        ];
        for (ml, en) in cases {
            assert_eq!(to_english(ml), en, "{ml}");
        }
    }
}
