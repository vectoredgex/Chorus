# chorus-refparse

The Bible-reference engine for Chorus. It reads live sermon transcripts and
finds every Bible reference the preacher speaks, with a confidence level for
each one. Pure Rust, no dependencies, so it drops into the Chorus (Rhema)
Tauri backend as a crate under `src-tauri/crates/`.

## What it understands

- Written citations: `John 3:16`, `1 Cor 10:13`, `Proverbs 3:5-6`, `John 3.16`
- Spoken numbers: "John three sixteen", "Psalm one hundred and nineteen verse
  one hundred and five", "Romans eight, twenty-eight to thirty-nine"
- Chapter/verse words: "Romans chapter 8 at verse 28", "verses 3 to 12",
  "verses eight and nine"
- Numbered books: "first Corinthians", "2nd Timothy", "third John"
- One-chapter books: "Jude 24", "Philemon verse 6"
- "Of" phrasing: "the third chapter of John", "verse 16 of John 3",
  "the twenty-third psalm"
- Book name variants: "John's gospel", "the gospel according to Luke",
  "Acts of the Apostles", "Revelations", "Song of Songs"
- Common speech-to-text mistakes: "Philippines", "Mathew", "Ezekial",
  "Isiah", "Ecclesiastics", "Habakuk", "Zachariah"
- Half verses: "verses 4 through 8a"
- Follow-ups using the last reference: "look at verse 28", "the next verse",
  "in chapter four verse twenty four", "the first verse"

## Malayalam

Malayalam transcripts (any text containing Malayalam script) are rewritten
into English word order first (`src/malayalam.rs`), so the same grammar
handles them: "യോഹന്നാൻ മൂന്നാം അദ്ധ്യായം പതിനാറാം വാക്യം" → John 3:16.
It knows the book names of the 1910 Sathyavedapusthakam (and modern
spellings, with case endings), number words 1-176, അദ്ധ്യായം/വാക്യം,
ranges ("നാലു മുതൽ ഏഴു വരെ", "എട്ടും ഒമ്പതും"), numbered books, and
recovers words the speech engine mishears by their sound. A book name it
cannot make out never borrows the previous book. Tests: `tests/cases_ml.txt`.

## What it refuses

- Everyday uses of book names: "mark my words", "my job is…", "James and
  John were brothers", "Peter denied him three times"
- Chapters that don't exist: `John 22:1`, `Psalm 151`, `Jude 2:1`
- Follow-ups with no earlier reference, or after a low-confidence one

## Confidence

| Level  | Meaning | Suggested behaviour |
|--------|---------|---------------------|
| High   | Book, chapter and verse clearly spoken | Can go on screen automatically |
| Medium | Likely (chapter only, or a follow-up like "verse 17") | Operator confirms |
| Low    | Weak evidence ("Mark 3" with nothing else) | Show to operator only |

Only Medium and High update the context used for follow-ups.

## Usage

```rust
use chorus_refparse::{Detector, Confidence};

let mut det = Detector::new(); // one per service
for d in det.detect("turn with me to Romans chapter eight verse twenty eight") {
    println!("{} {:?}", d.reference, d.confidence); // Romans 8:28 High
}
```

If the operator manually puts a verse on screen, call
`det.set_last(Some(reference))` so follow-ups resolve against it.

## Tests

`tests/cases.txt` holds 210+ real sermon phrases with the exact expected
result, including false-trigger traps and multi-line sermon context.
Add any phrase that goes wrong in a real service there, then run:

```
cargo test
```

To try your own phrases, one per line (a blank line resets context):

```
cargo run --example probe < phrases.txt
```

## Not yet covered

- Verse counts per chapter: invalid chapters are rejected, but a verse
  beyond the end of a real chapter (e.g. `John 3:99`) is only rejected if it
  is above 176. In Chorus this is checked against the Bible database.
- Cross-chapter ranges ("John 3:16 to 4:2").
- Malayalam (Phase 2).
