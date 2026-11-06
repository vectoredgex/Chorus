//! Confirmation of references read from partial (mid-sentence) transcripts.
//!
//! A live transcript of a few seconds of audio misreads numbers more often
//! than the final one ("John 3, verse 28" a moment before "John 3 verse 16").
//! A reference from a partial is therefore released only when the next
//! partial of the same utterance reads it the same way. A final transcript
//! is trusted at once.

use crate::types::Detection;

/// Holds the references seen in the previous partial transcript.
#[derive(Debug, Default)]
pub struct PartialConfirmer {
    seen: Vec<(i32, i32, i32)>,
}

impl PartialConfirmer {
    pub const fn new() -> Self {
        Self { seen: Vec::new() }
    }

    /// Filter detections from one transcript event.
    pub fn filter(&mut self, results: Vec<Detection>, is_final: bool) -> Vec<Detection> {
        if is_final {
            self.seen.clear();
            return results;
        }
        let key = |d: &Detection| (d.verse_ref.book_number, d.verse_ref.chapter, d.verse_ref.verse_start);
        let current: Vec<(i32, i32, i32)> = results.iter().map(key).collect();
        let confirmed: Vec<Detection> = results.into_iter().filter(|d| self.seen.contains(&key(d))).collect();
        self.seen = current;
        confirmed
    }
}

#[cfg(test)]
mod tests {
    use super::PartialConfirmer;
    use crate::types::{Detection, DetectionSource, VerseRef};

    fn det(chapter: i32, verse: i32) -> Detection {
        Detection {
            verse_ref: VerseRef {
                book_number: 43,
                book_name: "John".into(),
                chapter,
                verse_start: verse,
                verse_end: None,
            },
            verse_id: None,
            confidence: 1.0,
            source: DetectionSource::DirectReference,
            transcript_snippet: String::new(),
            detected_at: 0,
            is_chapter_only: false,
        }
    }

    #[test]
    fn partial_needs_a_second_agreeing_partial() {
        let mut c = PartialConfirmer::new();
        assert!(c.filter(vec![det(3, 28)], false).is_empty()); // misheard, held
        assert!(c.filter(vec![det(3, 16)], false).is_empty()); // different, held
        assert_eq!(c.filter(vec![det(3, 16)], false).len(), 1); // agrees, shown
    }

    #[test]
    fn final_is_trusted_and_resets() {
        let mut c = PartialConfirmer::new();
        assert!(c.filter(vec![det(3, 16)], false).is_empty());
        assert_eq!(c.filter(vec![det(3, 16)], true).len(), 1);
        assert!(c.filter(vec![det(3, 16)], false).is_empty()); // new utterance
    }
}
