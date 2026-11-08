//! Clean-up of Whisper output.
//!
//! Whisper sometimes loops and repeats what it just wrote ("Romans chapter
//! 8 verse 28, Romans chapter 8 verse 28, Romans chapter ..."). This removes
//! a run of three or more words that immediately repeats, keeping the first
//! copy. Shorter repeats ("holy, holy, holy", "amen amen") are left alone
//! because people really say them.

/// Lowercase and strip punctuation for comparing words.
fn norm(w: &str) -> String {
    w.chars()
        // Malayalam vowel signs are not "alphanumeric" but are part of the word.
        .filter(|c| c.is_alphanumeric() || ('\u{0D00}'..='\u{0D7F}').contains(c))
        .flat_map(char::to_lowercase)
        .collect()
}

/// Remove immediately repeated phrases of at least three words.
pub fn remove_repeats(text: &str) -> String {
    const MIN_WORDS: usize = 3;
    let words: Vec<&str> = text.split_whitespace().collect();
    let keys: Vec<String> = words.iter().map(|w| norm(w)).collect();
    let mut keep = vec![true; words.len()];

    let mut i = 0;
    while i < words.len() {
        let mut advanced = false;
        // Longest repeat first, so a whole repeated sentence goes at once.
        let max_len = (words.len() - i) / 2;
        for len in (MIN_WORDS..=max_len).rev() {
            if keys[i..i + len] == keys[i + len..i + 2 * len] {
                // Drop every further copy that follows directly.
                let mut j = i + len;
                while j + len <= words.len() && keys[i..i + len] == keys[j..j + len] {
                    keep[j..j + len].iter_mut().for_each(|k| *k = false);
                    j += len;
                }
                // A trailing partial copy ("... Romans chapter") goes too.
                let tail = words.len() - j;
                if tail > 0 && tail < len && keys[j..] == keys[i..i + tail] {
                    keep[j..].iter_mut().for_each(|k| *k = false);
                    j = words.len();
                }
                i = j;
                advanced = true;
                break;
            }
        }
        if !advanced {
            i += 1;
        }
    }

    let kept: Vec<&str> = words
        .iter()
        .zip(&keep)
        .filter_map(|(w, k)| k.then_some(*w))
        .collect();
    let mut out = kept.join(" ");
    // Repeats were often separated by a comma; drop one left dangling.
    while out.ends_with(',') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::remove_repeats;

    #[test]
    fn removes_loops() {
        assert_eq!(
            remove_repeats("Romans chapter 8 verse 28, Romans chapter 8 verse 28, Romans chapter 8 verse 28, Romans chapter"),
            "Romans chapter 8 verse 28,"
                .trim_end_matches(',')
        );
        assert_eq!(
            remove_repeats("1 Corinthians 13.4 1 Corinthians 13.4 1 Corinthians 13.4"),
            "1 Corinthians 13.4"
        );
        assert_eq!(
            remove_repeats("So we look at Ephesians 2 verse 8 and 9. For my grace you have been saved through faith. For my grace you have been saved through faith."),
            "So we look at Ephesians 2 verse 8 and 9. For my grace you have been saved through faith."
        );
        assert_eq!(
            remove_repeats("Turn with me to Psalm 23. Turn with me to Psalm 23."),
            "Turn with me to Psalm 23."
        );
    }

    #[test]
    fn keeps_real_speech() {
        assert_eq!(
            remove_repeats("Holy, holy, holy is the Lord"),
            "Holy, holy, holy is the Lord"
        );
        assert_eq!(remove_repeats("amen amen"), "amen amen");
        assert_eq!(
            remove_repeats("John 3:16 and John 3:17"),
            "John 3:16 and John 3:17"
        );
        assert_eq!(remove_repeats(""), "");
    }
}
