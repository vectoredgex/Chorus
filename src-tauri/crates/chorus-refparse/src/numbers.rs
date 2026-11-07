//! Spoken-number parsing: "16", "sixteen", "one hundred and nineteen",
//! "twenty third", "a hundred and five".
//!
//! Units never merge with units, so "John three sixteen" reads as 3 then
//! 16, and "Genesis one twenty six" as 1 then 26.

fn unit(tok: &str) -> Option<(u32, bool)> {
    let v = match tok {
        "zero" => (0, false),
        "one" => (1, false), "first" => (1, true),
        "two" => (2, false), "second" => (2, true),
        "three" => (3, false), "third" => (3, true),
        "four" => (4, false), "fourth" => (4, true),
        "five" => (5, false), "fifth" => (5, true),
        "six" => (6, false), "sixth" => (6, true),
        "seven" => (7, false), "seventh" => (7, true),
        "eight" => (8, false), "eighth" => (8, true),
        "nine" => (9, false), "ninth" => (9, true),
        "ten" => (10, false), "tenth" => (10, true),
        "eleven" => (11, false), "eleventh" => (11, true),
        "twelve" => (12, false), "twelfth" => (12, true),
        "thirteen" => (13, false), "thirteenth" => (13, true),
        "fourteen" => (14, false), "fourteenth" => (14, true),
        "fifteen" => (15, false), "fifteenth" => (15, true),
        "sixteen" => (16, false), "sixteenth" => (16, true),
        "seventeen" => (17, false), "seventeenth" => (17, true),
        "eighteen" => (18, false), "eighteenth" => (18, true),
        "nineteen" => (19, false), "nineteenth" => (19, true),
        _ => return None,
    };
    Some(v)
}

fn tens(tok: &str) -> Option<(u32, bool)> {
    let v = match tok {
        "twenty" => (20, false), "twentieth" => (20, true),
        "thirty" => (30, false), "thirtieth" => (30, true),
        "forty" => (40, false), "fortieth" => (40, true),
        "fifty" => (50, false), "fiftieth" => (50, true),
        "sixty" => (60, false), "sixtieth" => (60, true),
        "seventy" => (70, false), "seventieth" => (70, true),
        "eighty" => (80, false), "eightieth" => (80, true),
        "ninety" => (90, false), "ninetieth" => (90, true),
        _ => return None,
    };
    Some(v)
}

/// A parsed number: value, tokens consumed, whether it was an ordinal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Num {
    pub value: u32,
    pub len: usize,
    pub ordinal: bool,
}

/// Below 100: "seven", "twenty", "twenty three", "23".
fn below_100(t: &[String], i: usize) -> Option<Num> {
    let tok = t.get(i)?;
    if let Some((v, o)) = unit(tok) {
        return Some(Num { value: v, len: 1, ordinal: o });
    }
    if let Some((v, o)) = tens(tok) {
        if !o {
            if let Some(next) = t.get(i + 1) {
                if let Some((u, uo)) = unit(next) {
                    if (1..=9).contains(&u) {
                        return Some(Num { value: v + u, len: 2, ordinal: uo });
                    }
                }
            }
        }
        return Some(Num { value: v, len: 1, ordinal: o });
    }
    None
}

/// Parse a number starting at token `i`. Digits pass straight through.
pub fn parse(t: &[String], i: usize) -> Option<Num> {
    let tok = t.get(i)?;
    if let Ok(v) = tok.parse::<u32>() {
        return Some(Num { value: v, len: 1, ordinal: false });
    }
    // half-verse notation: "8a", "16b"
    if let Some(d) = tok.strip_suffix(['a', 'b', 'c']) {
        if !d.is_empty() && d.bytes().all(|c| c.is_ascii_digit()) {
            return Some(Num { value: d.parse().ok()?, len: 1, ordinal: false });
        }
    }
    // "23rd", "1st", "16th"
    for suf in ["st", "nd", "rd", "th"] {
        if let Some(d) = tok.strip_suffix(suf) {
            if !d.is_empty() && d.bytes().all(|c| c.is_ascii_digit()) {
                return Some(Num { value: d.parse().ok()?, len: 1, ordinal: true });
            }
        }
    }
    // "a hundred and five" / "one hundred nineteen" / "hundred and fiftieth"
    let (hundreds, mut j) = if tok == "hundred" {
        (1, i + 1)
    } else if (tok == "a" || tok == "one") && t.get(i + 1).map(|s| s.as_str()) == Some("hundred") {
        (1, i + 2)
    } else if tok == "hundredth" {
        return Some(Num { value: 100, len: 1, ordinal: true });
    } else {
        return below_100(t, i);
    };
    let mut value = hundreds * 100;
    if t.get(j).map(|s| s.as_str()) == Some("and") {
        // only consume "and" when a number follows
        if below_100(t, j + 1).is_some() {
            j += 1;
        }
    }
    let mut ordinal = false;
    if let Some(rest) = below_100(t, j) {
        value += rest.value;
        ordinal = rest.ordinal;
        j += rest.len;
    }
    Some(Num { value, len: j - i, ordinal })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn p(s: &str) -> Option<(u32, usize)> {
        let t: Vec<String> = s.split_whitespace().map(String::from).collect();
        parse(&t, 0).map(|n| (n.value, n.len))
    }
    #[test]
    fn numbers() {
        assert_eq!(p("16"), Some((16, 1)));
        assert_eq!(p("sixteen"), Some((16, 1)));
        assert_eq!(p("three sixteen"), Some((3, 1)));
        assert_eq!(p("twenty three"), Some((23, 2)));
        assert_eq!(p("twenty"), Some((20, 1)));
        assert_eq!(p("one hundred and nineteen"), Some((119, 4)));
        assert_eq!(p("a hundred and five"), Some((105, 4)));
        assert_eq!(p("hundred fifty"), Some((150, 2)));
        assert_eq!(p("one hundred"), Some((100, 2)));
        assert_eq!(p("one twenty"), Some((1, 1)));
        assert_eq!(p("twenty third"), Some((23, 2)));
        assert_eq!(p("one hundred and twenty one"), Some((121, 5)));
        assert_eq!(p("word"), None);
    }
}
