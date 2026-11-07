//! Runs every phrase in tests/cases.txt and reports all failures at once.

use chorus_refparse::{Confidence, Detector};

fn conf_tag(c: Confidence) -> &'static str {
    match c {
        Confidence::High => "H",
        Confidence::Medium => "M",
        Confidence::Low => "L",
    }
}

#[test]
fn sermon_phrases() {
    run(include_str!("cases.txt"));
}

#[test]
fn malayalam_phrases() {
    run(include_str!("cases_ml.txt"));
}

fn run(data: &str) {
    let mut det = Detector::new();
    let mut failures = Vec::new();
    let mut total = 0;

    for (lineno, line) in data.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            det = Detector::new();
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        let (input, expected) = line.split_once("=>").expect("missing =>");
        let (input, expected) = (input.trim(), expected.trim());
        total += 1;

        let found = det.detect(input);
        let wants_low = expected.contains("[L]");
        let considered: Vec<_> = found
            .iter()
            .filter(|d| wants_low || d.confidence >= Confidence::Medium)
            .collect();

        let ok = if expected == "NONE" {
            considered.is_empty()
        } else {
            let exp: Vec<&str> = expected.split(" | ").map(str::trim).collect();
            exp.len() == considered.len()
                && exp.iter().zip(&considered).all(|(e, d)| {
                    let (r, tag) = match e.split_once(" [") {
                        Some((r, t)) => (r, Some(t.trim_end_matches(']'))),
                        None => (*e, None),
                    };
                    d.reference.to_string() == r && tag.is_none_or(|t| t == conf_tag(d.confidence))
                })
        };
        if !ok {
            let got: Vec<String> = found
                .iter()
                .map(|d| format!("{} [{}]", d.reference, conf_tag(d.confidence)))
                .collect();
            failures.push(format!(
                "line {}: {:?}\n    expected: {}\n    got:      {}",
                lineno + 1,
                input,
                expected,
                if got.is_empty() { "nothing".into() } else { got.join(" | ") }
            ));
        }
    }

    println!("{} of {} phrases correct", total - failures.len(), total);
    if !failures.is_empty() {
        panic!("{} failures:\n{}", failures.len(), failures.join("\n"));
    }
}
