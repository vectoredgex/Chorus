use chorus_refparse::Detector;
fn main() {
    let input = std::io::read_to_string(std::io::stdin()).unwrap();
    let mut d = Detector::new();
    for line in input.lines() {
        if line.trim().is_empty() { d = Detector::new(); println!(); continue; }
        let got: Vec<String> = d.detect(line).iter().map(|x| format!("{} [{:?}]", x.reference, x.confidence)).collect();
        println!("{:<70} -> {}", line, if got.is_empty() { "-".into() } else { got.join(" | ") });
    }
}
