//! Prints the English word order each Malayalam line is rewritten into.
//!
//! Usage: echo "യോഹന്നാൻ മൂന്നാം അദ്ധ്യായം" | cargo run --example ml_debug
fn main() {
    for line in std::io::stdin().lines() {
        let (english, lost) = chorus_refparse::malayalam::rewrite(&line.unwrap());
        println!("{english}{}", if lost { "   [unknown book dropped]" } else { "" });
    }
}
