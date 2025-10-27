#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Part {
    Text(String),
    Number(u64),
}

pub fn natural_key(s: &str) -> Vec<Part> {
    let mut parts = Vec::new();
    let mut buf = String::new();
    let mut in_number = false;

    for ch in s.chars() {
        if ch.is_ascii_digit() {
            if !in_number {
                if !buf.is_empty() {
                    parts.push(Part::Text(buf.clone()));
                    buf.clear();
                }
                in_number = true;
            }
            buf.push(ch);
        } else {
            if in_number {
                let num: u64 = buf.parse().unwrap_or(0);
                parts.push(Part::Number(num));
                buf.clear();
                in_number = false;
            }
            buf.push(ch);
        }
    }
    if !buf.is_empty() {
        if in_number {
            let num: u64 = buf.parse().unwrap_or(0);
            parts.push(Part::Number(num));
        } else {
            parts.push(Part::Text(buf));
        }
    }
    parts
}
