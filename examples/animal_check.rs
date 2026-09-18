//! Diagnostic tool: `cargo run --example animal_check -- <image>...`
use std::env;
use std::path::Path;

use photo2cull::animal::detect_animals;
use photo2cull::photo::decode_photo;

fn main() {
    for arg in env::args().skip(1) {
        match decode_photo(Path::new(&arg), 1600) {
            Ok(img) => {
                let found = detect_animals(&img);
                println!("{arg}: {}x{}, {} animal(s)", img.width(), img.height(), found.len());
                for b in found {
                    println!("  score={:.2} box=({:.0},{:.0})-({:.0},{:.0})", b.score, b.x1, b.y1, b.x2, b.y2);
                }
            }
            Err(e) => println!("{arg}: ERR {e}"),
        }
    }
}
