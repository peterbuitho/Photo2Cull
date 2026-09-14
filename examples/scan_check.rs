//! Diagnostic tool: `cargo run --example scan_check -- <folder>`
use std::env;
use std::path::PathBuf;
use std::time::Instant;

use photo2cull::raw::{decode_raw, is_raw_file};
use photo2cull::sharpness::{score, PhotoMode};

fn main() {
    let folder = env::args().nth(1).expect("usage: scan_check <folder>");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&folder)
        .expect("read_dir failed")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| is_raw_file(p))
        .collect();
    files.sort();

    for path in files {
        let start = Instant::now();
        match decode_raw(&path, 1600) {
            Ok(img) => {
                let s = score(&img, PhotoMode::Landscape);
                println!(
                    "{}: OK {}x{} score={:.0} ({:?})",
                    path.file_name().unwrap().to_string_lossy(),
                    img.width(),
                    img.height(),
                    s,
                    start.elapsed()
                );
            }
            Err(e) => println!("{}: ERR {e}", path.file_name().unwrap().to_string_lossy()),
        }
    }
}
