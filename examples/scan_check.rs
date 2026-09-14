//! Diagnostic tool: `cargo run --example scan_check -- <folder>`
use std::env;
use std::path::PathBuf;
use std::time::Instant;

use photo2cull::classify::detect_main_face;
use photo2cull::raw::{decode_raw, is_raw_file};
use photo2cull::sharpness::{guess_landscape_or_object, score, PhotoMode};

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
                let best_face = detect_main_face(&img);
                let mode = match &best_face {
                    Some(_) => PhotoMode::Portrait,
                    None => guess_landscape_or_object(&img),
                };
                let s = score(&img, mode, best_face.as_ref());
                println!(
                    "{}: OK {}x{} mode={:?} face={} score={:.0} ({:?})",
                    path.file_name().unwrap().to_string_lossy(),
                    img.width(),
                    img.height(),
                    mode,
                    best_face.is_some(),
                    s,
                    start.elapsed()
                );
            }
            Err(e) => println!("{}: ERR {e}", path.file_name().unwrap().to_string_lossy()),
        }
    }
}
