//! Diagnostic tool: `cargo run --example scan_check -- <folder>`
use std::env;
use std::path::PathBuf;
use std::time::Instant;

use photo2cull::animal::detect_main_animal;
use photo2cull::classify::detect_main_face;
use photo2cull::metrics::{self, overall_score, Weights};
use photo2cull::photo::{decode_photo, is_supported_photo};
use photo2cull::sharpness::{guess_landscape_or_object, score, PhotoMode, SharpnessMethod};

fn main() {
    let folder = env::args().nth(1).expect("usage: scan_check <folder>");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&folder)
        .expect("read_dir failed")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| is_supported_photo(p))
        .collect();
    files.sort();

    for path in files {
        let start = Instant::now();
        match decode_photo(&path, 1600) {
            Ok(img) => {
                let (mode, subject) = if let Some(face) = detect_main_face(&img) {
                    (PhotoMode::Portrait, Some(face))
                } else if let Some(animal) = detect_main_animal(&img) {
                    (PhotoMode::Animal, Some(animal))
                } else {
                    (guess_landscape_or_object(&img), None)
                };
                let s = score(&img, mode, subject.as_ref(), SharpnessMethod::FixedRegion);
                let s_sharpest =
                    score(&img, mode, subject.as_ref(), SharpnessMethod::SharpestRegion);
                let m = metrics::compute(&img, s, subject.as_ref());
                let overall = overall_score(&m, &Weights::default());
                println!(
                    "{}: OK {}x{} mode={:?} subject={} score={:.0} sharpest_region={:.0} sharp={:.0} exp={:.0} contrast={:.0} color={:.0} comp={:.0} subj={} overall={:.0} ({:?})",
                    path.file_name().unwrap().to_string_lossy(),
                    img.width(),
                    img.height(),
                    mode,
                    subject.is_some(),
                    s,
                    s_sharpest,
                    m.sharpness,
                    m.exposure,
                    m.contrast,
                    m.color,
                    m.composition.unwrap_or(-1.0),
                    m.subject.map(|v| format!("{v:.0}")).unwrap_or_else(|| "-".to_string()),
                    overall,
                    start.elapsed()
                );
            }
            Err(e) => println!("{}: ERR {e}", path.file_name().unwrap().to_string_lossy()),
        }
    }
}
