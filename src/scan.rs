use std::path::PathBuf;
use std::sync::mpsc::Sender;

use rayon::prelude::*;
use walkdir::WalkDir;

use crate::classify::detect_faces;
use crate::raw::{decode_raw, is_raw_file};
use crate::sharpness::{guess_landscape_or_object, score, PhotoMode};

/// Resolution cap for decoding: enough detail for a meaningful sharpness
/// score without paying to decode full sensor resolution.
const SCORE_MAX_DIM: usize = 1600;
/// Cap for the preview thumbnail shown in the UI.
const THUMB_MAX_DIM: u32 = 220;

/// What sharpness-scoring mode to use for each photo in a scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanMode {
    /// Classify each photo individually (face detection for Portrait,
    /// falling back to a sharpness-uniformity heuristic for Landscape vs
    /// Object).
    Auto,
    /// Force every photo in the scan to the same mode.
    Fixed(PhotoMode),
}

pub struct PhotoResult {
    pub path: PathBuf,
    pub mode: PhotoMode,
    pub score: f64,
    pub thumb_w: u32,
    pub thumb_h: u32,
    pub thumb_rgb: Vec<u8>,
}

pub enum ScanEvent {
    Found(usize),
    Photo(PhotoResult),
    Failed(PathBuf, String),
    Done,
}

/// Walks `root` for RAW files and scores each in parallel, streaming results
/// back over `tx` as they complete. Intended to run on its own thread so the
/// UI thread stays responsive; call from a `std::thread::spawn`.
pub fn run_scan(root: PathBuf, scan_mode: ScanMode, tx: Sender<ScanEvent>) {
    let files: Vec<PathBuf> = WalkDir::new(&root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file() && is_raw_file(e.path()))
        .map(|e| e.path().to_path_buf())
        .collect();

    let _ = tx.send(ScanEvent::Found(files.len()));

    files.par_iter().for_each_with(tx.clone(), |tx, path| {
        match decode_raw(path, SCORE_MAX_DIM) {
            Ok(img) => {
                let (mode, face) = match scan_mode {
                    ScanMode::Fixed(m) => (m, None),
                    ScanMode::Auto => {
                        let best_face = detect_faces(&img).into_iter().max_by(|a, b| {
                            let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
                            let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
                            area_a.partial_cmp(&area_b).unwrap()
                        });
                        match best_face {
                            Some(f) => (PhotoMode::Portrait, Some(f)),
                            None => (guess_landscape_or_object(&img), None),
                        }
                    }
                };
                let s = score(&img, mode, face.as_ref());
                // `DynamicImage::resize` (the method) fits within the box,
                // preserving aspect ratio; the free functions
                // `imageops::resize`/`thumbnail` both stretch to it exactly.
                let thumb = image::DynamicImage::ImageRgb8(img.clone())
                    .resize(
                        THUMB_MAX_DIM,
                        THUMB_MAX_DIM,
                        image::imageops::FilterType::Triangle,
                    )
                    .to_rgb8();
                let _ = tx.send(ScanEvent::Photo(PhotoResult {
                    path: path.clone(),
                    mode,
                    score: s,
                    thumb_w: thumb.width(),
                    thumb_h: thumb.height(),
                    thumb_rgb: thumb.into_raw(),
                }));
            }
            Err(e) => {
                let _ = tx.send(ScanEvent::Failed(path.clone(), e.to_string()));
            }
        }
    });

    let _ = tx.send(ScanEvent::Done);
}
