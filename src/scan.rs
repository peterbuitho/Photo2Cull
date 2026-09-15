use std::path::PathBuf;
use std::sync::mpsc::Sender;

use image::RgbImage;
use rayon::prelude::*;
use walkdir::WalkDir;

use crate::classify::detect_main_face;
use crate::metrics::{self, Metrics};
use crate::photo::{decode_photo, is_supported_photo};
use crate::sharpness::{guess_landscape_or_object, score, PhotoMode};

/// Resolution cap for decoding: enough detail for a meaningful sharpness
/// score without paying to decode full sensor resolution.
const SCORE_MAX_DIM: usize = 1600;
/// Cap for the preview thumbnail shown in the UI.
const THUMB_MAX_DIM: u32 = 220;

/// Name of the subfolder (created inside the scanned root) that
/// disqualified photos get moved into. Scans skip it entirely, so a photo
/// moved there doesn't reappear (flagged all over again) on the next scan.
pub const DISQUALIFIED_DIR: &str = "disqualified";

/// What sharpness-scoring mode to use for a photo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanMode {
    /// Classify each photo individually (face detection for Portrait,
    /// falling back to a sharpness-uniformity heuristic for Landscape vs
    /// Object).
    Auto,
    /// Force this mode. Portrait still tries face detection (for a tighter,
    /// more accurate scoring region) and falls back to a centered crop if
    /// no face is found.
    Fixed(PhotoMode),
}

/// Classify (if `Auto`) and score a single already-decoded photo: the
/// existing absolute sharpness score plus the Phase-1 Overall-score
/// factors (exposure/contrast/color, and sharpness normalized for blending
/// -- see `metrics::compute`).
fn classify_and_score(img: &RgbImage, scan_mode: ScanMode) -> (PhotoMode, f64, Metrics) {
    let (mode, face) = match scan_mode {
        ScanMode::Auto => match detect_main_face(img) {
            Some(f) => (PhotoMode::Portrait, Some(f)),
            None => (guess_landscape_or_object(img), None),
        },
        ScanMode::Fixed(m) => {
            let face = if m == PhotoMode::Portrait {
                detect_main_face(img)
            } else {
                None
            };
            (m, face)
        }
    };
    let s = score(img, mode, face.as_ref());
    let metrics = metrics::compute(img, s);
    (mode, s, metrics)
}

pub struct PhotoResult {
    pub path: PathBuf,
    pub mode: PhotoMode,
    pub score: f64,
    pub metrics: Metrics,
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

/// Walks `root` for supported photos (RAW or standard formats) and scores
/// each in parallel, streaming results back over `tx` as they complete.
/// Intended to run on its own thread so the UI thread stays responsive;
/// call from a `std::thread::spawn`.
pub fn run_scan(root: PathBuf, scan_mode: ScanMode, tx: Sender<ScanEvent>) {
    let files: Vec<PathBuf> = WalkDir::new(&root)
        .into_iter()
        .filter_entry(|e| {
            !(e.file_type().is_dir() && e.file_name().to_str() == Some(DISQUALIFIED_DIR))
        })
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file() && is_supported_photo(e.path()))
        .map(|e| e.path().to_path_buf())
        .collect();

    let _ = tx.send(ScanEvent::Found(files.len()));

    files.par_iter().for_each_with(tx.clone(), |tx, path| {
        match decode_photo(path, SCORE_MAX_DIM) {
            Ok(img) => {
                let (mode, s, metrics) = classify_and_score(&img, scan_mode);
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
                    metrics,
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

pub struct RecomputeResult {
    pub path: PathBuf,
    pub score: f64,
    pub metrics: Metrics,
}

pub enum RecomputeEvent {
    Result(RecomputeResult),
    Done,
}

/// Re-scores specific (path, forced mode) pairs -- used after the user
/// manually overrides a photo's type -- without re-walking the folder or
/// regenerating thumbnails (the crop used for those doesn't depend on mode).
pub fn run_recompute(items: Vec<(PathBuf, PhotoMode)>, tx: Sender<RecomputeEvent>) {
    items.par_iter().for_each_with(tx.clone(), |tx, (path, mode)| {
        if let Ok(img) = decode_photo(path, SCORE_MAX_DIM) {
            let (_, s, metrics) = classify_and_score(&img, ScanMode::Fixed(*mode));
            let _ = tx.send(RecomputeEvent::Result(RecomputeResult {
                path: path.clone(),
                score: s,
                metrics,
            }));
        }
    });
    let _ = tx.send(RecomputeEvent::Done);
}
