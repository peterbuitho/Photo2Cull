use std::path::PathBuf;
use std::sync::mpsc::Sender;

use rayon::prelude::*;
use walkdir::WalkDir;

use crate::raw::{decode_raw, is_raw_file};
use crate::sharpness::{score, PhotoMode};

/// Resolution cap for decoding: enough detail for a meaningful sharpness
/// score without paying to decode full sensor resolution.
const SCORE_MAX_DIM: usize = 1600;
/// Cap for the preview thumbnail shown in the UI.
const THUMB_MAX_DIM: u32 = 220;

pub struct PhotoResult {
    pub path: PathBuf,
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
pub fn run_scan(root: PathBuf, mode: PhotoMode, tx: Sender<ScanEvent>) {
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
                let s = score(&img, mode);
                let thumb = image::imageops::thumbnail(&img, THUMB_MAX_DIM, THUMB_MAX_DIM);
                let _ = tx.send(ScanEvent::Photo(PhotoResult {
                    path: path.clone(),
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
