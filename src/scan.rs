use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;

use image::RgbImage;
use rayon::prelude::*;
use walkdir::WalkDir;

use crate::animal::detect_main_animal;
use crate::classify::detect_main_face;
use crate::dedupe::dhash;
use crate::metrics::{self, Metrics};
use crate::photo::{decode_photo, is_supported_photo};
use crate::sharpness::{guess_landscape_or_object, score, PhotoMode, SharpnessMethod};

/// Resolution cap for decoding: enough detail for a meaningful sharpness
/// score without paying to decode full sensor resolution.
const SCORE_MAX_DIM: usize = 1600;
/// Cap for the preview thumbnail shown in the UI.
const THUMB_MAX_DIM: u32 = 220;

/// Name of the subfolder (created inside each scanned root) that
/// disqualified photos get moved into. Scans skip it entirely, so a photo
/// moved there doesn't reappear (flagged all over again) on the next scan.
pub const DISQUALIFIED_DIR: &str = "disqualified";

/// Where a disqualified photo gets moved to: inside the `disqualified`
/// folder of the chosen root that contains it (the deepest one, if chosen
/// roots are nested), keeping its path relative to that root so subfolder
/// structure is preserved and each root keeps its own rejects.
pub fn disqualified_path_for(roots: &[PathBuf], photo: &Path) -> Option<PathBuf> {
    let root = roots
        .iter()
        .filter(|r| photo.starts_with(r))
        .max_by_key(|r| r.components().count())?;
    let relative = photo.strip_prefix(root).ok()?;
    Some(root.join(DISQUALIFIED_DIR).join(relative))
}

/// Extensions of "sidecar" files that travel with a photo: develop settings
/// or metadata written by external editors, e.g. Lightroom/Capture One/
/// darktable (`.xmp`), DxO PhotoLab (`.dop`), ON1 PhotoRAW (`.on1`) and
/// RawTherapee (`.pp3`).
const SIDECAR_EXTENSIONS: &[&str] = &["xmp", "dop", "on1", "pp3"];

/// Sidecar files sitting next to `photo` in the same folder, so they can be
/// moved alongside it. Matches both naming conventions in use: sidecar
/// stripped down to just the photo's stem (`IMG_0001.xmp`, Lightroom/Capture
/// One/darktable-style) and sidecar appended to the photo's full filename
/// (`IMG_0001.CR2.dop`, DxO/ON1-style).
pub fn sidecar_files_for(photo: &Path) -> Vec<PathBuf> {
    let Some(dir) = photo.parent() else {
        return Vec::new();
    };
    let Some(photo_name) = photo.file_name().and_then(|n| n.to_str()) else {
        return Vec::new();
    };
    let Some(photo_stem) = photo.file_stem().and_then(|n| n.to_str()) else {
        return Vec::new();
    };
    let Ok(read_dir) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    read_dir
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p != photo)
        .filter(|p| {
            let Some(ext) = p.extension().and_then(|e| e.to_str()) else {
                return false;
            };
            if !SIDECAR_EXTENSIONS
                .iter()
                .any(|s| s.eq_ignore_ascii_case(ext))
            {
                return false;
            }
            let Some(stem) = p.file_stem().and_then(|s| s.to_str()) else {
                return false;
            };
            stem.eq_ignore_ascii_case(photo_name) || stem.eq_ignore_ascii_case(photo_stem)
        })
        .collect()
}

/// What sharpness-scoring mode to use for a photo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanMode {
    /// Classify each photo individually (face detection for Portrait, then
    /// animal detection for Animal, falling back to a sharpness-uniformity
    /// heuristic for Landscape vs Object).
    Auto,
    /// Force this mode. Portrait and Animal still run their detector (for a
    /// tighter, more accurate scoring region) and fall back to the usual
    /// region if nothing is found.
    Fixed(PhotoMode),
}

/// Classify (if `Auto`) and score a single already-decoded photo: the
/// existing absolute sharpness score plus the full set of Overall-score
/// factors (see `metrics::compute`).
fn classify_and_score(
    img: &RgbImage,
    scan_mode: ScanMode,
    method: SharpnessMethod,
) -> (PhotoMode, f64, Metrics) {
    let (mode, subject) = match scan_mode {
        ScanMode::Auto => {
            if let Some(face) = detect_main_face(img) {
                (PhotoMode::Portrait, Some(face))
            } else if let Some(animal) = detect_main_animal(img) {
                (PhotoMode::Animal, Some(animal))
            } else {
                (guess_landscape_or_object(img), None)
            }
        }
        ScanMode::Fixed(m) => {
            let subject = match m {
                PhotoMode::Portrait => detect_main_face(img),
                PhotoMode::Animal => detect_main_animal(img),
                PhotoMode::Landscape | PhotoMode::Object => None,
            };
            (m, subject)
        }
    };
    let s = score(img, mode, subject.as_ref(), method);
    let metrics = metrics::compute(img, s, subject.as_ref());
    (mode, s, metrics)
}

pub struct PhotoResult {
    pub path: PathBuf,
    pub mode: PhotoMode,
    pub score: f64,
    pub metrics: Metrics,
    /// Perceptual hash for duplicate/burst detection (see `dedupe`).
    pub phash: u64,
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

/// Walks every folder in `roots` for supported photos (RAW or standard
/// formats) and scores each in parallel, streaming results back over `tx`
/// as they complete.
/// Intended to run on its own thread so the UI thread stays responsive;
/// call from a `std::thread::spawn`.
pub fn run_scan(
    roots: Vec<PathBuf>,
    scan_mode: ScanMode,
    method: SharpnessMethod,
    tx: Sender<ScanEvent>,
) {
    // Chosen folders may overlap (one inside another); a photo is only
    // scanned once.
    let mut seen = HashSet::new();
    let files: Vec<PathBuf> = roots
        .iter()
        .flat_map(|root| {
            WalkDir::new(root)
                .into_iter()
                .filter_entry(|e| {
                    !(e.file_type().is_dir() && e.file_name().to_str() == Some(DISQUALIFIED_DIR))
                })
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file() && is_supported_photo(e.path()))
                .map(|e| e.path().to_path_buf())
        })
        .filter(|p| seen.insert(p.clone()))
        .collect();

    let _ = tx.send(ScanEvent::Found(files.len()));

    files.par_iter().for_each_with(tx.clone(), |tx, path| {
        match decode_photo(path, SCORE_MAX_DIM) {
            Ok(img) => {
                let (mode, s, metrics) = classify_and_score(&img, scan_mode, method);
                let phash = dhash(&img);
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
                    phash,
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
pub fn run_recompute(
    items: Vec<(PathBuf, PhotoMode)>,
    method: SharpnessMethod,
    tx: Sender<RecomputeEvent>,
) {
    items.par_iter().for_each_with(tx.clone(), |tx, (path, mode)| {
        if let Ok(img) = decode_photo(path, SCORE_MAX_DIM) {
            let (_, s, metrics) = classify_and_score(&img, ScanMode::Fixed(*mode), method);
            let _ = tx.send(RecomputeEvent::Result(RecomputeResult {
                path: path.clone(),
                score: s,
                metrics,
            }));
        }
    });
    let _ = tx.send(RecomputeEvent::Done);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disqualified_path_keeps_structure_per_chosen_root() {
        let roots = vec![PathBuf::from("/a"), PathBuf::from("/b/trip")];
        assert_eq!(
            disqualified_path_for(&roots, Path::new("/a/sub/x.jpg")),
            Some(PathBuf::from("/a").join(DISQUALIFIED_DIR).join("sub/x.jpg"))
        );
        assert_eq!(
            disqualified_path_for(&roots, Path::new("/b/trip/y.jpg")),
            Some(PathBuf::from("/b/trip").join(DISQUALIFIED_DIR).join("y.jpg"))
        );
        assert_eq!(disqualified_path_for(&roots, Path::new("/c/z.jpg")), None);
    }

    #[test]
    fn nested_roots_use_the_deepest() {
        let roots = vec![PathBuf::from("/a"), PathBuf::from("/a/inner")];
        assert_eq!(
            disqualified_path_for(&roots, Path::new("/a/inner/x.jpg")),
            Some(PathBuf::from("/a/inner").join(DISQUALIFIED_DIR).join("x.jpg"))
        );
    }

    #[test]
    fn sidecar_files_matches_both_naming_conventions() {
        let dir = std::env::temp_dir().join(format!(
            "photo2cull-sidecar-test-{:?}",
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let photo = dir.join("IMG_0001.CR2");
        for name in [
            "IMG_0001.CR2",     // the photo itself, must be excluded
            "IMG_0001.xmp",     // Lightroom/darktable/Capture One-style
            "IMG_0001.CR2.dop", // DxO PhotoLab-style
            "IMG_0001.CR2.on1", // ON1 PhotoRAW-style
            "IMG_0001.CR2.pp3", // RawTherapee-style
            "IMG_0001.jpg",     // unrelated derived file, not a sidecar
            "IMG_0002.xmp",     // sidecar of a different photo
        ] {
            std::fs::write(dir.join(name), b"").unwrap();
        }

        let mut found: Vec<String> = sidecar_files_for(&photo)
            .into_iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        found.sort();

        assert_eq!(
            found,
            vec![
                "IMG_0001.CR2.dop",
                "IMG_0001.CR2.on1",
                "IMG_0001.CR2.pp3",
                "IMG_0001.xmp",
            ]
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
