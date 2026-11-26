use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;

use image::RgbImage;
use rayon::prelude::*;
use walkdir::WalkDir;

use crate::animal::detect_main_animal;
use crate::classify::detect_main_face;
use crate::dedupe::dhash;
use crate::metrics::{self, Metrics};
use crate::photo::{decode_photo, is_supported_photo};
use crate::sharpness::{
    PhotoMode, SharpnessMethod, guess_landscape_or_object_gray, score_gray,
};

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
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .map(|e| e.path())
        .filter(|p| p != photo)
        .filter(|p| is_sidecar_for(p, photo_name, photo_stem).is_some())
        .collect()
}

fn is_sidecar_for(candidate: &Path, photo_name: &str, photo_stem: &str) -> Option<bool> {
    let ext = candidate.extension().and_then(|e| e.to_str())?;
    if !SIDECAR_EXTENSIONS
        .iter()
        .any(|s| s.eq_ignore_ascii_case(ext))
    {
        return None;
    }
    let stem = candidate.file_stem().and_then(|s| s.to_str())?;
    if stem.eq_ignore_ascii_case(photo_name) {
        Some(true) // full-filename match (e.g. IMG_0001.CR2.dop)
    } else if stem.eq_ignore_ascii_case(photo_stem) {
        Some(false) // stem-only match (e.g. IMG_0001.xmp)
    } else {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoveItem {
    pub src: PathBuf,
    pub dest: PathBuf,
    pub is_sidecar: bool,
}

/// Plans all file moves for `disqualified` photos and their eligible sidecars:
/// - Scans each parent folder at most once.
/// - Deduplicates sidecars when multiple disqualified photos (e.g. a RAW+JPEG
///   pair) share one stem-matched sidecar (`IMG_0001.xmp`).
/// - Leaves a stem-only sidecar in place if a non-disqualified photo in the
///   same folder still shares that stem (while always moving exact
///   full-filename sidecars like `IMG_0001.CR2.dop`).
pub fn plan_disqualified_moves(roots: &[PathBuf], disqualified: &[PathBuf]) -> Vec<MoveItem> {
    let disqualified_set: HashSet<&Path> = disqualified.iter().map(PathBuf::as_path).collect();
    let mut by_dir: Vec<(PathBuf, Vec<PathBuf>)> = Vec::new();
    let mut dir_index: HashMap<PathBuf, usize> = HashMap::new();
    for photo in disqualified {
        let dir = photo.parent().unwrap_or_else(|| Path::new("")).to_path_buf();
        if let Some(&idx) = dir_index.get(&dir) {
            by_dir[idx].1.push(photo.clone());
        } else {
            dir_index.insert(dir.clone(), by_dir.len());
            by_dir.push((dir, vec![photo.clone()]));
        }
    }

    let mut items = Vec::new();
    for (dir, photos) in by_dir {
        let regular_files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flat_map(|rd| rd.filter_map(|e| e.ok()))
            .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
            .map(|e| e.path())
            .collect();

        let kept_stems: HashSet<String> = regular_files
            .iter()
            .filter(|p| is_supported_photo(p) && !disqualified_set.contains(p.as_path()))
            .filter_map(|p| p.file_stem().and_then(|s| s.to_str()))
            .map(|s| s.to_ascii_lowercase())
            .collect();

        let mut seen_sidecars: HashSet<PathBuf> = HashSet::new();
        for photo in photos {
            if let Some(dest) = disqualified_path_for(roots, &photo) {
                items.push(MoveItem {
                    src: photo.clone(),
                    dest,
                    is_sidecar: false,
                });
            }
            let Some(photo_name) = photo.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some(photo_stem) = photo.file_stem().and_then(|n| n.to_str()) else {
                continue;
            };
            let stem_is_kept = kept_stems.contains(&photo_stem.to_ascii_lowercase());

            for cand in &regular_files {
                if cand == &photo {
                    continue;
                }
                let Some(is_full_name_match) = is_sidecar_for(cand, photo_name, photo_stem) else {
                    continue;
                };
                if !is_full_name_match && stem_is_kept {
                    continue;
                }
                if seen_sidecars.insert(cand.clone())
                    && let Some(dest) = disqualified_path_for(roots, cand)
                {
                    items.push(MoveItem {
                        src: cand.clone(),
                        dest,
                        is_sidecar: true,
                    });
                }
            }
        }
    }

    items
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
    let gray = image::imageops::grayscale(img);
    let full_lap = imageproc::filter::laplacian_filter(&gray);
    let (mode, subject) = match scan_mode {
        ScanMode::Auto => {
            if let Some(face) = detect_main_face(img) {
                (PhotoMode::Portrait, Some(face))
            } else if let Some(animal) = detect_main_animal(img) {
                (PhotoMode::Animal, Some(animal))
            } else {
                (guess_landscape_or_object_gray(&gray, &full_lap), None)
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
    let s = score_gray(&gray, Some(&full_lap), mode, subject.as_ref(), method);
    let metrics = metrics::compute_with_gray(img, &gray, &full_lap, s, subject.as_ref());
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
/// as they complete. Stops early if `cancel` is set.
/// Intended to run on its own thread so the UI thread stays responsive;
/// call from a `std::thread::spawn`.
pub fn run_scan(
    roots: Vec<PathBuf>,
    scan_mode: ScanMode,
    method: SharpnessMethod,
    cancel: Arc<AtomicBool>,
    tx: Sender<ScanEvent>,
) {
    // Chosen folders may overlap (one inside another); a photo is only
    // scanned once.
    let mut seen = HashSet::new();
    let mut files = Vec::new();
    for root in &roots {
        for entry in WalkDir::new(root)
            .into_iter()
            .filter_entry(|e| {
                !(e.depth() > 0
                    && e.file_type().is_dir()
                    && e.file_name()
                        .to_str()
                        .is_some_and(|n| n.eq_ignore_ascii_case(DISQUALIFIED_DIR)))
            })
            .filter_map(|e| e.ok())
        {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            if entry.file_type().is_file() && is_supported_photo(entry.path()) {
                let path = entry.path().to_path_buf();
                if seen.insert(path.clone()) {
                    files.push(path);
                }
            }
        }
        if cancel.load(Ordering::Relaxed) {
            break;
        }
    }

    if !cancel.load(Ordering::Relaxed) {
        let _ = tx.send(ScanEvent::Found(files.len()));

        files.par_iter().for_each_with(tx.clone(), |tx, path| {
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            match decode_photo(path, SCORE_MAX_DIM) {
                Ok(img) => {
                    let (mode, s, metrics) = classify_and_score(&img, scan_mode, method);
                    let phash = dhash(&img);
                    // `DynamicImage::resize` (the method) fits within the box,
                    // preserving aspect ratio; the free functions
                    // `imageops::resize`/`thumbnail` both stretch to it exactly.
                    let thumb = image::DynamicImage::ImageRgb8(img)
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
    }

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

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            static COUNTER: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "photo2cull-{label}-{}-{id}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn sidecar_files_matches_both_naming_conventions() {
        let dir = TempDir::new("sidecar");
        let photo = dir.0.join("IMG_0001.CR2");
        for name in [
            "IMG_0001.CR2",     // the photo itself, must be excluded
            "IMG_0001.xmp",     // Lightroom/darktable/Capture One-style
            "IMG_0001.CR2.dop", // DxO PhotoLab-style
            "IMG_0001.CR2.on1", // ON1 PhotoRAW-style
            "IMG_0001.CR2.pp3", // RawTherapee-style
            "IMG_0001.jpg",     // unrelated derived file, not a sidecar
            "IMG_0002.xmp",     // sidecar of a different photo
        ] {
            std::fs::write(dir.0.join(name), b"").unwrap();
        }
        // A directory named like a sidecar must not be treated as a sidecar file.
        std::fs::create_dir_all(dir.0.join("IMG_0001.pp3")).unwrap();

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
    }

    #[test]
    fn plan_moves_keeps_shared_xmp_when_raw_sibling_is_kept() {
        let dir = TempDir::new("raw-jpeg-kept");
        let cr2 = dir.0.join("IMG_0001.CR2");
        let jpg = dir.0.join("IMG_0001.JPG");
        let shared_xmp = dir.0.join("IMG_0001.xmp");
        let jpg_dop = dir.0.join("IMG_0001.JPG.dop");
        for p in [&cr2, &jpg, &shared_xmp, &jpg_dop] {
            std::fs::write(p, b"").unwrap();
        }

        let items = plan_disqualified_moves(std::slice::from_ref(&dir.0), std::slice::from_ref(&jpg));
        let mut srcs: Vec<PathBuf> = items.iter().map(|i| i.src.clone()).collect();
        srcs.sort();
        assert_eq!(srcs, vec![jpg, jpg_dop]);
    }

    #[test]
    fn plan_moves_deduplicates_shared_xmp_when_both_siblings_disqualified() {
        let dir = TempDir::new("raw-jpeg-both");
        let cr2 = dir.0.join("IMG_0001.CR2");
        let jpg = dir.0.join("IMG_0001.JPG");
        let shared_xmp = dir.0.join("IMG_0001.xmp");
        for p in [&cr2, &jpg, &shared_xmp] {
            std::fs::write(p, b"").unwrap();
        }
        // Directory with sidecar name should be ignored.
        std::fs::create_dir_all(dir.0.join("IMG_0001.dop")).unwrap();

        let items = plan_disqualified_moves(std::slice::from_ref(&dir.0), &[cr2.clone(), jpg.clone()]);
        let mut srcs: Vec<PathBuf> = items.iter().map(|i| i.src.clone()).collect();
        srcs.sort();
        assert_eq!(srcs, vec![cr2, jpg, shared_xmp]);
        assert_eq!(items.iter().filter(|i| i.is_sidecar).count(), 1);
    }
}
