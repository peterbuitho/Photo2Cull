//! The Overall-score ranking system: exposure, contrast, color,
//! composition and subject quality, plus the weighted blend that combines
//! them with sharpness into a single Overall score.
//!
//! Unlike the existing absolute sharpness score (unbounded, used as-is for
//! the single-photo focus cull), everything here is normalized to a 0-100
//! range so the factors can be meaningfully weighted and summed. These
//! normalization curves are calibrated by eye against a handful of real
//! test photos, not a large corpus -- expect to retune them.
//!
//! Overall is deliberately *not* computed once during the scan and stored:
//! it's recomputed from the stored per-factor `Metrics` on demand (cheap --
//! a handful of float multiplications), so adjusting weights re-ranks
//! instantly without redecoding anything.
//!
//! Composition and subject quality are the two deliberately-approximate
//! factors, both agreed with the user as "cheap heuristic now, upgrade to
//! a real model later":
//!
//! - Composition scores rule-of-thirds alignment only. The original sketch
//!   also listed symmetry and negative-space balance, but those pull
//!   against thirds-alignment in a single blended number (a centered,
//!   symmetric subject is the *opposite* of a thirds-aligned one), so
//!   combining them produced a score that mostly cancelled itself out.
//! - Subject quality scores subject prominence (size + centering) for
//!   Portraits (face) and Animals (detected animal) only. The planned
//!   eyes-open check would need sourcing a second ONNX model with the same
//!   license/verification rigor as the face detector; deferred rather than
//!   rushed.

use image::{GrayImage, Luma, RgbImage};

use crate::classify::FaceBox;

/// Per-photo factor scores, each normalized to roughly 0-100 (higher is
/// better). `subject` is `None` unless a face (Portrait) or animal (Animal)
/// was detected (no well-defined subject otherwise); `overall_score`
/// renormalizes over
/// whatever's present.
#[derive(Debug, Clone, Copy)]
pub struct Metrics {
    /// Normalized sharpness (see [`normalize_sharpness`]) -- distinct from
    /// the raw, unbounded score `sharpness::score` returns.
    pub sharpness: f64,
    pub exposure: f64,
    pub contrast: f64,
    pub color: f64,
    pub composition: Option<f64>,
    pub subject: Option<f64>,
}

/// Blend weights for the six factors. Need not sum to 1.0 -- `overall_score`
/// renormalizes over whatever factors are actually available.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weights {
    pub sharpness: f32,
    pub exposure: f32,
    pub contrast: f32,
    pub color: f32,
    pub composition: f32,
    pub subject: f32,
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            sharpness: 0.25,
            exposure: 0.15,
            contrast: 0.10,
            color: 0.10,
            composition: 0.25,
            subject: 0.15,
        }
    }
}

/// Weighted average of the available factors in `m`, in the same 0-100
/// space as each individual factor.
pub fn overall_score(m: &Metrics, w: &Weights) -> f64 {
    let mut total_weight = 0.0_f64;
    let mut weighted_sum = 0.0_f64;
    let mut add = |weight: f32, value: f64| {
        total_weight += weight as f64;
        weighted_sum += weight as f64 * value;
    };

    add(w.sharpness, m.sharpness);
    add(w.exposure, m.exposure);
    add(w.contrast, m.contrast);
    add(w.color, m.color);
    if let Some(c) = m.composition {
        add(w.composition, c);
    }
    if let Some(s) = m.subject {
        add(w.subject, s);
    }

    if total_weight <= 0.0 {
        0.0
    } else {
        weighted_sum / total_weight
    }
}

/// Compute every factor from an already-decoded photo, given its precomputed
/// grayscale image, full-frame Laplacian buffer, raw (absolute) sharpness
/// score, and, for Portraits/Animals, the detected main subject box.
pub(crate) fn compute_with_gray(
    rgb: &RgbImage,
    gray: &GrayImage,
    full_lap: &image::ImageBuffer<Luma<i16>, Vec<i16>>,
    raw_sharpness: f64,
    subject: Option<&FaceBox>,
) -> Metrics {
    let (exposure, contrast) = exposure_and_contrast_score(gray);
    Metrics {
        sharpness: normalize_sharpness(raw_sharpness),
        exposure,
        contrast,
        color: color_score(rgb),
        composition: Some(composition_score_from_lap(full_lap)),
        subject: subject.map(|b| subject_quality_score(b, rgb.width(), rgb.height())),
    }
}

/// Compute every factor from an already-decoded photo, given its raw
/// (absolute) sharpness score and, for Portraits/Animals, the detected main
/// subject box.
pub fn compute(rgb: &RgbImage, raw_sharpness: f64, subject: Option<&FaceBox>) -> Metrics {
    let gray = image::imageops::grayscale(rgb);
    let full_lap = imageproc::filter::laplacian_filter(&gray);
    compute_with_gray(rgb, &gray, &full_lap, raw_sharpness, subject)
}

/// Maps the unbounded raw variance-of-Laplacian score to 0-100 with a
/// saturating (no hard ceiling) curve: raw=100 -> 50, raw=300 -> 75,
/// raw=900 -> 90. Loosely calibrated against real scores, which have run
/// roughly 24-330 across the photos tested so far.
fn normalize_sharpness(raw: f64) -> f64 {
    if raw <= 0.0 {
        0.0
    } else {
        100.0 * raw / (raw + 100.0)
    }
}

/// Computes `(exposure, contrast)` from a single pass-pair over `gray`:
/// - Exposure: 100 = midtone-centered with no clipping; penalized for both
///   deviation from a midtone mean and for blown-highlight/crushed-shadow
///   pixel fractions. A crude, scene-blind proxy -- a legitimately bright
///   (snow) or dark (night) scene will score low despite being "correctly"
///   exposed.
/// - Contrast: 100 = strong tonal spread, using luminance standard deviation
///   scaled so ~64 maps to 100.
fn exposure_and_contrast_score(gray: &GrayImage) -> (f64, f64) {
    let pixels = gray.as_raw();
    let n = pixels.len() as f64;
    if n == 0.0 {
        return (0.0, 0.0);
    }

    let mut sum = 0.0_f64;
    let mut clipped_count = 0usize;
    for &p in pixels {
        sum += p as f64;
        if p <= 3 || p >= 252 {
            clipped_count += 1;
        }
    }
    let mean = sum / n;
    let clipped = clipped_count as f64 / n;

    let brightness_score = 100.0 * (1.0 - (mean - 128.0).abs() / 128.0);
    let clipping_penalty = 100.0 * clipped;
    let exposure = (brightness_score - clipping_penalty).clamp(0.0, 100.0);

    let variance = pixels
        .iter()
        .map(|&p| {
            let d = p as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let std_dev = variance.sqrt();
    let contrast = (100.0 * std_dev / 64.0).clamp(0.0, 100.0);

    (exposure, contrast)
}

/// 100 = highly colorful/vivid, via the Hasler-Süsstrunk colorfulness
/// metric (a well-established, purely algorithmic measure -- not a
/// judgment of whether the colors are "correct", just how vivid they are).
fn color_score(rgb: &RgbImage) -> f64 {
    let n = (rgb.width() as u64 * rgb.height() as u64) as f64;
    if n == 0.0 {
        return 0.0;
    }

    let (mut sum_rg, mut sum_yb, mut sum_rg2, mut sum_yb2) = (0.0, 0.0, 0.0, 0.0);
    for px in rgb.pixels() {
        let r = px[0] as f64;
        let g = px[1] as f64;
        let b = px[2] as f64;
        let rg = r - g;
        let yb = 0.5 * (r + g) - b;
        sum_rg += rg;
        sum_yb += yb;
        sum_rg2 += rg * rg;
        sum_yb2 += yb * yb;
    }

    let mean_rg = sum_rg / n;
    let mean_yb = sum_yb / n;
    let std_rg = (sum_rg2 / n - mean_rg * mean_rg).max(0.0).sqrt();
    let std_yb = (sum_yb2 / n - mean_yb * mean_yb).max(0.0).sqrt();

    let colorfulness =
        (std_rg.powi(2) + std_yb.powi(2)).sqrt() + 0.3 * (mean_rg.powi(2) + mean_yb.powi(2)).sqrt();

    // The original paper's "extremely colorful" bucket starts around 109-110.
    (100.0 * colorfulness / 110.0).clamp(0.0, 100.0)
}

/// 100 = edge/detail energy concentrates near the four rule-of-thirds
/// gridlines; 50 = no particular alignment (energy proportional to area);
/// 0 = energy concentrates away from the gridlines (e.g. dead-centered or
/// pushed into a corner). Uses the same Laplacian-magnitude map as
/// sharpness as a crude "visual interest" proxy -- not a judgment of
/// whether the composition is actually good, just whether interesting
/// detail happens to sit near the classic thirds lines.
fn composition_score_from_lap(lap: &image::ImageBuffer<Luma<i16>, Vec<i16>>) -> f64 {
    let (lw, lh) = lap.dimensions();
    if lw < 4 || lh < 4 {
        return 50.0;
    }

    // Half-widths so each rule-of-thirds band covers 10% of its axis
    // (total width 2 * half_band = 0.10 * axis), centered on 1/3 and 2/3.
    let half_band_w = (lw as f64 * 0.05).max(0.5);
    let half_band_h = (lh as f64 * 0.05).max(0.5);
    let third_x = [lw as f64 / 3.0, 2.0 * lw as f64 / 3.0];
    let third_y = [lh as f64 / 3.0, 2.0 * lh as f64 / 3.0];

    let mut total = 0.0_f64;
    let mut near_thirds = 0.0_f64;
    let mut near_pixels = 0_u64;
    for (x, y, px) in lap.enumerate_pixels() {
        let e = (px[0] as f64).abs();
        total += e;
        let near_x = third_x.iter().any(|&tx| (x as f64 - tx).abs() <= half_band_w);
        let near_y = third_y.iter().any(|&ty| (y as f64 - ty).abs() <= half_band_h);
        if near_x || near_y {
            near_thirds += e;
            near_pixels += 1;
        }
    }
    if total <= 0.0 {
        return 50.0;
    }

    let fraction = near_thirds / total;
    // Exact discrete fraction of pixels covered by the thirds bands, so
    // uniform edge energy scores 50.0 regardless of image dimensions.
    let baseline = near_pixels as f64 / (lw as f64 * lh as f64);

    let lift = (fraction - baseline) / (1.0 - baseline).max(1e-6);
    (50.0 + 50.0 * lift).clamp(0.0, 100.0)
}

/// Portrait-only: how prominent the detected face is in the frame, as a
/// blend of size (bigger = more deliberate a subject, saturating around a
/// fairly close 15%-of-frame crop) and centering (closer to the frame
/// center scores higher). Purely geometric -- see the module doc for why
/// eyes-open detection isn't part of this.
fn subject_quality_score(face: &FaceBox, img_w: u32, img_h: u32) -> f64 {
    let img_w = img_w as f64;
    let img_h = img_h as f64;
    if img_w <= 0.0 || img_h <= 0.0 {
        return 50.0;
    }

    let face_w = (face.x2 - face.x1).max(0.0) as f64;
    let face_h = (face.y2 - face.y1).max(0.0) as f64;
    let area_frac = (face_w * face_h) / (img_w * img_h);
    let size_score = (100.0 * area_frac / 0.15).clamp(0.0, 100.0);

    let face_cx = (face.x1 + face.x2) as f64 / 2.0;
    let face_cy = (face.y1 + face.y2) as f64 / 2.0;
    let dx = face_cx - img_w / 2.0;
    let dy = face_cy - img_h / 2.0;
    let dist = (dx * dx + dy * dy).sqrt();
    let max_dist = (img_w * img_w + img_h * img_h).sqrt() / 2.0;
    let centering_score = (100.0 * (1.0 - dist / max_dist.max(1.0))).clamp(0.0, 100.0);

    0.6 * size_score + 0.4 * centering_score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_checkerboard_scores_near_neutral_composition() {
        let img = RgbImage::from_fn(300, 300, |x, y| {
            if (x / 5 + y / 5) % 2 == 0 {
                image::Rgb([220, 220, 220])
            } else {
                image::Rgb([80, 80, 80])
            }
        });
        let m = compute(&img, 100.0, None);
        let comp = m.composition.unwrap();
        assert!(
            (45.0..=55.0).contains(&comp),
            "uniform edge energy should score ~50, got {comp}"
        );
    }

    #[test]
    fn stripe_on_rule_of_thirds_scores_high_composition() {
        let img = RgbImage::from_fn(300, 300, |x, _y| {
            if (x as i32 - 100).abs() <= 2 {
                image::Rgb([240, 240, 240])
            } else {
                image::Rgb([40, 40, 40])
            }
        });
        let m = compute(&img, 100.0, None);
        let comp = m.composition.unwrap();
        assert!(comp > 75.0, "expected high composition score, got {comp}");
    }

    #[test]
    fn dead_center_detail_scores_low_composition() {
        let img = RgbImage::from_fn(300, 300, |x, y| {
            if (140..160).contains(&x) && (140..160).contains(&y) && (x + y) % 2 == 0 {
                image::Rgb([240, 240, 240])
            } else {
                image::Rgb([100, 100, 100])
            }
        });
        let m = compute(&img, 100.0, None);
        let comp = m.composition.unwrap();
        assert!(comp < 40.0, "expected low composition score, got {comp}");
    }

    #[test]
    fn normalize_sharpness_curve_matches_calibration_points() {
        assert_eq!(normalize_sharpness(-5.0), 0.0);
        assert_eq!(normalize_sharpness(0.0), 0.0);
        assert!((normalize_sharpness(100.0) - 50.0).abs() < 1e-9);
        assert!((normalize_sharpness(300.0) - 75.0).abs() < 1e-9);
    }

    #[test]
    fn overall_score_renormalizes_and_handles_zero_weights() {
        let m = Metrics {
            sharpness: 80.0,
            exposure: 60.0,
            contrast: 40.0,
            color: 20.0,
            composition: Some(50.0),
            subject: None,
        };
        let w = Weights::default();
        // Present weights: 0.25 + 0.15 + 0.10 + 0.10 + 0.25 = 0.85
        let expected = (0.25 * 80.0 + 0.15 * 60.0 + 0.10 * 40.0 + 0.10 * 20.0 + 0.25 * 50.0) / 0.85;
        assert!((overall_score(&m, &w) - expected).abs() < 1e-6);

        let zero_w = Weights {
            sharpness: 0.0,
            exposure: 0.0,
            contrast: 0.0,
            color: 0.0,
            composition: 0.0,
            subject: 0.0,
        };
        assert_eq!(overall_score(&m, &zero_w), 0.0);
    }
}
