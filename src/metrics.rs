//! Phase 1 of the Overall-score ranking system: exposure, contrast, and
//! color, plus the weighted blend that combines them with sharpness (and,
//! later, composition and subject quality) into a single Overall score.
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

use image::{GrayImage, RgbImage};

/// Per-photo factor scores, each normalized to roughly 0-100 (higher is
/// better). `composition`/`subject` are `None` until later phases add them;
/// `overall_score` renormalizes over whatever's present.
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

/// Compute the Phase-1 factors (exposure, contrast, color) plus normalized
/// sharpness from `raw_sharpness` (the existing absolute score). Composition
/// and subject quality aren't implemented yet.
pub fn compute(rgb: &RgbImage, raw_sharpness: f64) -> Metrics {
    Metrics {
        sharpness: normalize_sharpness(raw_sharpness),
        exposure: exposure_score(rgb),
        contrast: contrast_score(rgb),
        color: color_score(rgb),
        composition: None,
        subject: None,
    }
}

fn to_gray(rgb: &RgbImage) -> GrayImage {
    image::DynamicImage::ImageRgb8(rgb.clone()).to_luma8()
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

/// 100 = midtone-centered with no clipping; penalized for both deviation
/// from a midtone mean and for blown-highlight/crushed-shadow pixel
/// fractions. A crude, scene-blind proxy -- a legitimately bright (snow) or
/// dark (night) scene will score low despite being "correctly" exposed.
fn exposure_score(rgb: &RgbImage) -> f64 {
    let gray = to_gray(rgb);
    let pixels = gray.as_raw();
    let n = pixels.len() as f64;
    if n == 0.0 {
        return 0.0;
    }

    let mean = pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let clipped = pixels.iter().filter(|&&p| p <= 3 || p >= 252).count() as f64 / n;

    let brightness_score = 100.0 * (1.0 - (mean - 128.0).abs() / 128.0);
    let clipping_penalty = 100.0 * clipped;
    (brightness_score - clipping_penalty).clamp(0.0, 100.0)
}

/// 100 = strong tonal spread. Uses the luminance histogram's standard
/// deviation, scaled so ~64 (a commonly-cited "good contrast" ballpark)
/// maps to 100.
fn contrast_score(rgb: &RgbImage) -> f64 {
    let gray = to_gray(rgb);
    let pixels = gray.as_raw();
    let n = pixels.len() as f64;
    if n == 0.0 {
        return 0.0;
    }

    let mean = pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let variance = pixels
        .iter()
        .map(|&p| {
            let d = p as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let std_dev = variance.sqrt();

    (100.0 * std_dev / 64.0).clamp(0.0, 100.0)
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
