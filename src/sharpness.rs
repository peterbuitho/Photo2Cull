use image::{GrayImage, Luma, RgbImage};
use imageproc::filter::laplacian_filter;

use crate::classify::FaceBox;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhotoMode {
    Landscape,
    Portrait,
    Object,
    Animal,
}

impl PhotoMode {
    pub const ALL: [PhotoMode; 4] = [
        PhotoMode::Landscape,
        PhotoMode::Portrait,
        PhotoMode::Object,
        PhotoMode::Animal,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            PhotoMode::Landscape => "Landscape",
            PhotoMode::Portrait => "Portrait",
            PhotoMode::Object => "Object",
            PhotoMode::Animal => "Animal",
        }
    }

    /// Fraction of width/height kept (centered) when scoring; 1.0 = whole frame.
    ///
    /// Used when there's no detected subject box to score instead (see
    /// `score`): landscapes are scored edge-to-edge, while a centered/tighter
    /// crop stands in for "where the subject probably is" for
    /// portraits/objects/animals, so a deliberately blurred background
    /// doesn't tank the score.
    fn region_fraction(&self) -> f32 {
        match self {
            PhotoMode::Landscape => 1.0,
            PhotoMode::Object | PhotoMode::Animal => 0.6,
            PhotoMode::Portrait => 0.45,
        }
    }
}

/// How sharpness is measured when there's no detected face/animal box to
/// score instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharpnessMethod {
    /// A fixed crop per photo type (whole frame for Landscape, the centre
    /// for Object/Animal/Portrait).
    FixedRegion,
    /// The sharpest tiles anywhere in the frame, regardless of type. Suits
    /// shallow depth of field (macro) and off-centre subjects, where the
    /// in-focus part is small and not necessarily in the middle.
    SharpestRegion,
}

impl SharpnessMethod {
    pub const ALL: [SharpnessMethod; 2] =
        [SharpnessMethod::FixedRegion, SharpnessMethod::SharpestRegion];

    pub fn label(&self) -> &'static str {
        match self {
            SharpnessMethod::FixedRegion => "Fixed region",
            SharpnessMethod::SharpestRegion => "Sharpest region",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            SharpnessMethod::FixedRegion => {
                "When no face or animal is found, scores a fixed crop chosen by photo type (whole frame for Landscape, centre otherwise)."
            }
            SharpnessMethod::SharpestRegion => {
                "When no face or animal is found, scores the sharpest tiles anywhere in the frame. Better for macro and off-centre subjects; scores run higher than Fixed region."
            }
        }
    }
}

/// Tiles along the longer edge for [`SharpnessMethod::SharpestRegion`].
const TILE_GRID: u32 = 16;
/// Fraction of tiles (the sharpest ones) averaged into the score. Small
/// enough that a thin in-focus sliver dominates, large enough that one
/// high-contrast tile (a specular highlight, a hard edge) doesn't.
const TOP_TILE_FRACTION: f64 = 0.05;

fn center_crop(img: &GrayImage, fraction: f32) -> GrayImage {
    if fraction >= 1.0 {
        return img.clone();
    }
    let (w, h) = img.dimensions();
    let cw = ((w as f32) * fraction).round().max(1.0) as u32;
    let ch = ((h as f32) * fraction).round().max(1.0) as u32;
    let x = (w - cw) / 2;
    let y = (h - ch) / 2;
    image::imageops::crop_imm(img, x, y, cw, ch).to_image()
}

/// Crop around a detected subject box (a face or an animal), padded out to
/// ~1.7x its size (centered on the box) so the sample includes a bit of
/// surrounding context rather than just the tight rectangle, clamped to
/// the image bounds.
fn face_crop(img: &GrayImage, face: &FaceBox) -> GrayImage {
    let (iw, ih) = img.dimensions();
    let cx = (face.x1 + face.x2) / 2.0;
    let cy = (face.y1 + face.y2) / 2.0;
    let w = ((face.x2 - face.x1) * 1.7).max(1.0);
    let h = ((face.y2 - face.y1) * 1.7).max(1.0);

    let x = (cx - w / 2.0).clamp(0.0, iw as f32 - 1.0);
    let y = (cy - h / 2.0).clamp(0.0, ih as f32 - 1.0);
    let w = w.min(iw as f32 - x).max(1.0) as u32;
    let h = h.min(ih as f32 - y).max(1.0) as u32;
    image::imageops::crop_imm(img, x as u32, y as u32, w, h).to_image()
}

fn variance(values: impl Iterator<Item = i16> + Clone) -> f64 {
    let n = values.clone().count() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = values.clone().map(|v| v as f64).sum::<f64>() / n;
    values
        .map(|v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n
}

fn variance_of_laplacian(region: &GrayImage) -> f64 {
    let lap: image::ImageBuffer<Luma<i16>, Vec<i16>> = laplacian_filter(region);
    variance(lap.as_raw().iter().copied())
}

/// Mean Laplacian variance of the sharpest [`TOP_TILE_FRACTION`] of tiles.
/// The Laplacian is computed once over the whole frame, then variance is
/// taken per tile.
fn sharpest_region_score(gray: &GrayImage) -> f64 {
    let (w, h) = gray.dimensions();
    let tile = (w.max(h) / TILE_GRID).max(8);
    if w < tile || h < tile {
        return variance_of_laplacian(gray);
    }
    let lap: image::ImageBuffer<Luma<i16>, Vec<i16>> = laplacian_filter(gray);
    let raw = lap.as_raw();

    let mut tile_scores: Vec<f64> = Vec::new();
    for ty in (0..h).step_by(tile as usize) {
        for tx in (0..w).step_by(tile as usize) {
            let (x_end, y_end) = ((tx + tile).min(w), (ty + tile).min(h));
            // Skip thin leftover strips at the right/bottom edge.
            if x_end - tx < tile / 2 || y_end - ty < tile / 2 {
                continue;
            }
            let values = (ty..y_end).flat_map(|y| {
                let row = (y * w) as usize;
                raw[row + tx as usize..row + x_end as usize].iter().copied()
            });
            tile_scores.push(variance(values));
        }
    }
    if tile_scores.is_empty() {
        return variance_of_laplacian(gray);
    }
    tile_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let k = ((tile_scores.len() as f64 * TOP_TILE_FRACTION).ceil() as usize).max(1);
    tile_scores[..k].iter().sum::<f64>() / k as f64
}

/// Variance-of-Laplacian sharpness score. Higher means sharper. If `subject`
/// is given (a face for Portrait, an animal for Animal), scores that box.
/// Otherwise scores per `method`: a centered crop sized per `mode`, or the
/// sharpest tiles anywhere in the frame.
pub fn score(
    rgb: &RgbImage,
    mode: PhotoMode,
    subject: Option<&FaceBox>,
    method: SharpnessMethod,
) -> f64 {
    let gray: GrayImage = image::DynamicImage::ImageRgb8(rgb.clone()).to_luma8();
    if let Some(b) = subject {
        return variance_of_laplacian(&face_crop(&gray, b));
    }
    match method {
        SharpnessMethod::FixedRegion => {
            variance_of_laplacian(&center_crop(&gray, mode.region_fraction()))
        }
        SharpnessMethod::SharpestRegion => sharpest_region_score(&gray),
    }
}

/// When no face is found, guess Landscape vs Object from how uniform the
/// sharpness is across the frame. A single object shot with shallow depth
/// of field is much sharper in the center than at the edges; a landscape
/// (typically a small aperture, front-to-back focus) is comparatively
/// uniform. This is a coarse heuristic, not real scene classification.
pub fn guess_landscape_or_object(rgb: &RgbImage) -> PhotoMode {
    let gray: GrayImage = image::DynamicImage::ImageRgb8(rgb.clone()).to_luma8();
    let full = variance_of_laplacian(&gray);
    if full <= 1.0 {
        return PhotoMode::Landscape;
    }
    let center = variance_of_laplacian(&center_crop(&gray, 0.5));
    if center / full > 1.6 {
        PhotoMode::Object
    } else {
        PhotoMode::Landscape
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flat mid-gray frame with a small high-contrast checkerboard patch in
    /// the top-left corner: a stand-in for a tiny in-focus subject
    /// off-centre against a smooth (blurred) background.
    fn frame_with_corner_detail() -> RgbImage {
        RgbImage::from_fn(640, 480, |x, y| {
            if x < 80 && y < 80 && (x / 4 + y / 4) % 2 == 0 {
                image::Rgb([230, 230, 230])
            } else {
                image::Rgb([100, 100, 100])
            }
        })
    }

    #[test]
    fn sharpest_region_finds_off_centre_detail_the_center_crop_misses() {
        let img = frame_with_corner_detail();
        let fixed = score(&img, PhotoMode::Object, None, SharpnessMethod::FixedRegion);
        let sharpest = score(&img, PhotoMode::Object, None, SharpnessMethod::SharpestRegion);
        assert!(fixed < 1.0, "centre crop is flat, got {fixed}");
        assert!(sharpest > 100.0, "corner detail should register, got {sharpest}");
    }

    #[test]
    fn sharpest_region_stays_low_for_a_flat_frame() {
        let img = RgbImage::from_pixel(640, 480, image::Rgb([120, 120, 120]));
        let s = score(&img, PhotoMode::Landscape, None, SharpnessMethod::SharpestRegion);
        assert!(s < 1.0, "got {s}");
    }
}
