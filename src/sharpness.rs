use image::{GrayImage, Luma, RgbImage};
use imageproc::filter::laplacian_filter;

use crate::classify::FaceBox;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhotoMode {
    Landscape,
    Portrait,
    Object,
}

impl PhotoMode {
    pub const ALL: [PhotoMode; 3] = [PhotoMode::Landscape, PhotoMode::Portrait, PhotoMode::Object];

    pub fn label(&self) -> &'static str {
        match self {
            PhotoMode::Landscape => "Landscape",
            PhotoMode::Portrait => "Portrait",
            PhotoMode::Object => "Object",
        }
    }

    /// Fraction of width/height kept (centered) when scoring; 1.0 = whole frame.
    ///
    /// Used when there's no detected face to score instead (see `score`):
    /// landscapes are scored edge-to-edge, while a centered/tighter crop
    /// stands in for "where the subject probably is" for portraits/objects,
    /// so a deliberately blurred background doesn't tank the score.
    fn region_fraction(&self) -> f32 {
        match self {
            PhotoMode::Landscape => 1.0,
            PhotoMode::Object => 0.6,
            PhotoMode::Portrait => 0.45,
        }
    }
}

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

/// Crop around a detected face, padded out to ~1.7x its size (centered on
/// the face) so the sample includes a bit of surrounding context rather
/// than just the tight face rectangle, clamped to the image bounds.
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

fn variance_of_laplacian(region: &GrayImage) -> f64 {
    let lap: image::ImageBuffer<Luma<i16>, Vec<i16>> = laplacian_filter(region);
    let values = lap.as_raw();
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n;
    values
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n
}

/// Variance-of-Laplacian sharpness score. Higher means sharper. If `face` is
/// given (a Portrait with a detected face), scores that face region;
/// otherwise falls back to a centered crop sized per `mode`.
pub fn score(rgb: &RgbImage, mode: PhotoMode, face: Option<&FaceBox>) -> f64 {
    let gray: GrayImage = image::DynamicImage::ImageRgb8(rgb.clone()).to_luma8();
    let region = match face {
        Some(f) => face_crop(&gray, f),
        None => center_crop(&gray, mode.region_fraction()),
    };
    variance_of_laplacian(&region)
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
