use image::{GrayImage, Luma, RgbImage};
use imageproc::filter::laplacian_filter;

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
    /// This is a composition heuristic, not subject detection: landscapes
    /// are scored edge-to-edge, while portraits/objects assume the subject
    /// is roughly centered and shot with shallower depth of field, so we
    /// don't penalize a deliberately blurred background. Real face/eye
    /// detection would replace this, but that needs a model (e.g. via a
    /// pure-Rust ONNX runtime) which is future work, not a v1 requirement.
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

/// Variance-of-Laplacian sharpness score. Higher means sharper; the scored
/// region is centered and sized according to `mode` (see [`PhotoMode::region_fraction`]).
pub fn score(rgb: &RgbImage, mode: PhotoMode) -> f64 {
    let gray: GrayImage = image::DynamicImage::ImageRgb8(rgb.clone()).to_luma8();
    let region = center_crop(&gray, mode.region_fraction());
    let lap: image::ImageBuffer<Luma<i16>, Vec<i16>> = laplacian_filter(&region);

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
