use std::path::Path;

use anyhow::Context;
use image::{metadata::Orientation, DynamicImage, ImageDecoder, ImageReader, RgbImage};

use crate::raw::{self, decode_raw};

/// Standard (already-demosaiced) image formats, matched case-insensitively.
/// HEIC/HEIF isn't included: it needs H.265/HEVC decoding, which the
/// `image` crate doesn't support, and the available pure-Rust decoders are
/// all early (0.1-0.2) with real risk of failing on real-world files.
pub const STANDARD_EXTENSIONS: &[&str] =
    &["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"];

pub fn is_standard_image(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| STANDARD_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Any photo format this app can decode -- RAW or standard.
pub fn is_supported_photo(path: &Path) -> bool {
    raw::is_raw_file(path) || is_standard_image(path)
}

/// Decode a standard-format image to RGB8, downsized so neither edge
/// exceeds `max_dim`, with EXIF orientation applied.
fn decode_standard(path: &Path, max_dim: usize) -> anyhow::Result<RgbImage> {
    let mut decoder = ImageReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .into_decoder()
        .with_context(|| format!("unrecognized image format: {}", path.display()))?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);

    let mut dynamic = DynamicImage::from_decoder(decoder)
        .with_context(|| format!("failed to decode {}", path.display()))?;
    dynamic.apply_orientation(orientation);

    let resized = dynamic.resize(
        max_dim as u32,
        max_dim as u32,
        image::imageops::FilterType::Triangle,
    );
    Ok(resized.to_rgb8())
}

/// Decode any supported photo (RAW or standard format) to RGB8, downsized
/// so neither edge exceeds `max_dim`.
pub fn decode_photo(path: &Path, max_dim: usize) -> anyhow::Result<RgbImage> {
    if raw::is_raw_file(path) {
        decode_raw(path, max_dim)
    } else {
        decode_standard(path, max_dim)
    }
}
