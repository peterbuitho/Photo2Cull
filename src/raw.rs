use std::path::Path;

use anyhow::Context;
use image::{DynamicImage, RgbImage};
use rawler::decoders::RawDecodeParams;
use rawler::rawsource::RawSource;

/// File extensions treated as RAW photos. Matched case-insensitively.
pub const RAW_EXTENSIONS: &[&str] = &[
    "cr2", "cr3", "crw", "nef", "nrw", "arw", "srf", "sr2", "orf", "rw2",
    "raf", "dng", "pef", "ptx", "srw", "x3f", "3fr", "erf", "kdc", "mrw",
    "raw", "mos", "iiq", "rwl", "dcr",
];

pub fn is_raw_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| RAW_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Decode a RAW file to RGB8, downsized so neither edge exceeds `max_dim`.
///
/// Rather than demosaicing the sensor's CFA data ourselves, this extracts
/// the camera's embedded preview JPEG (every RAW file carries at least one,
/// for the camera's own rear-LCD display and for fast previews in other
/// tools). That sidesteps needing per-camera demosaic/calibration support
/// for sharpness scoring, which matters because that database lags real
/// camera releases by years (e.g. it lacked the Fujifilm X-T3 entirely) --
/// embedded-preview extraction only needs to recognize the container
/// format, not calibrate the sensor.
pub fn decode_raw(path: &Path, max_dim: usize) -> anyhow::Result<RgbImage> {
    let source =
        RawSource::new(path).with_context(|| format!("failed to open {}", path.display()))?;
    let decoder = rawler::get_decoder(&source)
        .map_err(|e| anyhow::anyhow!("unsupported RAW file {}: {e}", path.display()))?;
    let params = RawDecodeParams::default();

    let dynamic = decoder
        .full_image(&source, &params)
        .ok()
        .flatten()
        .or_else(|| decoder.preview_image(&source, &params).ok().flatten())
        .or_else(|| decoder.thumbnail_image(&source, &params).ok().flatten())
        .ok_or_else(|| anyhow::anyhow!("no usable preview found in {}", path.display()))?;

    // The embedded preview is stored in the sensor's native (landscape)
    // orientation; a portrait shot needs the EXIF Orientation tag applied
    // to come out upright. This matters beyond just display: face
    // detection and the center-crop scoring regions both assume an
    // upright image, so an un-rotated portrait would hurt both.
    let orientation = decoder
        .raw_metadata(&source, &params)
        .ok()
        .and_then(|m| m.exif.orientation);
    let dynamic = apply_exif_orientation(dynamic, orientation);

    let resized = dynamic.resize(
        max_dim as u32,
        max_dim as u32,
        image::imageops::FilterType::Triangle,
    );
    Ok(resized.to_rgb8())
}

/// Rotate/flip `img` per the EXIF Orientation tag (values 1-8) so it comes
/// out upright. Values 5 and 7 (mirrored + rotated) are vanishingly rare in
/// real camera output -- effectively only ever produced by some scanning
/// software -- so they're handled but not as rigorously verified as the
/// plain-rotation cases (3/6/8), which dominate in practice.
fn apply_exif_orientation(img: DynamicImage, orientation: Option<u16>) -> DynamicImage {
    match orientation {
        Some(2) => img.fliph(),
        Some(3) => img.rotate180(),
        Some(4) => img.flipv(),
        Some(5) => img.fliph().rotate270(),
        Some(6) => img.rotate90(),
        Some(7) => img.fliph().rotate90(),
        Some(8) => img.rotate270(),
        _ => img,
    }
}
