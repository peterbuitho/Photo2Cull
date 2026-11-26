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

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;

    /// 3x2 image with distinct pixel values so every orientation 1..=8 maps
    /// corners and dimensions uniquely:
    ///   (0,0)=1  (1,0)=2  (2,0)=3
    ///   (0,1)=4  (1,1)=5  (2,1)=6
    fn sample_3x2() -> DynamicImage {
        let mut img = RgbImage::new(3, 2);
        for y in 0..2u32 {
            for x in 0..3u32 {
                let v = (y * 3 + x + 1) as u8;
                img.put_pixel(x, y, image::Rgb([v, v * 10, v * 20]));
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    fn grid_ids(img: &DynamicImage) -> Vec<Vec<u8>> {
        let (w, h) = img.dimensions();
        (0..h)
            .map(|y| (0..w).map(|x| img.get_pixel(x, y)[0]).collect())
            .collect()
    }

    #[test]
    fn exif_orientations_transform_as_expected() {
        let src = sample_3x2();

        assert_eq!(grid_ids(&apply_exif_orientation(src.clone(), None)), vec![vec![1, 2, 3], vec![4, 5, 6]]);
        assert_eq!(grid_ids(&apply_exif_orientation(src.clone(), Some(1))), vec![vec![1, 2, 3], vec![4, 5, 6]]);
        assert_eq!(grid_ids(&apply_exif_orientation(src.clone(), Some(2))), vec![vec![3, 2, 1], vec![6, 5, 4]]);
        assert_eq!(grid_ids(&apply_exif_orientation(src.clone(), Some(3))), vec![vec![6, 5, 4], vec![3, 2, 1]]);
        assert_eq!(grid_ids(&apply_exif_orientation(src.clone(), Some(4))), vec![vec![4, 5, 6], vec![1, 2, 3]]);
        assert_eq!(
            grid_ids(&apply_exif_orientation(src.clone(), Some(5))),
            vec![vec![1, 4], vec![2, 5], vec![3, 6]]
        );
        assert_eq!(
            grid_ids(&apply_exif_orientation(src.clone(), Some(6))),
            vec![vec![4, 1], vec![5, 2], vec![6, 3]]
        );
        assert_eq!(
            grid_ids(&apply_exif_orientation(src.clone(), Some(7))),
            vec![vec![6, 3], vec![5, 2], vec![4, 1]]
        );
        assert_eq!(
            grid_ids(&apply_exif_orientation(src, Some(8))),
            vec![vec![3, 6], vec![2, 5], vec![1, 4]]
        );
    }
}
