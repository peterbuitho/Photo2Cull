use std::path::Path;

use image::RgbImage;

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
/// Decoding at a capped resolution (rather than full sensor resolution,
/// often 40-100MP) is what keeps scanning thousands of files fast; sharpness
/// scoring doesn't need full resolution to be meaningful.
pub fn decode_raw(path: &Path, max_dim: usize) -> anyhow::Result<RgbImage> {
    let out = imagepipe::simple_decode_8bit(path, max_dim, max_dim)
        .map_err(|e| anyhow::anyhow!("failed to decode {}: {e}", path.display()))?;
    RgbImage::from_raw(out.width as u32, out.height as u32, out.data)
        .ok_or_else(|| anyhow::anyhow!("decoded buffer size mismatch for {}", path.display()))
}
