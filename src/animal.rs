//! Animal detection via a bundled YOLOX-Nano COCO model, used to find the
//! subject of an animal photo so sharpness is scored on the animal rather
//! than on a fixed crop (see `assets/models/NOTICE.md` for the model's
//! license and I/O layout).
//!
//! Only COCO's ten animal classes are recognized (bird, cat, dog, horse,
//! sheep, cow, elephant, bear, zebra, giraffe). Other wildlife -- insects,
//! reptiles, most non-livestock mammals -- won't be detected and fall
//! through to the usual Landscape/Object heuristic.

use std::sync::OnceLock;

use image::RgbImage;
use ndarray::Array4;
use tract::prelude::*;

use crate::classify::{FaceBox, nms};

tract::impl_ndarray_interop!();

const MODEL_BYTES: &[u8] = include_bytes!("../assets/models/yolox_nano.onnx");
const INPUT_SIZE: u32 = 416;
const STRIDES: [u32; 3] = [8, 16, 32];
const NUM_CLASSES: usize = 80;
const ROW_LEN: usize = 5 + NUM_CLASSES;
/// COCO class indices for bird..giraffe.
const ANIMAL_CLASSES: std::ops::RangeInclusive<usize> = 14..=23;
/// Minimum objectness * class score to count as a detection.
const CONF_THRESHOLD: f32 = 0.35;
/// Ignore detections smaller than this fraction of the frame area. Lower
/// than the face cutoff: a bird in a wide frame is still the subject.
const MIN_AREA_FRACTION: f32 = 0.003;
const PAD_VALUE: u8 = 114;

fn model() -> &'static Runnable {
    static MODEL: OnceLock<Runnable> = OnceLock::new();
    MODEL.get_or_init(|| {
        tract::onnx()
            .and_then(|onnx| onnx.load_buffer(MODEL_BYTES))
            .and_then(|m| m.into_model())
            .and_then(|m| m.into_runnable())
            .expect("bundled animal detection model failed to load")
    })
}

/// (grid_x, grid_y, stride) per output row, in the order the model emits
/// them: the stride-8 grid row-major, then stride 16, then stride 32.
fn grid() -> &'static [(f32, f32, f32)] {
    static GRID: OnceLock<Vec<(f32, f32, f32)>> = OnceLock::new();
    GRID.get_or_init(|| {
        let mut cells = Vec::new();
        for stride in STRIDES {
            let n = INPUT_SIZE / stride;
            for y in 0..n {
                for x in 0..n {
                    cells.push((x as f32, y as f32, stride as f32));
                }
            }
        }
        cells
    })
}

/// Detect animals in `rgb`, in `rgb`'s own pixel coordinates, most
/// confident first. `FaceBox` doubles as the generic subject box here.
/// Returns an empty vec on any inference failure, same as face detection.
pub fn detect_animals(rgb: &RgbImage) -> Vec<FaceBox> {
    let (img_w, img_h) = rgb.dimensions();
    if img_w == 0 || img_h == 0 {
        return Vec::new();
    }

    // Letterbox: fit inside INPUT_SIZE^2 keeping aspect, anchored top-left.
    let scale = (INPUT_SIZE as f32 / img_w as f32).min(INPUT_SIZE as f32 / img_h as f32);
    let new_w = ((img_w as f32 * scale).round() as u32).clamp(1, INPUT_SIZE);
    let new_h = ((img_h as f32 * scale).round() as u32).clamp(1, INPUT_SIZE);
    let resized = image::imageops::resize(rgb, new_w, new_h, image::imageops::FilterType::Triangle);

    let input = Array4::from_shape_fn(
        (1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize),
        |(_, c, y, x)| {
            let (x, y) = (x as u32, y as u32);
            if x < new_w && y < new_h {
                // The model wants BGR.
                resized.get_pixel(x, y)[2 - c] as f32
            } else {
                PAD_VALUE as f32
            }
        },
    );

    let Ok(tensor) = input.tract() else {
        return Vec::new();
    };
    let Ok(result) = model().run([tensor]) else {
        return Vec::new();
    };
    let Some(Ok(out)) = result.first().map(|t| t.as_slice::<f32>()) else {
        return Vec::new();
    };

    let grid = grid();
    let n = (out.len() / ROW_LEN).min(grid.len());
    let mut candidates: Vec<(f32, [f32; 4])> = Vec::new();
    for (i, &(gx, gy, stride)) in grid.iter().enumerate().take(n) {
        let row = &out[i * ROW_LEN..(i + 1) * ROW_LEN];
        let objectness = row[4];
        let best_class = ANIMAL_CLASSES
            .map(|c| row[5 + c])
            .fold(0.0_f32, f32::max);
        let score = objectness * best_class;
        if score < CONF_THRESHOLD {
            continue;
        }
        let cx = (row[0] + gx) * stride;
        let cy = (row[1] + gy) * stride;
        let w = row[2].exp() * stride;
        let h = row[3].exp() * stride;
        candidates.push((score, [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]));
    }

    let frame_area = img_w as f32 * img_h as f32;
    nms(candidates)
        .into_iter()
        .map(|(score, [x1, y1, x2, y2])| FaceBox {
            x1: (x1 / scale).clamp(0.0, img_w as f32),
            y1: (y1 / scale).clamp(0.0, img_h as f32),
            x2: (x2 / scale).clamp(0.0, img_w as f32),
            y2: (y2 / scale).clamp(0.0, img_h as f32),
            score,
        })
        .filter(|b| {
            let area = (b.x2 - b.x1).max(0.0) * (b.y2 - b.y1).max(0.0);
            area / frame_area >= MIN_AREA_FRACTION
        })
        .collect()
}

/// The largest detected animal, on the assumption that it's the subject.
pub fn detect_main_animal(rgb: &RgbImage) -> Option<FaceBox> {
    detect_animals(rgb).into_iter().max_by(|a, b| {
        let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        area_a.partial_cmp(&area_b).unwrap()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decode grid must line up 1:1 with the model's output rows.
    #[test]
    fn grid_count_matches_model_output() {
        assert_eq!(grid().len(), 3549);
    }
}
