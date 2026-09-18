//! Per-photo subject classification: face detection (for Portrait) via a
//! bundled ONNX model, plus a cheap sharpness-uniformity heuristic to tell
//! Landscape from Object when no face is found.
//!
//! Model: "RFB-320" from Ultra-Light-Fast-Generic-Face-Detector-1MB
//! (MIT licensed, see assets/models/NOTICE.md). Its exported ONNX graph
//! outputs raw, un-decoded SSD regression deltas relative to a fixed set of
//! prior (anchor) boxes -- despite the reference PyTorch source suggesting
//! the decode is baked in, it measurably isn't for this exported file
//! (verified: output box values range roughly -5..5, not the 0..1 a decoded
//! normalized box would have). So prior generation and box decoding are
//! reimplemented here from the original repo's box_utils.py / fd_config.py.

use std::sync::OnceLock;

use image::RgbImage;
use ndarray::Array4;
use tract::prelude::*;

tract::impl_ndarray_interop!();

const MODEL_BYTES: &[u8] = include_bytes!("../assets/models/version-RFB-320.onnx");
const INPUT_W: u32 = 320;
const INPUT_H: u32 = 240;
const CONF_THRESHOLD: f32 = 0.7;
const IOU_THRESHOLD: f32 = 0.3;
const CENTER_VARIANCE: f32 = 0.1;
const SIZE_VARIANCE: f32 = 0.2;
/// Ignore detections smaller than this fraction of the frame area: a tiny
/// face in the background shouldn't turn an otherwise-landscape shot into
/// a "Portrait". Kept deliberately low -- real environmental portraits
/// (subject not tightly cropped) can have the face at well under 1% of
/// frame area while still clearly being the intended subject; verified
/// against real photos with a 99%+ confidence face at 0.6-1.4% area that
/// an earlier, stricter 2% cutoff was wrongly rejecting. The confidence
/// threshold above does most of the real discriminating work (background
/// faces in those same photos scored well under 0.5).
const MIN_FACE_AREA_FRACTION: f32 = 0.001;

/// A detected face, in pixel coordinates of the image it was detected in.
#[derive(Debug, Clone, Copy)]
pub struct FaceBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

fn model() -> &'static Runnable {
    static MODEL: OnceLock<Runnable> = OnceLock::new();
    MODEL.get_or_init(|| {
        tract::onnx()
            .and_then(|onnx| onnx.load_buffer(MODEL_BYTES))
            .and_then(|m| m.into_model())
            .and_then(|m| m.into_runnable())
            .expect("bundled face detection model failed to load")
    })
}

/// (center_x, center_y, w, h), normalized 0..1, matching the model's prior
/// (anchor) box layout for its 320x240 input: 4 feature-map layers, each
/// with its own grid size and set of anchor box sizes (in pixels).
fn generate_priors() -> Vec<[f32; 4]> {
    const LAYERS: [(u32, u32, &[f32]); 4] = [
        (40, 30, &[10.0, 16.0, 24.0]),
        (20, 15, &[32.0, 48.0]),
        (10, 8, &[64.0, 96.0]),
        (5, 4, &[128.0, 192.0, 256.0]),
    ];

    let mut priors = Vec::new();
    for &(fw, fh, min_boxes) in &LAYERS {
        for j in 0..fh {
            for i in 0..fw {
                let x_center = (i as f32 + 0.5) / fw as f32;
                let y_center = (j as f32 + 0.5) / fh as f32;
                for &min_box in min_boxes {
                    priors.push([
                        x_center,
                        y_center,
                        (min_box / INPUT_W as f32).clamp(0.0, 1.0),
                        (min_box / INPUT_H as f32).clamp(0.0, 1.0),
                    ]);
                }
            }
        }
    }
    priors
}

fn priors() -> &'static [[f32; 4]] {
    static PRIORS: OnceLock<Vec<[f32; 4]>> = OnceLock::new();
    PRIORS.get_or_init(generate_priors)
}

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let ix1 = a[0].max(b[0]);
    let iy1 = a[1].max(b[1]);
    let ix2 = a[2].min(b[2]);
    let iy2 = a[3].min(b[3]);
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    inter / (area_a + area_b - inter + 1e-5)
}

pub(crate) fn nms(mut candidates: Vec<(f32, [f32; 4])>) -> Vec<(f32, [f32; 4])> {
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut kept: Vec<(f32, [f32; 4])> = Vec::new();
    for (score, b) in candidates {
        if kept.iter().all(|&(_, k)| iou(k, b) <= IOU_THRESHOLD) {
            kept.push((score, b));
        }
    }
    kept
}

/// Detect faces in `rgb`, returned in `rgb`'s own pixel coordinates, most
/// confident first, with overlapping detections suppressed. Returns an
/// empty vec (rather than erroring) on any inference failure -- classifying
/// a photo as non-portrait is a reasonable degradation, not worth failing
/// the whole scan over.
pub fn detect_faces(rgb: &RgbImage) -> Vec<FaceBox> {
    let (img_w, img_h) = rgb.dimensions();
    if img_w == 0 || img_h == 0 {
        return Vec::new();
    }
    let resized = image::imageops::resize(
        rgb,
        INPUT_W,
        INPUT_H,
        image::imageops::FilterType::Triangle,
    );

    let input = Array4::from_shape_fn((1, 3, INPUT_H as usize, INPUT_W as usize), |(_, c, y, x)| {
        let px = resized.get_pixel(x as u32, y as u32);
        (px[c] as f32 - 127.0) / 128.0
    });

    let Ok(tensor) = input.tract() else {
        return Vec::new();
    };
    let Ok(result) = model().run([tensor]) else {
        return Vec::new();
    };
    let Some((Ok(confidences), Ok(boxes))) = result
        .first()
        .zip(result.get(1))
        .map(|(c, b)| (c.as_slice::<f32>(), b.as_slice::<f32>()))
    else {
        return Vec::new();
    };

    let priors = priors();
    let n = (confidences.len() / 2).min(priors.len());
    let mut candidates: Vec<(f32, [f32; 4])> = Vec::new();
    for i in 0..n {
        let face_score = confidences[i * 2 + 1];
        if face_score <= CONF_THRESHOLD {
            continue;
        }
        let [pcx, pcy, pw, ph] = priors[i];
        let lx = boxes[i * 4];
        let ly = boxes[i * 4 + 1];
        let lw = boxes[i * 4 + 2];
        let lh = boxes[i * 4 + 3];

        let cx = lx * CENTER_VARIANCE * pw + pcx;
        let cy = ly * CENTER_VARIANCE * ph + pcy;
        let w = (lw * SIZE_VARIANCE).exp() * pw;
        let h = (lh * SIZE_VARIANCE).exp() * ph;

        candidates.push((
            face_score,
            [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
        ));
    }

    let frame_area = img_w as f32 * img_h as f32;
    nms(candidates)
        .into_iter()
        .map(|(score, [x1, y1, x2, y2])| FaceBox {
            x1: x1.clamp(0.0, 1.0) * img_w as f32,
            y1: y1.clamp(0.0, 1.0) * img_h as f32,
            x2: x2.clamp(0.0, 1.0) * img_w as f32,
            y2: y2.clamp(0.0, 1.0) * img_h as f32,
            score,
        })
        .filter(|f| {
            let area = (f.x2 - f.x1).max(0.0) * (f.y2 - f.y1).max(0.0);
            area / frame_area >= MIN_FACE_AREA_FRACTION
        })
        .collect()
}

/// Convenience wrapper around [`detect_faces`] for callers that just want
/// "the" subject face, if any -- the largest detection by area, on the
/// assumption that the main subject of a portrait is usually the most
/// prominent face in frame.
pub fn detect_main_face(rgb: &RgbImage) -> Option<FaceBox> {
    detect_faces(rgb).into_iter().max_by(|a, b| {
        let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        area_a.partial_cmp(&area_b).unwrap()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guards the prior-box layout against silent drift: it must line up
    /// 1:1 with the model's actual output row count (4420, confirmed by
    /// inspecting the loaded model's output shape), or box decoding
    /// silently misaligns priors with the wrong regression deltas.
    #[test]
    fn prior_count_matches_model_output() {
        assert_eq!(generate_priors().len(), 4420);
    }
}
