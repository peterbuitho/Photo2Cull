`version-RFB-320.onnx` is the "RFB-320 simplified" face detection model from
[Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
by linzai, MIT licensed (see `LICENSE` in this directory).

- Input: 320x240 RGB, `(pixel - 127) / 128` per channel, CHW, batch-of-1, float32.
- Outputs: `confidences` `[1, N, 2]` (background, face — already softmaxed) and
  `boxes` `[1, N, 4]` (`x1, y1, x2, y2`, normalized 0..1). Prior-box decoding is
  already baked into the exported graph, so no separate anchor-decode step is
  needed downstream — only score thresholding + NMS.

---

`yolox_nano.onnx` is the YOLOX-Nano COCO detector from
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) by Megvii, Apache 2.0
licensed (see `LICENSE-YOLOX` in this directory), release 0.1.1rc0.

- Input: 416x416 BGR, 0-255 (no mean/std), CHW, batch-of-1, float32. The
  image is letterboxed (aspect preserved, top-left aligned, padded with 114).
- Output: `[1, 3549, 85]` — per anchor `cx, cy, w, h` (raw, still needing the
  grid/stride decode for strides 8/16/32), objectness, then 80 COCO class
  scores (objectness and class scores already sigmoided). Only the animal
  classes (bird, cat, dog, horse, sheep, cow, elephant, bear, zebra,
  giraffe) are used.
