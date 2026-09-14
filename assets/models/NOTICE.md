`version-RFB-320.onnx` is the "RFB-320 simplified" face detection model from
[Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
by linzai, MIT licensed (see `LICENSE` in this directory).

- Input: 320x240 RGB, `(pixel - 127) / 128` per channel, CHW, batch-of-1, float32.
- Outputs: `confidences` `[1, N, 2]` (background, face — already softmaxed) and
  `boxes` `[1, N, 4]` (`x1, y1, x2, y2`, normalized 0..1). Prior-box decoding is
  already baked into the exported graph, so no separate anchor-decode step is
  needed downstream — only score thresholding + NMS.
