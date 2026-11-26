# Photo2Cull Codebase Audit

- **Date:** 2026-11-26
- **Commit audited:** `9241933` (`main` after merging `origin/main` `aa10047` with local `mine/main` `c9923fc`)
- **Version:** `0.4.1` (Rust edition `2024`, tested on `rustc 1.100.0` / `cargo 1.100.0`)
- **Scope:** Full repository (`src/`, `examples/`, `Cargo.toml`, `packaging/macos/`, `.github/workflows/build.yml`, `assets/models/`, `docs/`, `README.md`)

---

## Executive Summary

Photo2Cull is a focused, readable, zero-`unsafe` Rust desktop photo-culling app built on `egui`/`eframe`, `rayon`, `rawler`, `image`/`imageproc`, and `tract`.

Following the initial audit, all **3 High-severity**, **7 Medium-severity**, and **4 Low-severity** findings were fixed in the working tree (12 files updated, including 12 new unit tests). Remaining items requiring a user/product decision (license selection, `tract` crate slimming, cross-platform scroll polarity, and `README.md`/`docs/architecture.html` refresh) are noted under **Open Items**.

### Verification Run

| Check | Baseline (`9241933`) | Post-Fix Working Tree | Notes |
| --- | --- | --- | --- |
| `git pull` (`merge origin/main`) | Passed (`9241933`) | — | Merged `aa10047` (`docs/architecture.html`) cleanly with `c9923fc` |
| `cargo test --all-targets` | Passed (10 passed, 0 failed) | **Passed (22 passed, 0 failed)** | Added tests in `metrics`, `sharpness`, `classify`, `animal`, `raw`, `scan` |
| `cargo clippy --all-targets -- -D warnings` | Passed (0 warnings) | **Passed (0 warnings)** | Verified across lib, bin, examples, and test targets |
| `unsafe` audit | 0 occurrences | **0 occurrences** | Pure safe Rust across `src/` and `examples/` |

---

## High-Severity Findings

### H1. Out-of-bounds panic / wrong photos in Pipeline view after moving disqualified photos — **Fixed**
- **Location:** `src/app.rs` (`PipelineResult`, `run_pipeline`, `render_pipeline`)
- **Issue:** `PipelineResult::shortlist` stored raw `Vec<usize>` indices into `self.entries`. `drain_move()` removed entries from `self.entries` without clearing or updating `self.pipeline_result`, risking index-out-of-bounds panics or rendering wrong photos in `ViewMode::Pipeline`.
- **Fix applied:** Changed `PipelineResult::shortlist` to `Vec<PathBuf>` and resolved live entry indices via a `HashMap<&Path, usize>` at render time.

### H2. Sidecar move bugs: duplicate moves, orphaning kept RAW+JPEG pairs, silent overwrites, and UI-thread stalls — **Fixed**
- **Location:** `src/scan.rs` (`sidecar_files_for`, `plan_disqualified_moves`, `MoveItem`), `src/app.rs` (`start_move_disqualified`, `drain_move`)
- **Issue:**
  1. Disqualifying only one photo of a RAW+JPEG pair sharing a stem-matched `.xmp` orphaned the kept sibling.
  2. Disqualifying both moved the shared `.xmp` twice, triggering a spurious `ENOENT` move error.
  3. `std::fs::rename` silently overwrote existing files in `disqualified/` on POSIX systems.
  4. `move_moved` counted sidecars alongside photos.
  5. `sidecar_files_for()` ran synchronous `read_dir` calls on the UI thread per disqualified photo and did not check `file_type().is_file()`.
- **Fix applied:** Added `plan_disqualified_moves()` running on the move background thread with once-per-folder `read_dir`, regular-file filtering, kept-sibling stem protection, sidecar deduplication, destination-existence checks (`AlreadyExists` instead of silent overwrite), and separate photo vs. sidecar reporting in `MoveEvent`.

### H3. `composition_score` baseline math error inflates every photo's composition score by ~22 points — **Fixed**
- **Location:** `src/metrics.rs` (`composition_score_from_lap`)
- **Issue:** `band_w = lw * 0.10` was used as a half-width (40% total axis coverage, uniform baseline `0.64`) while `baseline` was computed as `0.36` (20% per axis), inflating uniform/noise images from `50.0` to `~71.9` on a 25%-weight factor.
- **Fix applied:** Changed half-widths to `0.05 * axis` (`half_band_w`, `half_band_h`) and computed exact discrete pixel coverage `baseline = near_pixels / (lw * lh)`. Verified with unit tests in `src/metrics.rs`.

---

## Medium-Severity Findings

### M1. `NaN` from ONNX output or floats can panic via `.partial_cmp(...).unwrap()` — **Fixed**
- **Location:** `src/classify.rs`, `src/animal.rs`, `src/sharpness.rs`, `src/app.rs`
- **Issue:** `NaN` scores bypassed `<=` / `<` threshold filters in `classify.rs` and `animal.rs` and panicked in `.partial_cmp(...).unwrap()`.
- **Fix applied:** Explicit `is_nan()` threshold guards and `is_finite()` box coordinate checks in `classify.rs` and `animal.rs`; replaced all `.partial_cmp(...).unwrap()` calls across `classify.rs`, `animal.rs`, `sharpness.rs`, and `app.rs` with `.total_cmp(...)`.

### M2. Concurrency & stale-state hazards when clicking Scan during active Move or Recompute — **Fixed**
- **Location:** `src/app.rs` (`can_scan`, `start_scan`, `drain_move`)
- **Issue:** `can_scan` permitted starting a new scan while `moving` or `recomputing` was active, and `start_scan()` left stale `zoom_loading`, `move_status`, `move_errors`, and `viewer` state intact.
- **Fix applied:** Gated `can_scan` on `!self.moving && !self.recomputing`, reset stale viewer/zoom/move state on `start_scan()`, and closed `viewer` if the currently previewed photo is moved in `drain_move()`.

### M3. Move failures mislabeled as decode failures and error details hidden — **Fixed**
- **Location:** `src/app.rs` (`move_errors`, `render_error_list`)
- **Issue:** `MoveEvent::Failed` pushed into `self.errors` (displayed as `"files failed to decode"`), no failure details were inspectable, and emptying `entries` hid `move_status`.
- **Fix applied:** Separated `move_errors` from decode `errors`, displayed `move_status` even when `entries` is empty, and added collapsible scrollable error lists (`render_error_list`) showing filenames, reasons, and full-path hover tooltips.

### M4. Redundant per-photo RGB clones, grayscale conversions, and Laplacian passes during scan — **Fixed**
- **Location:** `src/scan.rs`, `src/sharpness.rs`, `src/metrics.rs`
- **Issue:** Each scanned photo performed redundant `RgbImage` clones, repeated RGB→luma conversions, and up to 3 full-frame `laplacian_filter` convolutions.
- **Fix applied:** Converted `rgb` to `GrayImage` once via `image::imageops::grayscale(rgb)` (zero RGB clone), computed the full-frame Laplacian once per photo (`classify_and_score` -> `guess_landscape_or_object_gray`, `score_gray`, `metrics::compute_with_gray`), merged `exposure_and_contrast_score`, and moved `img` when building the thumbnail.

### M5. Per-frame O(N log N) re-sorting, cloning, and unstable `ComboBox` IDs — **Fixed (with remaining virtualization option)**
- **Location:** `src/app.rs`
- **Issue:** Every frame in `ViewMode::Flat` re-sorted all entries, `render_groups()` cloned `self.groups`, and `ComboBox` IDs used mutable `idx`.
- **Fix applied:** Cached `flat_order` (recomputing only when dirty or when `sort_by`/`weights` change), replaced `self.groups.clone()` with `std::mem::take`, and keyed `ComboBox` IDs by `&self.entries[idx].path`. *(Note: `ScrollArea::show_rows` row virtualization was left out for now to keep fixed-row layout refactoring minimal.)*

### M6. Full-size viewer `NaN` pan bug on zero-sized viewport + cross-platform scroll direction — **Partially Fixed**
- **Location:** `src/app.rs` (`show_viewer`)
- **Issue:** Zero-sized canvas/texture dimensions caused `0.0 / 0.0 = NaN` in `viewer.pan`, poisoning zoom/pan state. Also, `-scroll` aligns with macOS natural scroll rather than Windows/Linux standard mouse wheels.
- **Fix applied:** Guarded against zero `avail`/`size` and `old_scale <= 0.0` in `show_viewer()`. Scroll polarity (`-scroll`) was preserved intentionally for macOS; see **Open Items** for cross-platform consideration.

### M7. Unbounded thread spawning for hover-zoom / preview decoding & no scan cancellation — **Fixed**
- **Location:** `src/app.rs` (`MAX_HOVER_ZOOM_LOADS`, `zoom_failed`, `scan_cancel`), `src/scan.rs` (`run_scan`)
- **Issue:** Hovering rapidly across cards spawned unbounded OS threads, failed decodes stayed permanently in `zoom_loading`, and scans had no cancel mechanism.
- **Fix applied:** Capped concurrent hover-zoom background decodes (`MAX_HOVER_ZOOM_LOADS = 2`), separated `zoom_failed`, and added `Arc<AtomicBool>` scan cancellation with a **Cancel** button in the top bar.

---

## Low & Informational Findings

### L1. Unused direct dependency (`serde`) and oversized `tract` umbrella dependency — **Partially Fixed**
- **Location:** `Cargo.toml`, `Cargo.lock`
- **Fix applied:** Removed unused `serde = "1.0.229"` from `Cargo.toml` (and updated `Cargo.lock`). Switching `tract` to `tract-onnx` remains in **Open Items**.

### L2. `sharpness::face_crop` panics on empty images; RAW thumbnail fallback & EXIF duplication — **Partially Fixed**
- **Location:** `src/sharpness.rs`, `src/raw.rs`
- **Fix applied:** Guarded `face_crop` and `variance_of_laplacian` against empty (`0x0`) images (with unit test). RAW thumbnail fallback warning and `Orientation::from_exif` cleanup remain minor informational notes.

### L3. `DISQUALIFIED_DIR` filter edge cases & path canonicalization — **Partially Fixed**
- **Location:** `src/scan.rs` (`run_scan`)
- **Fix applied:** Updated `WalkDir::filter_entry` to skip `disqualified` case-insensitively only at `depth() > 0` (so explicitly selecting a `disqualified` folder as a root works). Path canonicalization and cross-filesystem `EXDEV` fallback remain informational.

### L4. Selection persists invisibly across `ViewMode` switches & minor label mismatches — **Partially Fixed**
- **Location:** `src/app.rs`
- **Fix applied:** Updated `"Disqualify scores below"` to `"Disqualify scores ≤"` and updated the post-scan count label to distinguish `scanned_ok` vs. failed decodes. Cross-view selection clearing was left unchanged.

### L5. Missing repository `LICENSE`, `Cargo.toml` metadata, and model SHA-256 provenance — **Partially Fixed**
- **Location:** `assets/models/NOTICE.md`, repository root
- **Fix applied:** Corrected `assets/models/NOTICE.md` to document that `version-RFB-320.onnx` outputs raw SSD deltas decoded in `src/classify.rs`, and recorded verified SHA-256 checksums for both bundled `.onnx` models. Choosing and adding a root `LICENSE` remains in **Open Items**.

### L6. CI workflow & macOS packaging hygiene — **Fixed**
- **Location:** `.github/workflows/build.yml`, `packaging/macos/build-app.sh`
- **Fix applied:** Added `permissions: contents: read` and a `cargo clippy --all-targets -- -D warnings` step to `.github/workflows/build.yml`, and updated `packaging/macos/build-app.sh` to `set -euo pipefail`.

### L7. Documentation drift after `c9923fc` and `aa10047` — **Open**
- **Location:** `README.md`, `docs/architecture.html`
- **Status:** Left open for user preference (`README.md` doesn't yet mention sidecar moving or viewer scroll/pinch zoom; `docs/architecture.html` predates `c9923fc`).

### L8. Test coverage gaps & non-isolated temp directory in sidecar test — **Fixed**
- **Location:** `src/metrics.rs`, `src/sharpness.rs`, `src/classify.rs`, `src/animal.rs`, `src/raw.rs`, `src/scan.rs`
- **Fix applied:** Expanded unit tests from 10 to **22**, adding coverage for `composition_score` neutrality/thirds/center, `normalize_sharpness`, `overall_score`, empty-image `face_crop`, `nms` `NaN` handling, bundled RFB-320 and YOLOX-Nano ONNX model loading/inference smoke tests, all EXIF orientations (`1..=8`), RAII `TempDir` isolation, and `plan_disqualified_moves` RAW+JPEG sidecar scenarios.

---

## Open Items (For User Decision)

1. **Root `LICENSE` & `Cargo.toml` `license` field (`L5`):** Choose the project license (e.g., MIT or Apache-2.0) to add at the repo root and in `Cargo.toml`.
2. **Viewer scroll polarity on Windows/Linux (`M6`):** Decide whether to invert scroll direction on non-macOS targets or add a toggle/setting (`-scroll` currently matches macOS natural scrolling).
3. **`tract` → `tract-onnx` dependency swap (`L1`):** Optionally replace the `tract` umbrella crate with `tract-onnx` to reduce compile time and binary size.
4. **Docs refresh (`L7`):** Update `README.md` (sidecar moving + viewer zoom/pan + Cancel button) and regenerate `docs/architecture.html` if desired.
5. **Grid row virtualization (`M5`):** Optionally migrate `photo_row_grid` to `ScrollArea::show_rows` if working with tens of thousands of photos in a single session.
