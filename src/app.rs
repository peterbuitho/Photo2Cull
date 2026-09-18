use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;
use std::time::{Duration, Instant};

use egui::{ColorImage, TextureHandle, TextureOptions};

use crate::dedupe::group_duplicates;
use crate::metrics::{Metrics, Weights, overall_score};
use crate::scan::{
    DISQUALIFIED_DIR, RecomputeEvent, ScanEvent, ScanMode, disqualified_path_for, run_recompute,
    run_scan,
};
use crate::sharpness::{PhotoMode, SharpnessMethod};

/// Fixed square thumbnail slot on the left of each photo card. Fixed rather
/// than sized to the image's own aspect ratio so the info column to its
/// right always starts at the same x offset -- otherwise a mix of portrait
/// and landscape thumbnails makes the info jump around from card to card.
const THUMB_BOX: f32 = 120.0;

/// Fixed width of the info column to the right of the thumbnail (filename,
/// mode dropdown, score badge, overall). Fixed rather than "whatever's
/// left" so the card doesn't balloon out to fill the row and leave a slab
/// of empty space to the right of short filenames -- long filenames wrap
/// instead of growing the card.
const INFO_WIDTH: f32 = 130.0;

/// Width (excluding the group's frame/margin) of each photo card, shared by
/// the flat grid and the duplicate-groups view. Only used to plan how many
/// cards fit per row -- the card's actual rendered width is `THUMB_BOX` +
/// spacing + `INFO_WIDTH`, which this tracks.
const CARD_WIDTH: f32 = THUMB_BOX + INFO_WIDTH;

/// Raw (absolute, unbounded) sharpness-score boundaries for the
/// blurry/soft/sharp badge on each photo card.
const SHARPNESS_BLURRY_MAX: f64 = 40.0;
const SHARPNESS_SOFT_MAX: f64 = 100.0;

/// Background fill + readable text color for a raw sharpness score's
/// blurry/soft/sharp bucket, for the score badge on each photo card.
fn sharpness_bucket_colors(raw: f64) -> (egui::Color32, egui::Color32) {
    if raw <= SHARPNESS_BLURRY_MAX {
        // Blurry: red background, white text.
        (egui::Color32::from_rgb(200, 60, 60), egui::Color32::WHITE)
    } else if raw <= SHARPNESS_SOFT_MAX {
        // Soft: yellow background, black text.
        (egui::Color32::from_rgb(230, 200, 60), egui::Color32::BLACK)
    } else {
        // Sharp: green background, white text.
        (egui::Color32::from_rgb(80, 160, 80), egui::Color32::WHITE)
    }
}

struct PhotoEntry {
    path: PathBuf,
    mode: PhotoMode,
    score: f64,
    metrics: Metrics,
    /// Perceptual hash for duplicate/burst detection (see `dedupe`).
    phash: u64,
    /// True if `mode` was changed (manually, or by a rescan) since `score`
    /// (and `metrics`) was last computed for it -- i.e. the values shown
    /// are stale.
    dirty: bool,
    /// The user's manual mark (`Some(true)`) or unmark (`Some(false)`) of
    /// this photo as disqualified, overriding the score threshold. `None`
    /// follows the threshold (see `Photo2CullApp::is_disqualified`).
    manual_dq: Option<bool>,
    texture: TextureHandle,
}

/// Which set of photos is shown in the central panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewMode {
    /// Every scanned photo, in one sortable grid (the original view).
    Flat,
    /// Only photos belonging to a duplicate/burst group, one section per
    /// group, from the last "Group duplicates" click.
    Grouped,
    /// The funnel result from the last "Rank now" click: technical
    /// filter -> dedupe (best-of-group) -> rank by Overall -> top N.
    Pipeline,
}

/// Result of running the full funnel (technical filter -> dedupe -> rank)
/// down to a shortlist, from the last "Rank now" click.
struct PipelineResult {
    total: usize,
    after_technical: usize,
    after_dedupe: usize,
    /// Entry indices, ranked by Overall score (best first), truncated to
    /// the requested shortlist size.
    shortlist: Vec<usize>,
}

/// Which score the grid is sorted (and, for `Overall`, flagged) by. The
/// absolute-sharpness cull/Recalculate/Move-disqualified flow always
/// operates on `score` regardless of this -- Overall is a separate,
/// additional way to look at the same photos, not a replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SortBy {
    Sharpness,
    Overall,
}

/// The image cap for the double-click "full size" preview. Not the true
/// original resolution (which could be 40-100MP and slow/wasteful to
/// decode and hold as a GPU texture just to fit it on screen) -- big enough
/// to judge focus at full-window size.
const PREVIEW_MAX_DIM: usize = 2400;

/// Image cap for the hover-zoom lens: much sharper than the 220px thumbnail
/// (which would just turn to mush when magnified), but cheaper to decode
/// than the full-size viewer's `PREVIEW_MAX_DIM`.
const HOVER_ZOOM_MAX_DIM: usize = 1200;
/// Decoded hover-zoom textures kept around so re-hovering a photo is instant.
const HOVER_ZOOM_CACHE: usize = 8;
/// Vertices around the lens rim; enough that the circle looks smooth.
const LENS_SEGMENTS: usize = 64;

/// The thumbnail the pointer is currently resting on.
struct HoverZoom {
    path: PathBuf,
    /// The card's thumbnail, magnified until the sharper image loads.
    thumb: TextureHandle,
    /// Where the (aspect-fitted) thumbnail is on screen.
    thumb_rect: egui::Rect,
    since: Instant,
}

struct Viewer {
    path: PathBuf,
    /// `None` while the full-size decode is still in flight.
    texture: Option<TextureHandle>,
}

enum PreviewEvent {
    Loaded {
        path: PathBuf,
        w: u32,
        h: u32,
        rgb: Vec<u8>,
    },
    Failed {
        path: PathBuf,
    },
}

enum MoveEvent {
    Moved(PathBuf),
    Failed(PathBuf, String),
    Done,
}

pub struct Photo2CullApp {
    /// Folders chosen for the next scan.
    roots: Vec<PathBuf>,
    /// Folders the current results were scanned from. Kept separate from
    /// `roots` so editing the list after a scan doesn't change where
    /// disqualified photos are moved to.
    scanned_roots: Vec<PathBuf>,
    scan_mode: ScanMode,
    /// How sharpness is measured for the next scan.
    sharpness_method: SharpnessMethod,
    /// Method the current results were scored with; recalculation reuses it
    /// so type overrides stay comparable with the rest of the scan.
    scanned_method: SharpnessMethod,
    entries: Vec<PhotoEntry>,
    errors: Vec<(PathBuf, String)>,
    total_found: usize,
    processed: usize,
    scanning: bool,
    rx: Option<Receiver<ScanEvent>>,
    recompute_rx: Option<Receiver<RecomputeEvent>>,
    recomputing: bool,
    cull_threshold: f64,
    viewer: Option<Viewer>,
    /// Magnifier lens strength, in percent of the thumbnail's on-screen size.
    zoom_percent: f32,
    /// Seconds the pointer must rest on a thumbnail before the lens shows.
    zoom_delay_secs: f32,
    /// On-screen diameter of the magnifier lens, in points.
    lens_diameter: f32,
    /// Photos selected in the grid (click / Ctrl+click / Shift+click, like
    /// files in Explorer). Keyed by path so it survives moves and re-sorts.
    selection: HashSet<PathBuf>,
    /// The photo a Shift+click range extends from.
    selection_anchor: Option<PathBuf>,
    /// Entry indices in the order the cards were drawn last frame, across
    /// whichever view is showing; what a Shift+click range walks over.
    visible_order: Vec<usize>,
    /// The same, being built up during the current frame.
    visible_order_next: Vec<usize>,
    hover_zoom: Option<HoverZoom>,
    /// Set by `photo_card` whenever a thumbnail is hovered this frame, so
    /// `show_hover_zoom` can tell when the pointer has left them all.
    hover_zoom_seen: bool,
    zoom_cache: Vec<(PathBuf, TextureHandle)>,
    zoom_loading: HashSet<PathBuf>,
    zoom_tx: Sender<PreviewEvent>,
    zoom_rx: Receiver<PreviewEvent>,
    preview_rx: Option<Receiver<PreviewEvent>>,
    moving: bool,
    move_rx: Option<Receiver<MoveEvent>>,
    move_moved: usize,
    move_failed: usize,
    move_status: Option<String>,
    weights: Weights,
    sort_by: SortBy,
    view_mode: ViewMode,
    /// Duplicate/burst clusters (2+ members each) from the last "Group
    /// duplicates" click; empty until then. Stored as paths rather than
    /// entry indices so it stays valid (modulo dissolving groups down to
    /// <2 present members) across moves, which is checked at render time.
    groups: Vec<Vec<PathBuf>>,
    /// Max dHash Hamming distance (0-64) for two photos to be considered
    /// duplicates/burst-mates.
    group_threshold: u32,
    /// How many photos the pipeline's final shortlist keeps.
    pipeline_top_n: usize,
    pipeline_result: Option<PipelineResult>,
}

impl Photo2CullApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (zoom_tx, zoom_rx) = channel();
        Self {
            roots: Vec::new(),
            scanned_roots: Vec::new(),
            scan_mode: ScanMode::Auto,
            sharpness_method: SharpnessMethod::FixedRegion,
            scanned_method: SharpnessMethod::FixedRegion,
            entries: Vec::new(),
            errors: Vec::new(),
            total_found: 0,
            processed: 0,
            scanning: false,
            rx: None,
            recompute_rx: None,
            recomputing: false,
            cull_threshold: 32.0,
            viewer: None,
            zoom_percent: 800.0,
            zoom_delay_secs: 1.2,
            lens_diameter: 300.0,
            selection: HashSet::new(),
            selection_anchor: None,
            visible_order: Vec::new(),
            visible_order_next: Vec::new(),
            hover_zoom: None,
            hover_zoom_seen: false,
            zoom_cache: Vec::new(),
            zoom_loading: HashSet::new(),
            zoom_tx,
            zoom_rx,
            preview_rx: None,
            moving: false,
            move_rx: None,
            move_moved: 0,
            move_failed: 0,
            move_status: None,
            weights: Weights::default(),
            sort_by: SortBy::Sharpness,
            view_mode: ViewMode::Flat,
            groups: Vec::new(),
            group_threshold: 6,
            pipeline_top_n: 150,
            pipeline_result: None,
        }
    }

    /// Cluster the current entries into duplicate/burst groups by dHash
    /// distance. Cheap enough (a single XOR+popcount per pair) to run
    /// synchronously from a button click even for several thousand photos.
    fn compute_groups(&mut self) {
        let photos: Vec<(PathBuf, u64)> = self
            .entries
            .iter()
            .map(|e| (e.path.clone(), e.phash))
            .collect();
        self.groups = group_duplicates(&photos, self.group_threshold);
    }

    /// The full funnel: drop disqualified photos (at or below the sharpness
    /// cull threshold, or manually marked), cluster survivors into duplicate/burst groups and keep
    /// only the highest-Overall member of each (plus any ungrouped
    /// singletons), then rank the rest by Overall and keep the top N.
    /// Cheap enough (same cost as `compute_groups` plus a sort) to run
    /// synchronously from a button click.
    fn run_pipeline(&mut self) {
        let total = self.entries.len();
        let survivors: Vec<usize> = (0..total)
            .filter(|&i| !self.is_disqualified(&self.entries[i]))
            .collect();
        let after_technical = survivors.len();

        let photos: Vec<(PathBuf, u64)> = survivors
            .iter()
            .map(|&i| (self.entries[i].path.clone(), self.entries[i].phash))
            .collect();
        let groups = group_duplicates(&photos, self.group_threshold);

        let path_to_idx: HashMap<PathBuf, usize> = survivors
            .iter()
            .map(|&i| (self.entries[i].path.clone(), i))
            .collect();
        let grouped_paths: std::collections::HashSet<PathBuf> =
            groups.iter().flatten().cloned().collect();

        let mut keepers: Vec<usize> = Vec::new();
        for group in &groups {
            let best = group
                .iter()
                .filter_map(|p| path_to_idx.get(p).copied())
                .max_by(|&a, &b| {
                    overall_score(&self.entries[a].metrics, &self.weights)
                        .partial_cmp(&overall_score(&self.entries[b].metrics, &self.weights))
                        .unwrap()
                });
            keepers.extend(best);
        }
        for &i in &survivors {
            if !grouped_paths.contains(&self.entries[i].path) {
                keepers.push(i);
            }
        }
        let after_dedupe = keepers.len();

        keepers.sort_by(|&a, &b| {
            overall_score(&self.entries[b].metrics, &self.weights)
                .partial_cmp(&overall_score(&self.entries[a].metrics, &self.weights))
                .unwrap()
        });
        keepers.truncate(self.pipeline_top_n);

        self.pipeline_result = Some(PipelineResult {
            total,
            after_technical,
            after_dedupe,
            shortlist: keepers,
        });
    }

    /// Move every currently-disqualified photo into a
    /// `disqualified` subfolder of the scanned folder each one came from
    /// (keeping its subfolder structure), in the background.
    fn start_move_disqualified(&mut self) {
        if self.moving {
            return;
        }
        let items: Vec<(PathBuf, PathBuf)> = self
            .entries
            .iter()
            .filter(|e| self.is_disqualified(e))
            .filter_map(|e| {
                disqualified_path_for(&self.scanned_roots, &e.path)
                    .map(|dest| (e.path.clone(), dest))
            })
            .collect();
        if items.is_empty() {
            return;
        }

        self.moving = true;
        self.move_moved = 0;
        self.move_failed = 0;
        self.move_status = None;
        let (tx, rx) = channel();
        self.move_rx = Some(rx);
        thread::spawn(move || {
            for (path, dest) in items {
                let result = match dest.parent() {
                    Some(dir) => std::fs::create_dir_all(dir)
                        .map_err(|e| {
                            std::io::Error::other(format!(
                                "couldn't create {}: {e}",
                                dir.display()
                            ))
                        })
                        .and_then(|()| std::fs::rename(&path, &dest)),
                    None => Err(std::io::Error::other("photo path has no parent folder")),
                };
                let _ = tx.send(match result {
                    Ok(()) => MoveEvent::Moved(path),
                    Err(e) => MoveEvent::Failed(path, e.to_string()),
                });
            }
            let _ = tx.send(MoveEvent::Done);
        });
    }

    fn drain_move(&mut self) {
        let Some(rx) = &self.move_rx else {
            return;
        };
        let mut done = false;
        while let Ok(event) = rx.try_recv() {
            match event {
                MoveEvent::Moved(path) => {
                    self.entries.retain(|e| e.path != path);
                    self.selection.remove(&path);
                    // Entry indices shifted; a stale order would mis-select.
                    self.visible_order.clear();
                    self.move_moved += 1;
                }
                MoveEvent::Failed(path, err) => {
                    self.errors.push((path, err));
                    self.move_failed += 1;
                }
                MoveEvent::Done => done = true,
            }
        }
        if done {
            self.moving = false;
            self.move_rx = None;
            self.move_status = Some(if self.move_failed > 0 {
                format!(
                    "Moved {} to {DISQUALIFIED_DIR}/, {} failed",
                    self.move_moved, self.move_failed
                )
            } else {
                format!("Moved {} to {DISQUALIFIED_DIR}/", self.move_moved)
            });
        }
    }

    /// Load `path` at a larger size for the double-click "full size" view.
    fn start_preview_load(&mut self, path: PathBuf) {
        self.viewer = Some(Viewer {
            path: path.clone(),
            texture: None,
        });
        let (tx, rx) = channel();
        self.preview_rx = Some(rx);
        thread::spawn(move || {
            let event = match crate::photo::decode_photo(&path, PREVIEW_MAX_DIM) {
                Ok(img) => PreviewEvent::Loaded {
                    path,
                    w: img.width(),
                    h: img.height(),
                    rgb: img.into_raw(),
                },
                Err(_) => PreviewEvent::Failed { path },
            };
            let _ = tx.send(event);
        });
    }

    fn drain_preview(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.preview_rx else {
            return;
        };
        // Only ever one message per load; take the last if somehow more.
        let mut last = None;
        while let Ok(event) = rx.try_recv() {
            last = Some(event);
        }
        let Some(event) = last else {
            return;
        };
        self.preview_rx = None;
        match event {
            PreviewEvent::Loaded { path, w, h, rgb } => {
                if self.viewer.as_ref().is_some_and(|v| v.path == path) {
                    let image = ColorImage::from_rgb([w as usize, h as usize], &rgb);
                    let texture = ctx.load_texture(
                        format!("preview-{}", path.display()),
                        image,
                        TextureOptions::LINEAR,
                    );
                    if let Some(v) = &mut self.viewer {
                        v.texture = Some(texture);
                    }
                }
            }
            PreviewEvent::Failed { path } => {
                if self.viewer.as_ref().is_some_and(|v| v.path == path) {
                    self.viewer = None;
                }
            }
        }
    }

    fn start_scan(&mut self) {
        if self.roots.is_empty() {
            return;
        }
        let roots = self.roots.clone();
        self.scanned_roots = roots.clone();
        self.hover_zoom = None;
        self.zoom_cache.clear();
        self.selection.clear();
        self.selection_anchor = None;
        self.visible_order.clear();
        self.entries.clear();
        self.errors.clear();
        self.total_found = 0;
        self.processed = 0;
        self.scanning = true;
        self.groups.clear();
        self.pipeline_result = None;
        self.view_mode = ViewMode::Flat;

        let (tx, rx) = channel();
        self.rx = Some(rx);
        let scan_mode = self.scan_mode;
        let method = self.sharpness_method;
        self.scanned_method = method;
        thread::spawn(move || run_scan(roots, scan_mode, method, tx));
    }

    /// Re-score every entry whose type was manually changed since its last
    /// score, in the background, without re-walking the folder.
    fn start_recompute(&mut self) {
        if self.recomputing {
            return;
        }
        let items: Vec<(PathBuf, PhotoMode)> = self
            .entries
            .iter()
            .filter(|e| e.dirty)
            .map(|e| (e.path.clone(), e.mode))
            .collect();
        if items.is_empty() {
            return;
        }
        self.recomputing = true;
        let (tx, rx) = channel();
        self.recompute_rx = Some(rx);
        let method = self.scanned_method;
        thread::spawn(move || run_recompute(items, method, tx));
    }

    fn drain_events(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.rx {
            let mut done = false;
            while let Ok(event) = rx.try_recv() {
                match event {
                    ScanEvent::Found(n) => self.total_found = n,
                    ScanEvent::Photo(p) => {
                        self.processed += 1;
                        let image = ColorImage::from_rgb(
                            [p.thumb_w as usize, p.thumb_h as usize],
                            &p.thumb_rgb,
                        );
                        let texture = ctx.load_texture(
                            p.path.to_string_lossy().to_string(),
                            image,
                            TextureOptions::LINEAR,
                        );
                        self.entries.push(PhotoEntry {
                            path: p.path,
                            mode: p.mode,
                            score: p.score,
                            metrics: p.metrics,
                            phash: p.phash,
                            dirty: false,
                            manual_dq: None,
                            texture,
                        });
                    }
                    ScanEvent::Failed(path, err) => {
                        self.processed += 1;
                        self.errors.push((path, err));
                    }
                    ScanEvent::Done => done = true,
                }
            }
            if done {
                self.scanning = false;
                self.rx = None;
            }
        }

        if let Some(rx) = &self.recompute_rx {
            let mut done = false;
            while let Ok(event) = rx.try_recv() {
                match event {
                    RecomputeEvent::Result(r) => {
                        if let Some(entry) = self.entries.iter_mut().find(|e| e.path == r.path) {
                            entry.score = r.score;
                            entry.metrics = r.metrics;
                            entry.dirty = false;
                        }
                    }
                    RecomputeEvent::Done => done = true,
                }
            }
            if done {
                self.recomputing = false;
                self.recompute_rx = None;
            }
        }
    }

    /// Decode `path` at hover-zoom size in the background; the result comes
    /// back over `zoom_rx` and is cached by `drain_zoom`.
    fn start_zoom_load(&mut self, path: PathBuf) {
        if !self.zoom_loading.insert(path.clone()) {
            return;
        }
        let tx = self.zoom_tx.clone();
        thread::spawn(move || {
            let event = match crate::photo::decode_photo(&path, HOVER_ZOOM_MAX_DIM) {
                Ok(img) => PreviewEvent::Loaded {
                    path,
                    w: img.width(),
                    h: img.height(),
                    rgb: img.into_raw(),
                },
                Err(_) => PreviewEvent::Failed { path },
            };
            let _ = tx.send(event);
        });
    }

    fn drain_zoom(&mut self, ctx: &egui::Context) {
        while let Ok(event) = self.zoom_rx.try_recv() {
            match event {
                PreviewEvent::Loaded { path, w, h, rgb } => {
                    self.zoom_loading.remove(&path);
                    let image = ColorImage::from_rgb([w as usize, h as usize], &rgb);
                    let texture = ctx.load_texture(
                        format!("hover-zoom-{}", path.display()),
                        image,
                        TextureOptions::LINEAR,
                    );
                    if self.zoom_cache.len() >= HOVER_ZOOM_CACHE {
                        drop(self.zoom_cache.remove(0));
                    }
                    self.zoom_cache.push((path, texture));
                }
                // Left in `zoom_loading` so a photo that fails to decode
                // isn't retried on every frame of the hover.
                PreviewEvent::Failed { .. } => {}
            }
        }
    }

    /// Track which thumbnail (if any) the pointer has rested on, and once
    /// it has been there for `zoom_delay_secs` draw a circular magnifier
    /// lens centred on the pointer, showing the part of the photo under it
    /// at `zoom_percent`. Called once per frame after the cards are drawn.
    fn show_hover_zoom(&mut self, ctx: &egui::Context) {
        let hovering = std::mem::take(&mut self.hover_zoom_seen);
        if !hovering || self.viewer.is_some() {
            self.hover_zoom = None;
            return;
        }
        let Some(hover) = &self.hover_zoom else {
            return;
        };
        let Some(pointer) = ctx.pointer_hover_pos() else {
            return;
        };

        let delay = Duration::from_secs_f32(self.zoom_delay_secs.max(0.0));
        let waited = hover.since.elapsed();
        if waited < delay {
            // A still pointer produces no input events, so ask for the
            // repaint that will notice the delay has elapsed.
            ctx.request_repaint_after(delay - waited);
            return;
        }

        let path = hover.path.clone();
        let thumb = hover.thumb.clone();
        let thumb_rect = hover.thumb_rect;
        let cached = self
            .zoom_cache
            .iter()
            .find(|(p, _)| *p == path)
            .map(|(_, t)| t.clone());
        if cached.is_none() {
            self.start_zoom_load(path);
            // The decode thread can't wake the UI, so poll for its result.
            ctx.request_repaint_after(Duration::from_millis(100));
        }
        let texture = cached.unwrap_or(thumb);

        let zoom = (self.zoom_percent / 100.0).max(1.0);
        let radius = self.lens_diameter / 2.0;
        let size = thumb_rect.size();
        // How much of the image (as a 0..1 fraction of each axis) the lens
        // shows.
        let half_uv = egui::vec2(radius / (zoom * size.x), radius / (zoom * size.y));
        // Keep the visible window inside the image, so near an edge the lens
        // shows the edge rather than smearing stretched border pixels.
        let clamp_axis = |v: f32, half: f32| {
            if half >= 0.5 {
                0.5
            } else {
                v.clamp(half, 1.0 - half)
            }
        };
        let center_uv = egui::pos2(
            clamp_axis((pointer.x - thumb_rect.left()) / size.x, half_uv.x),
            clamp_axis((pointer.y - thumb_rect.top()) / size.y, half_uv.y),
        );

        let mut mesh = egui::Mesh::with_texture(texture.id());
        mesh.vertices.push(egui::epaint::Vertex {
            pos: pointer,
            uv: center_uv,
            color: egui::Color32::WHITE,
        });
        for i in 0..=LENS_SEGMENTS {
            let angle = i as f32 / LENS_SEGMENTS as f32 * std::f32::consts::TAU;
            let dir = egui::vec2(angle.cos(), angle.sin());
            mesh.vertices.push(egui::epaint::Vertex {
                pos: pointer + dir * radius,
                uv: center_uv + dir * half_uv,
                color: egui::Color32::WHITE,
            });
            if i > 0 {
                mesh.add_triangle(0, i as u32, i as u32 + 1);
            }
        }

        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Tooltip,
            egui::Id::new("hover-lens"),
        ));
        painter.circle_filled(pointer, radius + 2.0, egui::Color32::BLACK);
        painter.add(egui::Shape::mesh(mesh));
        painter.circle_stroke(pointer, radius, egui::Stroke::new(2.0, egui::Color32::WHITE));
        ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
    }

    fn show_viewer(&mut self, ctx: &egui::Context) {
        let Some(viewer) = &self.viewer else {
            return;
        };
        let name = viewer
            .path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        let mut close = false;
        egui::Window::new("photo-viewer")
            .title_bar(false)
            .resizable(false)
            .movable(false)
            .collapsible(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .frame(egui::Frame::popup(&ctx.style_of(ctx.theme())))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading(name);
                    if ui.button("✕").clicked() {
                        close = true;
                    }
                });
                ui.separator();
                match &viewer.texture {
                    Some(tex) => {
                        let avail = ctx.content_rect().size() * 0.85;
                        let size = tex.size_vec2();
                        let scale = (avail.x / size.x).min(avail.y / size.y).min(1.0);
                        ui.image((tex.id(), size * scale));
                    }
                    None => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Loading full size…");
                        });
                    }
                }
            });

        if close || ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.viewer = None;
        }
    }

    /// Apply an Explorer-style click on the card at `idx`: a plain click
    /// selects just it, Ctrl/Cmd+click toggles it, Shift+click selects the
    /// range from the anchor to it (added to the selection with Ctrl too).
    fn select_click(&mut self, idx: usize, mods: egui::Modifiers) {
        let path = self.entries[idx].path.clone();

        if mods.shift {
            let position = |order: &[usize], entries: &[PhotoEntry], p: &PathBuf| {
                order.iter().position(|&i| entries[i].path == *p)
            };
            let from = self
                .selection_anchor
                .as_ref()
                .and_then(|a| position(&self.visible_order, &self.entries, a));
            let to = position(&self.visible_order, &self.entries, &path);
            if let (Some(a), Some(b)) = (from, to) {
                if !mods.command {
                    self.selection.clear();
                }
                let (lo, hi) = (a.min(b), a.max(b));
                for &i in &self.visible_order[lo..=hi] {
                    self.selection.insert(self.entries[i].path.clone());
                }
                return;
            }
        }

        if mods.command {
            if !self.selection.remove(&path) {
                self.selection.insert(path.clone());
            }
        } else {
            self.selection.clear();
            self.selection.insert(path.clone());
        }
        self.selection_anchor = Some(path);
    }

    /// Mark (`true`) or unmark (`false`) every selected photo as
    /// disqualified by hand. A photo whose new state matches what the score
    /// threshold gives goes back to following the threshold.
    fn set_selected_disqualified(&mut self, dq: bool) {
        let threshold = self.cull_threshold;
        for e in &mut self.entries {
            if self.selection.contains(&e.path) {
                e.manual_dq = (dq != (e.score <= threshold)).then_some(dq);
            }
        }
    }

    /// Whether `entry` is currently disqualified: the user's manual mark or
    /// unmark if they made one, otherwise whether its score is at or below
    /// the cull threshold (so photos below the limit are marked
    /// automatically after a scan, and follow the threshold as it changes).
    fn is_disqualified(&self, entry: &PhotoEntry) -> bool {
        entry
            .manual_dq
            .unwrap_or(entry.score <= self.cull_threshold)
    }

    /// (min, max) score across all entries, for the threshold input's hint.
    fn score_range(&self) -> Option<(f64, f64)> {
        let mut iter = self.entries.iter().map(|e| e.score);
        let first = iter.next()?;
        Some(iter.fold((first, first), |(lo, hi), s| (lo.min(s), hi.max(s))))
    }

    /// One photo card: thumbnail (double-click to open the full-size
    /// preview) on the left, with filename (red if disqualified), the type
    /// dropdown, the absolute sharpness score, the Overall score, and a
    /// checkbox to manually mark/unmark the photo as disqualified, stacked
    /// in a column to its right. Shared by the flat grid and the
    /// duplicate-groups view so they can't drift apart. `badge`, if given,
    /// is drawn as a colored label under the filename (e.g. marking the
    /// best-of-group pick). The thumbnail sits in a fixed-size square slot
    /// (see `THUMB_BOX`) regardless of the photo's own aspect ratio, so the
    /// info column lines up the same way across a mix of portrait and
    /// landscape photos instead of jumping around card to card.
    fn photo_card(
        &mut self,
        ui: &mut egui::Ui,
        idx: usize,
        badge: Option<(egui::Color32, &str)>,
        open_request: &mut Option<PathBuf>,
    ) {
        let card = ui.group(|ui| {
            ui.horizontal_top(|ui| {
                let entry = &self.entries[idx];
                let size = entry.texture.size_vec2();
                let scale = (THUMB_BOX / size.x.max(size.y)).min(1.0);
                let img_size = size * scale;
                let (slot_rect, _) =
                    ui.allocate_exact_size(egui::vec2(THUMB_BOX, THUMB_BOX), egui::Sense::hover());
                let image_rect = egui::Rect::from_center_size(slot_rect.center(), img_size);
                // Disqualified photos are dimmed a little (60%) and struck through
                // with a red diagonal: the line keeps the photo readable (you
                // may be checking whether it was wrongly disqualified), the
                // dim makes the whole grid scannable at a glance.
                let disqualified = self.is_disqualified(entry);
                let tint = if disqualified {
                    egui::Color32::from_gray(153)
                } else {
                    egui::Color32::WHITE
                };
                let image_response = ui.put(
                    image_rect,
                    egui::Image::new((entry.texture.id(), img_size))
                        .tint(tint)
                        .sense(egui::Sense::click()),
                );
                if disqualified {
                    ui.painter().line_segment(
                        [image_rect.right_top(), image_rect.left_bottom()],
                        egui::Stroke::new(
                            3.0,
                            egui::Color32::from_rgba_unmultiplied(220, 60, 60, 210),
                        ),
                    );
                }
                if image_response.double_clicked() {
                    *open_request = Some(entry.path.clone());
                }
                if image_response.hovered() {
                    let (path, thumb) = (entry.path.clone(), entry.texture.clone());
                    self.hover_zoom_seen = true;
                    match &mut self.hover_zoom {
                        Some(h) if h.path == path => h.thumb_rect = image_rect,
                        _ => {
                            self.hover_zoom = Some(HoverZoom {
                                path,
                                thumb,
                                thumb_rect: image_rect,
                                since: Instant::now(),
                            });
                        }
                    }
                }
                if image_response.clicked() {
                    let mods = ui.input(|i| i.modifiers);
                    self.select_click(idx, mods);
                }

                ui.vertical(|ui| {
                    ui.set_width(INFO_WIDTH);
                    let entry = &self.entries[idx];
                    let name = entry
                        .path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    if self.is_disqualified(entry) {
                        ui.colored_label(egui::Color32::from_rgb(220, 90, 90), name);
                    } else {
                        ui.label(name);
                    }
                    if let Some((color, text)) = badge {
                        ui.colored_label(color, text);
                    }

                    let mut mode = self.entries[idx].mode;
                    egui::ComboBox::from_id_salt(("photo-mode", idx))
                        .selected_text(mode.label())
                        .width(88.0)
                        .show_ui(ui, |ui| {
                            for m in PhotoMode::ALL {
                                ui.selectable_value(&mut mode, m, m.label());
                            }
                        });
                    if mode != self.entries[idx].mode {
                        self.entries[idx].mode = mode;
                        self.entries[idx].dirty = true;
                    }

                    let entry = &self.entries[idx];
                    if entry.dirty {
                        ui.colored_label(egui::Color32::from_rgb(200, 150, 60), "score: pending…");
                    } else {
                        let (bg, fg) = sharpness_bucket_colors(entry.score);
                        ui.label(
                            egui::RichText::new(format!("score: {:.0}", entry.score))
                                .color(fg)
                                .background_color(bg),
                        );
                        let overall = overall_score(&self.entries[idx].metrics, &self.weights);
                        ui.small(format!("overall: {overall:.0}"));
                    }

                    let entry = &self.entries[idx];
                    let auto_dq = entry.score <= self.cull_threshold;
                    let manual = entry.manual_dq.is_some();
                    let mut dq = self.is_disqualified(entry);
                    let response = ui.checkbox(
                        &mut dq,
                        if manual { "Disqualified*" } else { "Disqualified" },
                    );
                    if manual {
                        response.clone().on_hover_text(
                            "Set manually (overrides the score threshold). Click to \
                             go back to following the threshold if it now agrees.",
                        );
                    }
                    if response.changed() {
                        if self.selection.contains(&self.entries[idx].path) {
                            // Like Explorer: acting on one selected item acts
                            // on the whole selection.
                            self.set_selected_disqualified(dq);
                        } else {
                            // Back in line with the threshold -> stop
                            // overriding it.
                            self.entries[idx].manual_dq = (dq != auto_dq).then_some(dq);
                        }
                    }
                });
            });
        });
        if self.selection.contains(&self.entries[idx].path) {
            let accent = ui.visuals().selection.stroke.color;
            let painter = ui.painter();
            painter.rect_filled(
                card.response.rect,
                4.0,
                egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 28),
            );
            painter.rect_stroke(
                card.response.rect,
                4.0,
                egui::Stroke::new(2.5, accent),
                egui::StrokeKind::Outside,
            );
        }
    }

    /// Lay `indices` out as chunked (non-wrapping) rows of `photo_card`s --
    /// see the note on `horizontal_wrapped` corruption where this pattern
    /// is used in the flat grid for why it's not just `horizontal_wrapped`.
    /// `best_idx`, if given, badges that one entry -- used by the groups
    /// view, not the flat grid.
    fn photo_row_grid(
        &mut self,
        ui: &mut egui::Ui,
        indices: &[usize],
        best_idx: Option<usize>,
        open_request: &mut Option<PathBuf>,
    ) {
        self.visible_order_next.extend_from_slice(indices);
        let spacing = ui.spacing().item_spacing.x;
        let columns = ((ui.available_width() / (CARD_WIDTH + spacing)).floor() as usize).max(1);
        for row in indices.chunks(columns) {
            ui.horizontal(|ui| {
                for &idx in row {
                    let badge = (Some(idx) == best_idx)
                        .then_some((egui::Color32::from_rgb(90, 170, 90), "★ best of group"));
                    self.photo_card(ui, idx, badge, open_request);
                }
            });
        }
    }

    /// The duplicate/burst groups view: one section per cluster from the
    /// last "Group duplicates" click, each showing its members with the
    /// highest-Overall pick badged. Groups that have dissolved to fewer
    /// than 2 still-present members (e.g. after a move) are skipped.
    fn render_groups(&mut self, ui: &mut egui::Ui, open_request: &mut Option<PathBuf>) {
        if self.groups.is_empty() {
            ui.label(
                "No duplicate/burst groups yet -- set a threshold and click \"Group duplicates\".",
            );
            return;
        }

        let path_to_idx: HashMap<PathBuf, usize> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.path.clone(), i))
            .collect();
        let groups = self.groups.clone();
        let weights = self.weights;

        egui::ScrollArea::vertical()
            .auto_shrink([false, true])
            .show(ui, |ui| {
                for (gi, group) in groups.iter().enumerate() {
                    let indices: Vec<usize> = group
                        .iter()
                        .filter_map(|p| path_to_idx.get(p).copied())
                        .collect();
                    if indices.len() < 2 {
                        continue;
                    }

                    let best_idx = indices.iter().copied().max_by(|&a, &b| {
                        overall_score(&self.entries[a].metrics, &weights)
                            .partial_cmp(&overall_score(&self.entries[b].metrics, &weights))
                            .unwrap()
                    });

                    ui.label(format!("Group {} ({} photos)", gi + 1, indices.len()));
                    self.photo_row_grid(ui, &indices, best_idx, open_request);
                    ui.separator();
                }
            });
    }

    /// The pipeline shortlist view: the funnel summary line, then the
    /// ranked result grid, from the last "Rank now" click.
    fn render_pipeline(&mut self, ui: &mut egui::Ui, open_request: &mut Option<PathBuf>) {
        let Some(result) = &self.pipeline_result else {
            ui.label("Click \"Rank now\" to filter, dedupe, and rank down to a shortlist.");
            return;
        };
        let summary = format!(
            "{} photos → {} after removing disqualified → {} after dedupe → top {} shown",
            result.total,
            result.after_technical,
            result.after_dedupe,
            result.shortlist.len()
        );
        let shortlist = result.shortlist.clone();

        ui.label(summary);
        egui::ScrollArea::vertical()
            .auto_shrink([false, true])
            .show(ui, |ui| {
                self.photo_row_grid(ui, &shortlist, None, open_request);
            });
    }
}

impl eframe::App for Photo2CullApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.drain_events(&ctx);
        self.drain_preview(&ctx);
        self.drain_zoom(&ctx);
        self.drain_move();
        if self.scanning
            || self.recomputing
            || self.moving
            || self.viewer.as_ref().is_some_and(|v| v.texture.is_none())
        {
            ctx.request_repaint();
        }

        egui::Panel::top("top").show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Add folders…").clicked()
                    && let Some(folders) = rfd::FileDialog::new().pick_folders()
                {
                    for folder in folders {
                        if !self.roots.contains(&folder) {
                            self.roots.push(folder);
                        }
                    }
                }
                if self.roots.is_empty() {
                    ui.label("No folder selected");
                } else if ui.button("Clear").clicked() {
                    self.roots.clear();
                }

                ui.separator();

                ui.label("Mode");
                egui::ComboBox::from_id_salt("scan-mode")
                    .selected_text(match self.scan_mode {
                        ScanMode::Auto => "Auto".to_string(),
                        ScanMode::Fixed(m) => m.label().to_string(),
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.scan_mode, ScanMode::Auto, "Auto");
                        for m in PhotoMode::ALL {
                            ui.selectable_value(&mut self.scan_mode, ScanMode::Fixed(m), m.label());
                        }
                    });

                ui.label("Sharpness");
                egui::ComboBox::from_id_salt("sharpness-method")
                    .selected_text(self.sharpness_method.label())
                    .show_ui(ui, |ui| {
                        for m in SharpnessMethod::ALL {
                            ui.selectable_value(&mut self.sharpness_method, m, m.label())
                                .on_hover_text(m.description());
                        }
                    })
                    .response
                    .on_hover_text(self.sharpness_method.description());
                if self.sharpness_method != self.scanned_method && !self.entries.is_empty() {
                    ui.small("(applies on next Scan)");
                }

                ui.separator();

                let can_scan = !self.roots.is_empty() && !self.scanning;
                let scan_label = if can_scan {
                    egui::RichText::new("Scan")
                        .strong()
                        .size(15.0)
                        .color(egui::Color32::WHITE)
                } else {
                    egui::RichText::new("Scan").strong().size(15.0)
                };
                let mut scan_button = egui::Button::new(scan_label);
                if can_scan {
                    scan_button = scan_button.fill(egui::Color32::from_rgb(40, 120, 200));
                }
                if ui.add_enabled(can_scan, scan_button).clicked() {
                    self.start_scan();
                }

                if self.scanning {
                    ui.spinner();
                    ui.label(format!("{} / {}", self.processed, self.total_found));
                } else if self.total_found > 0 {
                    ui.label(format!("{} photos scanned", self.total_found));
                }

                ui.separator();

                let dirty_count = self.entries.iter().filter(|e| e.dirty).count();
                if ui
                    .add_enabled(
                        dirty_count > 0 && !self.recomputing,
                        egui::Button::new("Recalculate"),
                    )
                    .clicked()
                {
                    self.start_recompute();
                }
                if self.recomputing {
                    ui.spinner();
                } else if dirty_count > 0 {
                    ui.label(format!(
                        "{dirty_count} type change{} pending",
                        if dirty_count == 1 { "" } else { "s" }
                    ));
                }
            });

            if !self.roots.is_empty() {
                let mut remove = None;
                ui.horizontal_wrapped(|ui| {
                    for (i, folder) in self.roots.iter().enumerate() {
                        if ui
                            .small_button("✕")
                            .on_hover_text("Remove this folder")
                            .clicked()
                        {
                            remove = Some(i);
                        }
                        ui.label(folder.display().to_string());
                        ui.add_space(8.0);
                    }
                });
                if let Some(i) = remove {
                    self.roots.remove(i);
                }
            }
        });

        egui::CentralPanel::default().show(ui, |ui| {
            if self.entries.is_empty() {
                ui.label(
                    "Add one or more folders and click Scan to check focus sharpness across your photos (RAW, PNG, JPG, TIFF, BMP, WebP).",
                );
                if !self.errors.is_empty() {
                    ui.label(format!("{} files failed to decode", self.errors.len()));
                }
                return;
            }

            let flagged_count = self
                .entries
                .iter()
                .filter(|e| self.is_disqualified(e))
                .count();
            let manual_count = self.entries.iter().filter(|e| e.manual_dq.is_some()).count();
            let dirty_count = self.entries.iter().filter(|e| e.dirty).count();

            // Three sections, each capped to a modest max width rather than
            // split into equal `ui.columns()` thirds: that helper grows
            // *all* columns to fit whichever one needs the most room, and
            // that growth compounds frame over frame (its own doc comment
            // calls this "make sure we fit everything next frame"), which
            // spirals a single too-wide line into every section ballooning
            // off the edge of the window. A small cap keeps each section
            // sized to its own content (wrapping long notes) and immune to
            // that feedback loop.
            let col_width = 150.0;
            ui.horizontal(|ui| {
                // Sharpness: the raw per-photo focus score, its cull
                // threshold/move-out workflow, and duplicate/burst grouping
                // (grouping runs against the same survivors the cull
                // threshold defines, so it lives alongside it).
                ui.vertical(|ui| {
                    ui.set_max_width(col_width);
                    ui.group(|ui| {
                        ui.strong("Sharpness");
                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.label("Disqualify scores below");
                            ui.add(
                                egui::DragValue::new(&mut self.cull_threshold)
                                    .speed(1.0)
                                    .range(0.0..=f64::MAX),
                            );
                        });
                        if let Some((min, max)) = self.score_range() {
                            ui.small(format!("(scanned scores range {min:.0}–{max:.0})"));
                        }

                        if !self.selection.is_empty() {
                            ui.horizontal_wrapped(|ui| {
                                ui.small(format!("{} selected:", self.selection.len()));
                                if ui.small_button("Disqualify").clicked() {
                                    self.set_selected_disqualified(true);
                                }
                                if ui.small_button("Keep").clicked() {
                                    self.set_selected_disqualified(false);
                                }
                                if ui.small_button("Deselect").clicked() {
                                    self.selection.clear();
                                }
                            });
                        }
                        if manual_count > 0
                            && ui
                                .small_button(format!("Clear {manual_count} manual marks"))
                                .on_hover_text(
                                    "Go back to the score threshold for every photo you \
                                     marked or unmarked by hand",
                                )
                                .clicked()
                        {
                            for e in &mut self.entries {
                                e.manual_dq = None;
                            }
                        }

                        ui.add_space(4.0);

                        let can_move = flagged_count > 0
                            && dirty_count == 0
                            && !self.moving
                            && !self.scanning
                            && !self.recomputing;
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(
                                    can_move,
                                    egui::Button::new(format!(
                                        "Move {flagged_count} disqualified"
                                    )),
                                )
                                .clicked()
                            {
                                self.start_move_disqualified();
                            }
                            if self.moving {
                                ui.spinner();
                            } else if let Some(status) = &self.move_status {
                                ui.label(status);
                            } else if dirty_count > 0 && flagged_count > 0 {
                                ui.label("recalculate pending changes first");
                            }
                        });

                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.label("Duplicate/burst threshold");
                            ui.add(
                                egui::DragValue::new(&mut self.group_threshold)
                                    .speed(1)
                                    .range(0..=64),
                            );
                        });
                        ui.small("(max hash distance, 0-64; lower = stricter)");

                        let can_group = !self.entries.is_empty() && !self.scanning;
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(can_group, egui::Button::new("Group duplicates"))
                                .clicked()
                            {
                                self.compute_groups();
                                self.view_mode = ViewMode::Grouped;
                            }
                            if !self.groups.is_empty() {
                                let grouped: usize =
                                    self.groups.iter().map(|g| g.len()).sum();
                                ui.label(format!(
                                    "{} groups ({grouped} photos)",
                                    self.groups.len()
                                ));
                            }
                        });
                    });
                });

                // Pipeline: the technical filter -> dedupe -> rank by
                // Overall -> top N funnel's own parameters, plus how the
                // results are displayed.
                ui.vertical(|ui| {
                    ui.set_max_width(col_width);
                    ui.group(|ui| {
                        ui.strong("Ranking pipeline");
                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.label("Shortlist size");
                            ui.add(
                                egui::DragValue::new(&mut self.pipeline_top_n)
                                    .speed(1)
                                    .range(1..=100_000),
                            );
                        });
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(
                                    "technical filter -> dedupe (best of group) -> rank \
                                     by Overall -> top N",
                                )
                                .small(),
                            )
                            .wrap(),
                        );

                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.label("Sort by");
                            egui::ComboBox::from_id_salt("sort-by")
                                .selected_text(match self.sort_by {
                                    SortBy::Sharpness => "Sharpness",
                                    SortBy::Overall => "Overall",
                                })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut self.sort_by,
                                        SortBy::Sharpness,
                                        "Sharpness",
                                    );
                                    ui.selectable_value(
                                        &mut self.sort_by,
                                        SortBy::Overall,
                                        "Overall",
                                    );
                                });
                        });
                        ui.horizontal(|ui| {
                            ui.label("View");
                            egui::ComboBox::from_id_salt("view-mode")
                                .selected_text(match self.view_mode {
                                    ViewMode::Flat => "All photos",
                                    ViewMode::Grouped => "Duplicate groups",
                                    ViewMode::Pipeline => "Pipeline shortlist",
                                })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut self.view_mode,
                                        ViewMode::Flat,
                                        "All photos",
                                    );
                                    ui.selectable_value(
                                        &mut self.view_mode,
                                        ViewMode::Grouped,
                                        "Duplicate groups",
                                    );
                                    ui.selectable_value(
                                        &mut self.view_mode,
                                        ViewMode::Pipeline,
                                        "Pipeline shortlist",
                                    );
                                });
                        });
                    });
                });

                // Weights: the ranking criteria behind Overall, and the
                // button that runs the pipeline with them.
                ui.vertical(|ui| {
                    ui.set_max_width(320.0);
                    ui.group(|ui| {
                        ui.strong("Ranking weights");
                        ui.separator();

                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(
                                    "Overall = weighted blend of these factors, each \
                                     0-100. Weights don't need to add to 100% -- \
                                     they're renormalized over whatever's scored.",
                                )
                                .small(),
                            )
                            .wrap(),
                        );
                        let pct = |ui: &mut egui::Ui, label: &str, w: &mut f32, note: &str| {
                            ui.label(label);
                            let drag = ui.add(
                                egui::DragValue::new(w)
                                    .speed(0.01)
                                    .range(0.0..=1.0)
                                    .custom_formatter(|v, _| format!("{:.0}%", v * 100.0))
                                    .custom_parser(|s| {
                                        s.trim_end_matches('%').parse::<f64>().ok().map(|v| v / 100.0)
                                    }),
                            );
                            if !note.is_empty() {
                                drag.on_hover_text(note);
                            }
                            ui.end_row();
                        };
                        // Two 3-row sub-columns rather than one tall list --
                        // reads more like a balanced table in the pane's
                        // narrow third column. Explanatory notes (e.g. for
                        // Composition/Subject) moved to a hover tooltip on
                        // the value, since there's no room for them inline
                        // at this width.
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                egui::Grid::new("weights-grid-left").num_columns(2).show(
                                    ui,
                                    |ui| {
                                        pct(ui, "Sharpness", &mut self.weights.sharpness, "");
                                        pct(ui, "Exposure", &mut self.weights.exposure, "");
                                        pct(ui, "Contrast", &mut self.weights.contrast, "");
                                    },
                                );
                            });
                            ui.vertical(|ui| {
                                egui::Grid::new("weights-grid-right").num_columns(2).show(
                                    ui,
                                    |ui| {
                                        pct(ui, "Color", &mut self.weights.color, "");
                                        pct(
                                            ui,
                                            "Composition",
                                            &mut self.weights.composition,
                                            "Rule-of-thirds heuristic",
                                        );
                                        pct(
                                            ui,
                                            "Subject",
                                            &mut self.weights.subject,
                                            "Portrait/Animal: subject prominence",
                                        );
                                    },
                                );
                            });
                        });

                        ui.add_space(4.0);

                        let can_run = !self.entries.is_empty() && !self.scanning;
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(can_run, egui::Button::new("Rank now"))
                                .clicked()
                            {
                                self.run_pipeline();
                                self.view_mode = ViewMode::Pipeline;
                            }
                        });
                    });
                });

                // Hover zoom: a magnifier lens over the thumbnail under the
                // pointer, for judging focus without opening the full-size
                // viewer.
                ui.vertical(|ui| {
                    ui.set_max_width(col_width);
                    ui.group(|ui| {
                        ui.strong("Hover zoom");
                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.label("Zoom");
                            ui.add(
                                egui::DragValue::new(&mut self.zoom_percent)
                                    .speed(5.0)
                                    .range(100.0..=1000.0)
                                    .max_decimals(0)
                                    .suffix("%"),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Delay");
                            ui.add(
                                egui::DragValue::new(&mut self.zoom_delay_secs)
                                    .speed(0.1)
                                    .range(0.0..=10.0)
                                    .max_decimals(1)
                                    .suffix(" s"),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Size");
                            ui.add(
                                egui::DragValue::new(&mut self.lens_diameter)
                                    .speed(2.0)
                                    .range(60.0..=500.0)
                                    .max_decimals(0)
                                    .suffix(" px"),
                            );
                        });
                        ui.small("(rest the pointer on a thumbnail for the delay to show the lens)");
                    });
                });
            });

            if !self.errors.is_empty() {
                ui.label(format!("{} files failed to decode", self.errors.len()));
            }
            ui.separator();

            let mut open_request: Option<PathBuf> = None;

            match self.view_mode {
                ViewMode::Flat => {
                    let mut order: Vec<usize> = (0..self.entries.len()).collect();
                    match self.sort_by {
                        SortBy::Sharpness => order.sort_by(|&a, &b| {
                            self.entries[a]
                                .score
                                .partial_cmp(&self.entries[b].score)
                                .unwrap()
                        }),
                        SortBy::Overall => order.sort_by(|&a, &b| {
                            let oa = overall_score(&self.entries[a].metrics, &self.weights);
                            let ob = overall_score(&self.entries[b].metrics, &self.weights);
                            // Higher Overall first -- unlike Sharpness, where
                            // worst-first matches "what should I cull", Overall
                            // is "what's best".
                            ob.partial_cmp(&oa).unwrap()
                        }),
                    }

                    // `auto_shrink` defaults to [true, true], meaning the area
                    // shrinks its *width* to fit its content -- pin it to the
                    // panel's, leaving only the vertical axis auto-sized.
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, true])
                        .show(ui, |ui| {
                            self.photo_row_grid(ui, &order, None, &mut open_request);
                        });
                }
                ViewMode::Grouped => {
                    self.render_groups(ui, &mut open_request);
                }
                ViewMode::Pipeline => {
                    self.render_pipeline(ui, &mut open_request);
                }
            }

            if let Some(path) = open_request {
                self.start_preview_load(path);
            }
        });

        self.visible_order = std::mem::take(&mut self.visible_order_next);

        if !ctx.egui_wants_keyboard_input() {
            let (select_all, escape) = ctx.input(|i| {
                (
                    i.modifiers.command && i.key_pressed(egui::Key::A),
                    i.key_pressed(egui::Key::Escape),
                )
            });
            if select_all {
                self.selection = self
                    .visible_order
                    .iter()
                    .map(|&i| self.entries[i].path.clone())
                    .collect();
            } else if escape && self.viewer.is_none() {
                // (Esc with the viewer open just closes the viewer.)
                self.selection.clear();
            }
        }

        self.show_viewer(&ctx);
        self.show_hover_zoom(&ctx);
    }
}
