use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, channel};
use std::thread;

use egui::{ColorImage, TextureHandle, TextureOptions};

use crate::dedupe::group_duplicates;
use crate::metrics::{overall_score, Metrics, Weights};
use crate::scan::{DISQUALIFIED_DIR, RecomputeEvent, ScanEvent, ScanMode, run_recompute, run_scan};
use crate::sharpness::PhotoMode;

/// Width (excluding the group's frame/margin) of each photo card, shared by
/// the flat grid and the duplicate-groups view.
const CARD_WIDTH: f32 = 180.0;

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
    /// The funnel result from the last "Run pipeline" click: technical
    /// filter -> dedupe (best-of-group) -> rank by Overall -> top N.
    Pipeline,
}

/// Result of running the full funnel (technical filter -> dedupe -> rank)
/// down to a shortlist, from the last "Run pipeline" click.
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
    root: Option<PathBuf>,
    scan_mode: ScanMode,
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
    preview_rx: Option<Receiver<PreviewEvent>>,
    moving: bool,
    move_rx: Option<Receiver<MoveEvent>>,
    move_moved: usize,
    move_failed: usize,
    move_status: Option<String>,
    weights: Weights,
    sort_by: SortBy,
    show_weights: bool,
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
        Self {
            root: None,
            scan_mode: ScanMode::Auto,
            entries: Vec::new(),
            errors: Vec::new(),
            total_found: 0,
            processed: 0,
            scanning: false,
            rx: None,
            recompute_rx: None,
            recomputing: false,
            cull_threshold: 40.0,
            viewer: None,
            preview_rx: None,
            moving: false,
            move_rx: None,
            move_moved: 0,
            move_failed: 0,
            move_status: None,
            weights: Weights::default(),
            sort_by: SortBy::Sharpness,
            show_weights: false,
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
        let photos: Vec<(PathBuf, u64)> =
            self.entries.iter().map(|e| (e.path.clone(), e.phash)).collect();
        self.groups = group_duplicates(&photos, self.group_threshold);
    }

    /// The full funnel: drop photos at or below the sharpness cull
    /// threshold, cluster survivors into duplicate/burst groups and keep
    /// only the highest-Overall member of each (plus any ungrouped
    /// singletons), then rank the rest by Overall and keep the top N.
    /// Cheap enough (same cost as `compute_groups` plus a sort) to run
    /// synchronously from a button click.
    fn run_pipeline(&mut self) {
        let total = self.entries.len();
        let cutoff = self.cull_threshold;
        let survivors: Vec<usize> =
            (0..total).filter(|&i| self.entries[i].score > cutoff).collect();
        let after_technical = survivors.len();

        let photos: Vec<(PathBuf, u64)> = survivors
            .iter()
            .map(|&i| (self.entries[i].path.clone(), self.entries[i].phash))
            .collect();
        let groups = group_duplicates(&photos, self.group_threshold);

        let path_to_idx: HashMap<PathBuf, usize> =
            survivors.iter().map(|&i| (self.entries[i].path.clone(), i)).collect();
        let grouped_paths: std::collections::HashSet<PathBuf> =
            groups.iter().flatten().cloned().collect();

        let mut keepers: Vec<usize> = Vec::new();
        for group in &groups {
            let best = group.iter().filter_map(|p| path_to_idx.get(p).copied()).max_by(
                |&a, &b| {
                    overall_score(&self.entries[a].metrics, &self.weights)
                        .partial_cmp(&overall_score(&self.entries[b].metrics, &self.weights))
                        .unwrap()
                },
            );
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

    /// Move every currently-flagged (likely out of focus) photo into a
    /// `disqualified` subfolder of the scanned root, in the background.
    fn start_move_disqualified(&mut self) {
        if self.moving {
            return;
        }
        let Some(root) = self.root.clone() else {
            return;
        };
        let Some(cutoff) = self.cull_cutoff() else {
            return;
        };
        let items: Vec<PathBuf> = self
            .entries
            .iter()
            .filter(|e| e.score <= cutoff)
            .map(|e| e.path.clone())
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
        let dest_dir = root.join(DISQUALIFIED_DIR);
        thread::spawn(move || {
            if let Err(e) = std::fs::create_dir_all(&dest_dir) {
                for path in items {
                    let _ = tx.send(MoveEvent::Failed(
                        path,
                        format!("couldn't create {}: {e}", dest_dir.display()),
                    ));
                }
                let _ = tx.send(MoveEvent::Done);
                return;
            }
            for path in items {
                let result = match path.file_name() {
                    Some(name) => std::fs::rename(&path, dest_dir.join(name)),
                    None => Err(std::io::Error::other("photo path has no file name")),
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
        let Some(root) = self.root.clone() else {
            return;
        };
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
        thread::spawn(move || run_scan(root, scan_mode, tx));
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
        thread::spawn(move || run_recompute(items, tx));
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

    /// Score below which entries are flagged. `None` while there's nothing
    /// scanned yet, so nothing shows as flagged before there's data.
    fn cull_cutoff(&self) -> Option<f64> {
        if self.entries.is_empty() {
            None
        } else {
            Some(self.cull_threshold)
        }
    }

    /// (min, max) score across all entries, for the threshold input's hint.
    fn score_range(&self) -> Option<(f64, f64)> {
        let mut iter = self.entries.iter().map(|e| e.score);
        let first = iter.next()?;
        Some(iter.fold((first, first), |(lo, hi), s| (lo.min(s), hi.max(s))))
    }

    /// One photo card: thumbnail (double-click to open the full-size
    /// preview), filename (red if `flagged`), the type dropdown, the
    /// absolute sharpness score, and the Overall score. Shared by the flat
    /// grid and the duplicate-groups view so they can't drift apart.
    /// `badge`, if given, is drawn as a colored label under the filename
    /// (e.g. marking the best-of-group pick).
    fn photo_card(
        &mut self,
        ui: &mut egui::Ui,
        idx: usize,
        flagged: bool,
        badge: Option<(egui::Color32, &str)>,
        open_request: &mut Option<PathBuf>,
    ) {
        ui.group(|ui| {
            ui.set_width(CARD_WIDTH);
            ui.vertical(|ui| {
                let entry = &self.entries[idx];
                let size = entry.texture.size_vec2();
                let max_dim = 160.0_f32;
                let scale = (max_dim / size.x.max(size.y)).min(1.0);
                let image_response = ui
                    .image((entry.texture.id(), size * scale))
                    .interact(egui::Sense::click());
                if image_response.double_clicked() {
                    *open_request = Some(entry.path.clone());
                }
                let name = entry
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                if flagged {
                    ui.colored_label(egui::Color32::from_rgb(220, 90, 90), name);
                } else {
                    ui.label(name);
                }
                if let Some((color, text)) = badge {
                    ui.colored_label(color, text);
                }

                ui.horizontal(|ui| {
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
                        ui.label(format!("score: {:.0}", entry.score));
                    }
                });
                if !self.entries[idx].dirty {
                    let overall = overall_score(&self.entries[idx].metrics, &self.weights);
                    ui.small(format!("overall: {overall:.0}"));
                }
            });
        });
    }

    /// Lay `indices` out as chunked (non-wrapping) rows of `photo_card`s --
    /// see the note on `horizontal_wrapped` corruption where this pattern
    /// is used in the flat grid for why it's not just `horizontal_wrapped`.
    /// `cutoff`, if given, flags (red name) entries scoring at or below it
    /// -- used by the flat grid, not the groups view. `best_idx`, if given,
    /// badges that one entry -- used by the groups view, not the flat grid.
    fn photo_row_grid(
        &mut self,
        ui: &mut egui::Ui,
        indices: &[usize],
        cutoff: Option<f64>,
        best_idx: Option<usize>,
        open_request: &mut Option<PathBuf>,
    ) {
        let spacing = ui.spacing().item_spacing.x;
        let columns = ((ui.available_width() / (CARD_WIDTH + spacing)).floor() as usize).max(1);
        for row in indices.chunks(columns) {
            ui.horizontal(|ui| {
                for &idx in row {
                    let flagged = cutoff.map(|c| self.entries[idx].score <= c).unwrap_or(false);
                    let badge = (Some(idx) == best_idx)
                        .then_some((egui::Color32::from_rgb(90, 170, 90), "★ best of group"));
                    self.photo_card(ui, idx, flagged, badge, open_request);
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
                    self.photo_row_grid(ui, &indices, None, best_idx, open_request);
                    ui.separator();
                }
            });
    }

    /// The pipeline shortlist view: the funnel summary line, then the
    /// ranked result grid, from the last "Run pipeline" click.
    fn render_pipeline(&mut self, ui: &mut egui::Ui, open_request: &mut Option<PathBuf>) {
        let Some(result) = &self.pipeline_result else {
            ui.label(
                "Click \"Run pipeline\" to filter, dedupe, and rank down to a shortlist.",
            );
            return;
        };
        let summary = format!(
            "{} photos → {} after technical filter (score > {:.0}) → {} after dedupe → top {} shown",
            result.total,
            result.after_technical,
            self.cull_threshold,
            result.after_dedupe,
            result.shortlist.len()
        );
        let shortlist = result.shortlist.clone();

        ui.label(summary);
        egui::ScrollArea::vertical()
            .auto_shrink([false, true])
            .show(ui, |ui| {
                self.photo_row_grid(ui, &shortlist, None, None, open_request);
            });
    }
}

impl eframe::App for Photo2CullApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.drain_events(&ctx);
        self.drain_preview(&ctx);
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
                if ui.button("Choose folder…").clicked()
                    && let Some(folder) = rfd::FileDialog::new().pick_folder()
                {
                    self.root = Some(folder);
                }
                match &self.root {
                    Some(p) => {
                        ui.label(p.display().to_string());
                    }
                    None => {
                        ui.label("No folder selected");
                    }
                }

                ui.separator();

                egui::ComboBox::from_label("Mode")
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

                ui.separator();

                let can_scan = self.root.is_some() && !self.scanning;
                if ui
                    .add_enabled(can_scan, egui::Button::new("Scan"))
                    .clicked()
                {
                    self.start_scan();
                }

                if self.scanning {
                    ui.spinner();
                    ui.label(format!("{} / {}", self.processed, self.total_found));
                } else if self.total_found > 0 {
                    ui.label(format!("{} photos scanned", self.total_found));
                }
            });
        });

        egui::CentralPanel::default().show(ui, |ui| {
            if self.entries.is_empty() {
                ui.label(
                    "Pick a folder and click Scan to check focus sharpness across your photos (RAW, PNG, JPG, TIFF, BMP, WebP).",
                );
                if !self.errors.is_empty() {
                    ui.label(format!("{} files failed to decode", self.errors.len()));
                }
                return;
            }

            let cutoff = self.cull_cutoff();
            let flagged_count = cutoff
                .map(|c| self.entries.iter().filter(|e| e.score <= c).count())
                .unwrap_or(0);
            let dirty_count = self.entries.iter().filter(|e| e.dirty).count();

            ui.horizontal(|ui| {
                ui.label("Disqualify scores below");
                ui.add(
                    egui::DragValue::new(&mut self.cull_threshold)
                        .speed(1.0)
                        .range(0.0..=f64::MAX),
                );
                if let Some((min, max)) = self.score_range() {
                    ui.label(format!("(scanned scores range {min:.0}–{max:.0})"));
                }

                ui.separator();

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

                ui.separator();

                let can_move = flagged_count > 0
                    && dirty_count == 0
                    && !self.moving
                    && !self.scanning
                    && !self.recomputing;
                if ui
                    .add_enabled(
                        can_move,
                        egui::Button::new(format!("Move {flagged_count} disqualified")),
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

            ui.horizontal(|ui| {
                ui.label("Sort by");
                egui::ComboBox::from_id_salt("sort-by")
                    .selected_text(match self.sort_by {
                        SortBy::Sharpness => "Sharpness",
                        SortBy::Overall => "Overall",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.sort_by, SortBy::Sharpness, "Sharpness");
                        ui.selectable_value(&mut self.sort_by, SortBy::Overall, "Overall");
                    });
                ui.checkbox(&mut self.show_weights, "Weights");
            });

            ui.horizontal(|ui| {
                ui.label("Duplicate/burst threshold");
                ui.add(
                    egui::DragValue::new(&mut self.group_threshold)
                        .speed(1)
                        .range(0..=64),
                );
                ui.small("(max hash distance, 0-64; lower = stricter)");

                ui.separator();

                let can_group = !self.entries.is_empty() && !self.scanning;
                if ui
                    .add_enabled(can_group, egui::Button::new("Group duplicates"))
                    .clicked()
                {
                    self.compute_groups();
                    self.view_mode = ViewMode::Grouped;
                }
                if !self.groups.is_empty() {
                    let grouped: usize = self.groups.iter().map(|g| g.len()).sum();
                    ui.label(format!("{} groups ({grouped} photos)", self.groups.len()));
                }

                ui.separator();

                egui::ComboBox::from_id_salt("view-mode")
                    .selected_text(match self.view_mode {
                        ViewMode::Flat => "All photos",
                        ViewMode::Grouped => "Duplicate groups",
                        ViewMode::Pipeline => "Pipeline shortlist",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.view_mode, ViewMode::Flat, "All photos");
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

            ui.horizontal(|ui| {
                ui.label("Pipeline shortlist size");
                ui.add(
                    egui::DragValue::new(&mut self.pipeline_top_n)
                        .speed(1)
                        .range(1..=100_000),
                );
                ui.small("technical filter -> dedupe (best of group) -> rank by Overall -> top N");

                ui.separator();

                let can_run = !self.entries.is_empty() && !self.scanning;
                if ui
                    .add_enabled(can_run, egui::Button::new("Run pipeline"))
                    .clicked()
                {
                    self.run_pipeline();
                    self.view_mode = ViewMode::Pipeline;
                }
            });

            if self.show_weights {
                ui.group(|ui| {
                    ui.label(
                        "Overall = weighted blend of these factors, each 0-100. \
                         Weights don't need to add to 100% -- they're renormalized \
                         over whatever's scored.",
                    );
                    egui::Grid::new("weights-grid").num_columns(2).show(ui, |ui| {
                        let pct = |ui: &mut egui::Ui, label: &str, w: &mut f32, note: &str| {
                            ui.label(label);
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::DragValue::new(w)
                                        .speed(0.01)
                                        .range(0.0..=1.0)
                                        .custom_formatter(|v, _| format!("{:.0}%", v * 100.0))
                                        .custom_parser(|s| {
                                            s.trim_end_matches('%').parse::<f64>().ok().map(|v| v / 100.0)
                                        }),
                                );
                                if !note.is_empty() {
                                    ui.small(note);
                                }
                            });
                            ui.end_row();
                        };
                        pct(ui, "Sharpness", &mut self.weights.sharpness, "");
                        pct(ui, "Exposure", &mut self.weights.exposure, "");
                        pct(ui, "Contrast", &mut self.weights.contrast, "");
                        pct(ui, "Color", &mut self.weights.color, "");
                        pct(
                            ui,
                            "Composition",
                            &mut self.weights.composition,
                            "(rule-of-thirds heuristic)",
                        );
                        pct(
                            ui,
                            "Subject",
                            &mut self.weights.subject,
                            "(Portrait only: face prominence)",
                        );
                    });
                });
            }

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
                            self.photo_row_grid(ui, &order, cutoff, None, &mut open_request);
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

        self.show_viewer(&ctx);
    }
}
