use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};
use std::thread;

use egui::{ColorImage, TextureHandle, TextureOptions};

use crate::scan::{run_recompute, run_scan, RecomputeEvent, ScanEvent, ScanMode};
use crate::sharpness::PhotoMode;

struct PhotoEntry {
    path: PathBuf,
    mode: PhotoMode,
    score: f64,
    /// True if `mode` was changed (manually, or by a rescan) since `score`
    /// was last computed for it -- i.e. the score shown is stale.
    dirty: bool,
    texture: TextureHandle,
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

/// Name of the subfolder (created inside the scanned root) that
/// disqualified photos get moved into.
const DISQUALIFIED_DIR: &str = "disqualified";

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
            cull_threshold: 0.0,
            viewer: None,
            preview_rx: None,
            moving: false,
            move_rx: None,
            move_moved: 0,
            move_failed: 0,
            move_status: None,
        }
    }

    /// Move every currently-flagged (likely out of focus) photo into a
    /// `disqualified` subfolder of the scanned root, in the background.
    fn start_move_disqualified(&mut self) {
        if self.moving {
            return;
        }
        let Some(root) = self.root.clone() else { return };
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
            let event = match crate::raw::decode_raw(&path, PREVIEW_MAX_DIM) {
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
        let Some(root) = self.root.clone() else { return };
        self.entries.clear();
        self.errors.clear();
        self.total_found = 0;
        self.processed = 0;
        self.scanning = true;

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
                            ui.selectable_value(
                                &mut self.scan_mode,
                                ScanMode::Fixed(m),
                                m.label(),
                            );
                        }
                    });

                ui.separator();

                let can_scan = self.root.is_some() && !self.scanning;
                if ui.add_enabled(can_scan, egui::Button::new("Scan")).clicked() {
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
                    "Pick a folder and click Scan to check focus sharpness across your RAW files.",
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
            if !self.errors.is_empty() {
                ui.label(format!("{} files failed to decode", self.errors.len()));
            }
            ui.separator();

            let mut order: Vec<usize> = (0..self.entries.len()).collect();
            order.sort_by(|&a, &b| {
                self.entries[a]
                    .score
                    .partial_cmp(&self.entries[b].score)
                    .unwrap()
            });

            let mut open_request: Option<PathBuf> = None;

            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    for idx in order {
                        let flagged = cutoff.map(|c| self.entries[idx].score <= c).unwrap_or(false);
                        ui.group(|ui| {
                            ui.set_width(180.0);
                            ui.vertical(|ui| {
                                let entry = &self.entries[idx];
                                let size = entry.texture.size_vec2();
                                let max_dim = 160.0_f32;
                                let scale = (max_dim / size.x.max(size.y)).min(1.0);
                                let image_response = ui
                                    .image((entry.texture.id(), size * scale))
                                    .interact(egui::Sense::click());
                                if image_response.double_clicked() {
                                    open_request = Some(entry.path.clone());
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
                                        ui.colored_label(
                                            egui::Color32::from_rgb(200, 150, 60),
                                            "score: pending…",
                                        );
                                    } else {
                                        ui.label(format!("score: {:.0}", entry.score));
                                    }
                                });
                            });
                        });
                    }
                });
            });

            if let Some(path) = open_request {
                self.start_preview_load(path);
            }
        });

        self.show_viewer(&ctx);
    }
}
