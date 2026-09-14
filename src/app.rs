use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};
use std::thread;

use egui::{ColorImage, TextureHandle, TextureOptions};

use crate::scan::{run_scan, ScanEvent};
use crate::sharpness::PhotoMode;

struct PhotoEntry {
    path: PathBuf,
    score: f64,
    texture: TextureHandle,
}

pub struct Photo2CullApp {
    root: Option<PathBuf>,
    mode: PhotoMode,
    entries: Vec<PhotoEntry>,
    errors: Vec<(PathBuf, String)>,
    total_found: usize,
    processed: usize,
    scanning: bool,
    rx: Option<Receiver<ScanEvent>>,
    cull_percentile: f32,
}

impl Photo2CullApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            root: None,
            mode: PhotoMode::Landscape,
            entries: Vec::new(),
            errors: Vec::new(),
            total_found: 0,
            processed: 0,
            scanning: false,
            rx: None,
            cull_percentile: 20.0,
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
        let mode = self.mode;
        thread::spawn(move || run_scan(root, mode, tx));
    }

    fn drain_events(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.rx else { return };
        let mut done = false;
        while let Ok(event) = rx.try_recv() {
            match event {
                ScanEvent::Found(n) => self.total_found = n,
                ScanEvent::Photo(p) => {
                    self.processed += 1;
                    let image =
                        ColorImage::from_rgb([p.thumb_w as usize, p.thumb_h as usize], &p.thumb_rgb);
                    let texture = ctx.load_texture(
                        p.path.to_string_lossy().to_string(),
                        image,
                        TextureOptions::LINEAR,
                    );
                    self.entries.push(PhotoEntry {
                        path: p.path,
                        score: p.score,
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

    /// Score below which entries are flagged, given the current percentile.
    fn cull_cutoff(&self) -> Option<f64> {
        if self.entries.is_empty() {
            return None;
        }
        let mut scores: Vec<f64> = self.entries.iter().map(|e| e.score).collect();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((self.cull_percentile / 100.0) * scores.len() as f32).round() as usize;
        let idx = idx.min(scores.len() - 1);
        Some(scores[idx])
    }
}

impl eframe::App for Photo2CullApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.drain_events(&ctx);
        if self.scanning {
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
                    .selected_text(self.mode.label())
                    .show_ui(ui, |ui| {
                        for m in PhotoMode::ALL {
                            ui.selectable_value(&mut self.mode, m, m.label());
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

            ui.horizontal(|ui| {
                ui.label("Flag bottom");
                ui.add(
                    egui::Slider::new(&mut self.cull_percentile, 0.0..=100.0)
                        .suffix("%")
                        .fixed_decimals(0),
                );
                ui.label("as likely out of focus");
            });
            if !self.errors.is_empty() {
                ui.label(format!("{} files failed to decode", self.errors.len()));
            }
            ui.separator();

            let cutoff = self.cull_cutoff();
            let mut sorted: Vec<&PhotoEntry> = self.entries.iter().collect();
            sorted.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    for entry in sorted {
                        let flagged = cutoff.map(|c| entry.score <= c).unwrap_or(false);
                        ui.group(|ui| {
                            ui.set_width(180.0);
                            ui.vertical(|ui| {
                                let size = entry.texture.size_vec2();
                                let max_dim = 160.0_f32;
                                let scale = (max_dim / size.x.max(size.y)).min(1.0);
                                ui.image((entry.texture.id(), size * scale));
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
                                ui.label(format!("score: {:.0}", entry.score));
                            });
                        });
                    }
                });
            });
        });
    }
}
