#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use photo2cull::app;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 750.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Photo2Cull",
        options,
        Box::new(|cc| Ok(Box::new(app::Photo2CullApp::new(cc)))),
    )
}
