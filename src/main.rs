mod app;
mod raw;
mod scan;
mod sharpness;

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
