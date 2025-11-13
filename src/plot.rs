use plotters::prelude::*;

use anyhow::Result;

pub fn output_pareto(filename: &str, points: &[(f64, f64)]) -> Result<()> {
    assert!(!points.is_empty());

    let root = SVGBackend::new(filename, (1024, 768)).into_drawing_area();

    root.fill(&WHITE)?;

    let (x_max, y_max) = points.iter().fold((0.0f64, 0.0f64), |(mx, my), (x, y)| {
        (mx.max(*x), my.max(*y))
    });

    let mut scatter_ctx = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(80)
        .build_cartesian_2d(0.0f64..x_max, 0.0f64..y_max)?;

    scatter_ctx
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()?;

    scatter_ctx.draw_series(
        points
            .iter()
            .map(|(x, y)| Circle::new((*x, *y), 2, RED.filled())),
    )?;

    root.present().expect("failed to write graph");

    Ok(())
}
