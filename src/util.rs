use std::{fmt::Display, path::Path};

use egg::*;

pub fn render_egraph<L: Language + Display, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    dir: &str,
    name: &str,
) {
    // render the e-graph as a dot file
    let dot_filename = format!("{}/{}.dot", dir, name);
    let png_filename = format!("{}/{}.png", dir, name);

    let path = Path::new(&dot_filename);
    egraph
        .dot()
        .to_dot(path)
        .expect("Couldn't write e-graph to file");

    // render dot file into a png
    std::process::Command::new("dot")
        .arg("-Tpng")
        .arg(&dot_filename)
        .arg("-o")
        .arg(&png_filename)
        .output()
        .expect("Couldn't render dot file to png");
}
