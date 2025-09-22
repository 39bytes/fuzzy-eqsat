use anyhow::{Context, Error};
use std::{fmt::Display, path::Path, str::FromStr};

use egg::*;

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone, Copy, Default)]
struct MatrixDim {
    pub rows: usize,
    pub cols: usize,
}

impl MatrixDim {
    fn max_rank(&self) -> usize {
        self.rows.min(self.cols)
    }
}

impl Display for MatrixDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.rows, self.cols)
    }
}

impl FromStr for MatrixDim {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (rows, cols) = s
            .split_once("x")
            .context("Failed to parse matrix dimensions")?;

        Ok(MatrixDim {
            rows: rows.parse::<usize>()?,
            cols: cols.parse::<usize>()?,
        })
    }
}

define_language! {
    enum Linalg {
        "mat" = Mat([Id; 2]),
        "*" = Mul([Id; 2]),
        "svd_u" = SvdU(Id),
        "svd_d" = SvdD(Id),
        "svd_vt" = SvdVt(Id),
        Dimensions(MatrixDim),
        Var(Symbol),
    }
}

#[derive(Default)]
struct Dimensions;

impl Analysis<Linalg> for Dimensions {
    type Data = MatrixDim;

    fn merge(&mut self, _: &mut Self::Data, _: Self::Data) -> DidMerge {
        DidMerge(false, false)
    }

    fn make(egraph: &mut EGraph<Linalg, Self>, enode: &Linalg) -> Self::Data {
        let dim = |i: &Id| egraph[*i].data;
        match enode {
            Linalg::Dimensions(dim) => *dim,
            Linalg::Var(_) => MatrixDim::default(),
            Linalg::Mat([_, dims]) => dim(dims),
            Linalg::Mul([a, b]) => MatrixDim {
                rows: dim(a).rows,
                cols: dim(b).cols,
            },
            Linalg::SvdU(a) => {
                let dim = dim(a);
                let r = dim.max_rank();
                let r = 10;
                MatrixDim {
                    rows: dim.rows,
                    cols: r,
                }
            }
            Linalg::SvdD(a) => {
                let dim = dim(a);
                let r = dim.max_rank();
                let r = 10;
                MatrixDim { rows: r, cols: r }
            }
            Linalg::SvdVt(a) => {
                let dim = dim(a);
                let r = dim.max_rank();
                let r = 10;
                MatrixDim {
                    rows: r,
                    cols: dim.cols,
                }
            }
        }
    }
}

fn svd_cost(dims: MatrixDim) -> usize {
    let min = dims.rows.min(dims.cols);
    let max = dims.rows.max(dims.cols);
    min * min * max * 15
}

struct LinalgCost<'a> {
    egraph: &'a EGraph<Linalg, Dimensions>,
}

impl<'a> CostFunction<Linalg> for LinalgCost<'a> {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &Linalg, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let dim = |i: &Id| self.egraph[*i].data;
        let op_cost = match enode {
            Linalg::Mat(_) => 0,
            Linalg::Dimensions(_) => 0,
            Linalg::Var(_) => 0,
            Linalg::SvdU(a) | Linalg::SvdD(a) | Linalg::SvdVt(a) => svd_cost(dim(a)) / 3,
            Linalg::Mul([a, b]) => dim(a).rows * dim(a).cols * dim(b).cols,
        };
        enode.fold(op_cost, |sum, id| sum + costs(id))
    }
}

fn render_egraph<L: Language + Display, N: Analysis<L>>(
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

fn main() {
    let mut rules: Vec<Rewrite<Linalg, Dimensions>> = vec![];
    rules.extend(rewrite!("matmul-assoc"; "(* (* ?a ?b) ?c)" <=> "(* ?a (* ?b ?c))"));
    rules.push(rewrite!("svd"; "?a" => "(* (* (svd_u ?a) (svd_d ?a)) (svd_vt ?a))"));

    let expr = "(* (* (mat a 10000x100) (mat b 100x10000)) (mat c 10000x100))"
        .parse()
        .unwrap();
    println!("hi");
    println!("Running rules");
    let runner = Runner::<Linalg, Dimensions, ()>::default()
        .with_expr(&expr)
        .run(&rules);

    println!("Rendering");
    // render_egraph(&runner.egraph, ".", "graph");

    println!("Extracting");
    let extractor = Extractor::new(
        &runner.egraph,
        LinalgCost {
            egraph: &runner.egraph,
        },
    );

    println!("Extracting");
    let (cost, best_expr) = extractor.find_best(runner.roots[0]);
    println!("Before: {}", expr);
    println!("Found best: {}", best_expr);
    println!("Cost: {}", cost);
}
