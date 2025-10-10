use anyhow::Result;
use ndarray::Array2;
use std::collections::HashMap;
use std::rc::Rc;

use analysis::LinalgAnalysis;
use egg::*;
use ndarray_linalg::*;

use crate::analysis::{AnalysisData, MatrixDim, VarInfo};
use crate::cost::LinalgCost;
use crate::lang::Linalg;
use crate::load::{Model, load_in_vec, load_model};

mod analysis;
mod cost;
mod lang;
mod load;
mod util;

fn make_expr(
    model: Model,
    avg_vec: Array2<f64>,
) -> (Result<(HashMap<Symbol, VarInfo>, RecExpr<Linalg>)>) {
    let input_size = model.layers[0].input_shape.0;

    let mut expr: RecExpr<Linalg> = RecExpr::default();
    let in_vec = expr.add(Linalg::Mat(Symbol::from("in")));

    let mut var_info = HashMap::from_iter([(
        Symbol::from("in"),
        VarInfo {
            dim: MatrixDim::new(input_size, 1),
            singular_values: None,
            value: Some(avg_vec.into()),
        },
    )]);

    let mut prev_layer_out = in_vec;

    for (i, layer) in model.layers.into_iter().enumerate() {
        let (_, sigma, _) = layer.weights.svd(false, false)?;
        println!("{}", sigma);

        let (rows, cols) = layer.weights.dim();

        let weight_mat = Symbol::from(format!("w{}", i));
        let bias = Symbol::from(format!("b{}", i));

        var_info.insert(
            weight_mat,
            VarInfo {
                dim: MatrixDim::new(rows, cols),
                singular_values: Some(sigma.into()),
                value: Some(layer.weights.into()),
            },
        );

        var_info.insert(
            bias,
            VarInfo {
                dim: MatrixDim::new(rows, 1),
                singular_values: None,
                value: Some(layer.biases.into()),
            },
        );

        let w = expr.add(Linalg::Mat(weight_mat));
        let b = expr.add(Linalg::Mat(bias));
        let mul = expr.add(Linalg::Mul([w, prev_layer_out]));
        let add = expr.add(Linalg::Add([mul, b]));
        prev_layer_out = expr.add(Linalg::Relu(add));
    }

    Ok((var_info, expr))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SvdApplier {
    a: Var,
}

impl Applier<Linalg, LinalgAnalysis> for SvdApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Linalg, LinalgAnalysis>,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<Linalg>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let a = subst[self.a];
        let rank = match &egraph[a].data {
            AnalysisData::Mat {
                singular_values: Some(vals),
                ..
            } => vals.len(),
            _ => return vec![],
        };

        let mut changed = vec![];

        for k in 1..rank {
            let k = egraph.add(Linalg::Num(k as i32));
            let u = egraph.add(Linalg::SvdU([a, k]));
            let d = egraph.add(Linalg::SvdD([a, k]));
            let vt = egraph.add(Linalg::SvdVt([a, k]));

            let ud = egraph.add(Linalg::Mul([u, d]));
            let udvt = egraph.add(Linalg::Mul([ud, vt]));

            if egraph.union(eclass, udvt) {
                changed.push(udvt);
            }
        }

        changed
    }
}

fn main() -> Result<()> {
    let mut rules: Vec<Rewrite<Linalg, LinalgAnalysis>> = vec![];
    rules.extend(rewrite!("matmul-assoc"; "(* (* ?a ?b) ?c)" <=> "(* ?a (* ?b ?c))"));
    rules.push(rewrite!("svd"; "?a" => {
        SvdApplier {
            a: "?a".parse().unwrap(),
        }
    }));

    let model = load_model("mnist_model_params.json").unwrap();
    let in_vec = load_in_vec("vec.json").unwrap();
    let (var_info, expr) = make_expr(model, in_vec)?;

    println!("{}", expr.pretty(30));

    let runner = Runner::<Linalg, LinalgAnalysis, ()>::new(LinalgAnalysis { var_info })
        .with_expr(&expr)
        .run(&rules);

    println!("Rendering");

    println!("Extracting");
    let extractor = Extractor::new(
        &runner.egraph,
        LinalgCost {
            egraph: &runner.egraph,
            max_rel_error: 0.1,
        },
    );

    let (cost, best_expr) = extractor.find_best(runner.roots[0]);
    println!("Before: {}", expr);
    println!("Found best: {}", best_expr);
    println!("Cost: {:?}", cost);

    Ok(())
}
