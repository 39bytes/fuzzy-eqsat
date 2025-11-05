use anyhow::Result;
use ndarray::Array2;
use std::collections::HashMap;
use std::rc::Rc;

use analysis::LinalgAnalysis;
use egg::*;
use ndarray_linalg::*;

use crate::analysis::{AnalysisData, MatrixDim, TrueValue, VarInfo};
use crate::cost::LinalgCost;
use crate::extract::MyExtractor;
use crate::lang::Linalg;
use crate::model::{Model, export_params, load_model, load_test_set};

mod analysis;
mod cost;
mod extract;
mod lang;
mod math;
mod model;
mod util;

fn make_expr(
    model: Model,
    test_set: Array2<f64>,
) -> Result<(HashMap<Symbol, VarInfo>, RecExpr<Linalg>)> {
    let input_size = model.layers[0].weights.shape()[0];

    let mut expr: RecExpr<Linalg> = RecExpr::default();
    let in_mat = expr.add(Linalg::Mat(Symbol::from("in")));

    let (u, sigma, vt) = test_set.svd(true, true)?;

    let mut var_info = HashMap::from_iter([(
        Symbol::from("in"),
        VarInfo {
            dim: MatrixDim::new(input_size, 1),
            svd: (u.unwrap().into(), sigma.into(), vt.unwrap().into()),
            value: test_set.into(),
        },
    )]);

    let mut prev_layer_out = in_mat;
    let n_layers = model.layers.len();

    for (i, layer) in model.layers.into_iter().enumerate() {
        let (u, sigma, vt) = layer.weights.svd(true, true)?;

        let (rows, cols) = layer.weights.dim();

        let weight_mat = Symbol::from(format!("w{}", i));
        let bias = Symbol::from(format!("b{}", i));

        var_info.insert(
            weight_mat,
            VarInfo {
                dim: MatrixDim::new(rows, cols),
                svd: (u.unwrap().into(), sigma.into(), vt.unwrap().into()),
                value: layer.weights.into(),
            },
        );

        let (u, sigma, vt) = layer.biases.svd(true, true)?;
        var_info.insert(
            bias,
            VarInfo {
                dim: MatrixDim::new(rows, 1),
                svd: (u.unwrap().into(), sigma.into(), vt.unwrap().into()),
                value: layer.biases.into(),
            },
        );

        let w = expr.add(Linalg::Mat(weight_mat));
        let b = expr.add(Linalg::Mat(bias));
        let mul = expr.add(Linalg::Mul([w, prev_layer_out]));
        let add = expr.add(Linalg::Add([mul, b]));
        prev_layer_out = if i < n_layers - 1 {
            expr.add(Linalg::Relu(add))
        } else {
            expr.add(Linalg::Softmax(add))
        };
    }

    Ok((var_info, expr))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SvdApplier {
    a: Var,
    b: Var,
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
        let b = subst[self.b];
        let rank = match &egraph[a].data {
            AnalysisData::Mat {
                true_value: Some(TrueValue { svd, .. }),
                ..
            } => svd.1.iter().filter(|x| **x > 0.0).count(),

            _ => return vec![],
        };

        let step = (rank.ilog2() - 2) as usize;
        let mut changed = vec![];
        let mut k = rank - step;

        while k > 0 {
            let k_node = egraph.add(Linalg::Num(k as i32));
            println!("K ({}) node ID: {}", k, k_node);
            let svd_mul_k = egraph.add(Linalg::SVDMul([a, b, k_node]));

            if egraph.union(eclass, svd_mul_k) {
                changed.push(svd_mul_k);
            }

            k -= step;
        }

        changed
    }
}

fn main() -> Result<()> {
    let mut rules: Vec<Rewrite<Linalg, LinalgAnalysis>> = vec![];
    rules.extend(rewrite!("matmul-assoc"; "(* (* ?a ?b) ?c)" <=> "(* ?a (* ?b ?c))"));
    rules.push(rewrite!("svd-mul"; "(* ?a ?b)" => {
        SvdApplier {
            a: "?a".parse().unwrap(),
            b: "?b".parse().unwrap(),
        }
    }));

    let model = load_model("mnist_model_params.json").unwrap();
    let in_vec = load_test_set("test_set.json", (10000, 784)).unwrap();
    let (var_info, expr) = make_expr(model, in_vec)?;

    println!("{}", expr.pretty(30));
    let var_info = Rc::new(var_info);
    for (sym, info) in var_info.iter() {
        println!("{}: [dim: {:?}]", sym, info.dim);
    }

    let runner = Runner::<Linalg, LinalgAnalysis, ()>::new(LinalgAnalysis {
        var_info: var_info.clone(),
    })
    .with_expr(&expr)
    .run(&rules);

    println!("Number of eclasses: {}", runner.egraph.classes().len());

    println!("Rendering");
    util::render_egraph(&runner.egraph, ".", "test2");

    println!("Extracting");
    let extractor = MyExtractor::new(
        &runner.egraph,
        LinalgCost {
            egraph: &runner.egraph,
            var_info: var_info.clone(),
            max_rel_error: 0.01,
        },
    );

    let (cost, best_expr) = extractor.find_best(runner.roots[0]);
    println!("Before: {}", expr);
    println!("Found best: {}", best_expr);
    println!("Cost: {:?}", cost);

    export_params(&best_expr, &var_info, "optimized.json")?;

    Ok(())
}
