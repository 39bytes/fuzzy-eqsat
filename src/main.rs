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
use crate::load::{Model, load_in_vec, load_model};

mod analysis;
mod cost;
mod extract;
mod lang;
mod load;
mod util;

fn make_expr(
    model: Model,
    avg_vec: Array2<f64>,
) -> Result<(HashMap<Symbol, VarInfo>, RecExpr<Linalg>)> {
    let input_size = model.layers[0].input_shape.0;

    let mut expr: RecExpr<Linalg> = RecExpr::default();
    let in_vec = expr.add(Linalg::Mat(Symbol::from("in")));

    let (u, sigma, vt) = avg_vec.svd(true, true)?;

    let mut var_info = HashMap::from_iter([(
        Symbol::from("in"),
        VarInfo {
            dim: MatrixDim::new(input_size, 1),
            svd: (u.unwrap().into(), sigma.into(), vt.unwrap().into()),
            value: avg_vec.into(),
        },
    )]);

    let mut prev_layer_out = in_vec;

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
        prev_layer_out = expr.add(Linalg::Relu(add));
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
    // let a = array![[0.0, 0.0], [0.0, 0.0]];
    // let (_, sigma, _) = a.svd(false, false).unwrap();
    // println!("{}", sigma);
    // Ok(())

    let mut rules: Vec<Rewrite<Linalg, LinalgAnalysis>> = vec![];
    rules.extend(rewrite!("matmul-assoc"; "(* (* ?a ?b) ?c)" <=> "(* ?a (* ?b ?c))"));
    rules.push(rewrite!("svd-mul"; "(* ?a ?b)" => {
        SvdApplier {
            a: "?a".parse().unwrap(),
            b: "?b".parse().unwrap(),
        }
    }));

    let model = load_model("mnist_model_params.json").unwrap();
    let in_vec = load_in_vec("vec.json").unwrap();
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
            var_info,
            max_rel_error: 0.20,
        },
        20,
    );

    // println!("{:?}", extractor.costs);

    for eclass in runner.egraph.classes() {
        let best_k = extractor.find_best_k(eclass.id);

        for (cost, node) in best_k {
            if node.to_string().parse::<f64>().is_ok() {
                continue;
            }
            println!("eclass: {} term: {}", eclass.id, node);
            println!("{}", cost);
        }
    }

    let (cost, best_expr) = extractor.find_best(runner.roots[0]);
    println!("Before: {}", expr);
    println!("Found best: {}", best_expr);
    println!("Cost: {:?}", cost);

    Ok(())
}
