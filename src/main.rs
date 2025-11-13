use anyhow::Result;
use log::{debug, info};
use ndarray::Array2;
use std::collections::HashMap;
use std::rc::Rc;

use analysis::LinalgAnalysis;
use egg::*;
use ndarray_linalg::*;

use crate::analysis::{AnalysisData, MatrixData, MatrixDim, TrueValue, VarInfo};
use crate::cost::{CostWithErrorBound, LinalgCost};
use crate::extract::MyExtractor;
use crate::lang::Linalg;
use crate::model::{ModelLayer, load_model, load_test_set, output_python_file};

mod analysis;
mod cost;
mod extract;
mod lang;
mod math;
mod model;
mod util;

fn make_expr(
    layers: Vec<ModelLayer>,
    test_set: Array2<f64>,
) -> Result<(HashMap<Symbol, VarInfo>, RecExpr<Linalg>)> {
    let input_size = layers[0].weights.shape()[0];

    let mut expr: RecExpr<Linalg> = RecExpr::default();
    let in_mat = expr.add(Linalg::Mat(Symbol::from("x")));

    let (u, sigma, vt) = test_set.svddc(JobSvd::Some)?;

    let mut var_info = HashMap::from_iter([(
        Symbol::from("x"),
        VarInfo {
            dim: MatrixDim::new(test_set.nrows(), test_set.ncols()),
            svd: (u.unwrap().into(), sigma.into(), vt.unwrap().into()),
            value: test_set.into(),
        },
    )]);

    let mut prev_layer_out = in_mat;
    let n_layers = layers.len();

    for (i, layer) in layers.into_iter().enumerate() {
        let (u, sigma, vt) = layer.weights.svddc(JobSvd::Some)?;

        let (rows, cols) = layer.weights.dim();

        let weight_mat = Symbol::from(format!("w[{}]", i));
        let bias = Symbol::from(format!("b[{}]", i));

        var_info.insert(
            weight_mat,
            VarInfo {
                dim: MatrixDim::new(rows, cols),
                svd: (u.unwrap().into(), sigma.into(), vt.unwrap().into()),
                value: layer.weights.into(),
            },
        );

        let (u, sigma, vt) = layer.biases.svddc(JobSvd::Some)?;
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
            AnalysisData::Mat(MatrixData {
                true_value: Some(TrueValue { svd, .. }),
                ..
            }) => svd.1.iter().filter(|x| **x > 0.0).count(),

            _ => return vec![],
        };

        let step = (rank.ilog2() - 2) as i32;
        let mut changed = vec![];
        let mut k: i32 = rank as i32 - step;

        while k > 0 {
            let k_node = egraph.add(Linalg::Num(k));
            let u_node = egraph.add(Linalg::SvdU([a, k_node]));
            let d_node = egraph.add(Linalg::SvdD([a, k_node]));
            let vt_node = egraph.add(Linalg::SvdVt([a, k_node]));

            let vtb = egraph.add(Linalg::Mul([vt_node, b]));
            let dvtb = egraph.add(Linalg::Mul([d_node, vtb]));
            let udvtb = egraph.add(Linalg::Mul([u_node, dvtb]));

            if egraph.union(eclass, udvtb) {
                changed.push(udvtb);
            }

            k -= step;
        }

        changed
    }
}

fn make_rules() -> Vec<Rewrite<Linalg, LinalgAnalysis>> {
    let mut rules: Vec<Rewrite<Linalg, LinalgAnalysis>> = vec![];
    // rules.extend(rewrite!("matmul-assoc"; "(* (* ?a ?b) ?c)" <=> "(* ?a (* ?b ?c))"));
    rules.push(rewrite!("svd-mul"; "(* ?a ?b)" => {
        SvdApplier {
            a: "?a".parse().unwrap(),
            b: "?b".parse().unwrap(),
        }
    }));

    rules
}

fn model_to_egg(
    model_path: &str,
    test_set_path: &str,
) -> Result<(HashMap<Symbol, VarInfo>, RecExpr<Linalg>)> {
    let model = load_model(model_path)?;
    let test_set = load_test_set(test_set_path, (10000, 784))?;

    let (var_info, expr) = make_expr(model, test_set)?;
    for (sym, info) in var_info.iter() {
        debug!("{}: [dim: {:?}]", sym, info.dim);
    }

    Ok((var_info, expr))
}

fn optimize(
    expr: RecExpr<Linalg>,
    var_info: Rc<HashMap<Symbol, VarInfo>>,
    max_rel_error: f64,
) -> Result<()> {
    let rules = make_rules();

    let runner = Runner::<Linalg, LinalgAnalysis, ()>::new(LinalgAnalysis {
        var_info: var_info.clone(),
    })
    .with_expr(&expr)
    .run(&rules);

    info!("Number of eclasses: {}", runner.egraph.classes().len());

    info!("Rendering");
    util::render_egraph(&runner.egraph, ".", "test2");

    info!("Extracting");
    let extractor = MyExtractor::new(
        &runner.egraph,
        LinalgCost {
            egraph: &runner.egraph,
            var_info: var_info.clone(),
            max_rel_error,
        },
    );

    // for expr in extractor.all_costs(runner.roots[0]) {
    //     let cost = &expr.cost;
    //     let expr = expr.node.build_recexpr(|id| expr.children[&id].clone());
    //     println!(
    //         "cost: {}, error: {:?}, expr: {}",
    //         cost.cost, cost.error, expr
    //     );
    // }

    // Ok(extractor.find_best(runner.roots[0]))
    let (best, best_expr) = extractor.find_best(runner.roots[0]);
    println!("Best cost: {:?}", best.cost);
    println!("Best: {}", best_expr);

    output_python_file(best, &best_expr, &runner.egraph, &var_info, "out.py")?;

    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();

    let (var_info, expr) = model_to_egg("mnist_model_params.json", "test_set.json")?;
    let var_info = Rc::new(var_info);
    optimize(expr, var_info.clone(), 0.05)?;

    Ok(())
}
