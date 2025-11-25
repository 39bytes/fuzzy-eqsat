use anyhow::Result;
use log::{debug, info};
use ndarray::Array2;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use analysis::LinalgAnalysis;
use egg::*;
use ndarray_linalg::*;
use std::time::Instant;

use crate::analysis::{AnalysisData, MODEL_INPUT, MatrixData, MatrixValue};
use crate::cost::LinalgCost;
use crate::extract::CompleteExtractor;
use crate::lang::Linalg;
use crate::math::prune;
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
) -> Result<(HashMap<Symbol, MatrixValue>, RecExpr<Linalg>)> {
    let mut expr: RecExpr<Linalg> = RecExpr::default();
    let input_sym = Symbol::from(MODEL_INPUT);
    let model_input = expr.add(Linalg::Mat(input_sym));

    let mut var_info = HashMap::from_iter([(input_sym, MatrixValue::new(test_set.into()))]);

    let mut prev_layer_out = model_input;
    let n_layers = layers.len();

    for (i, layer) in layers.into_iter().enumerate() {
        let weight_mat = Symbol::from(format!("w[{}]", i));
        let bias = Symbol::from(format!("b[{}]", i));

        var_info.insert(weight_mat, MatrixValue::new(layer.weights.into()));
        var_info.insert(bias, MatrixValue::new(layer.biases.into()));

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
        _: Option<&PatternAst<Linalg>>,
        _: Symbol,
    ) -> Vec<Id> {
        let a = subst[self.a];
        let b = subst[self.b];
        let rank = match &egraph[a].data {
            AnalysisData::Mat(MatrixData {
                canonical_value: Some(val),
                ..
            }) => val.svd().1.iter().filter(|x| **x > 0.0).count(),

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

struct PruneApplier {
    a: Var,
    // pruned_nodes: RefCell<HashMap<Symbol, VarInfo>>,
}

// fn make_pruned_vars(&mut ) {
//     return
// }

// fn prunable(var: &'static str) -> impl Fn(&mut EGraph<Linalg, LinalgAnalysis>, Id, &Subst) -> bool {
//     let var: Var = var.parse().unwrap();
//
//     move |egraph, _, subst| {
//         let eclass = &egraph[subst[var]];
//         match eclass.data {
//             AnalysisData::Mat(_) => {}
//             _ => return false,
//         };
//
//         !eclass
//             .nodes
//             .iter()
//             .any(|node| matches!(node, Linalg::Prune(_)))
//     }
// }

impl Applier<Linalg, LinalgAnalysis> for PruneApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Linalg, LinalgAnalysis>,
        eclass: Id,
        subst: &Subst,
        _: Option<&PatternAst<Linalg>>,
        _: Symbol,
    ) -> Vec<Id> {
        let a = subst[self.a];

        if egraph.analysis.pruned_eclasses.contains(&eclass) {
            return vec![];
        }

        // only prune const eclasses (that we can know at compile time)
        // (e.g weights/biases or their svd)
        let data = match &egraph[eclass].data {
            AnalysisData::Mat(data) if data.is_const => data.clone(),
            _ => return vec![],
        };

        let mut changed = vec![];

        for precision in -6..-2 {
            let precision_node = egraph.add(Linalg::Num(precision));

            let sym = Symbol::from(format!(
                "pruned-{} ({})",
                egraph.analysis.prune_count, precision
            ));
            // TODO:
            // let pruned_val = prune(&data., precision);
            // egraph
            //     .analysis
            //     .var_info
            //     .borrow_mut()
            //     .insert(sym, MatrixValue::new(pruned_val));
            let pruned_mat = egraph.add(Linalg::Mat(sym));

            let prune_node = egraph.add(Linalg::Pruned([pruned_mat, precision_node]));

            egraph.analysis.pruned_eclasses.insert(pruned_mat);

            if egraph.union(eclass, prune_node) {
                changed.push(prune_node);
            }
        }
        egraph.analysis.prune_count += 1;
        egraph.analysis.pruned_eclasses.insert(eclass);

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
    // rules.push(rewrite!("prune"; "?a" => {
    //     PruneApplier {
    //         a: "?a".parse().unwrap()
    //     }
    // }));

    rules
}

fn model_to_egg(
    model_path: &str,
    test_set_path: &str,
) -> Result<(HashMap<Symbol, MatrixValue>, RecExpr<Linalg>)> {
    let model = load_model(model_path)?;
    let test_set = load_test_set(test_set_path, (10000, 784))?;

    let (var_info, expr) = make_expr(model, test_set)?;
    for (sym, info) in var_info.iter() {
        debug!("{}: [dim: {:?}]", sym, info.dim());
    }

    Ok((var_info, expr))
}

fn optimize(
    expr: RecExpr<Linalg>,
    var_info: HashMap<Symbol, MatrixValue>,
    max_rel_error: f64,
) -> Result<()> {
    let rules = make_rules();
    let var_info = Rc::new(RefCell::new(var_info));

    let before = Instant::now();
    let runner = Runner::<Linalg, LinalgAnalysis, ()>::new(LinalgAnalysis {
        var_info: var_info.clone(),
        prune_count: 0,
        pruned_eclasses: HashSet::new(),
    })
    .with_expr(&expr)
    .run(&rules);

    println!("Analysis took: {}ms", before.elapsed().as_millis());

    info!("Number of eclasses: {}", runner.egraph.classes().len());

    info!("Rendering");
    util::render_egraph(&runner.egraph, ".", "test2");

    let before = Instant::now();
    info!("Extracting");
    let extractor = CompleteExtractor::new(
        &runner.egraph,
        LinalgCost {
            egraph: &runner.egraph,
            var_info: var_info.clone(),
            max_rel_error,
        },
    );
    println!("Extraction took: {}ms", before.elapsed().as_millis());

    let (best, best_expr) = extractor.find_best(runner.roots[0]);
    println!("Best: {}", best_expr);
    println!(
        "Cost: {}, Error: {}",
        best.cost.cost,
        best.cost.error.unwrap()
    );

    let before = Instant::now();
    output_python_file(best, &best_expr, &runner.egraph, &var_info, "out.py")?;
    println!("Outputting python took: {}ms", before.elapsed().as_millis());

    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();

    let (var_info, expr) = model_to_egg("mnist_model_params.json", "test_set.json")?;
    optimize(expr, var_info, 0.01)?;

    Ok(())
}
