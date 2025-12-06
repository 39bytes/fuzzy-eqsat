use anyhow::Result;
use csv::Writer;
use log::{debug, info};
use ndarray::{Array1, Array2, Axis};
use ndarray_stats::QuantileExt;
use serde::Serialize;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::rc::Rc;

use analysis::LinalgAnalysis;
use egg::*;
use std::time::Instant;

use crate::analysis::{AnalysisData, MODEL_INPUT, MatrixData};
use crate::cost::{CostWithErrorBound, LinalgCost};
use crate::extract::GeneticAlgorithmExtractor;
use crate::lang::Linalg;
use crate::math::{accuracy, prune};
use crate::matrix::MatrixValue;
use crate::model::{
    Activation, ModelLayer, TestSet, load_model, load_test_set, output_python_file,
};
use crate::plot::output_pareto;

mod analysis;
mod cost;
mod extract;
mod lang;
mod math;
mod matrix;
mod model;
mod plot;
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

    for (i, layer) in layers.into_iter().enumerate() {
        let weight_mat = Symbol::from(format!("w{}", i));
        let bias = Symbol::from(format!("b{}", i));

        var_info.insert(weight_mat, MatrixValue::new(layer.weights.into()));
        var_info.insert(bias, MatrixValue::new(layer.biases.into()));

        let w = expr.add(Linalg::Mat(weight_mat));
        let b = expr.add(Linalg::Mat(bias));
        let mul = expr.add(Linalg::Mul([w, prev_layer_out]));
        let add = expr.add(Linalg::Add([mul, b]));
        prev_layer_out = match layer.activation {
            Activation::Relu => expr.add(Linalg::Relu(add)),
            Activation::Tanh => expr.add(Linalg::Tanh(add)),
            Activation::Softmax => expr.add(Linalg::Softmax(add)),
        }
    }

    Ok((var_info, expr))
}

struct RewriteParameters {
    svd_step_scale: Option<i32>,
    prune_range: Option<(i32, i32)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SvdApplier {
    step_scale: i32,
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
        if egraph[a]
            .nodes
            .iter()
            .any(|n| matches!(n, Linalg::SvdU(_) | Linalg::SvdD(_) | Linalg::SvdVt(_)))
        {
            return vec![];
        }
        let rank = match &egraph[a].data {
            AnalysisData::Mat(MatrixData {
                canonical_value: Some(val),
                ..
            }) => val.svd().1.iter().filter(|x| **x > 0.0).count(),

            _ => return vec![],
        };

        let step = rank.ilog2() as i32 * self.step_scale;
        if step == 0 {
            return vec![];
        }
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

            k -= 20;
        }

        changed
    }
}

struct PruneApplier {
    a: Var,
    prune_low: i32,
    prune_high: i32,
}

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
        let const_value = match &egraph[eclass].data {
            AnalysisData::Mat(MatrixData {
                const_value: Some(val),
                ..
            }) => val.clone(),
            _ => return vec![],
        };

        let mut changed = vec![];

        for precision in self.prune_low..self.prune_high {
            let sym = Symbol::from(format!(
                "pruned-{} ({})",
                egraph.analysis.prune_count, precision
            ));
            let pruned_val = prune(const_value.val(), precision);
            egraph
                .analysis
                .var_info
                .borrow_mut()
                .insert(sym, MatrixValue::new(pruned_val));
            let pruned_mat = egraph.add(Linalg::Mat(sym));

            egraph.analysis.pruned_eclasses.insert(pruned_mat);

            if egraph.union(eclass, pruned_mat) {
                changed.push(pruned_mat);
            }
        }
        egraph.analysis.prune_count += 1;
        egraph.analysis.pruned_eclasses.insert(eclass);

        changed
    }
}

fn make_rules(params: RewriteParameters) -> Vec<Rewrite<Linalg, LinalgAnalysis>> {
    let mut rules: Vec<Rewrite<Linalg, LinalgAnalysis>> = vec![];
    rules.extend(rewrite!("matmul-assoc"; "(* (* ?a ?b) ?c)" <=> "(* ?a (* ?b ?c))"));
    if let Some(step_scale) = params.svd_step_scale {
        log::info!("Step scale: {}", step_scale);
        rules.push(rewrite!("svd-mul"; "(* ?a ?b)" => {
            SvdApplier {
                a: "?a".parse().unwrap(),
                b: "?b".parse().unwrap(),
                step_scale,
            }
        }));
    }

    if let Some((prune_low, prune_high)) = params.prune_range {
        rules.push(rewrite!("prune"; "?a" => {
            PruneApplier {
                a: "?a".parse().unwrap(),
                prune_low,
                prune_high,
            }
        }));
    }

    rules
}

fn model_to_egg(
    model_path: &str,
    test_set_path: &str,
    test_set_dim: (usize, usize),
    normalize: bool,
) -> Result<(HashMap<Symbol, MatrixValue>, RecExpr<Linalg>, Array1<usize>)> {
    let model = load_model(model_path)?;
    let test_set = load_test_set(test_set_path, test_set_dim, normalize)?;

    let (var_info, expr) = make_expr(model, test_set.mat)?;
    for (sym, info) in var_info.iter() {
        debug!("{}: [dim: {:?}]", sym, info.dim());
    }

    Ok((var_info, expr, test_set.labels))
}

// **Metrics**
// - Number of EClasses
// - Extraction time
// - Pareto for explored solutions
// - Cost
// - Relative error
// - Classification error
// - Execution time ?

#[derive(Serialize)]
struct Metrics {
    title: String,
    eclass_count: usize,
    extraction_time: u128,
    optimized_cost: usize,
    optimized_rel_error: f64,
    rel_cost_decrease: f64,
    classification_accuracy: f64,
}

struct EqSatResult {
    metrics: Metrics,
    optimized: Option<extract::Solution>,
    pareto_points: Vec<(f64, f64)>,
    root: Id,
    egraph: EGraph<Linalg, LinalgAnalysis>,
    var_info: Rc<RefCell<HashMap<Symbol, MatrixValue>>>,
}

fn optimize(
    title: String,
    expr: &RecExpr<Linalg>,
    var_info: HashMap<Symbol, MatrixValue>,
    test_set_labels: &Array1<usize>,
    max_rel_error: f64,
    rewrite_params: RewriteParameters,
) -> EqSatResult {
    let rules = make_rules(rewrite_params);
    let var_info = Rc::new(RefCell::new(var_info));

    let before = Instant::now();
    let runner = Runner::<Linalg, LinalgAnalysis, ()>::new(LinalgAnalysis {
        var_info: var_info.clone(),
        prune_count: 0,
        pruned_eclasses: HashSet::new(),
    })
    .with_expr(expr)
    .run(&rules);

    info!("Analysis took: {}ms", before.elapsed().as_millis());

    let eclass_count = runner.egraph.classes().len();
    info!("Number of eclasses: {}", eclass_count);

    // info!("Rendering");
    // util::render_egraph(&runner.egraph, ".", "test2");

    let before = Instant::now();
    info!("Extracting");
    let mut cf = LinalgCost {
        egraph: &runner.egraph,
        var_info: var_info.clone(),
        max_rel_error,
    };
    let initial_cost = cf.cost_rec(expr);
    info!("Initial: {}", expr);
    info!(
        "Initial cost: {}",
        initial_cost.cost / test_set_labels.len()
    );
    info!(
        "Initial classification accuracy: {}",
        accuracy(&initial_cost.val, test_set_labels)
    );

    let mut extractor = GeneticAlgorithmExtractor::new(&runner.egraph, cf);
    let root = runner.roots[0];
    let (best, best_sol, pareto_points) = extractor.find_best(root, expr.clone(), |c| {
        (c.error.unwrap(), (c.cost / test_set_labels.len()) as f64)
    });
    let extraction_time = before.elapsed().as_millis();
    info!("Extraction took: {}ms", extraction_time);

    let optimized_cost = best.cost / test_set_labels.len();
    let optimized_rel_error = best.error.unwrap();
    let rel_cost_decrease = (initial_cost.cost - best.cost) as f64 / initial_cost.cost as f64;
    info!(
        "Optimized cost: {}, Error: {}",
        optimized_cost, optimized_rel_error
    );
    let classification_accuracy = accuracy(&best.val, test_set_labels);
    info!("Classification accuracy: {}", classification_accuracy);

    EqSatResult {
        metrics: Metrics {
            title,
            eclass_count,
            extraction_time,
            optimized_cost,
            optimized_rel_error,
            rel_cost_decrease,
            classification_accuracy,
        },
        optimized: best_sol,
        pareto_points,
        root,
        egraph: runner.egraph,
        var_info,
    }
}

fn run_experiment(
    model_name: &str,
    model_path: &str,
    test_set_path: &str,
    test_set_dim: (usize, usize),
    normalize: bool,
) -> Result<()> {
    let (var_info, expr, test_set_labels) =
        model_to_egg(model_path, test_set_path, test_set_dim, normalize)?;
    let errs = [0.01, 0.02, 0.05, 0.10, 0.25];
    let rewrite_combs = [
        (Some(1), None),
        (None, Some((-3, 0))),
        (Some(1), Some((-3, 0))),
    ];

    let experiment_dir: PathBuf = ["experiments", model_name].iter().collect();

    fs::create_dir_all(&experiment_dir)?;

    let mut writer = Writer::from_path(experiment_dir.join("results.csv"))?;

    for (svd_scale, prune_range) in rewrite_combs {
        for (i, err) in errs.into_iter().enumerate() {
            let exp_name = format!(
                "{}::{}::{}::{}",
                model_name,
                i,
                svd_scale.map(|x| x.to_string()).unwrap_or("None".into()),
                prune_range
                    .map(|x| format!("[{}-{}]", x.0, x.1))
                    .unwrap_or("None".into()),
            );
            let experiment_dir = experiment_dir.join(&exp_name);
            fs::create_dir_all(&experiment_dir)?;
            let result = (0..3)
                .map(|_| {
                    log::info!("Running experiment {}", exp_name);
                    optimize(
                        exp_name.clone(),
                        &expr,
                        var_info.clone(),
                        &test_set_labels,
                        err,
                        RewriteParameters {
                            svd_step_scale: svd_scale,
                            prune_range,
                        },
                    )
                })
                .min_by_key(|x| x.metrics.optimized_cost)
                .unwrap();

            writer.serialize(result.metrics)?;

            output_pareto(experiment_dir.join("pareto.svg"), &result.pareto_points)?;

            fs::write(
                experiment_dir.join("pareto_points.json"),
                serde_json::to_string_pretty(&result.pareto_points)?,
            )?;

            output_python_file(
                result.root,
                result.optimized.as_ref(),
                &expr,
                &result.egraph,
                &result.var_info,
                &experiment_dir,
            )?;
        }
    }
    writer.flush()?;

    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();

    // optimize(
    //     "test",
    //     &expr,
    //     var_info.clone(),
    //     &test_set_labels,
    //     err,
    //     RewriteParameters {
    //         svd_step_scale: svd_scale,
    //         prune_range,
    //     },
    // )

    run_experiment(
        "lenet",
        "lenet_model_params.json",
        "lenet_test_set.json",
        (10000, 400),
        false,
    )?;

    Ok(())
}
