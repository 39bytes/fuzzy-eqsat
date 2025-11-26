use anyhow::Result;
use egg::{EGraph, Id, Language, RecExpr, Symbol};
use std::{
    cell::RefCell,
    collections::HashMap,
    fs::{self},
    rc::Rc,
};

use ndarray::{Array2, s};
use serde::{Deserialize, Serialize};

use crate::{
    analysis::LinalgAnalysis, cost::LinalgCost, extract::CandidateExpr, lang::Linalg,
    matrix::MatrixValue,
};

#[derive(Deserialize, Debug)]
pub struct RawLayer {
    weights: Vec<f64>,
    biases: Vec<f64>,
    input_shape: usize,
    output_shape: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ModelLayer {
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
}

#[derive(Deserialize, Debug)]
pub struct TestSet {
    pub mat: Vec<Vec<f64>>,
}

pub fn load_model(filename: &str) -> Result<Vec<ModelLayer>> {
    let contents = fs::read_to_string(filename)?;
    let layers: Vec<RawLayer> = serde_json::from_str(&contents)?;

    Ok(layers
        .into_iter()
        .map(|l| ModelLayer {
            weights: Array2::from_shape_vec((l.input_shape, l.output_shape), l.weights)
                .unwrap()
                .reversed_axes(),
            biases: Array2::from_shape_vec((l.output_shape, 1), l.biases).unwrap(),
        })
        .collect())
}

pub fn load_test_set(filename: &str, shape: (usize, usize)) -> Result<Array2<f64>> {
    let contents = fs::read_to_string(filename)?;
    let mat: TestSet = serde_json::from_str(&contents)?;

    let flat = mat.mat.into_iter().flatten().collect::<Vec<_>>();

    let out: Array2<f64> = Array2::from_shape_vec(shape, flat)?;
    Ok(out.map(|x| *x / 255.0).reversed_axes())
}

fn into_python(expr: &RecExpr<Linalg>) -> String {
    fn rec(expr: &RecExpr<Linalg>, cur: Id) -> String {
        match &expr[cur] {
            Linalg::Add([a, b]) => format!("{} + {}", rec(expr, *a), rec(expr, *b)),
            Linalg::Mul([a, b]) => format!("{} @ {}", rec(expr, *a), rec(expr, *b)),
            Linalg::SvdU([_, _]) | Linalg::SvdD([_, _]) | Linalg::SvdVt([_, _]) => {
                format!(
                    "param(\"{}\")",
                    expr[cur].build_recexpr(|id| expr[id].clone())
                )
            }
            Linalg::Relu(a) => format!("relu({})", rec(expr, *a)),
            Linalg::Softmax(a) => format!("softmax({})", rec(expr, *a)),
            Linalg::Num(i) => i.to_string(),
            Linalg::Mat(sym) => {
                if sym.as_str() == "x" {
                    "x".into()
                } else {
                    format!("param(\"{}\")", sym)
                }
            }
        }
    }

    rec(expr, expr.root())
}

#[derive(Serialize)]
struct Parameters {
    params: HashMap<String, Parameter>,
}

#[derive(Serialize)]
struct Parameter {
    shape: Vec<usize>,
    val: Vec<f64>,
}

pub fn export_params(
    expr: &CandidateExpr<LinalgCost, Linalg>,
    egraph: &EGraph<Linalg, LinalgAnalysis>,
    var_info: &Rc<RefCell<HashMap<Symbol, MatrixValue>>>,
    filename: &str,
) -> Result<()> {
    fn rec(
        node: &Linalg,
        expr: &CandidateExpr<LinalgCost, Linalg>,
        egraph: &EGraph<Linalg, LinalgAnalysis>,
        var_info: &Rc<RefCell<HashMap<Symbol, MatrixValue>>>,
    ) -> HashMap<String, Parameter> {
        match node {
            Linalg::Add([a, b]) | Linalg::Mul([a, b]) => {
                let mut m1 = rec(&expr.children[a], expr, egraph, var_info);
                m1.extend(rec(&expr.children[b], expr, egraph, var_info));
                m1
            }
            Linalg::SvdU([a, k]) => {
                let a = egraph[*a].data.unwrap_mat();
                let k = egraph[*k].data.unwrap_num() as usize;

                let trunc = a
                    .canonical_value
                    .as_ref()
                    .unwrap()
                    .svd()
                    .0
                    .clone()
                    .slice_move(s![.., ..k])
                    .to_owned();

                let e = node.build_recexpr(|id| expr.children[&id].clone());

                HashMap::from([(
                    e.to_string(),
                    Parameter {
                        shape: trunc.shape().to_vec(),
                        val: trunc.into_iter().collect(),
                    },
                )])
            }
            Linalg::SvdD([a, k]) => {
                let a = egraph[*a].data.unwrap_mat();
                let k = egraph[*k].data.unwrap_num() as usize;

                let trunc =
                    Array2::from_diag(&a.canonical_value.clone().unwrap().svd().1.slice(s![..k]));

                let e = node.build_recexpr(|id| expr.children[&id].clone());

                HashMap::from([(
                    e.to_string(),
                    Parameter {
                        shape: trunc.shape().to_vec(),
                        val: trunc.into_iter().collect(),
                    },
                )])
            }
            Linalg::SvdVt([a, k]) => {
                let a = egraph[*a].data.unwrap_mat();
                let k = egraph[*k].data.unwrap_num() as usize;

                let trunc = a
                    .canonical_value
                    .as_ref()
                    .unwrap()
                    .svd()
                    .2
                    .clone()
                    .slice_move(s![..k, ..])
                    .to_owned();

                let e = node.build_recexpr(|id| expr.children[&id].clone());

                HashMap::from([(
                    e.to_string(),
                    Parameter {
                        shape: trunc.shape().to_vec(),
                        val: trunc.into_iter().collect(),
                    },
                )])
            }
            Linalg::Relu(a) => rec(&expr.children[a], expr, egraph, var_info),
            Linalg::Softmax(a) => rec(&expr.children[a], expr, egraph, var_info),
            Linalg::Num(_) => HashMap::new(),
            Linalg::Mat(sym) => {
                if sym.as_str() != "x" {
                    let val = var_info.borrow()[sym].val().to_owned();

                    HashMap::from([(
                        sym.to_string(),
                        Parameter {
                            shape: val.shape().to_vec(),
                            val: val.into_iter().collect(),
                        },
                    )])
                } else {
                    HashMap::default()
                }
            }
        }
    }

    let params = Parameters {
        params: rec(&expr.node, expr, egraph, var_info),
    };

    fs::write(filename, serde_json::to_string_pretty(&params)?)?;

    Ok(())
}

pub fn output_python_file(
    best: &CandidateExpr<LinalgCost, Linalg>,
    expr: &RecExpr<Linalg>,
    egraph: &EGraph<Linalg, LinalgAnalysis>,
    var_info: &Rc<RefCell<HashMap<Symbol, MatrixValue>>>,
    output_path: &str,
) -> Result<()> {
    let py_expr = into_python(expr);

    export_params(best, egraph, var_info, "parameters.json")?;
    let prelude = include_str!("template/prelude.py");

    let out = format!(
        "{prelude}


def model(x: np.ndarray):
    return {py_expr}

def main():
    x_test, y_test = load_dataset()
    
    predictions = model(x_test.T).argmax(axis=0)
    labels = y_test.argmax(axis=1)
    
    print(f\"Accuracy: {{np.mean(predictions == labels)}}\")

if __name__ == \"__main__\":
    main()
"
    );

    fs::write(output_path, out)?;

    Ok(())
}
