use anyhow::{Error, Result, anyhow};
use egg::{EGraph, Id, Language, RecExpr, Symbol};
use std::{
    cell::RefCell,
    collections::HashMap,
    fs::{self},
    path::Path,
    rc::Rc,
};

use ndarray::{Array, Array1, Array2, s};
use serde::{Deserialize, Serialize};

use crate::{
    analysis::{LinalgAnalysis, MODEL_INPUT},
    extract,
    lang::Linalg,
    matrix::MatrixValue,
};

#[derive(Deserialize, Debug)]
struct RawLayer {
    weights: Vec<f64>,
    biases: Vec<f64>,
    input_shape: usize,
    output_shape: usize,
    activation: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Activation {
    Relu,
    Tanh,
    Softmax,
}

impl TryFrom<String> for Activation {
    type Error = Error;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        match value.as_str() {
            "relu" => Ok(Activation::Relu),
            "tanh" => Ok(Activation::Tanh),
            "softmax" => Ok(Activation::Softmax),
            s => Err(anyhow!("invalid activation function '{}'", s)),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ModelLayer {
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub activation: Activation,
}

#[derive(Deserialize, Debug)]
struct TestSetInternal {
    mat: Vec<Vec<f64>>,
    labels: Vec<usize>,
}

#[derive(Deserialize, Debug)]
pub struct TestSet {
    pub mat: Array2<f64>,
    pub labels: Array1<usize>,
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
            activation: l.activation.try_into().unwrap(),
        })
        .collect())
}

pub fn load_test_set(filename: &str, shape: (usize, usize), normalize: bool) -> Result<TestSet> {
    let contents = fs::read_to_string(filename)?;
    let test_set: TestSetInternal = serde_json::from_str(&contents)?;

    let flat = test_set.mat.into_iter().flatten().collect::<Vec<_>>();

    let mat: Array2<f64> = Array2::from_shape_vec(shape, flat)?;
    let labels: Array1<usize> = Array::from_vec(test_set.labels);

    let mat = if normalize {
        mat.map(|x| *x / 255.0)
    } else {
        mat
    }
    .reversed_axes();
    Ok(TestSet { mat, labels })
}

fn into_python(expr: &RecExpr<Linalg>) -> String {
    fn rec(expr: &RecExpr<Linalg>, cur: Id) -> String {
        match &expr[cur] {
            Linalg::Add([a, b]) => format!("np.add({},{})", rec(expr, *a), rec(expr, *b)),
            Linalg::Mul([a, b]) => format!("np.dot({},{})", rec(expr, *a), rec(expr, *b)),
            Linalg::DiagMul([a, b]) => {
                format!("np.multiply({}[:, None], {})", rec(expr, *b), rec(expr, *a))
            }
            Linalg::SvdU([_, _]) | Linalg::SvdD([_, _]) | Linalg::SvdVt([_, _]) => {
                format!(
                    "param(\'{}\')",
                    expr[cur].build_recexpr(|id| expr[id].clone())
                )
            }
            Linalg::Relu(a) => format!("relu({})", rec(expr, *a)),
            Linalg::Softmax(a) => format!("softmax({})", rec(expr, *a)),
            Linalg::Tanh(a) => format!("np.tanh({})", rec(expr, *a)),
            Linalg::Num(i) => i.to_string(),
            Linalg::Mat(sym) => {
                if sym.as_str() == "x" {
                    "x".into()
                } else {
                    format!("param(\'{}\')", sym)
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

fn make_recexpr(
    node: &Linalg,
    sol: &extract::Solution,
    egraph: &EGraph<Linalg, LinalgAnalysis>,
) -> RecExpr<Linalg> {
    node.build_recexpr(|id| egraph[id].nodes[sol[&id]].clone())
}

fn gather_params(
    id: Id,
    sol: &extract::Solution,
    egraph: &EGraph<Linalg, LinalgAnalysis>,
    var_info: &Rc<RefCell<HashMap<Symbol, MatrixValue>>>,
) -> HashMap<String, Parameter> {
    let node = &egraph[id].nodes[sol[&id]];
    match node {
        Linalg::Add([a, b]) | Linalg::Mul([a, b]) | Linalg::DiagMul([a, b]) => {
            let mut m1 = gather_params(*a, sol, egraph, var_info);
            m1.extend(gather_params(*b, sol, egraph, var_info));
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

            HashMap::from([(
                make_recexpr(node, sol, egraph).to_string(),
                Parameter {
                    shape: trunc.shape().to_vec(),
                    val: trunc.into_iter().collect(),
                },
            )])
        }
        Linalg::SvdD([a, k]) => {
            let a = egraph[*a].data.unwrap_mat();
            let k = egraph[*k].data.unwrap_num() as usize;

            let trunc = a
                .canonical_value
                .clone()
                .unwrap()
                .svd()
                .1
                .clone()
                .slice_move(s![..k]);

            HashMap::from([(
                make_recexpr(node, sol, egraph).to_string(),
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

            HashMap::from([(
                make_recexpr(node, sol, egraph).to_string(),
                Parameter {
                    shape: trunc.shape().to_vec(),
                    val: trunc.into_iter().collect(),
                },
            )])
        }
        Linalg::Relu(a) | Linalg::Softmax(a) | Linalg::Tanh(a) => {
            gather_params(*a, sol, egraph, var_info)
        }
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

pub fn export_params<T: AsRef<Path>>(
    root: Id,
    sol: Option<&extract::Solution>,
    egraph: &EGraph<Linalg, LinalgAnalysis>,
    var_info: &Rc<RefCell<HashMap<Symbol, MatrixValue>>>,
    filename: T,
) -> Result<()> {
    let params = match sol {
        None => var_info
            .borrow()
            .iter()
            .filter_map(|(sym, val)| {
                if sym.as_str() == MODEL_INPUT || sym.as_str().contains("pruned") {
                    None
                } else {
                    Some((
                        sym.to_string(),
                        Parameter {
                            shape: val.val().shape().to_vec(),
                            val: val.val().iter().copied().collect(),
                        },
                    ))
                }
            })
            .collect(),
        Some(sol) => gather_params(root, sol, egraph, var_info),
    };

    fs::write(
        filename,
        serde_json::to_string_pretty(&Parameters { params })?,
    )?;

    Ok(())
}

pub fn output_python_file<T: AsRef<Path>>(
    root: Id,
    sol: Option<&extract::Solution>,
    orig: &RecExpr<Linalg>,
    egraph: &EGraph<Linalg, LinalgAnalysis>,
    var_info: &Rc<RefCell<HashMap<Symbol, MatrixValue>>>,
    output_dir: T,
) -> Result<()> {
    let recexpr = match sol {
        Some(sol) => &make_recexpr(&egraph[root].nodes[sol[&root]], sol, egraph),
        None => orig,
    };

    let py_expr = into_python(recexpr);

    export_params(
        root,
        sol,
        egraph,
        var_info,
        output_dir.as_ref().join("parameters.json"),
    )?;
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

    fs::write(output_dir.as_ref().join("out.py"), out)?;

    Ok(())
}
