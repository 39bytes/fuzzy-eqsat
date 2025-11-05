use anyhow::Result;
use egg::{EGraph, Id, Language, RecExpr, Symbol};
use std::{
    collections::HashMap,
    fs::{self, File},
};

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::{
    analysis::{AnalysisData, LinalgAnalysis, VarInfo},
    lang::Linalg,
    math::make_truncated_svd,
};

#[derive(Deserialize, Debug)]
pub struct RawModel {
    layers: Vec<RawLayer>,
}

#[derive(Deserialize, Debug)]
pub struct RawLayer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    input_shape: (usize,),
    output_shape: (usize,),
    layer_type: String,
    activation: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ModelLayer {
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Model {
    pub layers: Vec<ModelLayer>,
}

#[derive(Deserialize, Debug)]
pub struct TestSet {
    pub mat: Vec<Vec<f64>>,
}

// relu(W0 * i + b0)
// |> relu(W1 * _ + b1)
// |> ...

pub fn load_model(filename: &str) -> Result<Model> {
    let contents = fs::read_to_string(filename)?;
    let model: RawModel = serde_json::from_str(&contents)?;

    Ok(Model {
        layers: model
            .layers
            .into_iter()
            .map(|l| ModelLayer {
                weights: Array2::from_shape_vec(
                    (l.output_shape.0, l.input_shape.0),
                    l.weights.into_iter().flatten().collect::<Vec<_>>(),
                )
                .unwrap(),
                biases: Array2::from_shape_vec((l.output_shape.0, 1), l.biases).unwrap(),
            })
            .collect(),
    })
}

pub fn load_test_set(filename: &str, shape: (usize, usize)) -> Result<Array2<f64>> {
    let contents = fs::read_to_string(filename)?;
    let mat: TestSet = serde_json::from_str(&contents)?;

    let flat = mat.mat.into_iter().flatten().collect::<Vec<_>>();

    let mut out = Array2::from_shape_vec(shape, flat)?;
    out.mapv_inplace(|x| x / 255.0);
    Ok(out.reversed_axes())
}

fn extract_params(
    expr: &RecExpr<Linalg>,
    info: &HashMap<Symbol, VarInfo>,
) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
    extract_params_rec(expr, expr.root(), info)
}

#[derive(Serialize)]
enum LayerWeight {
    Mat(Array2<f64>),
    Svd(Array2<f64>),
}

fn extract_params_rec(
    expr: &RecExpr<Linalg>,
    cur: Id,
    info: &HashMap<Symbol, VarInfo>,
) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
    let node = &expr[cur];
    println!("{:?}", cur);
    println!("{:?}", node);

    match node {
        Linalg::Softmax(a) | Linalg::Relu(a) => extract_params_rec(expr, *a, info),
        Linalg::SVDMul([a, b, k]) => {
            let svd = match &expr[*a] {
                Linalg::Mat(a) => &info[a].svd,
                _ => panic!("oops"),
            };
            let k = match &expr[*k] {
                Linalg::Num(k) => *k as usize,
                _ => panic!("oops"),
            };

            let (u_k, sigma_k, vt_k) = make_truncated_svd(&svd, k);
            let layer_weights = u_k.dot(&sigma_k.dot(&vt_k));

            let (mut weights, biases) = extract_params_rec(expr, *b, info);
            weights.push(layer_weights);

            (weights, biases)
        }
        Linalg::Add([a, b]) | Linalg::Mul([a, b]) => {
            let (mut ws1, mut bs1) = extract_params_rec(expr, *a, info);
            let (mut ws2, mut bs2) = extract_params_rec(expr, *b, info);
            ws1.append(&mut ws2);
            bs1.append(&mut bs2);
            (ws1, bs1)
        }
        Linalg::Num(_) => (vec![], vec![]),
        Linalg::Mat(sym) => match sym.to_string().chars().next().unwrap() {
            'w' => (vec![info[sym].value.to_owned()], vec![]),
            'b' => (vec![], vec![info[sym].value.to_owned()]),
            _ => (vec![], vec![]),
        },
    }
}

pub fn export_params(
    expr: &RecExpr<Linalg>,
    var_info: &HashMap<Symbol, VarInfo>,
    filename: &str,
) -> Result<()> {
    let (weights, biases) = extract_params(expr, var_info);

    let layers = weights
        .into_iter()
        .zip(biases)
        .map(|(w, b)| ModelLayer {
            weights: w,
            biases: b,
        })
        .collect::<Vec<_>>();

    let model = Model { layers };

    fs::write(filename, serde_json::to_string_pretty(&model)?)?;

    Ok(())
}
