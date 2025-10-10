use anyhow::Result;
use std::fs;

use ndarray::{Array1, Array2};
use serde::Deserialize;

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

#[derive(Deserialize, Debug)]
pub struct ModelLayer {
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub input_shape: (usize,),
    pub output_shape: (usize,),
    pub layer_type: String,
    pub activation: String,
}

#[derive(Deserialize, Debug)]
pub struct Model {
    pub layers: Vec<ModelLayer>,
}

#[derive(Deserialize, Debug)]
pub struct InVec {
    pub vec: Vec<f64>,
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
                    (l.input_shape.0, l.output_shape.0),
                    l.weights.into_iter().flatten().collect::<Vec<_>>(),
                )
                .unwrap(),
                biases: Array2::from_shape_vec((l.output_shape.0, 1), l.biases).unwrap(),
                input_shape: l.input_shape,
                output_shape: l.output_shape,
                layer_type: l.layer_type,
                activation: l.activation,
            })
            .collect(),
    })
}

pub fn load_in_vec(filename: &str) -> Result<Array2<f64>> {
    let contents = fs::read_to_string(filename)?;
    let vec: InVec = serde_json::from_str(&contents)?;

    Ok(Array2::from_shape_vec((784, 1), vec.vec)?)
}
