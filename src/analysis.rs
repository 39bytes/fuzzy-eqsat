use anyhow::{Context as _, Error};
use egg::*;
use ndarray::{ArcArray1, ArcArray2, Array1};
use std::{collections::HashMap, fmt::Display, str::FromStr};

use crate::lang::Linalg;

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone, Copy, Default)]
pub struct MatrixDim(usize, usize);

impl MatrixDim {
    pub fn new(rows: usize, cols: usize) -> Self {
        MatrixDim(rows, cols)
    }

    pub fn rows(&self) -> usize {
        self.0
    }

    pub fn cols(&self) -> usize {
        self.1
    }

    pub fn max_rank(&self) -> usize {
        self.rows().min(self.cols())
    }

    pub fn size(&self) -> usize {
        self.rows() * self.cols()
    }
}

impl Display for MatrixDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.rows(), self.cols())
    }
}

impl FromStr for MatrixDim {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (rows, cols) = s
            .split_once("x")
            .context("Failed to parse matrix dimensions")?;

        Ok(MatrixDim(rows.parse::<usize>()?, cols.parse::<usize>()?))
    }
}

pub struct VarInfo {
    pub dim: MatrixDim,
    pub singular_values: Option<ArcArray1<f64>>,
    pub value: Option<ArcArray2<f64>>,
}

#[derive(Default)]
pub struct LinalgAnalysis {
    pub var_info: HashMap<Symbol, VarInfo>,
}

#[derive(Debug, Clone)]
pub struct MatrixMetadata {
    pub dim: MatrixDim,
    pub singular_values: Option<Array1<f64>>,
}

#[derive(Debug, Clone)]
pub enum AnalysisData {
    Num(i32),
    Mat {
        dim: MatrixDim,
        singular_values: Option<ArcArray1<f64>>,
        true_value: Option<ArcArray2<f64>>,
    },
}

impl AnalysisData {
    pub fn get_inner_dim(&self) -> MatrixDim {
        match self {
            AnalysisData::Mat { dim, .. } => *dim,
            _ => panic!("Called get_inner_dim on non matrix analysis data"),
        }
    }

    pub fn get_inner_value(&self) -> Option<ArcArray2<f64>> {
        match self {
            AnalysisData::Mat { true_value, .. } => true_value.clone(),
            _ => panic!("Called get_inner_dim on non matrix analysis data"),
        }
    }
}

impl Analysis<Linalg> for LinalgAnalysis {
    type Data = AnalysisData;

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        match (a, b) {
            (
                AnalysisData::Mat {
                    true_value: va @ None,
                    ..
                },
                AnalysisData::Mat {
                    true_value: Some(val),
                    ..
                },
            ) => {
                *va = Some(val);
                DidMerge(true, false)
            }
            _ => DidMerge(false, false),
        }
    }

    fn make(egraph: &mut EGraph<Linalg, Self>, enode: &Linalg) -> Self::Data {
        let data = |i: &Id| &egraph[*i].data;
        match enode {
            Linalg::Num(val) => AnalysisData::Num(*val),
            Linalg::Mat(a) => {
                let info = egraph.analysis.var_info.get(a).expect("Unknown symbol");
                AnalysisData::Mat {
                    dim: info.dim,
                    singular_values: info.singular_values.clone(),
                    true_value: info.value.clone(),
                }
            }
            Linalg::Add([a, b]) => {
                let sum = data(a).get_inner_value().unwrap() + data(b).get_inner_value().unwrap();
                AnalysisData::Mat {
                    dim: data(a).get_inner_dim(),
                    singular_values: None,
                    true_value: Some(sum),
                }
            }
            Linalg::Mul([a, b]) => {
                let r = data(a).get_inner_dim().rows();
                let c = data(b).get_inner_dim().cols();

                let true_value = match (data(a), data(b)) {
                    (
                        AnalysisData::Mat {
                            true_value: Some(a),
                            ..
                        },
                        AnalysisData::Mat {
                            true_value: Some(b),
                            ..
                        },
                    ) => Some((a * b).to_shared()),
                    _ => None,
                };

                AnalysisData::Mat {
                    dim: MatrixDim::new(r, c),
                    singular_values: None,
                    true_value,
                }
            }
            Linalg::SvdU([a, k]) => {
                let dim = match (data(a), data(k)) {
                    (AnalysisData::Mat { dim, .. }, AnalysisData::Num(trunc)) => {
                        MatrixDim::new(dim.rows(), *trunc as usize)
                    }
                    _ => panic!("Invalid svd args"),
                };

                AnalysisData::Mat {
                    dim,
                    singular_values: None,
                    true_value: None,
                }
            }
            Linalg::SvdD([a, k]) => {
                let dim = match (data(a), data(k)) {
                    (AnalysisData::Mat { .. }, AnalysisData::Num(trunc)) => {
                        let r = *trunc as usize;
                        MatrixDim::new(r, r)
                    }
                    _ => panic!("Invalid svd args"),
                };

                AnalysisData::Mat {
                    dim,
                    singular_values: None,
                    true_value: None,
                }
            }
            Linalg::SvdVt([a, k]) => {
                let dim = match (data(a), data(k)) {
                    (AnalysisData::Mat { dim, .. }, AnalysisData::Num(trunc)) => {
                        let r = *trunc as usize;
                        MatrixDim::new(r, dim.cols())
                    }
                    _ => panic!("Invalid svd args"),
                };

                AnalysisData::Mat {
                    dim,
                    singular_values: None,
                    true_value: None,
                }
            }
            Linalg::Relu(a) => AnalysisData::Mat {
                dim: data(a).get_inner_dim(),
                singular_values: None,
                true_value: Some(
                    data(a)
                        .get_inner_value()
                        .unwrap()
                        .map(|x| x.max(0.0))
                        .to_shared(),
                ),
            },
        }
    }
}
