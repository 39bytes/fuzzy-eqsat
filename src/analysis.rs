use anyhow::{Context as _, Error};
use egg::*;
use ndarray::{ArcArray1, ArcArray2, Array1};
use ndarray_linalg::{Norm, SVD};
use std::{collections::HashMap, fmt::Display, rc::Rc, str::FromStr};

use crate::{
    lang::Linalg,
    math::{relu, softmax},
};

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

#[derive(Clone, Debug)]
pub struct VarInfo {
    pub dim: MatrixDim,
    pub svd: (ArcArray2<f64>, ArcArray1<f64>, ArcArray2<f64>),
    pub value: ArcArray2<f64>,
}

#[derive(Default, Debug)]
pub struct LinalgAnalysis {
    pub var_info: Rc<HashMap<Symbol, VarInfo>>,
}

#[derive(Debug, Clone)]
pub struct TrueValue {
    pub val: ArcArray2<f64>,
    pub svd: (ArcArray2<f64>, ArcArray1<f64>, ArcArray2<f64>),
}

#[derive(Debug, Clone)]
pub enum AnalysisData {
    Num(i32),
    Mat {
        dim: MatrixDim,
        true_value: Option<TrueValue>,
    },
}

impl AnalysisData {
    pub fn get_inner_dim(&self) -> MatrixDim {
        match self {
            AnalysisData::Mat { dim, .. } => *dim,
            _ => panic!("Called get_inner_dim on non matrix analysis data"),
        }
    }

    pub fn get_inner_value(&self) -> Option<TrueValue> {
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
                    true_value: Some(TrueValue {
                        val: info.value.clone(),
                        svd: info.svd.clone(),
                    }),
                }
            }
            Linalg::Add([a, b]) => {
                let sum =
                    data(a).get_inner_value().unwrap().val + data(b).get_inner_value().unwrap().val;

                let (u, sigma, vt) = sum.svd(true, true).unwrap();
                AnalysisData::Mat {
                    dim: data(a).get_inner_dim(),
                    true_value: Some(TrueValue {
                        val: sum,
                        svd: (u.unwrap().into(), sigma.into(), vt.unwrap().into()),
                    }),
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
                    ) => {
                        let product = a.val.dot(&b.val);
                        let (u, sigma, vt) = product.svd(true, true).unwrap();
                        Some(TrueValue {
                            val: product.into(),
                            svd: (u.unwrap().into(), sigma.into(), vt.unwrap().into()),
                        })
                    }
                    _ => None,
                };

                AnalysisData::Mat {
                    dim: MatrixDim::new(r, c),
                    true_value,
                }
            }
            Linalg::SVDMul([a, b, _]) => {
                let r = data(a).get_inner_dim().rows();
                let c = data(b).get_inner_dim().cols();

                AnalysisData::Mat {
                    dim: MatrixDim::new(r, c),
                    true_value: None,
                }
            }
            Linalg::Relu(a) => {
                let val = relu(&data(a).get_inner_value().unwrap().val).to_shared();
                let (u, sigma, vt) = val.svd(true, true).unwrap();

                AnalysisData::Mat {
                    dim: data(a).get_inner_dim(),
                    true_value: Some(TrueValue {
                        val,
                        svd: (u.unwrap().into(), sigma.into(), vt.unwrap().into()),
                    }),
                }
            }
            Linalg::Softmax(a) => {
                let val = softmax(&data(a).get_inner_value().unwrap().val).to_shared();
                let (u, sigma, vt) = val.svd(true, true).unwrap();

                AnalysisData::Mat {
                    dim: data(a).get_inner_dim(),
                    true_value: Some(TrueValue {
                        val,
                        svd: (u.unwrap().into(), sigma.into(), vt.unwrap().into()),
                    }),
                }
            }
        }
    }
}
