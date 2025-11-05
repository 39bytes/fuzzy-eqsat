use anyhow::{Context as _, Error};
use egg::*;
use ndarray::{ArcArray1, ArcArray2};
use ndarray_linalg::SVD as _;
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
pub struct MatrixData {
    pub dim: MatrixDim,
    pub true_value: Option<TrueValue>,
    pub diagonal: bool,
}

#[derive(Debug, Clone)]
pub enum AnalysisData {
    Num(i32),
    Mat(MatrixData),
}

impl AnalysisData {
    pub fn unwrap_mat(&self) -> &MatrixData {
        match self {
            AnalysisData::Mat(data) => data,
            _ => panic!("Called unwrap_mat on non matrix analysis data"),
        }
    }

    pub fn unwrap_num(&self) -> i32 {
        match self {
            AnalysisData::Num(x) => *x,
            _ => panic!("Called unwrap_num on non num analysis data"),
        }
    }
}

impl Analysis<Linalg> for LinalgAnalysis {
    type Data = AnalysisData;

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        match (a, b) {
            (
                AnalysisData::Mat(MatrixData {
                    true_value: va @ None,
                    ..
                }),
                AnalysisData::Mat(MatrixData {
                    true_value: Some(val),
                    ..
                }),
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
                AnalysisData::Mat(MatrixData {
                    dim: info.dim,
                    true_value: Some(TrueValue {
                        val: info.value.clone(),
                        svd: info.svd.clone(),
                    }),
                    diagonal: false,
                })
            }
            Linalg::Add([a, b]) => {
                let a = data(a).unwrap_mat();
                let b = data(b).unwrap_mat();

                let true_value = a
                    .true_value
                    .as_ref()
                    .zip(b.true_value.as_ref())
                    .map(|(a, b)| {
                        let sum = &a.val + &b.val;
                        let (u, sigma, vt) = sum.svd(true, true).unwrap();

                        TrueValue {
                            val: sum.into(),
                            svd: (u.unwrap().into(), sigma.into(), vt.unwrap().into()),
                        }
                    });

                AnalysisData::Mat(MatrixData {
                    dim: a.dim,
                    true_value,
                    diagonal: a.diagonal && b.diagonal,
                })
            }
            Linalg::Mul([a, b]) => {
                let a = data(a).unwrap_mat();
                let b = data(b).unwrap_mat();

                let true_value = a
                    .true_value
                    .as_ref()
                    .zip(b.true_value.as_ref())
                    .map(|(a, b)| {
                        let prod = a.val.dot(&b.val);
                        let (u, sigma, vt) = prod.svd(true, true).unwrap();

                        TrueValue {
                            val: prod.into(),
                            svd: (u.unwrap().into(), sigma.into(), vt.unwrap().into()),
                        }
                    });

                AnalysisData::Mat(MatrixData {
                    dim: MatrixDim::new(a.dim.rows(), b.dim.cols()),
                    true_value,
                    diagonal: a.diagonal && b.diagonal,
                })
            }
            Linalg::SvdU([a, k]) => {
                let r = data(a).unwrap_mat().dim.rows();
                let k = data(k).unwrap_num() as usize;

                AnalysisData::Mat(MatrixData {
                    dim: MatrixDim::new(r, k),
                    true_value: None,
                    diagonal: false,
                })
            }
            Linalg::SvdD([_, k]) => {
                let k = data(k).unwrap_num() as usize;

                AnalysisData::Mat(MatrixData {
                    dim: MatrixDim::new(k, k),
                    true_value: None,
                    diagonal: true,
                })
            }
            Linalg::SvdVt([a, k]) => {
                let c = data(a).unwrap_mat().dim.cols();
                let k = data(k).unwrap_num() as usize;

                AnalysisData::Mat(MatrixData {
                    dim: MatrixDim::new(k, c),
                    true_value: None,
                    diagonal: false,
                })
            }
            Linalg::Relu(a) => {
                let a = data(a).unwrap_mat();

                let true_value = a.true_value.as_ref().map(|a| {
                    let res = relu(&a.val);
                    let (u, sigma, vt) = res.svd(true, true).unwrap();
                    TrueValue {
                        val: res.into(),
                        svd: (u.unwrap().into(), sigma.into(), vt.unwrap().into()),
                    }
                });

                AnalysisData::Mat(MatrixData {
                    dim: a.dim,
                    true_value,
                    diagonal: a.diagonal,
                })
            }
            Linalg::Softmax(a) => {
                let a = data(a).unwrap_mat();

                let true_value = a.true_value.as_ref().map(|a| {
                    let res = softmax(&a.val);
                    let (u, sigma, vt) = res.svd(true, true).unwrap();
                    TrueValue {
                        val: res.into(),
                        svd: (u.unwrap().into(), sigma.into(), vt.unwrap().into()),
                    }
                });

                AnalysisData::Mat(MatrixData {
                    dim: a.dim,
                    true_value,
                    diagonal: false,
                })
            }
        }
    }
}
