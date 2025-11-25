use anyhow::{Context as _, Error};
use egg::*;
use ndarray::{ArcArray1, ArcArray2, Array2};
use ndarray_linalg::{JobSvd, SVDDC as _};
use serde::Serialize;
use std::{
    cell::{OnceCell, RefCell},
    collections::{HashMap, HashSet},
    fmt::Display,
    rc::Rc,
    str::FromStr,
};

use crate::{
    lang::Linalg,
    math::{relu, softmax},
};

pub const MODEL_INPUT: &str = "x";

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone, Copy, Default, Serialize)]
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

type SVD = (ArcArray2<f64>, ArcArray1<f64>, ArcArray2<f64>);

#[derive(Debug, Clone)]
pub struct MatrixValue {
    svd: OnceCell<SVD>,
    val: ArcArray2<f64>,
}

impl MatrixValue {
    pub fn new(val: ArcArray2<f64>) -> Self {
        MatrixValue {
            val: val.clone(),
            svd: OnceCell::new(),
        }
    }

    pub fn svd(&self) -> &SVD {
        self.svd.get_or_init(|| {
            let (u, sigma, vt) = self.val.svddc(JobSvd::Some).unwrap();
            (u.unwrap().into(), sigma.into(), vt.unwrap().into())
        })
    }

    pub fn val(&self) -> &ArcArray2<f64> {
        &self.val
    }

    pub fn dim(&self) -> MatrixDim {
        let (rows, cols) = self.val.dim();
        MatrixDim::new(rows, cols)
    }
}

#[derive(Default, Debug)]
pub struct LinalgAnalysis {
    pub var_info: Rc<RefCell<HashMap<Symbol, MatrixValue>>>,
    pub prune_count: usize,
    pub pruned_eclasses: HashSet<Id>,
}

#[derive(Debug, Clone)]
pub struct MatrixData {
    pub dim: MatrixDim,
    pub canonical_value: Option<MatrixValue>,
    pub is_const: bool,
    pub diagonal: bool,
    pub zeroes: usize,
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
                    canonical_value: va @ None,
                    ..
                }),
                AnalysisData::Mat(MatrixData {
                    canonical_value: Some(val),
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
                let var_info = egraph.analysis.var_info.borrow();
                let val = var_info
                    .get(a)
                    .unwrap_or_else(|| panic!("Unknown symbol {}", a));
                let is_const = a.as_str() != MODEL_INPUT;

                AnalysisData::Mat(MatrixData {
                    dim: val.dim(),
                    canonical_value: Some(val.clone()),
                    is_const,
                    diagonal: false,
                    zeroes: 0,
                })
            }
            Linalg::Add([a, b]) => {
                let a = data(a).unwrap_mat();
                let b = data(b).unwrap_mat();
                let canonical_value = compute_canonical2(a, b, |a, b| a + b);

                AnalysisData::Mat(MatrixData {
                    dim: a.dim,
                    canonical_value,
                    is_const: false,
                    diagonal: a.diagonal && b.diagonal,
                    zeroes: 0,
                })
            }
            Linalg::Mul([a, b]) => {
                let a = data(a).unwrap_mat();
                let b = data(b).unwrap_mat();

                let canonical_value = compute_canonical2(a, b, |a, b| a.dot(b));

                AnalysisData::Mat(MatrixData {
                    dim: MatrixDim::new(a.dim.rows(), b.dim.cols()),
                    canonical_value,
                    is_const: false,
                    diagonal: a.diagonal && b.diagonal,
                    zeroes: 0,
                })
            }
            Linalg::SvdU([a, k]) => {
                let a = data(a).unwrap_mat();
                let k = data(k).unwrap_num() as usize;

                AnalysisData::Mat(MatrixData {
                    dim: MatrixDim::new(a.dim.rows(), k),
                    canonical_value: None,
                    is_const: a.is_const,
                    diagonal: false,
                    zeroes: 0,
                })
            }
            Linalg::SvdD([a, k]) => {
                let a = data(a).unwrap_mat();
                let k = data(k).unwrap_num() as usize;

                AnalysisData::Mat(MatrixData {
                    dim: MatrixDim::new(k, k),
                    canonical_value: None,
                    is_const: a.is_const,
                    diagonal: true,
                    zeroes: 0,
                })
            }
            Linalg::SvdVt([a, k]) => {
                let a = data(a).unwrap_mat();
                let k = data(k).unwrap_num() as usize;

                AnalysisData::Mat(MatrixData {
                    dim: MatrixDim::new(k, a.dim.cols()),
                    canonical_value: None,
                    is_const: a.is_const,
                    diagonal: false,
                    zeroes: 0,
                })
            }
            Linalg::Pruned([a, k]) => {
                let a = data(a).unwrap_mat();
                let k = data(k).unwrap_num();

                // TODO:

                AnalysisData::Mat(MatrixData {
                    dim: a.dim,
                    canonical_value: None,
                    is_const: false,
                    diagonal: false,
                    zeroes: 0,
                })
            }
            Linalg::Relu(a) => {
                let a = data(a).unwrap_mat();
                let canonical_value = compute_canonical1(a, relu);

                AnalysisData::Mat(MatrixData {
                    dim: a.dim,
                    canonical_value,
                    is_const: false,
                    diagonal: a.diagonal,
                    zeroes: 0,
                })
            }
            Linalg::Softmax(a) => {
                let a = data(a).unwrap_mat();
                let canonical_value = compute_canonical1(a, softmax);

                AnalysisData::Mat(MatrixData {
                    dim: a.dim,
                    canonical_value,
                    is_const: false,
                    diagonal: false,
                    zeroes: 0,
                })
            }
        }
    }
}

fn compute_canonical2(
    a: &MatrixData,
    b: &MatrixData,
    op: impl Fn(&ArcArray2<f64>, &ArcArray2<f64>) -> Array2<f64>,
) -> Option<MatrixValue> {
    let canonical_value = a
        .canonical_value
        .as_ref()
        .zip(b.canonical_value.as_ref())
        .map(|(a, b)| {
            let res = op(&a.val, &b.val);
            MatrixValue::new(res.into())
        });

    canonical_value
}

fn compute_canonical1(
    a: &MatrixData,
    op: impl Fn(&ArcArray2<f64>) -> Array2<f64>,
) -> Option<MatrixValue> {
    let canonical_value = a.canonical_value.as_ref().map(|a| {
        let res = op(&a.val);
        MatrixValue::new(res.into())
    });

    canonical_value
}

fn count_zeroes(mat: &ArcArray2<f64>) -> usize {
    mat.iter().filter(|x| **x == 0.0).count()
}
