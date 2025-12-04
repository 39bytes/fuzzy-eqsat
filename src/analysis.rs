use egg::*;
use ndarray::{ArcArray2, Array2, s};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use crate::{
    lang::Linalg,
    math::{relu, softmax, tanh},
    matrix::{MatrixDim, MatrixValue},
};

pub const MODEL_INPUT: &str = "x";

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
    pub const_value: Option<MatrixValue>,
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
        let mut did_merge = DidMerge(false, false);
        if let (
            AnalysisData::Mat(MatrixData {
                canonical_value: canon1,
                const_value: const1,
                ..
            }),
            AnalysisData::Mat(MatrixData {
                canonical_value: canon2,
                const_value: const2,
                ..
            }),
        ) = (a, b)
        {
            if canon1.is_none() && canon2.is_some() {
                *canon1 = canon2;
                did_merge = DidMerge(true, false);
            }
            if const1.is_none() && const2.is_some() {
                *const1 = const2;
                did_merge = DidMerge(true, false);
            }
        }

        did_merge
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

                AnalysisData::Mat(MatrixData {
                    dim: val.dim(),
                    canonical_value: Some(val.clone()),
                    const_value: (a.as_str() != MODEL_INPUT).then(|| val.clone()),
                    diagonal: false,
                })
            }
            Linalg::Add([a, b]) => {
                let a = data(a).unwrap_mat();
                let b = data(b).unwrap_mat();
                let canonical_value = compute_canonical2(a, b, |a, b| a + b);

                AnalysisData::Mat(MatrixData {
                    dim: a.dim,
                    canonical_value,
                    const_value: None,
                    diagonal: a.diagonal && b.diagonal,
                })
            }
            Linalg::Mul([a, b]) => {
                let a = data(a).unwrap_mat();
                let b = data(b).unwrap_mat();

                let canonical_value = compute_canonical2(a, b, |a, b| a.dot(b));

                AnalysisData::Mat(MatrixData {
                    dim: MatrixDim::new(a.dim.rows(), b.dim.cols()),
                    canonical_value,
                    const_value: None,
                    diagonal: a.diagonal && b.diagonal,
                })
            }
            Linalg::SvdU([a, k]) => {
                let a = data(a).unwrap_mat();
                let k = data(k).unwrap_num() as usize;

                AnalysisData::Mat(MatrixData {
                    dim: MatrixDim::new(a.dim.rows(), k),
                    canonical_value: None,
                    // const_value: None,
                    const_value: a
                        .const_value
                        .as_ref()
                        .map(|a| MatrixValue::new(a.svd().0.clone().slice_move(s![.., ..k]))),
                    diagonal: false,
                })
            }
            Linalg::SvdD([a, k]) => {
                let a = data(a).unwrap_mat();
                let k = data(k).unwrap_num() as usize;

                AnalysisData::Mat(MatrixData {
                    dim: MatrixDim::new(k, k),
                    canonical_value: None,
                    // const_value: None,
                    const_value: a.const_value.as_ref().map(|a| {
                        MatrixValue::new(Array2::from_diag(&a.svd().1.slice(s![..k])).into())
                    }),
                    diagonal: true,
                })
            }
            Linalg::SvdVt([a, k]) => {
                let a = data(a).unwrap_mat();
                let k = data(k).unwrap_num() as usize;

                AnalysisData::Mat(MatrixData {
                    dim: MatrixDim::new(k, a.dim.cols()),
                    canonical_value: None,
                    // const_value: None,
                    const_value: a
                        .const_value
                        .as_ref()
                        .map(|a| MatrixValue::new(a.svd().2.clone().slice_move(s![..k, ..]))),
                    diagonal: false,
                })
            }
            Linalg::Relu(a) => {
                let a = data(a).unwrap_mat();
                let canonical_value = compute_canonical1(a, relu);

                AnalysisData::Mat(MatrixData {
                    dim: a.dim,
                    canonical_value,
                    const_value: None,
                    diagonal: a.diagonal,
                })
            }
            Linalg::Softmax(a) => {
                let a = data(a).unwrap_mat();
                let canonical_value = compute_canonical1(a, softmax);

                AnalysisData::Mat(MatrixData {
                    dim: a.dim,
                    canonical_value,
                    const_value: None,
                    diagonal: false,
                })
            }
            Linalg::Tanh(a) => {
                let a = data(a).unwrap_mat();
                let canonical_value = compute_canonical1(a, tanh);

                AnalysisData::Mat(MatrixData {
                    dim: a.dim,
                    canonical_value,
                    const_value: None,
                    diagonal: false,
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
    a.canonical_value
        .as_ref()
        .zip(b.canonical_value.as_ref())
        .map(|(a, b)| {
            let res = op(a.val(), b.val());
            MatrixValue::new(res.into())
        })
}

fn compute_canonical1(
    a: &MatrixData,
    op: impl Fn(&ArcArray2<f64>) -> Array2<f64>,
) -> Option<MatrixValue> {
    a.canonical_value.as_ref().map(|a| {
        let res = op(a.val());
        MatrixValue::new(res.into())
    })
}
