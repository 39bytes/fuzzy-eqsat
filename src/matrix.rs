use std::{cell::OnceCell, fmt::Display, str::FromStr};

use anyhow::{Context as _, Error};
use ndarray::{ArcArray1, ArcArray2};
use ndarray_linalg::{JobSvd, SVDDC as _};
use serde::Serialize;

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
