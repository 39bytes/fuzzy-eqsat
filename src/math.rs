use ndarray::{ArcArray1, ArcArray2, Array2, ArrayView2, s};

pub fn relu(a: &ArcArray2<f64>) -> Array2<f64> {
    a.map(|x| f64::max(*x, 0.0))
}

pub fn softmax(a: &ArcArray2<f64>) -> Array2<f64> {
    let max = a.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e_x = (a - max).exp();
    let sum = e_x.sum();
    e_x.map(|x| x / sum)
}

pub fn make_truncated_svd<'a>(
    (u, sigma, vt): &'a (ArcArray2<f64>, ArcArray1<f64>, ArcArray2<f64>),
    k: usize,
) -> (ArrayView2<'a, f64>, Array2<f64>, ArrayView2<'a, f64>) {
    let u_k = u.slice(s![.., ..k]);
    let sigma_k = Array2::from_diag(&sigma.slice(s![..k]));
    let vt_k = vt.slice(s![..k, ..]);

    (u_k, sigma_k, vt_k)
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::array;

    #[test]
    pub fn test_softmax() {
        let a = array![
            [0.0],
            [0.0],
            [2.0],
            [3.0],
            [4.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0]
        ]
        .to_shared();

        let res = softmax(&a);

        assert!((res.sum() - 1.0).abs() < 0.000001);
    }
}
