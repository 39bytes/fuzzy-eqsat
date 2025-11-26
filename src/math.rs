use ndarray::{ArcArray2, Array2};

pub fn relu(a: &ArcArray2<f64>) -> Array2<f64> {
    a.map(|x| f64::max(*x, 0.0))
}

pub fn softmax(input: &ArcArray2<f64>) -> Array2<f64> {
    let (rows, cols) = input.dim();
    let mut output = Array2::<f64>::zeros((rows, cols));

    // Apply softmax to each column
    for col_idx in 0..cols {
        let col = input.column(col_idx);

        // Find max value for numerical stability
        let max_val = col.fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp(x - max) for each element
        let exp_vals: Vec<f64> = col.iter().map(|&x| (x - max_val).exp()).collect();

        // Compute sum of exponentials
        let sum_exp: f64 = exp_vals.iter().sum();

        // Normalize and store in output
        for (row_idx, &exp_val) in exp_vals.iter().enumerate() {
            output[[row_idx, col_idx]] = exp_val / sum_exp;
        }
    }

    output
}

pub fn prune(mat: &ArcArray2<f64>, precision: i32) -> ArcArray2<f64> {
    let threshold = 10.0f64.powf(precision as f64);
    mat.map(|x| if x.abs() < threshold { 0.0 } else { *x })
        .into()
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_softmax() {
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let result = softmax(&input.into());

        // Check that each column sums to approximately 1.0
        for col_idx in 0..3 {
            let col_sum: f64 = result.column(col_idx).sum();
            assert!((col_sum - 1.0).abs() < 1e-10);
        }

        // Check that all values are positive
        assert!(result.iter().all(|&x| x > 0.0));
    }
}
