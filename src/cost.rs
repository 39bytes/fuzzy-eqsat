use std::{cmp::Ordering, collections::HashMap, fmt::Display, rc::Rc};

use egg::*;
use ndarray::{ArcArray2, Array2, s};
use ndarray_linalg::Norm;

use crate::{
    analysis::{AnalysisData, LinalgAnalysis, VarInfo},
    lang::Linalg,
};

#[derive(Debug)]
pub struct LinalgCost<'a> {
    pub egraph: &'a EGraph<Linalg, LinalgAnalysis>,
    pub var_info: Rc<HashMap<Symbol, VarInfo>>,
    pub max_rel_error: f64,
}

#[derive(PartialEq, Debug, Clone, Default)]
pub struct CostWithErrorBound {
    cost: usize,
    error: f64,
    max_rel_error: f64,
    val: ArcArray2<f64>,
}

impl Display for CostWithErrorBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cost: {}, error: {}", self.cost, self.error)
    }
}

impl PartialOrd for CostWithErrorBound {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let a = if self.error >= self.max_rel_error {
            usize::MAX
        } else {
            self.cost
        };
        let b = if other.error >= other.max_rel_error {
            usize::MAX
        } else {
            other.cost
        };
        a.partial_cmp(&b)
    }
}

impl<'a> LinalgCost<'a> {
    fn get_expected_spectral_norm(&self, enode: &Linalg) -> f64 {
        let id = self
            .egraph
            .lookup(enode.clone())
            .expect("enode should be in egraph");
        self.egraph[id]
            .data
            .get_inner_value()
            .expect("true value should exist in eclass")
            .svd
            .1[0]
    }

    fn relative_frobenius_norm_error(&self, enode: &Linalg, approx: &ArcArray2<f64>) -> f64 {
        let id = self
            .egraph
            .lookup(enode.clone())
            .expect("enode should be in egraph");
        let true_val = self.egraph[id]
            .data
            .get_inner_value()
            .expect("true value should exist in eclass")
            .val;

        let num = (&true_val - approx).norm_l2();
        let denom = true_val.norm_l2();

        println!("approx L2 ({}): {}", id, approx.norm_l2());
        println!("true value L2 ({}): {}", id, denom);

        let ret = (&true_val - approx).norm_l2() / true_val.norm_l2();
        // println!(
        //     "({}) err: {} ({}/{}) ({:?}, {:?})",
        //     id,
        //     ret,
        //     num,
        //     denom,
        //     true_val.dim(),
        //     approx.dim()
        // );
        ret
    }

    fn fold_costs(
        &self,
        enode: &Linalg,
        res: ArcArray2<f64>,
        op_cost: usize,
        child_costs: &[CostWithErrorBound],
    ) -> CostWithErrorBound {
        let total_cost = child_costs
            .iter()
            .fold(op_cost, |c, cost| c.saturating_add(cost.cost));

        let error = self.relative_frobenius_norm_error(enode, &res);

        CostWithErrorBound {
            cost: total_cost,
            max_rel_error: self.max_rel_error,
            error,
            val: res,
        }
    }
}

impl<'a> CostFunction<Linalg> for LinalgCost<'a> {
    type Cost = CostWithErrorBound;

    fn cost<C>(&mut self, enode: &Linalg, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let data = |i: &Id| self.egraph[*i].data.clone();
        match enode {
            Linalg::Mat(a) => CostWithErrorBound {
                cost: 0,
                error: 0.0,
                max_rel_error: self.max_rel_error,
                val: self.var_info[a].value.clone(),
            },
            Linalg::Num(_) => CostWithErrorBound::default(),
            Linalg::SVDMul([a, b, k]) => {
                let a_cost = costs(*a);
                let b_cost = costs(*b);
                let a_dim = data(a).get_inner_dim();
                let b_dim = data(b).get_inner_dim();
                let k = match data(k) {
                    AnalysisData::Num(x) => x,
                    _ => panic!("oops"),
                } as usize;

                // n x m * m x l
                // => n x k * k x k * k x m * m x l
                let op_cost = k * b_dim.rows() * b_dim.cols()
                    + k * b_dim.cols()
                    + a_dim.rows() * k * b_dim.cols();

                let (u, sigma, vt) = data(a).get_inner_value().unwrap().svd;

                let u_k = u.slice(s![.., ..k]);
                let sigma_k = Array2::from_diag(&sigma.slice(s![..k]));
                let vt_k = vt.slice(s![..k, ..]);

                let vtb = vt_k.dot(&b_cost.val);
                // println!("vtb: {:?}", vtb.dim());
                let scaled = sigma_k.dot(&vtb);
                // println!("scaled: {:?}", scaled.dim());
                let res = u_k.dot(&scaled).to_shared();
                // println!("res: {:?}", res.dim());

                self.fold_costs(enode, res, op_cost, &[a_cost, b_cost])
            }
            Linalg::Add([a, b]) => {
                let a_dim = data(a).get_inner_dim();
                let a_cost = costs(*a);
                let b_cost = costs(*b);

                let op_cost = a_dim.size();

                let res = &a_cost.val + &b_cost.val;

                self.fold_costs(enode, res.into(), op_cost, &[a_cost, b_cost])
            }
            Linalg::Relu(a) => {
                let a_dim = data(a).get_inner_dim();
                let a_cost = costs(*a);
                let op_cost = a_dim.size();

                let res = a_cost.val.map(|x| f64::max(*x, 0.0)).to_shared();

                self.fold_costs(enode, res, op_cost, &[a_cost])
            }
            Linalg::Mul([a, b]) => {
                let a_dim = data(a).get_inner_dim();
                let b_dim = data(b).get_inner_dim();

                let a_cost = costs(*a);
                let b_cost = costs(*b);
                let op_cost = a_dim.rows() * a_dim.cols() * b_dim.cols();

                let res = a_cost.val.dot(&b_cost.val).to_shared();

                self.fold_costs(enode, res, op_cost, &[a_cost, b_cost])
            }
        }
    }
}
