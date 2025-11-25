use std::{cell::RefCell, cmp::Ordering, collections::HashMap, fmt::Display, rc::Rc};

use egg::*;
use ndarray::{ArcArray2, Array2, s};
use ndarray_linalg::Norm;

use crate::{
    analysis::{LinalgAnalysis, MatrixValue},
    lang::Linalg,
    math::{prune, relu, softmax},
};

#[derive(Debug)]
pub struct LinalgCost<'a> {
    pub egraph: &'a EGraph<Linalg, LinalgAnalysis>,
    pub var_info: Rc<RefCell<HashMap<Symbol, MatrixValue>>>,
    pub max_rel_error: f64,
}

#[derive(PartialEq, Debug, Clone, Default)]
pub struct CostWithErrorBound {
    pub cost: usize,
    pub error: Option<f64>,
    pub max_rel_error: f64,
    pub val: ArcArray2<f64>,
}

impl Display for CostWithErrorBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cost: {}, error: {:?}", self.cost, self.error)
    }
}

impl PartialOrd for CostWithErrorBound {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let a = if self.error.unwrap_or(0.0) > self.max_rel_error {
            usize::MAX
        } else {
            self.cost
        };
        let b = if other.error.unwrap_or(0.0) > other.max_rel_error {
            usize::MAX
        } else {
            other.cost
        };

        a.partial_cmp(&b)
    }
}

impl<'a> LinalgCost<'a> {
    fn relative_frobenius_norm_error(
        &self,
        enode: &Linalg,
        approx: &ArcArray2<f64>,
    ) -> Option<f64> {
        let eclass_id = self
            .egraph
            .lookup(enode.clone())
            .expect("enode should be in egraph");

        let eclass = &self.egraph[eclass_id];

        eclass
            .data
            .unwrap_mat()
            .canonical_value
            .as_ref()
            .map(|x| (x.val() - approx).norm_l2() / x.val().norm_l2())
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
                error: Some(0.0),
                max_rel_error: self.max_rel_error,
                val: self.var_info.borrow()[a].val().clone(),
            },
            Linalg::Num(_) => CostWithErrorBound::default(),
            Linalg::SvdU([a, k]) => {
                let a_cost = costs(*a);
                let k = data(k).unwrap_num();

                let (u, _, _) = data(a)
                    .unwrap_mat()
                    .canonical_value
                    .as_ref()
                    .expect("a should have a true value")
                    .svd()
                    .clone();

                CostWithErrorBound {
                    cost: a_cost.cost,
                    error: None,
                    max_rel_error: self.max_rel_error,
                    val: u.slice_move(s![.., ..k]),
                }
            }
            Linalg::SvdD([a, k]) => {
                let a_cost = costs(*a);
                let k = data(k).unwrap_num();

                let (_, sigma, _) = data(a)
                    .unwrap_mat()
                    .canonical_value
                    .as_ref()
                    .expect("a should have a true value")
                    .svd()
                    .clone();

                CostWithErrorBound {
                    cost: a_cost.cost,
                    error: None,
                    max_rel_error: self.max_rel_error,
                    val: Array2::from_diag(&sigma.slice(s![..k])).into(),
                }
            }
            Linalg::SvdVt([a, k]) => {
                let a_cost = costs(*a);
                let k = data(k).unwrap_num();

                let (_, _, vt) = data(a)
                    .unwrap_mat()
                    .canonical_value
                    .as_ref()
                    .expect("a should have a true value")
                    .svd()
                    .clone();

                CostWithErrorBound {
                    cost: a_cost.cost,
                    error: None,
                    max_rel_error: self.max_rel_error,
                    val: vt.slice_move(s![..k, ..]),
                }
            }
            Linalg::Pruned([a, k]) => {
                let a_cost = costs(*a);
                let k = data(k).unwrap_num();

                // let res = prune(&a_cost.val, k);

                self.fold_costs(enode, a_cost.val.clone(), 0, &[a_cost])
            }
            Linalg::Add([a, b]) => {
                let a_dim = data(a).unwrap_mat().dim;
                let a_cost = costs(*a);
                let b_cost = costs(*b);

                let op_cost = a_dim.size();

                let res = &a_cost.val + &b_cost.val;

                self.fold_costs(enode, res.into(), op_cost, &[a_cost, b_cost])
            }
            Linalg::Relu(a) => {
                let a_dim = data(a).unwrap_mat().dim;
                let a_cost = costs(*a);
                let op_cost = a_dim.size();

                let res = relu(&a_cost.val).to_shared();

                self.fold_costs(enode, res, op_cost, &[a_cost])
            }
            Linalg::Softmax(a) => {
                let a_dim = data(a).unwrap_mat().dim;
                let a_cost = costs(*a);
                let op_cost = a_dim.size();

                let res = softmax(&a_cost.val).to_shared();

                self.fold_costs(enode, res, op_cost, &[a_cost])
            }
            Linalg::Mul([a, b]) => {
                let a_dim = data(a).unwrap_mat().dim;
                let b_dim = data(b).unwrap_mat().dim;

                let a_cost = costs(*a);
                let b_cost = costs(*b);
                let op_cost = if data(a).unwrap_mat().diagonal {
                    a_dim.cols() * b_dim.cols()
                } else {
                    a_dim.rows() * a_dim.cols() * b_dim.cols()
                };

                let res = a_cost.val.dot(&b_cost.val).to_shared();

                self.fold_costs(enode, res, op_cost, &[a_cost, b_cost])
            }
        }
    }
}
