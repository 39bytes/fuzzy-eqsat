use std::cmp::Ordering;

use egg::*;
use ndarray::ArcArray2;

use crate::{analysis::LinalgAnalysis, lang::Linalg};

pub struct LinalgCost<'a> {
    pub egraph: &'a EGraph<Linalg, LinalgAnalysis>,
    pub max_rel_error: f64,
}

struct PossibleCosts {
    costs: Vec<CostWithErrorBound>,
}

#[derive(PartialEq, Debug, Clone)]
struct CostWithErrorBound {
    cost: usize,
    max_error: f64,
    error: f64,
    result: ArcArray2<f64>,
}

impl PartialOrd for CostWithErrorBound {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.error > self.max_error {
            Some(Ordering::Greater)
        } else {
            self.cost.partial_cmp(&other.cost)
        }
    }
}

impl<'a> CostFunction<Linalg> for LinalgCost<'a> {
    type Cost = CostWithErrorBound;

    fn cost<C>(&mut self, enode: &Linalg, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let data = |i: &Id| self.egraph[*i].data;
        let op_cost = match enode {
            Linalg::Mat(a) => CostWithErrorBound {
                cost: 0,
                max_error: self.max_rel_error,
                error: 0,
                result: data(a).get_inner_value(),
            },
            Linalg::Num(_) => 0,
            Linalg::SvdU([a, _]) | Linalg::SvdD([a, _]) | Linalg::SvdVt([a, _]) => 0,
            Linalg::Add([a, _]) | Linalg::Relu(a) => dim(a).size(),
            Linalg::Mul([a, b]) => dim(a).rows() * dim(a).cols() * dim(b).cols(),
        };
        enode.fold(op_cost, |sum, id| sum + costs(id))
    }
}
