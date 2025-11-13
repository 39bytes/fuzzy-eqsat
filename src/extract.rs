use indexmap::IndexMap;
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
};

use egg::*;

pub struct MyExtractor<'a, CF: CostFunction<L>, L: Language, N: Analysis<L>> {
    cost_function: CF,
    costs: HashMap<Id, Vec<CandidateExpr<CF, L>>>,
    egraph: &'a EGraph<L, N>,
}

pub struct CandidateExpr<CF: CostFunction<L>, L: Language> {
    pub cost: CF::Cost,
    pub node: L,
    pub children: HashMap<Id, L>,
}

impl<CF: CostFunction<L>, L: Language> PartialEq for CandidateExpr<CF, L> {
    fn eq(&self, other: &Self) -> bool {
        self.cost.eq(&other.cost)
    }
}

impl<CF: CostFunction<L>, L: Language> PartialOrd for CandidateExpr<CF, L> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<CF: CostFunction<L>, L: Language> Eq for CandidateExpr<CF, L> {}

impl<CF: CostFunction<L>, L: Language> Ord for CandidateExpr<CF, L> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.partial_cmp(&other.cost).unwrap()
    }
}

impl<'a, CF, L, N> MyExtractor<'a, CF, L, N>
where
    CF: CostFunction<L>,
    L: Language,
    N: Analysis<L>,
{
    /// Create a new `MyExtractor` given an `EGraph` and a
    /// `CostFunction`.
    ///
    /// The extraction does all the work on creation, so this function
    /// performs the greedy search for cheapest representative of each
    /// eclass.
    pub fn new(egraph: &'a EGraph<L, N>, cost_function: CF) -> Self {
        let costs = HashMap::default();
        let mut extractor = MyExtractor {
            costs,
            egraph,
            cost_function,
        };
        extractor.find_costs();

        extractor
    }

    /// Find the cheapest (lowest cost) represented `RecExpr` in the
    /// given eclass.
    pub fn find_best(&self, eclass: Id) -> (&CandidateExpr<CF, L>, RecExpr<L>) {
        let all_possible_costs = &self.costs[&self.egraph.find(eclass)];

        let best = all_possible_costs.iter().min().unwrap();
        let expr = best.node.build_recexpr(|id| best.children[&id].clone());

        (best, expr)
    }

    pub fn all_costs(&self, eclass: Id) -> &Vec<CandidateExpr<CF, L>> {
        &self.costs[&self.egraph.find(eclass)]
    }

    /// Find the cheapest e-node in the given e-class.
    pub fn find_best_node(&self, eclass: Id) -> &L {
        let possible_costs = &self.costs[&self.egraph.find(eclass)];
        &possible_costs.iter().min().unwrap().node
    }

    /// Find the cost of the term that would be extracted from this e-class.
    pub fn find_best_cost(&self, eclass: Id) -> CF::Cost {
        let possible_costs = &self.costs[&self.egraph.find(eclass)];
        possible_costs.iter().min().unwrap().cost.clone()
    }

    fn has_all_costs(&self, node: &L) -> bool {
        node.all(|id| self.costs.contains_key(&self.egraph.find(id)))
    }

    fn node_total_cost(&mut self, node: &L) -> Vec<CandidateExpr<CF, L>> {
        let eg = &self.egraph;
        assert!(self.has_all_costs(node));

        let mut indices: IndexMap<Id, usize> =
            IndexMap::from_iter(node.children().iter().map(|i| (*i, 0usize)));

        let index_maxes = node
            .children()
            .iter()
            .map(|id| self.costs[&eg.find(*id)].len())
            .collect::<Vec<_>>();

        let mut all_costs = Vec::new();

        loop {
            let costs = &self.costs;
            let cost_f = |id| costs[&eg.find(id)][indices[&id]].cost.clone();

            if indices.is_empty() {
                all_costs.push(CandidateExpr {
                    cost: self.cost_function.cost(node, cost_f),
                    node: node.clone(),
                    children: HashMap::new(),
                });
                return all_costs;
            }

            let mut all_children = HashMap::new();
            for id in node.children() {
                let actual_child = &costs[&eg.find(*id)][indices[id]];
                all_children.insert(*id, actual_child.node.clone());
                all_children.extend(actual_child.children.clone());
            }

            all_costs.push(CandidateExpr {
                cost: self.cost_function.cost(node, cost_f),
                node: node.clone(),
                children: all_children,
            });

            if let Some(idxs) = next_permutation(indices, &index_maxes) {
                indices = idxs;
            } else {
                break;
            }
        }

        all_costs
    }

    fn find_costs(&mut self) {
        let mut computed = HashSet::new();

        let mut did_something = true;
        while did_something {
            did_something = false;

            for class in self.egraph.classes() {
                if computed.contains(&class.id) || !class.iter().all(|x| self.has_all_costs(x)) {
                    continue;
                }
                computed.insert(class.id);
                let pass = self.make_pass(class);
                self.costs.insert(class.id, pass);
                did_something = true;
            }
        }
    }

    fn make_pass(&mut self, eclass: &EClass<L, N::Data>) -> Vec<CandidateExpr<CF, L>> {
        eclass
            .iter()
            .flat_map(|node| self.node_total_cost(node))
            .collect()
    }
}

fn next_permutation(
    mut indices: IndexMap<Id, usize>,
    maxes: &[usize],
) -> Option<IndexMap<Id, usize>> {
    assert!(!indices.is_empty());
    let mut i = indices.len() - 1;

    loop {
        indices[i] += 1;
        if indices[i] >= maxes[i] {
            indices[i] = 0;
            if i == 0 {
                return None;
            }
            i -= 1;
        } else {
            return Some(indices);
        }
    }
}
