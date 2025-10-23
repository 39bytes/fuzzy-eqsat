use indexmap::IndexMap;
use std::collections::BTreeSet;
use std::{cmp::Ordering, collections::HashMap};

use egg::*;

#[derive(Clone)]
struct Cost<T: PartialOrd + Clone>(T);

impl<T: PartialOrd + Clone> PartialEq for Cost<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T: PartialOrd + Clone> Eq for Cost<T> {}

impl<T: PartialOrd + Clone> PartialOrd for Cost<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: PartialOrd + Clone> Ord for Cost<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

#[derive(Clone)]
struct CandidateExpr<T: PartialOrd + Clone, L: Language> {
    cost: Cost<T>,
    node: L,
}

impl<T: PartialOrd + Clone, L: Language> PartialEq for CandidateExpr<T, L> {
    fn eq(&self, other: &Self) -> bool {
        self.cost.eq(&other.cost)
    }
}

impl<T: PartialOrd + Clone, L: Language> PartialOrd for CandidateExpr<T, L> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: PartialOrd + Clone, L: Language> Ord for CandidateExpr<T, L> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.cmp(&other.cost)
    }
}

impl<T: PartialOrd + Clone, L: Language> Eq for CandidateExpr<T, L> {}

#[derive(PartialEq, Eq)]
struct TopK<T: Ord> {
    k: usize,
    elems: BTreeSet<T>,
}

impl<T: Ord> TopK<T> {
    pub fn new(k: usize) -> Self {
        TopK {
            k,
            elems: BTreeSet::new(),
        }
    }

    pub fn push(&mut self, item: T) {
        self.elems.insert(item);
        if self.elems.len() > self.k {
            self.elems.pop_last();
        }
    }

    pub fn len(&self) -> usize {
        self.elems.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elems.is_empty()
    }

    pub fn smallest(&self) -> Option<&T> {
        self.elems.first()
    }

    pub fn iter(&self) -> std::collections::btree_set::Iter<'_, T> {
        self.elems.iter()
    }
}

pub struct MyExtractor<'a, CF: CostFunction<L>, L: Language, N: Analysis<L>> {
    cost_function: CF,
    pub costs: HashMap<Id, TopK<CandidateExpr<CF::Cost, L>>>,
    egraph: &'a EGraph<L, N>,
    k: usize,
}

fn cmp<T: PartialOrd>(a: &Option<T>, b: &Option<T>) -> Ordering {
    // None is high
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(a), Some(b)) => a.partial_cmp(b).unwrap(),
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
    pub fn new(egraph: &'a EGraph<L, N>, cost_function: CF, k: usize) -> Self {
        let costs = HashMap::default();
        let mut extractor = MyExtractor {
            costs,
            egraph,
            cost_function,
            k,
        };
        extractor.find_costs();

        extractor
    }

    /// Find the cheapest (lowest cost) represented `RecExpr` in the
    /// given eclass.
    pub fn find_best(&self, eclass: Id) -> (CF::Cost, RecExpr<L>) {
        let topk = &self.costs[&self.egraph.find(eclass)];
        let best = topk.smallest().unwrap();

        let expr = best
            .node
            .build_recexpr(|id| self.find_best_node(id).clone());
        (best.cost.0.clone(), expr)
    }

    pub fn find_best_k(&self, eclass: Id) -> Vec<(CF::Cost, RecExpr<L>)> {
        let topk = &self.costs[&self.egraph.find(eclass)];

        topk.iter()
            .map(|c| {
                (
                    c.cost.0.clone(),
                    c.node.build_recexpr(|id| self.find_best_node(id).clone()),
                )
            })
            .collect()
    }

    /// Find the cheapest e-node in the given e-class.
    pub fn find_best_node(&self, eclass: Id) -> &L {
        &self.costs[&self.egraph.find(eclass)]
            .smallest()
            .unwrap()
            .node
    }

    /// Find the cost of the term that would be extracted from this e-class.
    pub fn find_best_cost(&self, eclass: Id) -> CF::Cost {
        let topk_costs = &self.costs[&self.egraph.find(eclass)];
        topk_costs.smallest().unwrap().cost.0.clone()
    }

    fn node_total_cost(&mut self, node: &L) -> Option<TopK<CandidateExpr<CF::Cost, L>>> {
        let eg = &self.egraph;
        let has_cost = |id| self.costs.contains_key(&eg.find(id));

        let mut top_k = TopK::new(self.k);
        let mut indices: IndexMap<Id, usize> =
            IndexMap::from_iter(node.children().iter().map(|i| (*i, 0usize)));

        if !node.all(has_cost) {
            return None;
        }

        let maxes = node
            .children()
            .iter()
            .map(|id| self.costs[&eg.find(*id)].len())
            .collect::<Vec<_>>();

        loop {
            let costs = &self.costs;
            let cost_f = |id| {
                costs[&eg.find(id)]
                    .elems
                    .iter()
                    .nth(indices[&id])
                    .unwrap()
                    .cost
                    .0
                    .clone()
            };

            if indices.is_empty() {
                top_k.push(CandidateExpr {
                    cost: Cost(self.cost_function.cost(node, cost_f)),
                    node: node.clone(),
                });
                return Some(top_k);
            }

            top_k.push(CandidateExpr {
                cost: Cost(self.cost_function.cost(node, cost_f)),
                node: node.clone(),
            });

            if let Some(idxs) = next_permutation(indices, &maxes) {
                indices = idxs;
            } else {
                break;
            }
        }

        Some(top_k)
    }

    fn find_costs(&mut self) {
        let mut did_something = true;
        while did_something {
            did_something = false;

            for class in self.egraph.classes() {
                let pass = self.make_pass(class);
                match (self.costs.get(&class.id), pass) {
                    (None, pass) if !pass.is_empty() => {
                        self.costs.insert(class.id, pass);
                        did_something = true;
                    }
                    (Some(old), pass) if *old != pass => {
                        self.costs.insert(class.id, pass);
                        did_something = true;
                    }
                    _ => (),
                }
            }
        }

        for class in self.egraph.classes() {
            if !self.costs.contains_key(&class.id) {}
        }
    }

    fn make_pass(&mut self, eclass: &EClass<L, N::Data>) -> TopK<CandidateExpr<CF::Cost, L>> {
        let mut top_k = TopK::new(self.k);
        for node in eclass.iter() {
            let candidates = self.node_total_cost(node);
            if let Some(candidates) = candidates {
                for candidate in candidates.iter() {
                    top_k.push(candidate.clone())
                }
            }
        }

        top_k
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
