use indexmap::IndexMap;
use rand::{Rng, rngs::ThreadRng};
use rand_distr::{Bernoulli, Distribution};
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};

use egg::*;

pub struct CompleteExtractor<'a, CF: CostFunction<L>, L: Language, N: Analysis<L>> {
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

#[allow(dead_code)]
impl<'a, CF, L, N> CompleteExtractor<'a, CF, L, N>
where
    CF: CostFunction<L>,
    L: Language,
    N: Analysis<L>,
{
    pub fn new(egraph: &'a EGraph<L, N>, cost_function: CF) -> Self {
        let costs = HashMap::default();
        let mut extractor = CompleteExtractor {
            costs,
            egraph,
            cost_function,
        };
        extractor.find_costs();

        extractor
    }

    pub fn find_best(&self, eclass: Id) -> (&CandidateExpr<CF, L>, RecExpr<L>) {
        let all_possible_costs = &self.costs[&self.egraph.find(eclass)];

        let best = all_possible_costs.iter().min().unwrap();
        let expr = best.node.build_recexpr(|id| best.children[&id].clone());

        (best, expr)
    }

    pub fn all_costs(&self, eclass: Id) -> &Vec<CandidateExpr<CF, L>> {
        &self.costs[&self.egraph.find(eclass)]
    }

    pub fn find_best_node(&self, eclass: Id) -> &L {
        let possible_costs = &self.costs[&self.egraph.find(eclass)];
        &possible_costs.iter().min().unwrap().node
    }

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

            let cost = self.cost_function.cost(node, cost_f);
            all_costs.push(CandidateExpr {
                cost,
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
pub struct GeneticAlgorithmExtractor<'a, CF: CostFunction<L>, L: Language, N: Analysis<L>> {
    cost_function: CF,
    egraph: &'a EGraph<L, N>,
    eclass_bounds: Vec<usize>,
    rng: ThreadRng,
    mutation: Bernoulli,
}

impl<'a, CF, L, N> GeneticAlgorithmExtractor<'a, CF, L, N>
where
    CF: CostFunction<L>,
    L: Language,
    N: Analysis<L>,
{
    const POPULATION_SIZE: usize = 100;
    const MAX_GENERATIONS: usize = 10;
    const SELECTION_COUNT: f64 = 0.20;
    const MUTATION_RATE: f64 = 0.05;

    pub fn new(egraph: &'a EGraph<L, N>, cost_function: CF) -> Self {
        let num_eclasses = egraph.classes().count();
        let eclass_bounds = (0..num_eclasses)
            .map(|i| egraph[i.into()].nodes.len())
            .collect();

        GeneticAlgorithmExtractor {
            egraph,
            cost_function,
            eclass_bounds,
            rng: rand::rng(),
            mutation: Bernoulli::new(Self::MUTATION_RATE).expect("Bernoulli creation failed"),
        }
    }

    pub fn find_best(&mut self, root: Id) -> (CF::Cost, RecExpr<L>) {
        let selection_cutoff =
            ((Self::POPULATION_SIZE as f64) * Self::SELECTION_COUNT).round() as usize;
        let mut generation = 0;

        let mut best: Option<(Vec<usize>, CF::Cost)> = None;

        let mut population: Vec<_> = (0..Self::POPULATION_SIZE)
            .map(|_| self.random_sol())
            .collect();

        while generation < Self::MAX_GENERATIONS {
            let mut population_with_costs: Vec<_> = population
                .iter()
                .enumerate()
                .map(|(i, sol)| (i, self.evaluate_sol(root, sol)))
                .collect();

            // TODO: Optimize using heap
            population_with_costs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            match best {
                None => {
                    best = Some((
                        population[population_with_costs[0].0].clone(),
                        population_with_costs[0].1.clone(),
                    ))
                }
                Some(ref mut existing) => {
                    if population_with_costs[0].1 < existing.1 {
                        *existing = (
                            population[population_with_costs[0].0].clone(),
                            population_with_costs[0].1.clone(),
                        )
                    }
                }
            }

            let fittest: Vec<_> = population_with_costs[..selection_cutoff]
                .iter()
                .map(|x| &population[x.0])
                .collect();
            let mut next_population = Vec::new();
            while next_population.len() < Self::POPULATION_SIZE {
                let parent1 = fittest[self.rng.random_range(0..fittest.len())];
                let parent2 = fittest[self.rng.random_range(0..fittest.len())];
                let (mut child1, mut child2) = self.crossover(parent1, parent2);
                self.mutate(&mut child1);
                self.mutate(&mut child2);
                next_population.push(child1);
                next_population.push(child2);
            }

            population = next_population;
            println!("Finished generation {}", generation);
            generation += 1;
        }

        let (best_sol, best_cost) = best.expect("There should be a best solution");
        let root_node = self.egraph[root].nodes[best_sol[usize::from(root)]].clone();
        let best_expr =
            root_node.build_recexpr(|id| self.egraph[id].nodes[best_sol[usize::from(id)]].clone());

        (best_cost, best_expr)
    }

    fn evaluate_sol(&mut self, root: Id, sol: &[usize]) -> CF::Cost {
        let mut costs: HashMap<Id, CF::Cost> = HashMap::new();

        let mut did_something = true;
        while did_something {
            did_something = false;

            let mut ids = self.egraph.classes().map(|c| c.id).collect::<Vec<_>>();
            ids.sort();
            println!("{:?}", ids);

            for class in self.egraph.classes() {
                let id = class.id;
                if costs.contains_key(&id) {
                    continue;
                }
                let enode = &self.egraph[id].nodes[sol[usize::from(id)]];
                if !enode.all(|id| costs.contains_key(&self.egraph.find(id))) {
                    continue;
                }

                let cost = self.cost_function.cost(enode, |id| costs[&id].clone());
                costs.insert(class.id, cost);
                did_something = true;
            }
        }

        costs[&root].clone()
    }

    fn crossover(&mut self, a: &[usize], b: &[usize]) -> (Vec<usize>, Vec<usize>) {
        assert!(a.len() == b.len());
        let crossover_point = self.rng.random_range(0..a.len());
        let mut child1 = Vec::from(a);
        let mut child2 = Vec::from(b);

        child1[crossover_point..].swap_with_slice(&mut child2[crossover_point..]);

        (child1, child2)
    }

    fn mutate(&mut self, sol: &mut [usize]) {
        for (i, x) in sol.iter_mut().enumerate() {
            if self.mutation.sample(&mut self.rng) {
                *x = self.rng.random_range(0..self.eclass_bounds[i]);
            }
        }
    }

    fn random_sol(&mut self) -> Vec<usize> {
        self.eclass_bounds
            .iter()
            .copied()
            .map(|x| self.rng.random_range(0..x))
            .collect()
    }
}
