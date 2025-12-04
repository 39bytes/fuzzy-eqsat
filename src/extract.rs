use indexmap::IndexMap;
use rand::{Rng, rngs::ThreadRng, seq::IndexedRandom};
use rand_distr::{Bernoulli, Distribution};
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet, VecDeque},
    fmt::Display,
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
    eclass_bounds: HashMap<Id, usize>,
    rng: ThreadRng,
    mutation: Bernoulli,
}

impl<'a, CF, L, N> GeneticAlgorithmExtractor<'a, CF, L, N>
where
    CF: CostFunction<L>,
    CF::Cost: Display,
    L: Language,
    N: Analysis<L>,
{
    const POPULATION_SIZE: usize = 100;
    const MAX_GENERATIONS: usize = 20;
    const SELECTION_COUNT: f64 = 0.10;
    const MUTATION_RATE: f64 = 0.05;
    const CONVERGENCE_GENERATIONS: usize = 5;

    pub fn new(egraph: &'a EGraph<L, N>, cost_function: CF) -> Self {
        let eclass_bounds = egraph
            .classes()
            .map(|c| (c.id, egraph[c.id].nodes.len()))
            .collect();

        log::debug!(
            "eclass IDs: {:?}",
            egraph.classes().map(|c| c.id).collect::<Vec<_>>()
        );

        GeneticAlgorithmExtractor {
            egraph,
            cost_function,
            eclass_bounds,
            rng: rand::rng(),
            mutation: Bernoulli::new(Self::MUTATION_RATE).expect("Bernoulli creation failed"),
        }
    }

    pub fn find_best(
        &mut self,
        root: Id,
        orig: RecExpr<L>,
        to_pareto_point: impl Fn(&CF::Cost) -> (f64, f64),
    ) -> (CF::Cost, RecExpr<L>, Vec<(f64, f64)>) {
        let selection_cutoff =
            ((Self::POPULATION_SIZE as f64) * Self::SELECTION_COUNT).round() as usize;
        let mut generation = 0;
        let mut generations_since_improvement = 0;

        let mut best_cost = self.cost_function.cost_rec(&orig);
        let mut best_expr = orig;

        let mut population: Vec<_> = (0..Self::POPULATION_SIZE)
            .map(|_| random_sol(&self.eclass_bounds, &mut self.rng))
            .collect();

        let mut pareto_points: Vec<(f64, f64)> = vec![];

        while generation < Self::MAX_GENERATIONS {
            let mut population_with_costs: Vec<_> = population
                .iter()
                .enumerate()
                .map(|(i, sol)| {
                    (
                        i,
                        evaluate_sol(self.egraph, &mut self.cost_function, root, sol),
                    )
                })
                .collect();

            pareto_points.extend(
                population_with_costs
                    .iter()
                    .map(|p| to_pareto_point(&p.1.0)),
            );

            // TODO: Optimize using heap
            population_with_costs.sort_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap());

            if population_with_costs[0].1.0 < best_cost {
                best_expr = population_with_costs[0].1.1.clone();
                best_cost = population_with_costs[0].1.0.clone();
                generations_since_improvement = 0;
                log::info!("Found better cost: {}", best_cost)
            } else {
                generations_since_improvement += 1;
                // if generations_since_improvement > Self::CONVERGENCE_GENERATIONS {
                //     log::info!(
                //         "Haven't found better solution in {} generations, stopping",
                //         Self::CONVERGENCE_GENERATIONS,
                //     );
                //     break;
                // }
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
            log::info!("Finished generation {}", generation);
            generation += 1;
        }

        (best_cost, best_expr, pareto_points)
    }

    fn crossover(
        &mut self,
        a: &HashMap<Id, usize>,
        b: &HashMap<Id, usize>,
    ) -> (HashMap<Id, usize>, HashMap<Id, usize>) {
        assert!(a.len() == b.len());
        let a_vec: Vec<_> = a.iter().collect();
        let b_vec: Vec<_> = b.iter().collect();

        let crossover_point = self.rng.random_range(0..a_vec.len());
        let mut child1 = a_vec;
        let mut child2 = b_vec;
        child1.sort_by_key(|c| c.0);
        child2.sort_by_key(|c| c.0);

        child1[crossover_point..].swap_with_slice(&mut child2[crossover_point..]);

        (
            child1.into_iter().map(|(k, v)| (*k, *v)).collect(),
            child2.into_iter().map(|(k, v)| (*k, *v)).collect(),
        )
    }

    fn mutate(&mut self, sol: &mut HashMap<Id, usize>) {
        for (i, x) in sol.iter_mut() {
            if self.mutation.sample(&mut self.rng) {
                *x = self.rng.random_range(0..self.eclass_bounds[i]);
            }
        }
    }
}

// pub struct SimulatedAnnealingExtractor<'a, CF: CostFunction<L>, L: Language, N: Analysis<L>> {
//     cost_function: CF,
//     egraph: &'a EGraph<L, N>,
//     eclass_bounds: HashMap<Id, usize>,
//     eclass_ids: Vec<Id>,
//     rng: ThreadRng,
// }
//
// impl<'a, CF, L, N> SimulatedAnnealingExtractor<'a, CF, L, N>
// where
//     CF: CostFunction<L>,
//     L: Language,
//     N: Analysis<L>,
// {
//     pub fn new(egraph: &'a EGraph<L, N>, cost_function: CF) -> Self {
//         let eclass_bounds: HashMap<Id, usize> = egraph
//             .classes()
//             .map(|c| (c.id, egraph[c.id].nodes.len()))
//             .collect();
//         let mut eclass_ids: Vec<_> = eclass_bounds.keys().copied().collect();
//         eclass_ids.sort();
//
//         log::debug!(
//             "eclass IDs: {:?}",
//             egraph.classes().map(|c| c.id).collect::<Vec<_>>()
//         );
//
//         SimulatedAnnealingExtractor {
//             egraph,
//             cost_function,
//             eclass_bounds,
//             eclass_ids,
//             rng: rand::rng(),
//         }
//     }
//
//     pub fn find_best(
//         &mut self,
//         root: Id,
//         orig: RecExpr<L>,
//         energy_function: impl Fn(&CF::Cost) -> f64,
//     ) -> (CF::Cost, RecExpr<L>) {
//         let mut sol = random_sol(&self.eclass_bounds, &mut self.rng);
//         let (cur_cost, _) = evaluate_sol(self.egraph, &mut self.cost_function, root, &sol);
//         let mut e = energy_function(&cur_cost);
//
//         let mut best = orig;
//         let mut best_e = e;
//         let mut best_cost = cur_cost;
//         let mut temp = 2000.0;
//         let decay = 0.995;
//
//         while temp > 1.0 {
//             log::info!("{}", temp);
//             let nxt = self.next_state(&sol);
//             let (cost, expr) = evaluate_sol(self.egraph, &mut self.cost_function, root, &nxt);
//             let nxt_e = energy_function(&cost);
//             if self.should_accept(e, nxt_e, temp) {
//                 sol = nxt;
//                 if nxt_e < best_e {
//                     best = expr;
//                     best_cost = cost;
//                     best_e = nxt_e;
//                 }
//                 e = nxt_e;
//             }
//             temp *= decay;
//         }
//
//         (best_cost, best)
//     }
//
//     fn should_accept(&mut self, e: f64, e_next: f64, t: f64) -> bool {
//         let p = f64::exp(-(e_next - e) / t);
//         if p.is_nan() {
//             false
//         } else if p > 1.0 {
//             true
//         } else {
//             self.rng.random_bool(p)
//         }
//     }
//
//     fn next_state(&mut self, sol: &HashMap<Id, usize>) -> HashMap<Id, usize> {
//         let mut nxt = sol.clone();
//         let id = self
//             .eclass_ids
//             .choose(&mut self.rng)
//             .expect("EClasses shouldn't be empty");
//         nxt.insert(*id, self.rng.random_range(0..self.eclass_bounds[id]));
//         nxt
//     }
// }

fn random_sol(eclass_bounds: &HashMap<Id, usize>, rng: &mut ThreadRng) -> HashMap<Id, usize> {
    eclass_bounds
        .iter()
        .map(|(id, bound)| (*id, rng.random_range(0..*bound)))
        .collect()
}

fn evaluate_sol<L, N, CF>(
    egraph: &EGraph<L, N>,
    cost_function: &mut CF,
    root: Id,
    sol: &HashMap<Id, usize>,
) -> (CF::Cost, RecExpr<L>)
where
    L: Language,
    N: Analysis<L>,
    CF: CostFunction<L>,
    CF::Cost: Display,
{
    let mut costs: HashMap<Id, CF::Cost> = HashMap::new();

    let mut in_degrees = HashMap::<Id, usize>::new();
    let mut reversed_edges = HashMap::<Id, Vec<Id>>::new();
    let mut visited = HashSet::new();
    let mut stack = vec![root];
    let mut queue = VecDeque::<Id>::new();

    while let Some(cur) = stack.pop() {
        visited.insert(cur);
        let children = egraph[cur].nodes[sol[&cur]].children();

        in_degrees.insert(cur, children.len());
        if children.is_empty() {
            queue.push_back(cur);
        }
        for child in children {
            if !visited.contains(child) {
                stack.push(*child);
            }
            reversed_edges.entry(*child).or_default().push(cur);
        }
    }

    while let Some(id) = queue.pop_front() {
        let enode = &egraph[id].nodes[sol[&id]];
        let cost = cost_function.cost(enode, |id| costs[&id].clone());
        costs.insert(id, cost);
        if let Some(neighbors) = reversed_edges.get(&id) {
            for neighbor in neighbors {
                let deg = in_degrees.get_mut(neighbor).expect("should exist");
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(*neighbor);
                }
            }
        }
    }

    // // TODO: optimize this? (toposort maybe?)
    // let mut did_something = true;
    // while did_something {
    //     did_something = false;
    //
    //     for class in egraph.classes() {
    //         let id = class.id;
    //         if costs.contains_key(&id) {
    //             continue;
    //         }
    //         let enode = &class.nodes[sol[&id]];
    //         if !enode.all(|id| costs.contains_key(&egraph.find(id))) {
    //             continue;
    //         }
    //
    //         let cost = cost_function.cost(enode, |id| costs[&id].clone());
    //         costs.insert(class.id, cost);
    //         did_something = true;
    //     }
    // }

    // TODO: avoid unnecessary building of recexpr every time?
    let root_node = egraph[root].nodes[sol[&root]].clone();
    let recexpr = root_node.build_recexpr(|id| egraph[id].nodes[sol[&id]].clone());
    // println!("{}", costs[&root]);
    (costs[&root].clone(), recexpr)
}
