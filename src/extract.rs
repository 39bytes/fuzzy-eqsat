use rand::{Rng, rngs::ThreadRng};
use rand_distr::{Bernoulli, Distribution, weighted::WeightedIndex};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::Display,
};

use egg::*;

pub type Solution = HashMap<Id, usize>;

pub struct GeneticAlgorithmExtractor<'a, CF: CostFunction<L>, L: Language, N: Analysis<L>> {
    cost_function: CF,
    egraph: &'a EGraph<L, N>,
    eclass_bounds: HashMap<Id, usize>,
    rng: ThreadRng,
    mutation_dist: Bernoulli,
    selection_dist: WeightedIndex<f64>,
}

impl<'a, CF, L, N> GeneticAlgorithmExtractor<'a, CF, L, N>
where
    CF: CostFunction<L>,
    CF::Cost: Display,
    L: Language,
    N: Analysis<L>,
{
    const POPULATION_SIZE: usize = 100;
    const MAX_GENERATIONS: usize = 25;
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

        let w: f64 = 0.9;
        let total: f64 = (1..=Self::POPULATION_SIZE)
            .map(|i| w.powf((Self::POPULATION_SIZE - i) as f64))
            .sum();
        let weights = (1..=Self::POPULATION_SIZE)
            .map(|i| w.powf((Self::POPULATION_SIZE - i) as f64) / total)
            .rev()
            .collect::<Vec<_>>();

        let selection_dist = WeightedIndex::new(&weights).expect("Selection dist creation failed");
        let mutation_dist = Bernoulli::new(Self::MUTATION_RATE).expect("Bernoulli creation failed");

        GeneticAlgorithmExtractor {
            egraph,
            cost_function,
            eclass_bounds,
            rng: rand::rng(),
            mutation_dist,
            selection_dist,
        }
    }

    pub fn find_best(
        &mut self,
        root: Id,
        orig: RecExpr<L>,
        to_pareto_point: impl Fn(&CF::Cost) -> (f64, f64),
    ) -> (CF::Cost, Option<Solution>, Vec<(f64, f64)>, usize) {
        let mut generation = 0;
        let mut generations_since_improvement = 0;

        let mut best_cost = self.cost_function.cost_rec(&orig);
        let mut best_sol: Option<Solution> = None;

        let mut population: Vec<_> = (0..Self::POPULATION_SIZE)
            .map(|_| random_sol(&self.eclass_bounds, &mut self.rng))
            .collect();

        let mut pareto_points: Vec<(f64, f64)> = vec![];

        while generation < Self::MAX_GENERATIONS {
            let mut costs: Vec<_> = population
                .iter()
                .enumerate()
                .map(|(i, sol)| {
                    (
                        i,
                        evaluate_sol(self.egraph, &mut self.cost_function, root, sol),
                    )
                })
                .collect();

            pareto_points.extend(costs.iter().map(|p| to_pareto_point(&p.1)));

            // TODO: Optimize using heap
            costs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            if costs[0].1 < best_cost {
                best_sol = Some(population[costs[0].0].clone());
                best_cost = costs[0].1.clone();
                generations_since_improvement = 0;
                log::info!("Found better cost: {}", best_cost);
            } else {
                generations_since_improvement += 1;
                if generations_since_improvement > Self::CONVERGENCE_GENERATIONS {
                    log::info!(
                        "Haven't found better solution in {} generations, stopping",
                        Self::CONVERGENCE_GENERATIONS,
                    );
                    break;
                }
            }

            let ranked: Vec<_> = costs.iter().map(|x| &population[x.0]).collect();
            let mut next_population = Vec::new();

            while next_population.len() < Self::POPULATION_SIZE {
                let parent1 = ranked[self.selection_dist.sample(&mut self.rng)];
                let parent2 = ranked[self.selection_dist.sample(&mut self.rng)];
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

        (best_cost, best_sol, pareto_points, generation)
    }

    fn crossover(&mut self, a: &Solution, b: &Solution) -> (Solution, Solution) {
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

    fn mutate(&mut self, sol: &mut Solution) {
        for (i, x) in sol.iter_mut() {
            if self.mutation_dist.sample(&mut self.rng) {
                *x = self.rng.random_range(0..self.eclass_bounds[i]);
            }
        }
    }
}

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
) -> CF::Cost
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

    costs[&root].clone()
}
