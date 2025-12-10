#import "@preview/charged-ieee:0.1.4": ieee
#import "@preview/cetz:0.4.2"
#import "@preview/simplebnf:0.1.1": *

#show: ieee.with(
  title: [fuzzy-eqsat: Equality saturation with approximations],
  abstract: [
    #lorem(100)
  ],
  authors: (
    (
      name: "Jeff Zhang",
      organization: [McGill University],
      location: [Montreal, Canada],
    ),
  ),
  // index-terms: ("Scientific writing", "Typesetting", "Document creation", "Syntax"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

#set text(
  size: 11pt,
)
#show raw: set text(font: "FiraCode Nerd Font", size: 8pt)
#show figure.caption: set align(center)
#set math.equation(numbering: none)

#let small-neural-net = cetz.canvas(length: 5pt, {
  import cetz.draw: *

  // Layer configuration
  let (n-input-neurons, n-hidden-neurons) = (5, 8)
  let (input-len, hidden-len) = (20, 30)

  // Draw neurons
  on-layer(1, {
    // Input layer
    for ii in range(n-input-neurons) {
      circle(
        (0, ii * input-len / (n-input-neurons - 1)),
        radius: 1,
        fill: none,
        stroke: black,
        name: "in-" + str(ii),
      )
    }
    // Hidden layer
    for ii in range(n-hidden-neurons) {
      let y-offset = (input-len - hidden-len) / 2
      circle(
        (10, ii * hidden-len / (n-hidden-neurons - 1) + y-offset),
        radius: 1,
        fill: none,
        stroke: black,
        name: "hidden-" + str(ii),
      )
    }
    // Output layer
    for ii in range(n-input-neurons) {
      circle(
        (20, ii * input-len / (n-input-neurons - 1)),
        radius: 1,
        fill: none,
        stroke: black,
        name: "out-" + str(ii),
      )
    }
  })

  // Draw connections
  on-layer(0, {
    for ii in range(n-input-neurons) {
      for jj in range(n-hidden-neurons) {
        line("in-" + str(ii), "hidden-" + str(jj), stroke: red.lighten(50%))
      }
    }
    for ii in range(n-hidden-neurons) {
      for jj in range(n-input-neurons) {
        line("hidden-" + str(ii), "out-" + str(jj), stroke: blue.lighten(50%))
      }
    }
  })
})

#let ga-crossover = cetz.canvas(length: 5pt, {
  import cetz.draw: *

  let cell-size = 3.5
  let n-cells = 8
  let crossover_point = 3

  let draw-cell(pos, value, fill: none) = {
    rect(
      pos,
      (pos.at(0) + cell-size, pos.at(1) + cell-size),
      fill: fill,
      stroke: .5pt,
    )
    if value != none {
      content(
        (pos.at(0) + cell-size / 2, pos.at(1) + cell-size / 2),
        $#value$,
      )
    }
  }

  content((12, 6), "Parents", anchor: "north")

  for i in range(n-cells) {
    draw-cell((i * cell-size - 2, 0), text(blue)[$a_#(i + 1)$])
  }

  for i in range(n-cells) {
    draw-cell((i * cell-size - 2, -5), text(green)[$b_#(i + 1)$])
  }

  content((12, -6), "Children", anchor: "north")

  for i in range(n-cells) {
    let cell-content = if i < crossover_point {
      text(blue)[$a_#(i + 1)$]
    } else {
      text(green)[$b_#(i + 1)$]
    }
    draw-cell((i * cell-size - 2, -12), cell-content)
  }

  for i in range(n-cells) {
    let cell-content = if i < crossover_point {
      text(green)[$b_#(i + 1)$]
    } else {
      text(blue)[$a_#(i + 1)$]
    }
    draw-cell((i * cell-size - 2, -17), cell-content)
  }
})

= Introduction
This paper explores the feasibility of introducing rewrite rules representing *approximations*
for linear algebra operations, such as matrix multiplication, using equality saturation. The concept of compilers automatically
inserting approximations for operations, trading precision for performance, is not new. A prominent
example of this is the `-ffast-math` flag in GCC #cite(<gcc-ffast-math>), which speeds up floating point operations by allowing
the compiler to make many assumptions that do not respect the IEEE standard.

The approach explored in this paper requires that values in the expression being optimized
are known ahead of time, which is highly restrictive. Nevertheless, this remains applicable for speeding up
neural network inference, as the model parameters are all known after training is complete.

= Background
== Equality saturation
The fundamental data structure used in equality saturation is the *e-graph*.
E-graphs are a graph data structure that contain a set of *e-classes* uniquely identified by an ID.
E-classes are equivalence classes each containing a set of equivalent *e-nodes*.
E-nodes represent a term in the language, with child terms being e-class IDs.

An e-graph compactly represents a very large number of equivalent expressions for a given expression subject to
a set of *rewrite rules*.
For example, we might have a rewrite rule that replaces division by 2 by a bitshift:
$ x / 2 -> x << 1 $

*[Add diagram]*

Equality saturation works by repeatedly applying these rewrite rules until:
1. The graph is saturated (applying further rewrites would not result in any changes)
2. A timeout is reached

Note that a saturated e-graph represents all possible equivalent expressions reachable from the initial expression
using the given rewrite rules.

Then, the best possible expression can be extracted from the e-graph according a specific *cost function* which outputs
a cost value given an e-node. However, the extraction problem is known to be NP-Hard #cite(<stepp2011>), as such
computing an exact solution is intractable for large e-graphs.

Common solutions are greedy approximation algorithm (taking the minimum cost e-node from each
e-class) and integer linear programming (computes an exact solution, but can be slow).

Due to the constraints of the cost model chosen in this paper, a genetic algorithm was used for extraction.


== Genetic algorithms
A genetic algorithm is a heuristic search method for optimization problems
inspired by the biological process of evolution.

The set of parameters that encode a solution are called a *chromosome*, and a single
parameter is called a *gene*.
It works by randomly generating a *population* of initial solutions.
These solutions are evaluated according to some *fitness function*.
Then, the best solutions are selected to "reproduce" by combining their chromosomes
via some *crossover* method to produce the next population, with the hope being that combining two good solutions
will produce an even better solution.
Finally, random *mutation* is applied gene by gene in the new offspring to introduce more diversity in the solutions. #cite(<mitchell1998ga>)

This process is repeated for some number of iterations, and the best solution is returned.

= Methods <sec:methods>
This approach to equality saturation comes with the restriction that all variable values in the expression are known
ahead of time, since otherwise it is not possible to compute a relative error for approximations. The method was applied
to the problem of neural network optimization for inference on two classification models.

1. A simple multilayer perception with 1 hidden layer
2. LeNet 5 #cite(<lenet5>), one of the earliest convolutional neural networks

Both of these models are trained on the task of handwritten digit recognition using the MNIST dataset #cite(<mnist>).

== Representation
The two neural network architectures used are both feedforward neural networks, as such they can easily
be converted into a single expression for optimization.

#figure(
  placement: none,
  small-neural-net,
  caption: [A multilayer perception with 1 hidden layer.],
) <fig:nn>

`fuzzy-eqsat` uses the `egg` equality saturation library written in Rust, with the following grammar
for expression rewrites.

#bnf(
  Prod(
    $n$,
    annot: $sans("Num")$,
    delim: none,
  ),
  [_integer literals_],
)
#bnf(
  Prod(
    $e$,
    annot: $sans("Expr")$,
    {
      Or[$x$][_variable_]
      Or[$e + e$][_matrix addition_]
      Or[$e * e$][_matrix multiplication_]
      Or[$"svd_u"(x, n)$][$U$ _matrix of truncated SVD_]
      Or[$"svd_d"(x, n)$][#sym.Sigma _matrix of truncated SVD_]
      Or[$"svd_vt"(x, n)$][$V^T$ _matrix of truncated SVD_]
      Or[$"ReLU"(e)$][_ReLU function_]
      Or[$tanh(e)$][_hyperbolic tangent function_]
      Or[$"softmax"(e)$][_softmax function_]
    },
  ),
)

The network shown in @fig:nn matches the architecture of the first classification model tested, albeit with fewer
neurons. In `egg`, which uses an s-expression syntax, this network corresponds to the expression:

$
  (#text("softmax") (+ space (* space w_1 space (#text("ReLU") space (+ space (* space w_0 space x) space b_0)) space b_1))
$

where $w_i$ and $b_i$ correspond to the weight matrix and bias vector for the $i$-th layer respectively,
and $x$ corresponds to the model input.

The model parameters are constants, but the model input by definition can't be known ahead of time. This can be resolved
by treating $x$ as a random variable which is defined by some distribution. The value assigned to $x$ is the test set matrix from
the classification problem's data set, since this approximates the distribution of possible model inputs.

== Rewrite rules
The two approximation rewrite rules used were truncated singular value decomposition (SVD) and pruning. In addition to these,
a basic rewrite rule for matrix multiplication associativity is also applied since this can give small cost reductions for free:
$ A (B C) -> (A B) C $

=== *Truncated SVD*
The singular value decomposition of an $m times n$ matrix $A$ is a decomposition of the form
$ A = U Sigma V^T $
where $U$ is an $m times r$ matrix, $Sigma$ is an $r times r$ diagonal matrix, and $V$ is an $n times r$ matrix, and $r = min(m, n)$.

The truncated SVD is obtained by taking only the first $k$ columns of each matrix, for some $1 <= k < r$, giving us an approximation for $A$

$ A_k = U_k Sigma_k V^T_k $

This gives the *best rank-k* approximation for $A$ by the Eckart-Young theorem.

Given a matrix multiplication $A B$, we rewrite $A$ into its truncated SVD.

$ A B -> U_k Sigma_k V^T_k B $

If A is $m times n$, and B is $n times p$, then the total cost is $m n p$ multiplications. When multiplying by the truncated
SVD instead, the total cost is
- $V^T_k B$: $k n p$ multiplications
- $Sigma_k (V^T_k B)$: $k p$ multiplications since $Sigma_k$ is diagonal
- $U_k (Sigma_k (V^T_k B))$: $k m p$ multiplications

$k n p + k p + k m p$ can be much smaller than $m n p$ if $k$ is sufficiently small.
In the rewrite implementation, the truncated SVD for several $k$ are inserted in a linearly decreasing manner.
The step size is equal to $floor(log_2(min(m, n)))$. For example, for a $784 times 100$ matrix, $floor(log_2(min(784, 100))) = 6$, thus
the truncated SVDs for $k = 94, 88, 82, ...$ are merged into the node's eclass. This rewrite is only applied once per e-class because
since applying an SVD multiple times is redundant. This rewrite is also only applied to terms whose values can be
known ahead of time, such as plain variables. Note that this means the cost of the svd_u, svd_d, svd_vt terms in the language is 0,
because the decomposition is computed ahead of time during equality saturation.

=== *Pruning*
Pruning is the process of truncating small values in a matrix to 0.

If $Z$ is the amount of zeroes in matrix $A$, then to compute $A B$ where $A$ is $m times n$, and $B$ is $n times p$,
the number of multiplications required is $(m n - Z)p$ since we don't have to consider the zero entries in $A$.

The pruning rewrite rule is applied to terms that can be known ahead of time such as plain variables or terms of their SVD. It simply
introduces a new variable for the pruned value. This is done instead of adding a prune term to the language
because the rewrite

$ A -> "prune"(A, k) $

introduces a cycle into the e-graph, which would complicate extraction significantly.
The actual rewrite is

$ A -> P_(i, k) $

where $P_(i, k)$ is a fresh variable name that has the value of matrix $A$ where all $x$ in $A$ such that $x < 10^k$ are truncated to 0.
Similar to the truncated SVD, multiple pruning thresholds are inserted into the eclass. The values of $k$ chosen were -3, -2, and -1.

== Cost model
The cost model used differs from typical equality saturation in many ways.

`egg` provides functionality to attach arbitrary data to eclasses during e-graph saturation via the `Analysis` trait.

First, during the analysis phase, a _canonical_ expression value is computed for each eclass,
which is derived from the variable values.
The root e-class's canonical value corresponds to the actual model output given the test set.

The cost function implemented not only outputs an integer cost, but also the actual value of the e-node and its relative error
compared to the e-class's canonical value.
The relative error $eta$ between the canonical value $A$ and its approximation $A'$ is computed in terms of the Frobenius norm:

$ eta = (||A - A'||_F)/(||A||_F) $

The cost function itself can be configured by passing a maximum allowed relative error. The ordering between costs
is determined using this value.

#figure(
  ```rust
  impl PartialOrd for CostWithErrorBound {
      fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
          if self.error > self.max_rel_error && other.error > self.max_rel_error {
              self.error.partial_cmp(&other.error)
          } else if a_err > self.max_rel_error {
              Some(Ordering::Greater)
          } else if b_err > self.max_rel_error {
              Some(Ordering::Less)
          } else {
              self.cost.partial_cmp(&other.cost)
          }
      }
  }
  ```,
)

== Genetic algorithm implementation
The e-graph extraction problem can be formulated as follows: Given an e-graph, pick
an enode from each e-class so that the resulting expression's cost is minimized.

*Chromosome*: A solution to the extraction problem is simply a mapping from e-class IDs to indices,
with the indices corresponding to a specific e-node in that e-class. This is represented using a hashmap.

*Population size*: The population size used was 50, which is somewhat small however the cost function is quite
expensive to compute due to the size of the matrices. 50 was chosen to strike balance between
extraction time and quality of solutions produced.

*Crossover*: Crossover is done using simple single point crossover. The 2 parent solutions are flattened
into an array, and a random index is chosen to be the crossover point. The two slices past the crossover point
are then swapped, producing 2 children.

#figure(
  placement: none,
  ga-crossover,
  caption: [Single point crossover.],
) <fig:ga-crossover>

*Mutation*: The mutation rate chosen was 5%, mutation simply randomly selects a different index
for an e-class.

= Experiments

= Results & Discussion

= Related Work

= Conclusion

