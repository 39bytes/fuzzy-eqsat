#import "@preview/charged-ieee:0.1.4": ieee
#import "@preview/cetz:0.4.2"
#import "@preview/simplebnf:0.1.1": *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import fletcher.shapes: diamond


#show: ieee.with(
  title: [fuzzy-eqsat: Equality saturation with approximations],
  abstract: [
    We present `fuzzy-eqsat`, a novel form of equality saturation that employs approximate rewrite
    rules to optimize programs at the cost of accuracy.
    Namely, we introduce truncated singular value decomposition and matrix pruning rewrites in order to
    speed up matrix multiplication.
    This paper evalutes the efficacy of the approach on two basic neural network classifiers for handwritten
    digit classification using the MNIST dataset.
    The experiments show large theoretical speedups for only a slight accuracy penalty.
  ],
  authors: (
    (
      name: "Jeff Zhang",
      organization: [McGill University],
      location: [Montreal, Canada],
    ),
  ),
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
        line(
          "in-" + str(ii),
          "hidden-" + str(jj),
          stroke: 1pt + gray.lighten(30%),
        )
      }
    }
    for ii in range(n-hidden-neurons) {
      for jj in range(n-input-neurons) {
        line(
          "hidden-" + str(ii),
          "out-" + str(jj),
          stroke: 1pt + gray.lighten(30%),
        )
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
are known ahead of time, which is highly restrictive. Nevertheless, we explore its applicability to the task of optimizing
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

#let eclass(enclose, name: none) = {
  node(
    enclose: enclose,
    stroke: (paint: gray, thickness: 1pt, dash: "dashed"),
    corner-radius: 5pt,
    snap: -1, // prioritise other nodes when auto-snapping
    name: name,
  )
}

#figure(
  diagram(
    node-stroke: 0.6pt,
    node-fill: white,
    node((1, 0), [$slash$], name: <div>, radius: 4mm),
    node((1.75, 0), [$>>$], name: <bitshift>, radius: 4mm),
    eclass((<div>, <bitshift>), name: <div-eclass>),
    node((1.75, -1), [$+$], name: <plus>, radius: 4mm),
    eclass((<plus>), name: <root-eclass>),
    node((0.5, 1), [$x$], name: <x>, radius: 4mm),
    eclass((<x>), name: <x-eclass>),
    node((1.5, 1), [$2$], name: <two>, radius: 4mm),
    eclass((<two>), name: <two-eclass>),
    node((2.5, 1), [$1$], name: <one>, radius: 4mm),
    eclass((<one>), name: <one-eclass>),
    edge(<div>, <x-eclass>, "->"),
    edge(<div>, <two-eclass>, "->"),
    edge(<bitshift>, <one-eclass>, "->"),
    edge(<bitshift>, <x-eclass>, "->"),
    edge(<plus>, <div-eclass>, "->"),
    edge(<plus>, <one-eclass>, "->", bend: 20deg),
  ),
  caption: [E-graph for the expression $x/2 + 1$ with the rewrite $x/2 -> x << 1$ applied. Dotted boxes represent e-classes, and circles represent e-nodes.],
)

Equality saturation works by repeatedly applying these rewrite rules until:
1. The graph is saturated (applying further rewrites would not result in any changes)
2. A timeout is reached

Note that a saturated e-graph represents all possible equivalent expressions reachable from the initial expression
using the given rewrite rules.

Then, the best possible expression can be extracted from the e-graph according a specific *cost function* which outputs
a cost value given an e-node. However, the extraction problem is known to be NP-Hard #cite(<stepp2011>), as such
computing an exact solution is intractable for large e-graphs.

Common solutions are a simple bottom-up greedy approximation algorithm #cite(<2021-egg>) which takes the minimum cost e-node from each
e-class, and integer linear programming #cite(<ilpextract1>) #cite(<ilpextract2>) which computes an exact solution, but can be very slow on large e-graphs.

Neither of these approaches are compatible with the cost model used in this paper, so we employ a genetic algorithm to tackle the extraction problem for ease of implementation.


== Genetic algorithms
A genetic algorithm is a heuristic search method for optimization problems
inspired by the biological process of evolution.

The set of parameters that encode a solution are called a *chromosome*, and a single
parameter is called a *gene*.
It works by randomly generating a *population* of initial solutions.
These solutions are evaluated according to some *fitness function*.
Then, the best solutions are *selected* via some to "reproduce" by combining their chromosomes
via some *crossover* method to produce the next population, with the hope being that combining two good solutions
will produce an even better solution.
Finally, random *mutation* is applied gene by gene in the new offspring to introduce more diversity in the solutions. #cite(<mitchell1998ga>)

This process is repeated for some number of iterations, and the best solution is returned.

#let blob(pos, label, tint: white, ..args) = node(
  pos,
  align(center, label),
  width: 28mm,
  fill: tint.lighten(60%),
  stroke: 1pt + tint.darken(20%),
  corner-radius: 5pt,
  ..args,
)

#set text(10pt)
#figure(
  diagram(
    spacing: 8pt,
    cell-size: (8mm, 10mm),
    edge-stroke: 1pt,
    edge-corner-radius: 5pt,
    mark-scale: 70%,
    edge(
      (0, 0),
      "d",
      "-|>",
      `Initial population`,
      label-pos: 0,
      label-side: center,
    ),
    blob((0, 1), `Evaluate`),
    edge("-|>"),
    node(
      (0, 2),
      align(center)[`Done?`],
      width: 22mm,
      shape: diamond,
      stroke: 1pt + gray.darken(20%),
    ),
    edge((0, 2), (0, 4), "--|>", `No`),
    edge((0, 2), (2, 2), "--|>", `Yes`),
    blob(
      (2, 2),
      `Output best`,
    ),
    node(
      (0, 4),
      align(center)[`Next population full?`],
      width: 22mm,
      shape: diamond,
      stroke: 1pt + gray.darken(20%),
    ),
    edge((0, 4), (2, 4), `No`, "--|>"),
    edge((0, 4), "l,uuu,r", `Yes`, "--|>", label-side: left),
    blob(
      (2, 4),
      `Select parents`,
    ),
    edge((2, 4), (2, 5), "-|>"),
    blob((2, 5), `Crossover`),
    edge((2, 5), (2, 6), "-|>"),
    blob((2, 6), `Mutate children`),
    edge(
      (2, 6),
      "ll,uu",
      `Add children`,
      "-|>",
      label-pos: 0.75,
      label-side: left,
    ),
  ),
  caption: [Genetic algorithm flowchart.],
)


= Methods <sec:methods>
This approach to equality saturation comes with the restriction that all variable values in the expression are known
ahead of time, since otherwise it is not possible to compute a relative error for approximations. We use `fuzzy-eqsat` to
optimize inference for two classification models.

1. A simple multilayer perception with 1 hidden layer (784 x 100 x 10 neurons)
2. LeNet 5 #cite(<lenet5>), one of the earliest convolutional neural networks

Both of these models are trained on the task of handwritten digit recognition using the MNIST dataset #cite(<mnist>).

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

This gives the *best rank-k* approximation for $A$ by the Eckart-Young theorem #cite(<eckart-young>).

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
known ahead of time, such as plain variables.
An example of an e-graph with this rewrite applied is shown in @fig:svd-egraph.

=== *Pruning*
Pruning truncates small values in a matrix to 0. If $Z$ is the amount of zeroes in matrix $A$, then to compute $A B$ where $A$ is $m times n$, and $B$ is $n times p$,
the number of multiplications required is $(m n - Z)p$ since we don't have to consider the zero entries in $A$ when using a sparse matrix representation.

The pruning rewrite rule is applied to terms that can be known ahead of time such as plain variables or terms of their SVD. It simply
introduces a new variable for the pruned value. This is done instead of adding a prune term to the language
because the rewrite

$ A -> "prune"(A, k) $

introduces a cycle into the e-graph, which would complicate extraction significantly.
The actual rewrite is

$ A -> P_(i, k) $

where $P_(i, k)$ is a fresh variable name that has the value of matrix $A$ where all $x$ in $A$ such that $x < 10^k$ are truncated to 0.
Similar to the truncated SVD, multiple pruning thresholds are inserted into the eclass. The values of $k$ chosen were -3, -2, and -1.

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
      Or[$"diag_mul"(e_1, e_2)$][_diagonal matrix mul._]
      Or[$"svd_u"(x, n)$][$U$ _matrix of truncated SVD_]
      Or[$"svd_d"(x, n)$][#sym.Sigma _matrix of truncated SVD_]
      Or[$"svd_vt"(x, n)$][$V^T$ _matrix of truncated SVD_]
      Or[$"ReLU"(e)$][_ReLU function_]
      Or[$tanh(e)$][_hyperbolic tangent function_]
      Or[$"softmax"(e)$][_softmax function_]
    },
  ),
)

The grammar includes standard matrix addition and multiplication,
`diag_mul` is a special case of matrix multiplication where $e_1$ is a diagonal matrix.
This is faster to compute than normal matrix multiplication so it is discounted in the cost model.

The `svd_u`, `svd_d` and `svd_vt` terms correspond to the $U_k, Sigma_k, V^T_k$ matrices of the truncated SVD respectively.
`ReLU`, `tanh`, and `softmax` are included since they are common activation functions, `ReLU` and `tanh` are applied element-wise, and `softmax` is applied column-wise to matrices.

The network shown in @fig:nn matches the architecture of the first classification model tested, albeit with fewer
neurons. In `egg`, which uses an s-expression syntax, this network corresponds to the expression:

$
  (#text("softmax") (+ space (* space w_1 space (#text("ReLU") space (+ space (* space w_0 space x) space b_0)) space b_1))
$

where $w_i$ and $b_i$ correspond to the weight matrix and bias vector for the $i$-th layer respectively,
and $x$ corresponds to the model input.

The model parameters are constants, but the model input evidently can't be known ahead of time. We resolve this
by treating $x$ as a random variable which is defined by some distribution. The value assigned to $x$ is the test set matrix from
the classification problem's data set, since this approximates the distribution of possible model inputs. This is functionally the same as
batch inference.

#figure(
  diagram(
    node-stroke: 0.6pt,
    node-fill: white,
    node((2.5, 0), [$*$], name: <mul1>, radius: 4mm),
    node((3.25, 0), [$*$], name: <mul2>, radius: 4mm),
    eclass((<mul1>, <mul2>), name: <root-eclass>),
    eclass((<mul1>, <mul2>), name: <root-eclass>),
    node((2, 2), [#text([svd_u], size: 7pt)], name: <svdu>, radius: 4mm),
    eclass((<svdu>), name: <svdu-eclass>),
    node((3, 2), [#text([svd_d], size: 7pt)], name: <svdd>, radius: 4mm),
    eclass((<svdd>), name: <svdd-eclass>),
    node((4, 2), [$*$], name: <mul3>, radius: 4mm),
    eclass((<mul3>), name: <mul3-eclass>),
    node((4, 3), [#text([svd_vt], size: 7pt)], name: <svdvt>, radius: 4mm),
    eclass((<svdvt>), name: <svdvt-eclass>),
    node((3, 3), [$a$], name: <a>, radius: 4mm),
    eclass((<a>), name: <a-eclass>),
    node((5, 3), [$b$], name: <b>, radius: 4mm),
    eclass((<b>), name: <b-eclass>),
    node((3, 4), [$5$], name: <five>, radius: 4mm),
    eclass((<five>), name: <five-eclass>),
    node(
      (3, 1),
      [#text([diag_mul], size: 5pt)],
      name: <diagmul>,
      radius: 4mm,
    ),
    eclass((<diagmul>), name: <diagmul-eclass>),
    edge(<svdu>, <a-eclass>, "->"),
    edge(<svdd>, <a-eclass>, "->"),
    edge(<svdvt>, <a-eclass>, "->"),
    edge(<diagmul>, <svdd-eclass>, "->"),
    edge(<diagmul>, <mul3-eclass>, "->"),

    edge(<mul1>, <svdu-eclass>, "->"),
    edge(<mul1>, <diagmul-eclass>, "->"),
    edge(<mul3>, <svdvt-eclass>, "->"),
    edge(<mul3>, <b-eclass>, "->"),
    edge(<svdu>, <five-eclass>, "->"),
    edge(<svdd>, <five-eclass>, "->", bend: -60deg),
    edge(<svdvt>, <five-eclass>, "->"),
    edge(<mul2>, <a-eclass>, "->", bend: 30deg),
    edge(<mul2>, <b-eclass>, "->", bend: 20deg),
  ),
  caption: [E-graph for the expression (\* a b) after applying the truncated SVD rewrite, with $k=5$],
) <fig:svd-egraph>

== Cost model
The cost model used differs from typical equality saturation in many ways.

`egg` provides functionality to attach arbitrary data to eclasses during e-graph saturation via the `Analysis` trait, we leverage this to
compute a _canonical_ expression value is computed for each eclass, which is derived from the variable values.
The root e-class's canonical value corresponds to the actual model output given the test set.

The cost function implemented not only outputs an integer cost, but also the actual value of the e-node and its relative error
compared to the e-class's canonical value.
The relative error $eta$ between the canonical value $A$ and its approximation $A'$ is computed in terms of the Frobenius norm:

$ eta = (||A - A'||_F)/(||A||_F) $

The cost function itself can be configured by passing a maximum allowed relative error $E_max$. The ordering between costs
is in part determined using this value.
Specifically, let $C_a$ and $C_b$ be the integer cost values for enodes $a$ and $b$, and $E_a$ and $E_b$ be their respective errors.
An ordering for the costs of $a$ and $b$ are determined as follows:
- If $E_a$ > $E_max$ and $E_b > E_max$ then compare by $E_a$ and $E_b$
- Otherwise if only one of $E_a$ or $E_b$ exceed $E_max$ then the one that does not exceed $E_max$ is smaller.
- Otherwise compare by $C_a$ and $C_b$.


== Genetic algorithm implementation
The e-graph extraction problem can be formulated as follows: Given an e-graph, pick
an enode from each e-class so that the resulting expression's cost is minimized.

*Chromosome*: A solution to the extraction problem is simply a mapping from e-class IDs to indices,
with the indices corresponding to a specific e-node in that e-class. This is represented using a hashmap.

*Population size*: The population size used was 100, which is somewhat small however the cost function is quite
expensive to compute due to the size of the matrices. 100 was chosen to strike a balance between
extraction time and quality of solutions produced.

*Convergence*: The algorithm stops once 5 iterations have passed without finding a new best solution,
or after a maximum of 20 total iterations.

*Selection*: Since costs only have a relative order and not a single fitness value, rank-based selection
is most suitable.
Exponential rank selection is employed to more heavily favor the best solutions.
The probability of choosing rank $i$ (where rank 1 is the best solution) is given by

$ P(i) = (0.9^(i-1))/(sum_(k=1)^n 0.9^(k-1)) $

where $n$ is the population size (in this case 100).

*Crossover*: Crossover is done using simple single point crossover. The 2 parent solutions are flattened
into an array, and a random index is chosen to be the crossover point. The two slices past the crossover point
are then swapped, producing 2 children, as shown in @fig:ga-crossover.

#figure(
  placement: none,
  ga-crossover,
  caption: [Single point crossover, with crossover point 3.],
) <fig:ga-crossover>

*Mutation*: The mutation rate chosen was 5%, mutation simply randomly selects a different index
for an e-class. This rate is rather arbitrary, but seems to produce good results.

= Experiments

Many different configurations were tested for both neural networks by:
- Varying the maximum allowed relative error (2%, 5%, 10%, 25%)
- Applying only the truncated SVD rewrite, only the pruning rewrite, and both rewrites

Extraction is run 3 times on each configuration, taking the best solution found out of the 3 runs,
in order to minimize noise due to the randomness inherent to genetic algorithms.

Note that in the case of LeNet 5, since the equality saturation includes no rewrites for
convolutions, the expression being optimized is only the last three fully connected layers of the network,
since those are the relevant parts that can be optimized. As such, the value used for the model input is
the output of the original model's convolutional layers on the test set.

The experiments were run on a Lenovo Thinkbook G16 with a AMD Ryzen 7 8845H, running Arch Linux 6.17.

#let results-table(results2, results5, results10, results25, caption: none) = {
  set table(stroke: none)

  table(
    columns: 13 * (auto,),
    table.vline(x: 0, start: 0),
    table.vline(x: 1, start: 0),
    table.vline(x: 5, start: 0),
    table.vline(x: 9, start: 0),
    table.vline(x: 13, start: 0),
    table.hline(y: 0, start: 0),
    table.hline(y: 1, start: 0),
    table.hline(y: 6, start: 0),
    "",
    table.cell(colspan: 4, [*Truncated SVD*]),
    table.cell(colspan: 4, [*Pruning*]),
    table.cell(colspan: 4, [*Truncated SVD + Pruning*]),
    table.header(
      [*Max rel. error*],
      [Cost],
      [Rel. error],
      [Rel. cost],
      [Accuracy],
      [Cost],
      [Rel. error],
      [Rel. cost],
      [Accuracy],
      [Cost],
      [Rel. error],
      [Rel. cost],
      [Accuracy],
    ),
    ..results2,
    ..results5,
    ..results10,
    ..results25,
  )
}
#let results-mlp-2 = (
  [*2%*],
  //
  "73790",
  "0.047%",
  "-7.3%",
  "97.02%",
  //
  "53806",
  "0.644%",
  "-32.42%",
  "97.03%",
  //
  "53806",
  "0.644%",
  "-32.42%",
  "97.03%",
)

#let results-mlp-5 = (
  [*5%*],
  "73790",
  "0.047%",
  "-7.3%",
  "97.02%",
  //
  "53806",
  "0.644%",
  "-32.42%",
  "97.03%",
  //
  "53806",
  "0.644%",
  "-32.42%",
  "97.03%",
  //
)

#let results-mlp-10 = (
  [*10%*],
  //
  "63170",
  "9.410%",
  "-20.66%",
  "97.03%",
  //
  "53806",
  "0.644%",
  "-32.42%",
  "97.03%",
  //
  "45424",
  "9.862%",
  "-42.95%",
  "97.05%",
)

#let results-mlp-25 = (
  [*25%*],
  //
  "36620",
  "23.70%",
  "-54.00%",
  "96.01%",
  //
  "53806",
  "0.644%",
  "-32.42%",
  "97.03%",
  //
  "24338",
  "24.42%",
  "-69.43%",
  "95.93%",
)

#figure(
  placement: bottom,
  scope: "parent",
  results-table(
    results-mlp-2,
    results-mlp-5,
    results-mlp-10,
    results-mlp-25,
  ),
  caption: [Cost and error of optimized MLP expression for each rewrite rule configuration\
    Base classification accuracy: 97.02% \
    Base cost: 79620 \
  ],
) <fig:mlp-results>

#let results-lenet-2 = (
  [*2%*],
  "55112",
  "1.988%",
  "-7.138%",
  "98.51%",
  //
  "54825",
  "1.463%",
  "-7.621%",
  "98.56%",
  //
  "54825",
  "1.480%",
  "-7.621%",
  "98.56%",
)

#let results-lenet-5 = (
  [*5%*],
  "32426",
  "4.950%",
  "-45.36%",
  "98.37%",
  //
  "50461",
  "4.027%",
  "-14.97%",
  "98.49%",
  //
  "29377",
  "4.925%",
  "-50.50%",
  "98.40%",
)

#let results-lenet-10 = (
  [*10%*],
  "16232",
  "9.305%",
  "-72.65%",
  "98.19%",
  //
  "20823",
  "8.780%",
  "-64.91%",
  "98.22%",
  //
  "14083",
  "9.434%",
  "-76.27%",
  "98.22%",
)

#let results-lenet-25 = (
  [*25%*],
  "12931",
  "23.13%",
  "-78.21%",
  "94.86%",
  //
  "20823",
  "9.058%",
  "-64.91%",
  "98.18%",
  //
  "9982",
  "22.16%",
  "-83.18%",
  "95.38%",
)

#figure(
  placement: top,
  scope: "parent",
  results-table(
    results-lenet-2,
    results-lenet-5,
    results-lenet-10,
    results-lenet-25,
  ),
  caption: [Metrics for optimized LeNet expression for each rewrite rule configuration\
    Base classification accuracy: 98.56% \
    Base cost: 59348 \
  ],
) <fig:lenet-results>

#place(
  top + center,
  scope: "parent",
  float: true,
  [#figure(
    grid(
      columns: (1fr, 1fr),
      // Two columns, each taking an equal fraction of the available space
      gutter: 1em,
      // Space between the columns
      image("mlp_plot.svg"), image("lenet_plot.svg"),
      text(size: 8pt, "a) MLP expression"),
      text(size: 8pt, "b) LeNet-5 expression"),
    ),
    caption: [
      Explored solutions for MLP and LeNet-5 with each rewrite configuration. \
      Points marked with a star and joined with a line mark the Pareto front.
    ],
  )<fig:solution-plots>],
)

= Discussion
Overall, `fuzzy-eqsat` is able to find good solutions for the two models that give significant reductions in cost without
sacrificing too much on model accuracy.

Looking at @fig:mlp-results, in the MLP case, classification accuracy sees no decrease at all until the relative error surpasses 10%, in some cases
even very slightly increasing, however this increase is insignificant. The truncated SVD rewrite by itself is fairly
effective, giving a 20.66% cost reduction without compromising on accuracy, and a 54% reduction with only a 1% accuracy loss.

When only applying pruning on the MLP, each error threshold gave exactly the same solution with a 32.42% cost reduction while only having
a relative error of 0.644%.

This indicates that there are many parameters in the MLP that have little impact on the final result, so pruning can be applied freely without tradeoffs.
Since the model is small and pruning does not make the e-graph much bigger, the solution space is quite small, which explains
why the same solution is obtained in all cases.
This can be seen in @fig:solution-plots a), there are only a few red points on the very left side.

The optimization is most effective when the two rewrite rules are combined, since a matrix can be decomposed into its SVD
and the resulting decomposition matrices can be pruned, further reducing cost. In the 2% and 5% max error cases, the solution
is the same as pruning by itself, however when the max error is relaxed to 10% or 25%, the extractor finds a solution
that is cheaper than either of the individual rewrite cases with similar accuracy.

In the case of LeNet-5, the accuracies in @fig:lenet-results show that this model is more sensitive to changes in the output,
as the accuracy begins to fall before reaching a 10% relative error unlike the MLP, and the model suffers heavy accuracy losses
of more than 3% when the relative error exceeds 20%.

Truncated SVD is notably more impactful in this case, giving a 72.65% reduction in cost while only incurring a 0.32% accuracy loss.
Pruning gives different solutions unlike the MLP case but with larger error, indicating that the parameters are more meaningful compared to the MLP.
This makes sense because LeNet-5 is a convolutional model and has fewer parameters than the MLP,
so each neuron activation "represents" more information.

Combining the two rewrites for LeNet-5 is noticeably less effective compared to the MLP, resulting only in a 4-5% cost reduction. Indeed,
@fig:solution-plots b) shows that the Pareto fronts for truncated SVD and truncated SVD + pruning are very close together, unlike the MLP case where there
is a sizable gap.

In all cases, it is evident that going past 10% error results in diminishing returns in terms of cost reduction. This can be seen in @fig:solution-plots b),
the Pareto front has a very steep downwards slope up to 10% error before flattening out quickly.
The MLP does not seem to hit these diminishing returns until around 40% error.

The scatter plot for the LeNet-5 case is significantly denser than the MLP plot, this is because the solution space is much larger
and thus more generations are required for convergence.

There is a clear pattern of vertical lines in both figures when only applying the truncated SVD rewrite (the blue points).
These are likely solutions where a small matrix is heavily truncated which strongly impacts the output while
not resulting a large cost decrease.

= Limitations

The most obvious limitation is the requirement that all values are known ahead of time. This greatly restricts the number of
possible usecases, though it is unclear how this requirement could be relaxed.

Additionally, the approach presented in this paper suffers from scaling issues, as more and more rewrite rules are introduced,
the number of e-classes in the e-graph explodes, leading to a very large solution space.
Due to the unreliable nature of genetic algorithms, it is likely that the extractor begins to fail to find good solutions,
and would require further tweaking of the parameters. The crossover method could also be improved
to be aware of the structure of the e-graph rather than simply swapping contiguous slices ordered
by e-class IDs.

Finally, the cost values of the optimized expressions is simply part of the model and may not actually be faster
in practice. Full performance data was not collected, but when running a few tests comparing the runtime of a model that was optimized using the
truncated SVD rewrite to the original model, the "optimized" model was slightly slower.

A variety of factors could be the cause of this, but the most likely explanation is the CPU cache.
A single matrix is stored in a contiguous block of memory which is very cache-friendly, while multiplying by the decomposed matrices
performs three separate matrix multiplications, resulting in more cache misses.

For similar reasons, pruning is unlikely to result in a real speedup unless the resulting matrix is sufficiently sparse.

The models tested in this paper are very small.
It is likely that the model being optimized would have to be much larger before a significant speedup could be observed.

= Related Work

Regarding introducing approximations into equality saturation, the idea appears to be novel and there is little to no
trace of it in the literature, likely due to challenges already discussed.

There has been a fair amount of research devoted to tackling the e-graph extraction problem.
Chen et al. #cite(<emorphic>) propose a solution based on simulated annealing with solution space pruning which achieves
impressive results on large e-graphs for hardware synthesis. Rui et al. #cite(<esaco>) take a similar approach also
based on simulated annealing, but combined with ant colony optimization.

Cai et al. #cite(<smoothe>) formulate the e-graph problem probabilistically, converting the discrete optimization problem into a continuous
differentiable form and applying gradient descent.

Both of these approaches could be promising alternatives to the genetic algorithm extractor used in this paper, though
the difficulty lies in coming up with a suitable cost model formulation for either of them to be usable since `fuzzy-eqsat`'s cost
model is rather unusual.

= Conclusion

This paper presents `fuzzy-eqsat` which introduces approximate rewrite rules to equality saturation, namely
truncated SVD and matrix pruning for linear algebra expressions. Experimental results show
large cost reductions on basic neural networks under the cost model used, though this does not lead to actual
runtime improvement in practice.

For future work, it would be interesting to rethink the cost model so that better extraction methods
could be used instead of genetic algorithms, improving reliability and scalability. Generalizing the truncated SVD rewrite to
tensor train decomposition #cite(<tensor-train>) could also be useful for applying this approach to larger models that have higher
dimensional parameter tensors.
