# Small LPs and QPs of the batch solver tests, shared by MadIPM/test/batch/solver.jl and the
# GPU batch tests of cuMadIPM. `QuadraticModel` is the helper of ../utils.jl.

_lp() = QuadraticModel(
    ones(2),
    Int[],
    Int[],
    Float64[];
    Arows = [1, 1],
    Acols = [1, 2],
    Avals = [1.0, 1.0],
    lcon = [1.0],
    ucon = [1.0],
    lvar = [0.0, 0.0],
    uvar = [Inf, Inf],
    x0 = ones(2),
)

_qp() = QuadraticModel(
    [1.0, -2.0, 0.5, 1.0],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [2.0, 1.0, 3.0, 1.5];
    Arows = [1, 1, 2, 2],
    Acols = [1, 2, 3, 4],
    Avals = [1.0, 1.0, 1.0, 1.0],
    lcon = [1.0, 0.5],
    ucon = [2.0, 1.5],
    lvar = zeros(4),
    uvar = fill(Inf, 4),
    x0 = ones(4),
)

_qp_ub() = QuadraticModel(
    [1.0, -1.0],
    [1, 2],
    [1, 2],
    [1.0, 1.0];
    Arows = [1, 1],
    Acols = [1, 2],
    Avals = [1.0, 1.0],
    lcon = [1.0],
    ucon = [1.0],
    lvar = [-Inf, -Inf],
    uvar = [5.0, 5.0],
    x0 = [2.5, 2.5],
)

_qp_db() = QuadraticModel(
    [1.0, -1.0],
    [1, 2],
    [1, 2],
    [1.0, 1.0];
    Arows = [1, 1],
    Acols = [1, 2],
    Avals = [1.0, 1.0],
    lcon = [1.0],
    ucon = [1.0],
    lvar = [0.0, 0.0],
    uvar = [5.0, 5.0],
    x0 = [2.5, 2.5],
)

_qp_free() = QuadraticModel(
    [1.0, -1.0],
    [1, 2],
    [1, 2],
    [1.0, 1.0];
    Arows = [1, 1],
    Acols = [1, 2],
    Avals = [1.0, 1.0],
    lcon = [1.0],
    ucon = [1.0],
    lvar = [-Inf, -Inf],
    uvar = [Inf, Inf],
    x0 = [0.5, 0.5],
)

_qp_scaled() = QuadraticModel(
    [1e3, -2e3, 5e2, 1e3],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [2e3, 1e3, 3e3, 1.5e3];
    Arows = [1, 1, 2, 2],
    Acols = [1, 2, 3, 4],
    Avals = [1e3, 1e3, 1e3, 1e3],
    lcon = [1e3, 5e2],
    ucon = [2e3, 1.5e3],
    lvar = zeros(4),
    uvar = fill(Inf, 4),
    x0 = ones(4),
)

_qp_fixed() = QuadraticModel(
    [1.0, 1.0, 1.0],
    [1, 2, 3],
    [1, 2, 3],
    [2.0, 1.0, 3.0];
    Arows = [1, 1],
    Acols = [1, 2],
    Avals = [1.0, 1.0],
    lcon = [1.0],
    ucon = [Inf],
    lvar = [0.0, 0.0, 2.0],
    uvar = [Inf, Inf, 2.0],
    x0 = [1.0, 1.0, 1.0],
)

_qp_mixed() = QuadraticModel(
    [1.0, -1.0, 0.5, 1.0],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [2.0, 1.0, 3.0, 1.5];
    Arows = [1, 1, 2, 2],
    Acols = [1, 2, 3, 4],
    Avals = [1.0, 1.0, 1.0, 1.0],
    lcon = [1.0, 0.5],
    ucon = [2.0, 1.5],
    lvar = [0.0, -Inf, 0.0, -Inf],
    uvar = [Inf, 5.0, 5.0, Inf],
    x0 = ones(4),
)

_qp_dense() = QuadraticModel(
    [1.0, -1.0],
    [1, 2, 2],
    [1, 1, 2],
    [4.0, 2.0, 3.0];
    Arows = [1, 1],
    Acols = [1, 2],
    Avals = [1.0, 1.0],
    lcon = [1.0],
    ucon = [1.0],
    lvar = [0.0, 0.0],
    uvar = [Inf, Inf],
    x0 = [0.5, 0.5],
)

_qp_A() = QuadraticModel(
    [1.0, -2.0, 0.5, 1.0],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [2.0, 1.0, 3.0, 1.5];
    Arows = [1, 1, 2, 2],
    Acols = [1, 2, 3, 4],
    Avals = [1.0, 1.0, 1.0, 1.0],
    lcon = [1.0, 0.5],
    ucon = [2.0, 1.5],
    lvar = zeros(4),
    uvar = fill(Inf, 4),
    x0 = ones(4),
)
_qp_B() = QuadraticModel(
    [2.0, 1.0, -1.0, 0.5],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [5.0, 2.0, 1.0, 4.0];
    Arows = [1, 1, 2, 2],
    Acols = [1, 2, 3, 4],
    Avals = [2.0, 1.0, 0.5, 1.0],
    lcon = [1.0, 0.5],
    ucon = [2.0, 1.5],
    lvar = zeros(4),
    uvar = fill(Inf, 4),
    x0 = ones(4),
)
