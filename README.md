# MadIPM.jl

MadIPM.jl is a GPU-accelerated optimization solver for linear and quadratic programming.
The solver implements the Mehrotra predictor-corrector method in pure Julia,
and supports the solution of large-scale linear programs on the GPU using NVIDIA cuDSS.

## Installation

MadIPM can be installed and tested through the Julia package manager:

```julia
julia> ]
pkg> add MadIPM
pkg> test MadIPM
```

The repository holds three packages, like MadNLP.jl:

| Package | What it is |
|---|---|
| `MadIPM` | MadIPM on the CPU, with the CPU linear solvers of MadNLP (MUMPS by default) and the JuMP/MathOptInterface interface |
| `cuMadIPM` | MadIPM on NVIDIA GPUs, with cuDSS by default |
| `MadIPMCore` | The algorithm alone, for packages built on MadIPM; it loads no linear solver |

## Basic usage

### JuMP

MadIPM supports JuMP models with an extension for `MathOptInterface.jl`.
For instance, you can solve any LP formulated with JuMP by using:

```julia
using JuMP
using MadIPM

c = rand(10)
model = Model(MadIPM.Optimizer)
@variable(model, 0 <= x[1:10], start=0.5)
@constraint(model, sum(x) == 1.0)
@objective(model, Min, c' * x)
JuMP.optimize!(model)
```

### QuadraticModels

We detail here how to solve a LP stored in a MPS file `mylp.mps` using [QPSReader](https://github.com/JuliaSmoothOptimizers/QPSReader.jl) and [QuadraticModels](https://github.com/JuliaSmoothOptimizers/QuadraticModels.jl).

```julia
using QPSReader
using QuadraticModels
using MadIPM

qpdat = readqps("mylp.mps")
qp = QuadraticModel(qpdat)
results = madipm(qp)
```

MadIPM uses MUMPS by default; pass the option `linear_solver` to use another solver, for example `linear_solver = MadCore.LDLSolver` after `import MadCore`.

### Custom usage

MadIPM takes as input any linear program (LP) or quadratic program (QP) represented as an `AbstractNLPModel`,
following the specification in [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl/).

For any `qp <: AbstractNLPModel`, you can pass it to MadIPM either directly with `madipm(qp)`, or in two steps as follows:

```julia
solver = MPCSolver(qp)
results = MadIPM.solve!(solver)
```

## Solving a LP with CUDA

MadIPM runs on NVIDIA GPUs with the package `cuMadIPM`, which uses NVIDIA cuDSS by default.
It requires specifying your problem in a `QuadraticProblem` first.

The data are moved to the GPU using:
```julia
using cuMadIPM, CUDA

qp_gpu = convert(QuadraticModel{Float64, CuVector{Float64}}, qp)
```
Then, you can pass the problem `qp_gpu` to MadIPM; models on the GPU use cuDSS:
```julia
results = madipm(qp_gpu)
```
As a result, all the solution happens on the GPU, with minimum data transfer
between the host and the device.

If you have a JUMP model, just set the array type for CUDA arrays:
```julia
using JuMP
using MadIPM, cuMadIPM, CUDA

c = rand(10)
model = Model(MadIPM.Optimizer)
set_optimizer_attribute(model, "array_type", CuVector{Float64})
set_optimizer_attribute(model, "linear_solver", cuMadIPM.CUDSSSolver)

@variable(model, 0 <= x[1:10], start=0.5)
@constraint(model, sum(x) == 1.0)
@objective(model, Min, c' * x)

JuMP.optimize!(model)
```

## Citing MadIPM.jl

If you use MadIPM.jl in your research, we would greatly appreciate your citing it.

```bibtex
@article{MadIPM,
  title   = {{GPU Implementation of Second-Order Linear and Nonlinear Programming Solvers}},
  author  = {Montoison, Alexis and Pacaud, Fran{\c{c}}ois and Shin, Sungho and Anitescu, Mihai},
  journal = {arXiv preprint arXiv:2508.16094},
  year    = {2025}
}
```
