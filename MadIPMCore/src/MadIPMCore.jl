"""
    MadIPMCore

The interior-point algorithm of MadIPM (Mehrotra predictor-corrector) for linear and quadratic
programs, on the KKT systems and linear solver interface of MadCore. It loads no linear solver:
the bundles `MadIPM` (CPU) and `cuMadIPM` (NVIDIA GPUs) load them and set
[`default_linear_solver`](@ref).
"""
module MadIPMCore

using Printf
using LinearAlgebra
import SparseArrays
import MadCore
import MadCore: full
# Internal names of MadCore (not in `MadCore.public_names()`) used by MadIPMCore.
import MadCore: create_linear_solver, is_factorized
import NLPModels
import SparseMatricesCOO: SparseMatrixCOO

include("models/Models.jl")
import .Models
import .Models:
    LinearModel,
    QuadraticModel,
    LPData,
    QPData,
    ObjRHSBatchQuadraticModel,
    batch_spmv!,
    sync_batch_operator!,
    batch_mapreduce!,
    batch_maximum!,
    operator_sparse_matrix
import .Models.Presolve:
    AbstractPresolver,
    BasicPresolver,
    NoPresolver,
    PRESOLVE_REDUCED,
    PRESOLVE_UNCHANGED,
    PRESOLVE_INFEASIBLE,
    PRESOLVE_UNBOUNDED,
    PRESOLVE_UNBOUNDED_OR_INFEASIBLE,
    PRESOLVE_SOLVED,
    apply_presolve,
    recover_solution

include("utils.jl")
include("structure.jl")
include("kernels.jl")
include("KKT/normalkkt.jl")
include("linear_solver.jl")
include("solver.jl")

include("batch/utils.jl")
include("batch/views.jl")
include("batch/madnlp/rhs.jl")
include("batch/madnlp/callback.jl")
include("batch/KKT/KKT.jl")
include("batch/structure.jl")
include("batch/madnlp/linear_solver.jl")
include("batch/madnlp/kernels.jl")
include("batch/madnlp/initialization.jl")
include("batch/madnlp/nlpmodels.jl")
include("batch/madipm/kernels.jl")
include("batch/madipm/solver.jl")

export MPCSolver, madipm, madipm_batch

"""
    default_linear_solver(nlp)

The linear solver `MPCSolver` uses for `nlp` when the option `linear_solver` is not given.
MadIPMCore falls back to `MadCore.default_sparse_solver(nlp)`; the bundles `MadIPM` and
`cuMadIPM` add methods for their array types.
"""
default_linear_solver(nlp) = MadCore.default_sparse_solver(nlp)

"""
    default_batch_linear_solver(bnlp)

The linear solver `UniformBatchMPCSolver` (and `madipm_batch`) uses for the batch model `bnlp`
when the option `uniformbatch_linear_solver` is not given. MadIPMCore falls back to
`LoopedBatchLinearSolver`, which factorizes each instance with `looped_linear_solver`; the
bundle `cuMadIPM` adds a method for models on the GPU.
"""
default_batch_linear_solver(bnlp) = LoopedBatchLinearSolver

MadCore.madsuite(::Val{:madipm}, args...; kwargs...) = madipm(args...; kwargs...)

end # module MadIPMCore
