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
# TODO(M0): MadCore names MadIPMCore uses that are not in `MadCore.public_names()`. They are
# called qualified (`MadCore._name`); each should become public with a documented
# signature, or stop being needed once the batch layer moves to MadCore.
#   _parse_indexes                     batch/madnlp/callback.jl
#   _treat_equality_initialize!        batch/madnlp/initialization.jl
#   _treat_fixed_variable_initialize!  batch/madnlp/initialization.jl
#   _initialize_variables!             batch/madnlp/initialization.jl
#   _get_primary_options               utils.jl
#   _set_aug_diagonal!                 kernels.jl
# Also missing from MadCore: the status UNBOUNDED_PROBLEM_DETECTED (solver.jl).
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

MadCore.madsuite(::Val{:madipm}, args...; kwargs...) = madipm(args...; kwargs...)

end # module MadIPMCore
