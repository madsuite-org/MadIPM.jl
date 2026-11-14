module MadIPM

using Printf
using LinearAlgebra
import SparseArrays
import MadCore
import MadCore: full
# TODO(M0): MadCore names MadIPM uses that are not in `MadCore.public_names()`. They are
# called qualified (`MadCore._name`); each should become public with a documented
# signature, or stop being needed once the batch layer moves to MadCore.
#   _parse_indexes                     batch/madnlp/callback.jl
#   _treat_equality_initialize!        batch/madnlp/initialization.jl
#   _treat_fixed_variable_initialize!  batch/madnlp/initialization.jl
#   _initialize_variables!             batch/madnlp/initialization.jl
#   _get_primary_options               utils.jl
#   _set_aug_diagonal!                 kernels.jl
# Also missing from MadCore: the status UNBOUNDED_PROBLEM_DETECTED (solver.jl).
import LDLFactorizations
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

MadCore.madsuite(::Val{:madipm}, args...; kwargs...) = madipm(args...; kwargs...)

end # module MadIPM
