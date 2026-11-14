"""
    cuMadIPM

MadIPM on NVIDIA GPUs: MadIPMCore with the CUDA sparse operators and kernels of MadIPM and the
GPU linear solvers of MadCoreGPU (cuDSS, cuSOLVER). Models on `CuVector` use cuDSS by default.
"""
module cuMadIPM

using Adapt
using LinearAlgebra
using SparseArrays
using NLPModels
using CUDA
using CUDA.CUSPARSE
using CUDSS
using KernelAbstractions
import Atomix
import SparseMatricesCOO: SparseMatrixCOO
import LinearAlgebra: BlasFloat, Symmetric, Transpose, mul!, tril
import MadCore
import MadCoreGPU
import MadIPMCore

import MadIPMCore: Models
import MadIPMCore.Models:
    AbstractSparseOperator, SparseOperator,
    BatchSparseOperator, HostBatchSparseOperator,
    ModelData, ScalarModel, LPData, QPData, LinearModel, QuadraticModel,
    BatchModel, BatchQuadraticModel, BatchLinearModel,
    sparse_operator, operator_sparse_matrix,
    batch_mapreduce!, sync_batch_operator!,
    _mul_jt!, _batch_spmv_impl!,
    _copy_sparse_structure!, _copy_sparse_values!,
    _sparse_structure, _sparse_values,
    _build_op,
    _adapt_batch_meta

# The cuDSS solver is implemented in the extension MadCoreGPUCUDSSExt and built with
# `MadCore.create_linear_solver(CUDSSSolver, A)`.
const CUDSSSolver = MadCoreGPU.CUDSSSolver

# cuMadIPM exports what MadIPMCore exports, and the GPU linear solvers.
using MadIPMCore
for name in names(MadIPMCore)
    name === :MadIPMCore && continue
    @eval export $name
end
using MadCoreGPU: LapackCUDASolver
export CUDSSSolver, LapackCUDASolver

# cuDSS is the default sparse linear solver for models on the GPU, as in cuMadNLP.
MadIPMCore.default_linear_solver(::NLPModels.AbstractNLPModel{T, <:CuVector{T}}) where {T} = CUDSSSolver

include("models/sparse_operator.jl")
include("models/scalar_models.jl")
include("models/mapreduce.jl")
include("models/batch_operator.jl")
include("models/batch_models.jl")
include("cuda_wrapper.jl")
include("cuda_batch_kernels.jl")

function MadIPMCore._csc_with_nzval(A::CUSPARSE.CuSparseMatrixCSC, nzval, n)
    return CUSPARSE.CuSparseMatrixCSC(A.colPtr, A.rowVal, nzval, (n, n))
end

#=
    Device constructors for SparseMatrixCOO sources
=#

function CUSPARSE.CuSparseMatrixCOO(A::SparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
    return CUSPARSE.CuSparseMatrixCOO{Tv, Ti}(
        CuVector(A.rows),
        CuVector(A.cols),
        CuVector(A.vals),
        size(A),
        nnz(A),
    )
end

function CUSPARSE.CuSparseMatrixCSR(A::SparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
    m, n = size(A)
    Ap, Ai, Ax = MadIPMCore.coo_to_csr(m, n, A.rows, A.cols, A.vals)
    return CUSPARSE.CuSparseMatrixCSR{Tv, Ti}(
        CuVector(Ap),
        CuVector(Ai),
        CuVector(Ax),
        size(A),
    )
end

end # module cuMadIPM
