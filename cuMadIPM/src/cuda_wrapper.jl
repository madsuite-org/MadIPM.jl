import LinearAlgebra: BlasFloat

function MadIPMCore._build_linear_solver(::Type{CUDSSSolver}, aug_com::SparseMatrixCSC{T, Cint}; opt) where T
    return MadCore.create_linear_solver(CUDSSSolver, CUSPARSE.CuSparseMatrixCSC(aug_com); opt = opt)
end

function MadCore.transfer!(
    dest::CUSPARSE.CuSparseMatrixCSC{Tv},
    src::MadCore.SparseMatrixCOO{Tv},
    map::CuVector{Int},
) where {Tv}
    return MadCore.transfer_values!(dest.nzVal, src.V, map)
end

function MadCore.compress_hessian!(
    kkt::MadCore.SparseKKTSystem{T,VT,MT},
) where {T,VT<:CuVector{T},MT<:CUSPARSE.CuSparseMatrixCSC{T,Int32}}
    MadCore.transfer!(kkt.hess_com, kkt.hess_raw, kkt.hess_csc_map)
end

function MadCore.compress_jacobian!(
    kkt::MadIPMCore.NormalKKTSystem{T,VT,MT},
) where {T,VT,MT<:CUSPARSE.CuSparseMatrixCSC{T,Int32}}
    n_slack = length(kkt.ind_ineq)
    kkt.A.V[end-n_slack+1:end] .= -1.0
    # Transfer to the matrix A stored in CSC format
    fill!(kkt.AT.nzVal, 0.0)
    kkt.AT.nzVal .= kkt.A.V[kkt.A_csr_map]
    return
end

function MadIPMCore.coo_to_csr(
    n_rows,
    n_cols,
    Ai::CuVector{Ti},
    Aj::CuVector{Ti},
    Ax::CuVector{Tv},
) where {Tv, Ti}
    @assert length(Ai) == length(Aj) == length(Ax)
    B = sparse(Ai, Aj, Ax, n_rows, n_cols; fmt=:csr)
    return (B.rowPtr, B.colVal, B.nzVal)
end



MadIPMCore.sparse_csc_format(::Type{<:CuArray}) = CuSparseMatrixCSC
MadIPMCore._colptr(A::CuSparseMatrixCSC) = A.colPtr
MadIPMCore._rowval(A::CuSparseMatrixCSC) = A.rowVal
MadIPMCore._nzval(A::CuSparseMatrixCSC) = A.nzVal

# we introduce a new constructor that takes the nzvals as a matrix explicitly
function MadIPMCore.create_batch_linear_solver(
    ::Type{CUDSSSolver},
    aug_com::CUSPARSE.CuSparseMatrixCSC{T,Cint},
    nzvals_mat::CuMatrix{T},
    n::Int;
    opt = MadCore.default_options(CUDSSSolver),
) where T
    batch_nzVal = vec(nzvals_mat)
    batch_aug_com = CUSPARSE.CuSparseMatrixCSC(
        aug_com.colPtr, aug_com.rowVal, batch_nzVal, size(aug_com),
    )
    solver = MadCore.create_linear_solver(CUDSSSolver, batch_aug_com; opt=opt)
    solver.tril.nzVal = batch_nzVal
    return solver
end

# The uniform batch solver on CUDA arrays is cuDSS.
function _check_cudss(s)
    (hasproperty(s, :inner) && s.inner isa CUDSS.CudssSolver) ||
        throw(ArgumentError("$(typeof(s)) does not support batch solves on CUDA arrays; use CUDSSSolver"))
    return
end

function MadIPMCore.factorize_active!(s::MadCore.AbstractLinearSolver, active::MadIPMCore.BatchView)
    _check_cudss(s)
    na = MadIPMCore.local_batch_size(active)
    # Recent cuDSS versions reject a new `ubatch_size` until the next analysis phase.
    if CUDSS.cudss_get(s.inner, "ubatch_size") != na
        CUDSS.cudss_set(s.inner, "ubatch_size", na)
        s.analyzed = false
    end
    MadCore.factorize!(s)
    return
end

function MadIPMCore.solve_active!(s::MadCore.AbstractLinearSolver{T}, rhs::CuMatrix{T}, active::MadIPMCore.BatchView) where T
    _check_cudss(s)
    na = MadIPMCore.local_batch_size(active)
    n = size(rhs, 1)
    rhs_active = unsafe_wrap(CuArray{T, 2}, pointer(rhs), (n, na))
    CUDSS.cudss_update(s.b_gpu, rhs_active)
    CUDSS.cudss_update(s.x_gpu, rhs_active)
    CUDSS.cudss("solve", s.inner, s.x_gpu, s.b_gpu, asynchronous=s.opt.cudss_asynchronous)
    return
end

# Disambiguation with the method for CPU and looped solvers.
MadIPMCore.solve_active!(s::MadIPMCore.LoopedBatchLinearSolver{T}, rhs::CuMatrix{T}, active::MadIPMCore.BatchView) where T =
    invoke(MadIPMCore.solve_active!, Tuple{MadIPMCore.LoopedBatchLinearSolver{T}, AbstractMatrix{T}, MadIPMCore.BatchView}, s, rhs, active)
