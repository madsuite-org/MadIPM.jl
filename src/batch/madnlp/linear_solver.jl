# FIXME: threads/polyester version

struct LoopedBatchLinearSolver{T,VT,LS<:MadCore.AbstractLinearSolver{T}} <:
       MadCore.AbstractLinearSolver{T}
    solvers::Vector{LS}
    batch_size::Int
end

@kwdef mutable struct LoopedBatchLinearSolverOptions{OT<:MadCore.AbstractOptions} <: MadCore.AbstractOptions
    looped_linear_solver::Type = MadCore.MumpsSolver
    # options of the solver of each instance, e.g. `mumps_pivtol`
    looped_linear_solver_options::OT = MadCore.default_options(looped_linear_solver)
end

MadCore.default_options(::Type{LoopedBatchLinearSolver}) = LoopedBatchLinearSolverOptions()

# The options of the instances' solver (e.g. `mumps_pivtol`) go to each of them, as they would
# to the solver of `MPCSolver`. Their type depends on `looped_linear_solver`, so they are built
# (unless given) once it is set, and the options struct is built from both.
function batch_linear_solver_options(::Type{LoopedBatchLinearSolver}, options)
    remaining = Dict{Symbol,Any}(options)
    opt_instance = pop!(remaining, :looped_linear_solver_options, nothing)
    opt = MadCore.default_options(LoopedBatchLinearSolver)
    remaining = MadCore.set_options!(opt, remaining)
    if isnothing(opt_instance)
        opt_instance = MadCore.default_options(opt.looped_linear_solver)
    end
    remaining = MadCore.set_options!(opt_instance, remaining)
    return LoopedBatchLinearSolverOptions(opt.looped_linear_solver, opt_instance), remaining
end

function LoopedBatchLinearSolver(
    aug_com,
    nzvals_mat::AbstractMatrix{T},
    n::Int;
    opt::LoopedBatchLinearSolverOptions = LoopedBatchLinearSolverOptions(),
) where {T}
    linear_solver = opt.looped_linear_solver
    per_instance_opt = opt.looped_linear_solver_options
    batch_size = size(nzvals_mat, 2)
    nnz_csc = size(nzvals_mat, 1)
    VT = typeof(similar(nzvals_mat, T, 0))
    individual_solvers = map(1:batch_size) do i
        nzval_i = _madnlp_unsafe_column_wrap(nzvals_mat, nnz_csc, (i - 1) * nnz_csc + 1, VT)
        csc_i = _csc_with_nzval(aug_com, nzval_i, n)
        _build_linear_solver(linear_solver, csc_i; opt = per_instance_opt)
    end
    LoopedBatchLinearSolver{T,VT,eltype(individual_solvers)}(individual_solvers, batch_size)
end

create_batch_linear_solver(::Type{LoopedBatchLinearSolver}, aug_com, nzvals_mat, n; opt) =
    LoopedBatchLinearSolver(aug_com, nzvals_mat, n; opt = opt)

function is_factorized!(
    failed_local_buffer::Vector{Int32},
    batch_linear_solver::LoopedBatchLinearSolver,
    factor_view::BatchView,
)
    nfailed = 0
    @inbounds for j = 1:factor_view.n
        if !MadCore.is_factorized(batch_linear_solver.solvers[j])
            nfailed += 1
            failed_local_buffer[nfailed] = j
        end
    end
    return nfailed
end

function factorize_active!(s::LoopedBatchLinearSolver, factor_view::BatchView)
    @inbounds for j = 1:factor_view.n
        MadCore.factorize!(s.solvers[j])
    end
    return
end

function solve_active!(
    s::LoopedBatchLinearSolver{T,VT},
    rhs::AbstractMatrix{T},
    active::BatchView,
) where {T,VT}
    na = local_batch_size(active)
    na == 0 && return
    n = size(rhs, 1)
    @inbounds for j = 1:na
        rhs_j = _madnlp_unsafe_column_wrap(rhs, n, (j - 1) * n + 1, VT)
        MadCore.solve_linear_system!(s.solvers[j], rhs_j)
    end
    return
end
is_factorized!(::Vector{Int32}, ::MadCore.AbstractLinearSolver, ::BatchView) = 0
