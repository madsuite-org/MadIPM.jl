
#=
    Interface to direct solver for solving KKT system
=#

function factorize_regularized_system!(solver)
    max_trials = 3
    for ntrial = 1:max_trials
        set_aug_diagonal_reg!(solver.kkt, solver)
        MadCore.factorize_wrapper!(solver)
        if MadCore.is_factorized(solver.kkt.linear_solver)
            break
        end
        solver.del_w *= 100.0
        solver.del_c *= 100.0
    end
end

function solve_system!(
    d::MadCore.UnreducedKKTVector{T},
    solver::MadCore.AbstractMadNLPSolver{T},
    p::MadCore.UnreducedKKTVector{T},
) where {T}
    opt = solver.opt
    copyto!(MadCore.full(d), MadCore.full(p))
    MadCore.solve_kkt!(solver.kkt, d)

    # Check residual
    w = solver._w1
    copyto!(MadCore.full(w), MadCore.full(p))
    mul!(w, solver.kkt, d, -one(T), one(T))
    norm_w = norm(MadCore.full(w), Inf)
    norm_p = norm(MadCore.full(p), Inf)

    residual_ratio = norm_w / max(one(T), norm_p)
    MadCore.@debug(
        solver.logger,
        @sprintf("Residual after linear solve: %6.2e", residual_ratio),
    )
    if isnan(residual_ratio) ||
       (opt.check_residual && (residual_ratio > opt.tol_linear_solve))
        throw(MadCore.SolveException)
    end
    return d
end
