abstract type AbstractBatchKKTSystem{T} end

function _kktmul!(
    w::BatchUnreducedKKTVector,
    x::BatchUnreducedKKTVector,
    kkt::AbstractBatchKKTSystem,
    alpha,
    beta,
)
    return _kktmul!(
        w, x,
        kkt.reg, du_diag(kkt), kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag,
        alpha, beta,
    )
end

@inbounds function _kktmul!(
    w::BatchUnreducedKKTVector,
    x::BatchUnreducedKKTVector,
    reg,
    du_diag_val,
    l_lower,
    u_lower,
    l_diag,
    u_diag,
    alpha,
    beta,
)
    MadCore.primal(w) .+= alpha .* reg .* MadCore.primal(x)
    MadCore.dual(w) .+= alpha .* du_diag_val .* MadCore.dual(x)
    xp_lr(w) .-= alpha .* MadCore.dual_lb(x)
    xp_ur(w) .+= alpha .* MadCore.dual_ub(x)
    MadCore.dual_lb(w) .=
        beta .* MadCore.dual_lb(w) .+
        alpha .* (xp_lr(x) .* l_lower .- MadCore.dual_lb(x) .* l_diag)
    MadCore.dual_ub(w) .=
        beta .* MadCore.dual_ub(w) .+
        alpha .* (xp_ur(x) .* u_lower .+ MadCore.dual_ub(x) .* u_diag)
    return
end

include("Sparse/augmented.jl")
