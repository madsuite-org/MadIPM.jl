struct BatchUnreducedKKTVector{T,MT<:AbstractMatrix{T},VI,SV,IV}
    values::MT
    n::Int
    m::Int
    nlb::Int
    nub::Int
    ind_lb::VI
    ind_ub::VI
    _primal::SV
    _dual::SV
    _primal_dual::SV
    _dual_lb::SV
    _dual_ub::SV
    _xp_lr::IV
    _xp_ur::IV
end

function BatchUnreducedKKTVector(
    ::Type{MT},
    ::Type{VT},
    n::Int,
    m::Int,
    nlb::Int,
    nub::Int,
    batch_size::Int,
    ind_lb,
    ind_ub,
) where {T,MT<:AbstractMatrix{T},VT<:AbstractVector{T}}
    total = n + m + nlb + nub
    values = MT(undef, total, batch_size)
    fill!(values, zero(T))

    primal = view(values, 1:n, :)
    xp_lr = view(values, ind_lb, :)
    SV = typeof(primal)
    IV = typeof(xp_lr)
    return BatchUnreducedKKTVector{T,MT,typeof(ind_lb),SV,IV}(
        values,
        n,
        m,
        nlb,
        nub,
        ind_lb,
        ind_ub,
        primal,
        view(values, (n+1):(n+m), :),
        view(values, 1:(n+m), :),
        view(values, (n+m+1):(n+m+nlb), :),
        view(values, (n+m+nlb+1):(n+m+nlb+nub), :),
        xp_lr,
        view(values, ind_ub, :),
    )
end

MadCore.full(bv::BatchUnreducedKKTVector) = bv.values
MadCore.primal(bv::BatchUnreducedKKTVector) = bv._primal
MadCore.dual(bv::BatchUnreducedKKTVector) = bv._dual
MadCore.primal_dual(bv::BatchUnreducedKKTVector) = bv._primal_dual
MadCore.dual_lb(bv::BatchUnreducedKKTVector) = bv._dual_lb
MadCore.dual_ub(bv::BatchUnreducedKKTVector) = bv._dual_ub
xp_lr(bv::BatchUnreducedKKTVector) = bv._xp_lr
xp_ur(bv::BatchUnreducedKKTVector) = bv._xp_ur

struct BatchPrimalVector{T,MT<:AbstractMatrix{T},VI,SV,IV}
    values::MT
    nx::Int
    ns::Int
    ind_lb::VI
    ind_ub::VI
    _variable::SV
    _slack::SV
    _lower::IV
    _upper::IV
end

function BatchPrimalVector(
    ::Type{MT},
    ::Type{VT},
    nx::Int,
    ns::Int,
    batch_size::Int,
    ind_lb,
    ind_ub,
) where {T,MT<:AbstractMatrix{T},VT<:AbstractVector{T}}
    total = nx + ns
    values = MT(undef, total, batch_size)
    fill!(values, zero(T))

    variable = view(values, 1:nx, :)
    lower = view(values, ind_lb, :)
    SV = typeof(variable)
    IV = typeof(lower)
    return BatchPrimalVector{T,MT,typeof(ind_lb),SV,IV}(
        values,
        nx,
        ns,
        ind_lb,
        ind_ub,
        variable,
        view(values, (nx+1):(nx+ns), :),
        lower,
        view(values, ind_ub, :),
    )
end

MadCore.variable(bpv::BatchPrimalVector) = bpv._variable
MadCore.slack(bpv::BatchPrimalVector) = bpv._slack
lower(bpv::BatchPrimalVector) = bpv._lower
upper(bpv::BatchPrimalVector) = bpv._upper
MadCore.full(bpv::BatchPrimalVector) = bpv.values
MadCore.primal(bpv::BatchPrimalVector) = bpv.values
