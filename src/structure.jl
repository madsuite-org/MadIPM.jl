mutable struct MPCSolver{
    T,
    VT<:AbstractVector{T},
    VI<:AbstractVector{Int},
    KKTSystem<:MadCore.AbstractKKTSystem{T},
    Model<:NLPModels.AbstractNLPModel{T,VT},
    CB<:MadCore.AbstractCallback{T},
} <: MadCore.AbstractMadNLPSolver{T}
    nlp::Model
    class::AbstractConicProblem
    cb::CB
    kkt::KKTSystem

    opt::IPMOptions
    cnt::MadCore.MadNLPCounters
    logger::MadCore.MadNLPLogger

    n::Int # number of variables (after reformulation)
    m::Int # number of cons
    nlb::Int
    nub::Int

    x::MadCore.PrimalVector{T,VT,VI} # primal (after reformulation)
    y::VT # dual
    zl::MadCore.PrimalVector{T,VT,VI} # dual (after reformulation)
    zu::MadCore.PrimalVector{T,VT,VI} # dual (after reformulation)
    xl::MadCore.PrimalVector{T,VT,VI} # primal lower bound (after reformulation)
    xu::MadCore.PrimalVector{T,VT,VI} # primal upper bound (after reformulation)

    obj_val::T
    f::MadCore.PrimalVector{T,VT,VI}
    c::VT

    jacl::VT

    d::MadCore.UnreducedKKTVector{T,VT}
    p::MadCore.UnreducedKKTVector{T,VT}

    # Buffers
    _w1::MadCore.UnreducedKKTVector{T,VT}
    _w2::MadCore.UnreducedKKTVector{T,VT}

    correction_lb::VT
    correction_ub::VT
    rhs::VT
    ind_ineq::VI
    ind_fixed::VI
    ind_lb::VI
    ind_ub::VI
    ind_llb::VI
    ind_uub::VI

    x_lr::MadCore.SubVector{T,VT,VI}
    x_ur::MadCore.SubVector{T,VT,VI}
    xl_r::MadCore.SubVector{T,VT,VI}
    xu_r::MadCore.SubVector{T,VT,VI}
    zl_r::MadCore.SubVector{T,VT,VI}
    zu_r::MadCore.SubVector{T,VT,VI}
    dx_lr::MadCore.SubVector{T,VT,VI}
    dx_ur::MadCore.SubVector{T,VT,VI}

    inf_pr::T
    inf_du::T
    inf_compl::T
    norm_b::T
    norm_c::T

    mu::T

    alpha_p::T
    alpha_d::T
    del_w::T
    del_c::T
    best_complementarity::T
    mu_curr::T
    status::MadCore.Status
end

for (k, attribute) in enumerate(fieldnames(MPCSolver))
    fname = "get_$(attribute)"
    sf = Symbol(fname)
    if isdefined(MadCore, sf)
        @eval begin
            @inline function MadCore.$(sf)(solver::MPCSolver)
                return getfield(solver, $k)
            end
        end
    else
        @eval begin
            @inline function $(sf)(solver::MPCSolver)
                return getfield(solver, $k)
            end
        end
    end
end

function MPCSolver(nlp::NLPModels.AbstractNLPModel{T,VT}; kwargs...) where {T,VT}
    options = load_options(nlp; kwargs...)

    ipm_opt = options.interior_point
    logger = options.logger
    @assert MadCore.is_supported(ipm_opt.linear_solver, T)

    cnt = MadCore.MadNLPCounters(start_time = time())
    cb = MadCore.create_callback(
        MadCore.SparseCallback,
        nlp;
        fixed_variable_treatment = ipm_opt.fixed_variable_treatment,
        equality_treatment = ipm_opt.equality_treatment,
    )

    # generic options
    MadCore.@trace(logger, "Initializing variables.")

    ind_lb = cb.ind_lb
    ind_ub = cb.ind_ub

    ns = length(cb.ind_ineq)
    nx = MadCore.n_variables(cb)
    n = nx+ns
    m = NLPModels.get_ncon(nlp)
    nlb = length(ind_lb)
    nub = length(ind_ub)

    MadCore.@trace(logger, "Initializing KKT system.")
    kkt = MadCore.create_kkt_system(
        ipm_opt.kkt_system,
        cb,
        ipm_opt.linear_solver;
        opt_linear_solver = options.linear_solver,
    )

    x = MadCore.PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    xl = MadCore.PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    xu = MadCore.PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    zl = MadCore.PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    zu = MadCore.PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    f = MadCore.PrimalVector(VT, nx, ns, ind_lb, ind_ub)

    d = MadCore.UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    p = MadCore.UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    _w1 = MadCore.UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    _w2 = MadCore.UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)

    # Buffers
    correction_lb = VT(undef, nlb)
    correction_ub = VT(undef, nub)
    jacl = VT(undef, n)
    c_trial = VT(undef, m)
    y = VT(undef, m)
    c = VT(undef, m)
    rhs = VT(undef, m)

    x_lr = view(full(x), cb.ind_lb)
    x_ur = view(full(x), cb.ind_ub)
    xl_r = view(full(xl), cb.ind_lb)
    xu_r = view(full(xu), cb.ind_ub)
    zl_r = view(full(zl), cb.ind_lb)
    zu_r = view(full(zu), cb.ind_ub)
    dx_lr = view(d.xp, cb.ind_lb)
    dx_ur = view(d.xp, cb.ind_ub)

    cnt.init_time = time() - cnt.start_time

    nnzh = MadCore.get_nnzh(nlp)
    # check if the problem is a LP or a QP
    class = iszero(nnzh) ? LinearProgram() : QuadraticProgram()

    return MPCSolver(
        nlp,
        class,
        cb,
        kkt,
        ipm_opt,
        cnt,
        options.logger,
        n,
        m,
        nlb,
        nub,
        x,
        y,
        zl,
        zu,
        xl,
        xu,
        zero(T),
        f,
        c,
        jacl,
        d,
        p,
        _w1,
        _w2,
        correction_lb,
        correction_ub,
        rhs,
        cb.ind_ineq,
        cb.ind_fixed,
        cb.ind_lb,
        cb.ind_ub,
        cb.ind_llb,
        cb.ind_uub,
        x_lr,
        x_ur,
        xl_r,
        xu_r,
        zl_r,
        zu_r,
        dx_lr,
        dx_ur,
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        typemax(T),
        zero(T),
        MadCore.INITIAL,
    )
end

function MadCore.print_iter(solver::MPCSolver; options...)
    obj_scale = solver.cb.obj_scale[]
    mod(solver.cnt.k, 10)==0 && MadCore.@info(
        solver.logger,
        @sprintf(
            "iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr"
        )
    )
    inf_du = solver.inf_du
    inf_pr = solver.inf_pr
    mu = log10(solver.mu)
    MadCore.@info(
        solver.logger,
        Printf.@sprintf(
            "%4i%s% 10.7e %6.2e %6.2e %5.1f %6.2e %s %6.2e %6.2e",
            solver.cnt.k,
            " ",
            solver.obj_val/obj_scale,
            inf_pr,
            inf_du,
            mu,
            solver.cnt.k == 0 ? 0.0 : norm(MadCore.primal(solver.d), Inf),
            solver.del_w == 0 ? "   - " : @sprintf("%5.1f", log(10, solver.del_w)),
            solver.alpha_d,
            solver.alpha_p
        )
    )
    return
end
