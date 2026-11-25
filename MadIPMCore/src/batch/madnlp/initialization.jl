function MadCore.set_initial_bounds!(
    xl::AbstractMatrix{T},
    xu::AbstractMatrix{T},
    tol,
) where {T}
    if tol > zero(T)
        xl .= xl .- max.(one(T), abs.(xl)) .* tol
        xu .= xu .+ max.(one(T), abs.(xu)) .* tol
    end
end

function MadCore.initialize!(
    bcb::UniformBatchCallback{T},
    x,
    xl,
    xu,
    y,
    rhs,
    ind_ineq,
    bx_buffer;
    tol = 1e-8,
    bound_push = 1e-2,
    bound_fac = 1e-2,
) where {T}
    x0 = MadCore.variable(x)
    lvar = MadCore.variable(xl)
    uvar = MadCore.variable(xu)

    x0 .= NLPModels.get_x0(bcb)
    lvar .= NLPModels.get_lvar(bcb)
    uvar .= NLPModels.get_uvar(bcb)
    y .= NLPModels.get_y0(bcb)
    lcon = copy(NLPModels.get_lcon(bcb))
    ucon = copy(NLPModels.get_ucon(bcb))

    MadCore.treat_fixed_variable_initialize!(bcb, x0, lvar, uvar)  # this is no-op, recall bcb <: MadCore.AbstractCallback. will need to be adjust for batchdensecallback

    MadCore.set_initial_bounds!(lvar, uvar, tol)
    x0 .= MadCore.initialize_variable.(x0, lvar, uvar, bound_push, bound_fac)

    x_full = MadCore.update_x!(bcb, x0)
    copyto!(bx_buffer, x_full)
    MadCore.eval_cons_callback!(bcb, bx_buffer, bcb.con_buffer)

    MadCore.slack(xl) .= view(lcon, ind_ineq, :)
    MadCore.slack(xu) .= view(ucon, ind_ineq, :)
    MadCore.treat_equality_initialize!(bcb.equality_handler, rhs, lcon, ucon)
    copyto!(MadCore.slack(x), view(bcb.con_buffer, ind_ineq, :))

    MadCore.set_initial_bounds!(MadCore.slack(xl), MadCore.slack(xu), tol)
    MadCore.slack(x) .= MadCore.initialize_variable.(
        MadCore.slack(x),
        MadCore.slack(xl),
        MadCore.slack(xu),
        bound_push,
        bound_fac,
    )

    return
end

function _set_con_scale_batch!(
    con_scale::MT,
    jac_I,
    jac_buffer,
    max_gradient,
) where {T,MT<:AbstractMatrix{T}}
    fill!(con_scale, one(T))
    _update_con_scale_batch!(con_scale, jac_I, jac_buffer)
    con_scale .= min.(one(T), max_gradient ./ con_scale)
    return con_scale
end

function _update_con_scale_batch!(
    con_scale::MT,
    jac_I,
    jac_buffer,
) where {T,MT<:AbstractMatrix{T}}
    nnzj = length(jac_I)
    bs = size(jac_buffer, 2)
    @inbounds for k = 1:nnzj
        row = jac_I[k]
        for j = 1:bs
            con_scale[row, j] = max(con_scale[row, j], abs(jac_buffer[k, j]))
        end
    end
    return con_scale
end

function _set_jac_scale_batch!(
    jac_scale::MT,
    con_scale,
    jac_I,
) where {T,MT<:AbstractMatrix{T}}
    return copyto!(jac_scale, @view(con_scale[jac_I, :]))
end

function _set_obj_scale_batch!(
    obj_scale,
    F::MT,
    max_gradient,
) where {T,MT<:AbstractMatrix{T}}
    return obj_scale .= min.(one(T), max_gradient ./ maximum(abs, F; dims = 1))
end

function MadCore.set_scaling!(
    cb::UniformBatchCallback,
    x,
    xl,
    xu,
    y,
    rhs,
    ind_ineq,
    nlp_scaling_max_gradient,
    bx_buffer,
)
    x0 = MadCore.variable(x)
    x_full = MadCore.update_x!(cb, x0)
    copyto!(bx_buffer, x_full)

    jac_free = MadCore.eval_jac_callback!(cb, bx_buffer, cb.jac_buffer)
    _set_con_scale_batch!(cb.con_scale, cb.jac_I, jac_free, nlp_scaling_max_gradient)
    _set_jac_scale_batch!(cb.jac_scale, cb.con_scale, cb.jac_I)

    MadCore.eval_grad_f_callback!(cb, bx_buffer, cb.grad_buffer)
    _set_obj_scale_batch!(cb.obj_scale, cb.grad_buffer, nlp_scaling_max_gradient)

    con_scale_slk = @view(cb.con_scale[ind_ineq, :])
    y ./= cb.con_scale
    rhs .*= cb.con_scale
    MadCore.slack(x) .*= con_scale_slk
    MadCore.slack(xl) .*= con_scale_slk
    MadCore.slack(xu) .*= con_scale_slk
    return
end
