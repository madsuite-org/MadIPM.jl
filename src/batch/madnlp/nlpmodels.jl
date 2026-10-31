function MadCore._update_x!(
    bcb::UniformBatchCallback{T,VT,MT,VI,BM,FH,EH},
    x,
) where {T,VT,MT,VI,BM,FH,EH}
    return x
end

function MadCore._update_x!(
    bcb::UniformBatchCallback{T,VT,MT,VI,BM,FH,EH},
    x,
) where {T,VT,MT,VI,BM,FH<:MadCore.MakeParameter,EH}
    fh = bcb.fixed_handler
    nvar_nlp = bcb.nlp.meta.nvar
    bs = bcb.batch_size
    BX = reshape(fh.x_full, nvar_nlp, bs)
    view(BX, fh.fixed, :) .= view(bcb.nlp.meta.lvar, fh.fixed, :)
    view(BX, fh.free, :) .= x
    return BX
end

function MadCore.unpack_x!(
    X_full::AbstractMatrix,
    bcb::UniformBatchCallback{T,VT,MT,VI,BM,FH,EH},
    x::BatchPrimalVector,
) where {T,VT,MT,VI,BM,FH,EH}
    X_full .= MadCore.variable(x)
end

function MadCore.unpack_x!(
    X_full::AbstractMatrix,
    bcb::UniformBatchCallback{T,VT,MT,VI,BM,FH,EH},
    x::BatchPrimalVector,
) where {T,VT,MT,VI,BM,FH<:MadCore.MakeParameter,EH}
    fh = bcb.fixed_handler
    X_full[fh.free, :] .= MadCore.variable(x)
    X_full[fh.fixed, :] .= view(bcb.nlp.meta.lvar, fh.fixed, :)
end

function MadCore.unpack_z!(
    Z_full::AbstractMatrix,
    bcb::UniformBatchCallback{T,VT,MT,VI,BM,FH,EH},
    z_free,
) where {T,VT,MT,VI,BM,FH,EH}
    Z_full .= z_free ./ bcb.obj_scale
end

function MadCore.unpack_z!(
    Z_full::AbstractMatrix,
    bcb::UniformBatchCallback{T,VT,MT,VI,BM,FH,EH},
    z_free,
) where {T,VT,MT,VI,BM,FH<:MadCore.MakeParameter,EH}
    fill!(Z_full, zero(T))
    Z_full[bcb.fixed_handler.free, :] .= z_free ./ bcb.obj_scale
end

function MadCore.get_x0(
    bcb::UniformBatchCallback{T,VT,MT,VI,BM,FH,EH},
) where {T,VT,MT,VI,BM,FH<:MadCore.MakeParameter,EH}
    view(bcb.nlp.meta.x0, bcb.fixed_handler.free, :)
end

MadCore.get_y0(bcb::UniformBatchCallback) = bcb.nlp.meta.y0

MadCore.get_lvar(
    bcb::UniformBatchCallback{T,VT,MT,VI,BM,FH,EH},
) where {T,VT,MT,VI,BM,FH,EH} = bcb.nlp.meta.lvar
function MadCore.get_lvar(
    bcb::UniformBatchCallback{T,VT,MT,VI,BM,FH,EH},
) where {T,VT,MT,VI,BM,FH<:MadCore.MakeParameter,EH}
    view(bcb.nlp.meta.lvar, bcb.fixed_handler.free, :)
end

MadCore.get_uvar(
    bcb::UniformBatchCallback{T,VT,MT,VI,BM,FH,EH},
) where {T,VT,MT,VI,BM,FH,EH} = bcb.nlp.meta.uvar
function MadCore.get_uvar(
    bcb::UniformBatchCallback{T,VT,MT,VI,BM,FH,EH},
) where {T,VT,MT,VI,BM,FH<:MadCore.MakeParameter,EH}
    view(bcb.nlp.meta.uvar, bcb.fixed_handler.free, :)
end

MadCore.get_lcon(bcb::UniformBatchCallback) = bcb.nlp.meta.lcon
MadCore.get_ucon(bcb::UniformBatchCallback) = bcb.nlp.meta.ucon

function MadCore.unpack_y!(y_full, bcb::UniformBatchCallback, y)
    @. y_full = y * bcb.con_scale * bcb.obj_sign / bcb.obj_scale
end

function unpack_obj!(dst, bcb::UniformBatchCallback, obj_val)
    dst_mat = reshape(dst, 1, length(dst))
    @. dst_mat = bcb.obj_sign * obj_val / bcb.obj_scale
end

function MadCore.unpack_cons!(c_full, bcb::UniformBatchCallback, c, rhs, ind_ineq, slack)
    c_full .= c ./ bcb.con_scale .+ rhs
    if length(ind_ineq) > 0
        view(c_full, ind_ineq, :) .+= slack
    end
end

function MadCore._eval_f_wrapper(
    bcb::UniformBatchCallback,
    bx::AbstractMatrix,
    bf::AbstractVector,
)
    NLPModels.obj!(bcb.nlp, bx, bf)
    bf .*= vec(bcb.obj_scale)
    return bf
end

function MadCore._eval_cons_wrapper!(
    bcb::UniformBatchCallback,
    bx::AbstractMatrix,
    bc_mat::AbstractMatrix,
)
    NLPModels.cons!(bcb.nlp, bx, bc_mat)
    bc_mat .*= bcb.con_scale
    return bc_mat
end

function MadCore._eval_grad_f_wrapper!(
    bcb::UniformBatchCallback{T,VT,MT,VI,BM,FH,EH},
    bx::AbstractMatrix,
    bg::AbstractMatrix,
) where {T,VT,MT,VI,BM,FH,EH}
    NLPModels.grad!(bcb.nlp, bx, bg)
    bg .*= bcb.obj_scale
    return bg
end

function MadCore._eval_grad_f_wrapper!(
    bcb::UniformBatchCallback{T,VT,MT,VI,BM,FH,EH},
    bx::AbstractMatrix,
    bg::AbstractMatrix,
) where {T,VT,MT,VI,BM,FH<:MadCore.MakeParameter,EH}
    fh = bcb.fixed_handler
    nvar_nlp = bcb.nlp.meta.nvar
    bs = bcb.batch_size
    GF = reshape(fh.g_full, nvar_nlp, bs)
    NLPModels.grad!(bcb.nlp, bx, GF)
    view(bg, 1:bcb.nvar, :) .= view(GF, fh.free, :) .* bcb.obj_scale
    return bg
end

function MadCore._eval_jac_wrapper!(
    bcb::UniformBatchCallback{T,VT,MT,VI,BM,FH,EH},
    bx::AbstractMatrix,
    jac_buffer::AbstractMatrix,
) where {T,VT,MT,VI,BM,FH,EH}
    NLPModels.jac_coord!(bcb.nlp, bx, jac_buffer)
    jac_buffer .*= bcb.jac_scale
    return jac_buffer
end

function MadCore._eval_jac_wrapper!(
    bcb::UniformBatchCallback{T,VT,MT,VI,BM,FH,EH},
    bx::AbstractMatrix,
    jac_buffer::AbstractMatrix,
) where {T,VT,MT,VI,BM,FH<:MadCore.MakeParameter,EH}
    NLPModels.jac_coord!(bcb.nlp, bx, jac_buffer)
    jac_free = view(jac_buffer, bcb.fixed_handler.ind_jac_free, :)
    jac_free .*= bcb.jac_scale
    return jac_free
end

function MadCore._eval_lag_hess_wrapper!(
    bcb::UniformBatchCallback{T,VT,MT,VI,BM,FH,EH},
    bx::AbstractMatrix,
    y_mat::AbstractMatrix,
    bv::AbstractMatrix,
    hess::AbstractMatrix;
    obj_weight::AbstractVector = vec(bcb.obj_scale),
) where {T,VT,MT,VI,BM,FH,EH}
    bv .= y_mat .* bcb.con_scale
    NLPModels.hess_coord!(bcb.nlp, bx, bv, obj_weight, hess)
    return
end

function MadCore._eval_lag_hess_wrapper!(
    bcb::UniformBatchCallback{T,VT,MT,VI,BM,FH,EH},
    bx::AbstractMatrix,
    y_mat::AbstractMatrix,
    bv::AbstractMatrix,
    hess::AbstractMatrix;
    obj_weight::AbstractVector = vec(bcb.obj_scale),
) where {T,VT,MT,VI,BM,FH<:MadCore.MakeParameter,EH}
    bv .= y_mat .* bcb.con_scale
    NLPModels.hess_coord!(bcb.nlp, bx, bv, obj_weight, bcb.hess_buffer)
    hess .= view(bcb.hess_buffer, bcb.fixed_handler.ind_hess_free, :)
    return
end

function MadCore.eval_f_wrapper(solver::AbstractBatchMPCSolver, bx::AbstractMatrix)
    ws = solver.workspace
    bcb = solver.bcb
    bcnt = solver.batch_cnt

    t = @elapsed begin
        MadCore._eval_f_wrapper(bcb, bx, ws.bf)
        ws.bf .*= vec(bcb.obj_sign)
        vec(ws.obj_val) .= ws.bf
    end
    bcnt.eval_function_time[] += t
    bcnt.obj_cnt[] += 1
    return
end

function MadCore.eval_cons_wrapper!(solver::AbstractBatchMPCSolver, bx::AbstractMatrix)
    ws = solver.workspace
    bcb = solver.bcb
    bcnt = solver.batch_cnt
    ind_ineq = bcb.ind_ineq
    ns = length(ind_ineq)

    t = @elapsed begin
        MadCore._eval_cons_wrapper!(bcb, bx, MadCore.full(solver.c))
        if ns > 0
            view(MadCore.full(solver.c), ind_ineq, :) .-= MadCore.slack(solver.x)
        end
        MadCore.full(solver.c) .-= MadCore.full(solver.rhs)
    end
    bcnt.eval_function_time[] += t
    bcnt.con_cnt[] += 1
    return
end

function MadCore.eval_grad_f_wrapper!(solver::AbstractBatchMPCSolver, bx::AbstractMatrix)
    ws = solver.workspace
    bcb = solver.bcb
    bcnt = solver.batch_cnt
    nvar = bcb.nvar

    t = @elapsed begin
        MadCore._eval_grad_f_wrapper!(bcb, bx, ws.bg)
        BG = view(ws.bg, 1:nvar, :)
        BG .*= bcb.obj_sign
        copyto!(MadCore.variable(solver.f), BG)
    end
    bcnt.eval_function_time[] += t
    bcnt.obj_grad_cnt[] += 1
    return
end
