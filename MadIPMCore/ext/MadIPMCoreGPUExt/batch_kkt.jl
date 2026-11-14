# Kernels of the batch KKT system: the constraint scaling, the bound reduction of the right-hand
# side and its inverse

@kernel function _set_con_scale_kernel!(con_scale, @Const(jac_I), @Const(jac_buffer))
    k = @index(Global, Linear)
    row = jac_I[k]
    bs = size(jac_buffer, 2)
    m = size(con_scale, 1)
    @inbounds for j in 1:bs
        _atomic_max!(con_scale, row + (j - 1) * m, abs(jac_buffer[k, j]))
    end
end

function MadCore.update_con_scale_sparse!(
    con_scale::AbstractGPUMatrix{T},
    jac_I::AbstractGPUVector{<:Integer},
    jac_buffer::AbstractGPUMatrix{T},
) where T
    nnzj = length(jac_I)
    if nnzj > 0
        backend = get_backend(con_scale)
        _set_con_scale_kernel!(backend)(con_scale, jac_I, jac_buffer; ndrange=nnzj)
    end
    return con_scale
end

# the same reduction for the lower and the upper bounds
@kernel function _reduce_rhs_kernel!(values, @Const(ind), off, @Const(diag))
    i, j = @index(Global, NTuple)
    @inbounds values[ind[i], j] -= values[off + i, j] / diag[i, j]
end

function MadIPMCore._reduce_rhs_batch!(values::AbstractGPUMatrix, ind_lb, lb_off, l_diag,
                                                      ind_ub, ub_off, u_diag)
    bs = size(values, 2); backend = get_backend(values)
    nlb = length(ind_lb)
    nlb > 0 && _reduce_rhs_kernel!(backend)(values, ind_lb, lb_off, l_diag; ndrange=(nlb, bs))
    nub = length(ind_ub)
    nub > 0 && _reduce_rhs_kernel!(backend)(values, ind_ub, ub_off, u_diag; ndrange=(nub, bs))
    return
end

@kernel function _finish_aug_solve_lb_kernel!(values, @Const(ind_lb), lb_off, @Const(l_lower), @Const(l_diag))
    i, j = @index(Global, NTuple)
    @inbounds values[lb_off + i, j] = (-values[lb_off + i, j] + l_lower[i, j] * values[ind_lb[i], j]) / l_diag[i, j]
end

@kernel function _finish_aug_solve_ub_kernel!(values, @Const(ind_ub), ub_off, @Const(u_lower), @Const(u_diag))
    i, j = @index(Global, NTuple)
    @inbounds values[ub_off + i, j] = (values[ub_off + i, j] - u_lower[i, j] * values[ind_ub[i], j]) / u_diag[i, j]
end

function MadIPMCore._finish_aug_solve_batch!(values::AbstractGPUMatrix, ind_lb, lb_off, l_lower, l_diag,
                                                            ind_ub, ub_off, u_lower, u_diag)
    bs = size(values, 2); backend = get_backend(values)
    nlb = length(ind_lb)
    nlb > 0 && _finish_aug_solve_lb_kernel!(backend)(values, ind_lb, lb_off, l_lower, l_diag; ndrange=(nlb, bs))
    nub = length(ind_ub)
    nub > 0 && _finish_aug_solve_ub_kernel!(backend)(values, ind_ub, ub_off, u_lower, u_diag; ndrange=(nub, bs))
    return
end
