# Column gathers and scatters of the active-set views of the batch solver

@kernel function _gather_batch_view_columns_kernel!(dst, @Const(src), @Const(local_to_root))
    i, j = @index(Global, NTuple)
    @inbounds dst[i, j] = src[i, local_to_root[j]]
end

@kernel function _scatter_batch_view_columns_kernel!(dst, @Const(src), @Const(local_to_root))
    i, j = @index(Global, NTuple)
    @inbounds dst[i, local_to_root[j]] = src[i, j]
end

function MadIPMCore.gather_batch_view_columns!(
    dst::AbstractGPUMatrix{TD},
    src::AbstractGPUMatrix{TS},
    batch_view::MadIPMCore.BatchView,
) where {TD, TS}
    na = MadIPMCore.local_batch_size(batch_view)
    backend = get_backend(dst)
    _gather_batch_view_columns_kernel!(backend)(dst, src, MadIPMCore.local_to_root_dev(batch_view); ndrange=(size(dst, 1), na))
    return dst
end

function MadIPMCore.scatter_batch_view_columns!(
    dst::AbstractGPUMatrix{TD},
    src::AbstractGPUMatrix{TS},
    batch_view::MadIPMCore.BatchView,
) where {TD, TS}
    na = MadIPMCore.local_batch_size(batch_view)
    backend = get_backend(dst)
    _scatter_batch_view_columns_kernel!(backend)(dst, src, MadIPMCore.local_to_root_dev(batch_view); ndrange=(size(src, 1), na))
    return dst
end

function MadIPMCore.compact_active_columns_inplace!(
    dst::AbstractGPUMatrix{T},
    batch_view::MadIPMCore.BatchView,
    scratch::AbstractGPUMatrix{T},
) where T
    na = MadIPMCore.local_batch_size(batch_view)
    na == 0 && return dst
    nrows = size(dst, 1)
    copyto!(view(scratch, 1:nrows, :), view(dst, 1:nrows, :))
    backend = get_backend(dst)
    _gather_batch_view_columns_kernel!(backend)(dst, scratch, MadIPMCore.local_to_root_dev(batch_view); ndrange=(nrows, na))
    return dst
end
