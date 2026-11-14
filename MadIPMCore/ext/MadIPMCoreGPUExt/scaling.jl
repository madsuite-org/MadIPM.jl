# Kernels of the Ruiz equilibration of the LP/QP models (Models.Scaling)

const GPUSparseCOO{T} = Models.Scaling.SparseCOO{T, Ti, V, Vi} where
  {Ti, V <: AbstractGPUVector{T}, Vi <: AbstractGPUVector{Ti}}

@inline _atomic_max!(dst, i, value) = begin
  old = dst[i]
  while value > old
    result = Atomix.@atomicreplace dst[i] old => value
    old = result.old
    result.success && break
  end
  return
end

@inline _atomic_min!(dst, i, value) = begin
  old = dst[i]
  while value < old
    result = Atomix.@atomicreplace dst[i] old => value
    old = result.old
    result.success && break
  end
  return
end

@kernel function _coo_scaled_maxabs_kernel!(rownrm, colnrm, @Const(rowpair), @Const(colpair),
                                            @Const(rowval), @Const(colval), @Const(nzval),
                                            @Const(drow), @Const(dcol))
  k = @index(Global, Linear)
  @inbounds begin
    i = rowval[k]; j = colval[k]
    ri = rowpair[i] > 0
    cj = colpair[j] > 0
    if ri || cj
      s = abs(nzval[k]) / (drow[i] * dcol[j])
      ri && _atomic_max!(rownrm, i, s)
      cj && _atomic_max!(colnrm, j, s)
    end
  end
end

@kernel function _coo_scaled_argmax_kernel!(rowcand, colcand, @Const(rownrm), @Const(colnrm),
                                            @Const(rowpair), @Const(colpair),
                                            @Const(rowval), @Const(colval), @Const(nzval),
                                            @Const(drow), @Const(dcol))
  k = @index(Global, Linear)
  @inbounds begin
    i = rowval[k]; j = colval[k]
    ri = rowpair[i] > 0 && rownrm[i] > 0
    cj = colpair[j] > 0 && colnrm[j] > 0
    if ri || cj
      s = abs(nzval[k]) / (drow[i] * dcol[j])
      ri && s == rownrm[i] && _atomic_min!(rowcand, i, Int(j))
      cj && s == colnrm[j] && _atomic_min!(colcand, j, Int(i))
    end
  end
end

function Models.Scaling.scaled_maxabs!(rownrm::AbstractGPUVector{T}, colnrm::AbstractGPUVector{T},
                                       rowpair::AbstractGPUVector{Int}, colpair::AbstractGPUVector{Int},
                                       A::GPUSparseCOO{T},
                                       drow::AbstractGPUVector{T}, dcol::AbstractGPUVector{T}) where {T <: AbstractFloat}
  isempty(A.nzval) && return rownrm, colnrm
  _coo_scaled_maxabs_kernel!(get_backend(rownrm))(
    rownrm, colnrm, rowpair, colpair, A.rowval, A.colval, A.nzval, drow, dcol;
    ndrange = length(A.nzval))
  return rownrm, colnrm
end

function Models.Scaling.scaled_argmax!(rowcand::AbstractGPUVector{Int}, colcand::AbstractGPUVector{Int},
                                       rownrm::AbstractGPUVector{T}, colnrm::AbstractGPUVector{T},
                                       rowpair::AbstractGPUVector{Int}, colpair::AbstractGPUVector{Int},
                                       A::GPUSparseCOO{T},
                                       drow::AbstractGPUVector{T}, dcol::AbstractGPUVector{T}) where {T <: AbstractFloat}
  isempty(A.nzval) && return rowcand, colcand
  _coo_scaled_argmax_kernel!(get_backend(rowcand))(
    rowcand, colcand, rownrm, colnrm, rowpair, colpair,
    A.rowval, A.colval, A.nzval, drow, dcol;
    ndrange = length(A.nzval))
  return rowcand, colcand
end

@kernel function _settle_pairs_kernel!(rowpair, colpair)
  j = @index(Global, Linear)
  @inbounds begin
    i = colpair[j]
    if i > 0 && rowpair[i] == j
      rowpair[i] = 0
      colpair[j] = 0
    end
  end
end

function Models.Scaling.settle_pairs!(rowpair::AbstractGPUVector{Int}, colpair::AbstractGPUVector{Int})
  isempty(colpair) && return nothing
  _settle_pairs_kernel!(get_backend(rowpair))(rowpair, colpair; ndrange = length(colpair))
  return nothing
end
