"""
    MadIPMCoreGPUExt

The GPU code of MadIPMCore that does not depend on a GPU vendor: KernelAbstractions kernels on
`AbstractGPUArray`s, launched on the backend of their arguments. It loads with the GPU
extensions of MadCore, as MadNLPCoreGPUExt does for MadNLPCore; the bundles (`cuMadIPM`) add
the vendor-specific code.
"""
module MadIPMCoreGPUExt

import MadCore, MadIPMCore
import MadIPMCore: Models
import Atomix
import GPUArraysCore: AbstractGPUArray, AbstractGPUVector, AbstractGPUMatrix
import KernelAbstractions: @kernel, @index, @Const, get_backend

# Atomic max and min of `dst[i]` by compare-and-swap, for element types without native atomics
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

include("batch_views.jl")
include("batch_kkt.jl")
include("scaling.jl")
include("normal_kkt.jl")

end # module MadIPMCoreGPUExt
