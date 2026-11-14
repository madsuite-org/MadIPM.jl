"""
    MadIPMCoreGPUExt

The GPU code of MadIPMCore that does not depend on a GPU vendor: KernelAbstractions kernels on
`AbstractGPUArray`s, launched on the backend of their arguments. It loads with MadCoreGPU, as
MadNLPCoreGPUExt does for MadNLPCore; the bundles (`cuMadIPM`) add the vendor-specific code.
"""
module MadIPMCoreGPUExt

import MadCore, MadCoreGPU, MadIPMCore
import MadIPMCore: Models
import Atomix
import GPUArraysCore: AbstractGPUArray, AbstractGPUVector, AbstractGPUMatrix
import KernelAbstractions: @kernel, @index, @Const, get_backend

include("batch_views.jl")
include("batch_kkt.jl")
include("scaling.jl")
include("normal_kkt.jl")

end # module MadIPMCoreGPUExt
