"""
    MadIPM

MadIPM on the CPU: MadIPMCore with the CPU linear solvers of MadCore (MUMPS, the HSL solvers,
LDLFactorizations, UMFPACK and CHOLMOD), the same set as MadNLP. MUMPS is the default sparse
linear solver. Loading MathOptInterface adds `MadIPM.Optimizer`.
"""
module MadIPM

import MadCore, MadIPMCore
import NLPModels
import HSL, Krylov, LDLFactorizations, MUMPS_seq_jll, OpenBLAS32_jll, SuiteSparse

# MadIPM bundles MadIPMCore with the CPU linear solvers. Every name of MadIPMCore also
# resolves as `MadIPM.name`, as before the split.
for name in MadCore.defined_names(MadIPMCore)
    name === :MadCore && continue
    @eval import MadIPMCore: $name
end

export MPCSolver, madipm, madipm_batch

# MUMPS is the default sparse linear solver for models on the CPU, as in MadNLP. Models on other
# arrays keep the fallback of MadIPMCore (a GPU bundle, or MadCore.default_sparse_solver).
MadIPMCore.default_linear_solver(::NLPModels.AbstractNLPModel{T, <:Vector{T}}) where {T} = MadCore.MumpsSolver

global Optimizer

end # module MadIPM
