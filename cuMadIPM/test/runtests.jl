using Test, LinearAlgebra, SparseArrays, NLPModels
using cuMadIPM, MadIPMCore, MadCore, CUDA
using MadIPM, MadNLP   # CPU references and the option values of MadNLP
using MadIPM.Models: BatchQuadraticModel

include(joinpath(@__DIR__, "..", "..", "MadIPM", "test", "utils.jl"))
include(joinpath(@__DIR__, "..", "..", "MadIPM", "test", "batch", "problems.jl"))
include(joinpath(@__DIR__, "..", "..", "MadIPMCore", "test", "gpu_ext.jl"))

@testset "cuMadIPM" begin
    @testset "Exports and the default linear solver" begin
        @test CUDSSSolver === MadCore.CUDSSSolver
        @test LapackCUDASolver === MadCore.LapackCUDASolver
        @test cuMadIPM.MPCSolver === MadIPMCore.MPCSolver
        @test cuMadIPM.madipm === MadIPMCore.madipm
        # cuDSS for models on the GPU, MUMPS (from the MadIPM bundle) for the others
        m = which(MadIPMCore.default_linear_solver, Tuple{NLPModels.AbstractNLPModel{Float64, CuVector{Float64}}})
        @test m.module === cuMadIPM
        @test MadIPMCore.default_linear_solver(simple_lp()) === MadCore.MumpsSolver
        mb = which(MadIPMCore.default_batch_linear_solver, Tuple{NLPModels.AbstractBatchNLPModel{Float64, CuMatrix{Float64}}})
        @test mb.module === cuMadIPM
        @test MadIPMCore.default_batch_linear_solver(BatchQuadraticModel([_lp(), _lp()])) === MadIPMCore.LoopedBatchLinearSolver
        # the vendor-independent GPU kernels load with the GPU extensions of MadCore
        @test Base.get_extension(MadIPMCore, :MadIPMCoreGPUExt) !== nothing
    end
    if CUDA.functional()
        test_madipmcore_gpu_ext(CuArray)
        include("test_gpu.jl")
        include("batch_gpu.jl")
    else
        @warn "No functional CUDA device: skipping the GPU solves"
    end
end
