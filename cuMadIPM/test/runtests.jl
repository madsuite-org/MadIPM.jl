using Test, LinearAlgebra, SparseArrays, NLPModels
using cuMadIPM, MadIPMCore, MadCore, MadCoreGPU, CUDA
using MadIPM, MadNLP   # CPU references and the option values of MadNLP

include(joinpath(@__DIR__, "..", "..", "MadIPM", "test", "utils.jl"))

@testset "cuMadIPM" begin
    @testset "Exports and the default linear solver" begin
        @test CUDSSSolver === MadCoreGPU.CUDSSSolver
        @test LapackCUDASolver === MadCoreGPU.LapackCUDASolver
        @test cuMadIPM.MPCSolver === MadIPMCore.MPCSolver
        @test cuMadIPM.madipm === MadIPMCore.madipm
        # cuDSS for models on the GPU, MUMPS (from the MadIPM bundle) for the others
        m = which(MadIPMCore.default_linear_solver, Tuple{NLPModels.AbstractNLPModel{Float64, CuVector{Float64}}})
        @test m.module === cuMadIPM
        @test MadIPMCore.default_linear_solver(simple_lp()) === MadCore.MumpsSolver
    end
    if CUDA.functional()
        include("test_gpu.jl")
        include("batch_gpu.jl")
    else
        @warn "No functional CUDA device: skipping the GPU solves"
    end
end
