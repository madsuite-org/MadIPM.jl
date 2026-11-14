# MadIPMCore on its own: no bundle, so the linear solver is passed explicitly.
using Test, MadIPMCore, MadCore, NLPModels, SparseArrays
import LDLFactorizations   # loads MadCore's LDLSolver
import MadIPMCore.Models: LPData, LinearModel

@testset "MadIPMCore" begin
    # min x₁ + x₂  s.t.  x₁ + x₂ = 1, x ≥ 0
    data = LPData(sparse([1, 1], [1, 2], [1.0, 1.0], 1, 2), ones(2); lvar = zeros(2), uvar = fill(Inf, 2), lcon = [1.0], ucon = [1.0])
    lp = LinearModel(data; x0 = ones(2), minimize = true, name = "simpleLP")
    stats = madipm(lp; linear_solver = MadCore.LDLSolver, print_level = MadCore.ERROR)
    @test stats.status == MadCore.SOLVE_SUCCEEDED
    @test stats.objective ≈ 1.0 atol = 1e-6

    # without a bundle there is no default linear solver: MadCore's placeholder, which errors when used
    @test MadIPMCore.default_linear_solver(lp) === MadCore.DummySolver
end
