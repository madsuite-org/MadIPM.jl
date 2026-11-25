# Tests of MadIPMCoreGPUExt for the GPU bundles: each kernel on the arrays of a backend against
# the CPU method of the same function. A bundle includes this file and calls
# `test_madipmcore_gpu_ext(ArrayType)`, e.g. cuMadIPM with `CuArray`. `T` is the element type of
# the floating-point data (Float32 for backends without Float64).
using Test, Random, SparseArrays
import MadCore, MadIPMCore

function test_madipmcore_gpu_ext(AT; T = Float64)
  Scaling = MadIPMCore.Models.Scaling
  gpu(x) = AT(x)
  rand_(rng, dims...) = rand(rng, T, dims...)
  randn_(rng, dims...) = randn(rng, T, dims...)
  @testset "MadIPMCoreGPUExt on $AT{$T}" begin
    @test Base.get_extension(MadIPMCore, :MadIPMCoreGPUExt) !== nothing
    rng = Random.Xoshiro(3)

    @testset "batch views" begin
        n, bs = 5, 6
        roots = Int32[2, 3, 5]
        view_ = MadIPMCore.BatchView(bs, 1, length(roots), roots, Int32.(1:length(roots)), gpu(roots))
        src = rand_(rng, n, bs)
        dst = zeros(T, n, length(roots)); dst_g = gpu(dst)
        MadIPMCore.gather_batch_view_columns!(dst, src, view_)
        MadIPMCore.gather_batch_view_columns!(dst_g, gpu(src), view_)
        @test Array(dst_g) == dst == src[:, roots]

        out = zeros(T, n, bs); out_g = gpu(out)
        MadIPMCore.scatter_batch_view_columns!(out, dst, view_)
        MadIPMCore.scatter_batch_view_columns!(out_g, gpu(dst), view_)
        @test Array(out_g) == out

        A = copy(src); A_g = gpu(A)
        MadIPMCore.compact_active_columns_inplace!(A, view_)
        MadIPMCore.compact_active_columns_inplace!(A_g, view_, gpu(similar(src)))
        @test Array(A_g)[:, 1:length(roots)] == A[:, 1:length(roots)]
    end

    @testset "batch KKT kernels" begin
        nv, bs = 8, 3
        ind_lb, ind_ub = [1, 3, 4], [2, 4]
        lb_off, ub_off = nv, nv + length(ind_lb)
        vals = rand_(rng, nv + length(ind_lb) + length(ind_ub), bs)
        l_diag, u_diag = rand_(rng, 3, bs)  .+ one(T), rand_(rng, 2, bs) .+ one(T)
        l_lower, u_lower = rand_(rng, 3, bs), rand_(rng, 2, bs)
        for (f!, args) in (
            (MadIPMCore._reduce_rhs_batch!, (ind_lb, lb_off, l_diag, ind_ub, ub_off, u_diag)),
            (MadIPMCore._finish_aug_solve_batch!, (ind_lb, lb_off, l_lower, l_diag, ind_ub, ub_off, u_lower, u_diag)),
        )
            v = copy(vals); v_g = gpu(vals)
            f!(v, args...)
            f!(v_g, (a isa AbstractArray ? gpu(a) : a for a in args)...)
            @test Array(v_g) ≈ v
        end

        jac_I = [1, 2, 2, 3, 1]
        jac_buffer = randn_(rng, length(jac_I), bs)
        cs = ones(T, 3, bs); cs_g = gpu(cs)
        MadCore.update_con_scale_sparse!(cs, jac_I, jac_buffer)
        MadCore.update_con_scale_sparse!(cs_g, gpu(jac_I), gpu(jac_buffer))
        @test Array(cs_g) == cs
    end

    @testset "Ruiz scaling kernels" begin
        m, n = 6, 5
        S = sprand(rng, T, m, n, 0.5) + sparse(1:min(m, n), 1:min(m, n), one(T), m, n)
        I, J, V = findnz(S)
        A = Scaling.SparseCOO(m, n, I, J, V)
        A_g = Scaling.SparseCOO(m, n, gpu(I), gpu(J), gpu(V))
        drow, dcol = rand_(rng, m) .+ T(0.5), rand_(rng, n) .+ T(0.5)
        rowpair, colpair = collect(1:m), collect(1:n)
        rowpair[2] = 0; colpair[4] = 0
        rn, cn = zeros(T, m), zeros(T, n); rn_g, cn_g = gpu(rn), gpu(cn)
        Scaling.scaled_maxabs!(rn, cn, rowpair, colpair, A, drow, dcol)
        Scaling.scaled_maxabs!(rn_g, cn_g, gpu(rowpair), gpu(colpair), A_g, gpu(drow), gpu(dcol))
        @test Array(rn_g) == rn && Array(cn_g) == cn

        rc, cc = fill(typemax(Int), m), fill(typemax(Int), n); rc_g, cc_g = gpu(rc), gpu(cc)
        Scaling.scaled_argmax!(rc, cc, rn, cn, rowpair, colpair, A, drow, dcol)
        Scaling.scaled_argmax!(rc_g, cc_g, rn_g, cn_g, gpu(rowpair), gpu(colpair), A_g, gpu(drow), gpu(dcol))
        @test Array(rc_g) == rc && Array(cc_g) == cc

        rp, cp = [3, 0, 1, 0, 2, 0], [3, 5, 1, 0, 0]
        rp_g, cp_g = gpu(rp), gpu(cp)
        Scaling.settle_pairs!(rp, cp)
        Scaling.settle_pairs!(rp_g, cp_g)
        @test Array(rp_g) == rp && Array(cp_g) == cp
    end

    @testset "NormalKKTSystem kernels" begin
        src, map = randn_(rng, 9), [1, 3, 3, 2, 5, 1, 4, 4, 2]
        dest = zeros(T, 5); dest_g = gpu(dest)
        MadCore.coo_to_csc_values!(dest, src, map)
        MadCore.coo_to_csc_values!(dest_g, gpu(src), gpu(map))
        @test Array(dest_g) ≈ dest

        # C = J D Jᵀ, with Jᵀ and C in CSR (the CSC arrays of their transposes)
        nr, nc = 6, 9
        J = sprand(rng, T, nr, nc, 0.4) + sparse(1:nr, 1:nr, one(T), nr, nc)
        Dx = rand_(rng, nc) .+ one(T)
        Jt = sparse(J')            # CSC of Jᵀ = CSR of J
        C = sparse((J * spdiagm(Dx) * J')')
        args(f) = (f(Jt.colptr), f(Jt.rowval), f(Jt.nzval), f(C.colptr), f(C.rowval))
        Cx = zeros(T, nnz(C)); Cx_g = gpu(Cx)
        MadIPMCore.assemble_normal_system!(nr, nc, args(identity)..., Cx, Dx)
        MadIPMCore.assemble_normal_system!(nr, nc, args(gpu)..., Cx_g, gpu(Dx))
        @test Array(Cx_g) ≈ Cx
        @test Cx ≈ nonzeros(C)
    end
end
end
