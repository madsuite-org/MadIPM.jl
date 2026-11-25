# Kernels of NormalKKTSystem: the assembly of A D Aᵀ. The COO to CSC transfer is MadCore's
# `coo_to_csc_values!`, which has a GPU method in the GPU extension of MadCore.

@kernel function assemble_normal_system_kernel!(@Const(n_rows), @Const(n_cols), @Const(Jtp), @Const(Jtj), @Const(Jtx),
                                                @Const(Cp), @Const(Cj), Cx, @Const(Dx), @Const(Tv))
    i = @index(Global, Linear)

    for c in Cp[i]:Cp[i+1]-1
        j = Cj[c]
        acc = zero(Tv)

        p1 = Jtp[i]
        p2 = Jtp[j]
        p1_end = Jtp[i+1] - 1
        p2_end = Jtp[j+1] - 1

        while p1 <= p1_end && p2 <= p2_end
            k1 = Jtj[p1]
            k2 = Jtj[p2]

            if k1 == k2
                acc += Jtx[p1] * Dx[k1] * Jtx[p2]
                p1 += 1
                p2 += 1
            elseif k1 < k2
                p1 += 1
            else
                p2 += 1
            end
        end

        Cx[c] = acc
    end
    nothing
end

function MadIPMCore.assemble_normal_system!(
    n_rows,
    n_cols,
    Jtp::AbstractGPUVector{Ti},
    Jtj::AbstractGPUVector{Ti},
    Jtx::AbstractGPUVector{Tv},
    Cp::AbstractGPUVector{Ti},
    Cj::AbstractGPUVector{Ti},
    Cx::AbstractGPUVector{Tv},
    Dx::AbstractGPUVector{Tv},
) where {Ti, Tv}
    backend = get_backend(Cx)
    kernel! = assemble_normal_system_kernel!(backend)
    kernel!(n_rows, n_cols, Jtp, Jtj, Jtx, Cp, Cj, Cx, Dx, Tv; ndrange = n_rows)
end
