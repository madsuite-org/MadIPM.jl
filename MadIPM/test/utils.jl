# Test helpers shared by the MadIPM and cuMadIPM tests
import MadIPM.Models: LPData, QPData, LinearModel, QuadraticModel
using SparseMatricesCOO: SparseMatrixCOO

function QuadraticModel(
    c::AbstractVector{T},
    Hrows::AbstractVector{<:Integer},
    Hcols::AbstractVector{<:Integer},
    Hvals::AbstractVector{T};
    Arows::AbstractVector{<:Integer} = Int[],
    Acols::AbstractVector{<:Integer} = Int[],
    Avals::AbstractVector{T} = T[],
    lcon::AbstractVector{T} = T[],
    ucon::AbstractVector{T} = T[],
    lvar::AbstractVector{T} = fill(T(-Inf), length(c)),
    uvar::AbstractVector{T} = fill(T(Inf), length(c)),
    c0::Real = zero(T),
    x0::AbstractVector{T} = zeros(T, length(c)),
    y0::AbstractVector{T} = T[],
    minimize::Bool = true,
    name::String = "QP",
) where {T}
    nvar = length(c)
    ncon = max(length(lcon), length(ucon), isempty(Arows) ? 0 : maximum(Arows))
    A = SparseMatrixCOO(
        ncon,
        nvar,
        Vector{Int}(Arows),
        Vector{Int}(Acols),
        Vector{T}(Avals),
    )
    H = SparseMatrixCOO(
        nvar,
        nvar,
        Vector{Int}(Hrows),
        Vector{Int}(Hcols),
        Vector{T}(Hvals),
    )
    lcon_ = isempty(lcon) ? fill(T(-Inf), ncon) : Vector{T}(lcon)
    ucon_ = isempty(ucon) ? fill(T(Inf), ncon) : Vector{T}(ucon)
    y0_ = length(y0) == ncon ? Vector{T}(y0) : zeros(T, ncon)
    data = QPData(
        A,
        Vector{T}(c),
        H;
        lvar = Vector{T}(lvar),
        uvar = Vector{T}(uvar),
        lcon = lcon_,
        ucon = ucon_,
        c0 = T(c0),
    )
    return QuadraticModel(
        data;
        x0 = Vector{T}(x0),
        y0 = y0_,
        minimize = minimize,
        name = name,
    )
end

function simple_lp()
    c = ones(2)
    Hrows = Int[]
    Hcols = Int[]
    Hvals = Float64[]
    Arows = [1, 1]
    Acols = [1, 2]
    Avals = [1.0; 1.0]
    c0 = 0.0
    lvar = [0.0; 0.0]
    uvar = [Inf; Inf]
    lcon = [1.0]
    ucon = [1.0]
    x0 = ones(2)

    return QuadraticModel(
        c,
        Hrows,
        Hcols,
        Hvals,
        Arows = Arows,
        Acols = Acols,
        Avals = Avals,
        lcon = lcon,
        ucon = ucon,
        lvar = lvar,
        uvar = uvar,
        c0 = c0,
        x0 = x0,
        name = "simpleLP",
    )
end
