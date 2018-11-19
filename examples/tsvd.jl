using Revise

using XLA
using Metalhead
using Flux
using TensorFlow
using StaticArrays
using TSVD
using LinearAlgebra

StaticArrays.similar_type(A::XRTArray{T}, s::Size{Sz}) where {T, Sz} = XRTArray{T, Sz, length(Sz)}
StaticArrays.similar_type(A::XRTArray{T}, ::Type{S}, s::Size{Sz}) where {T, S, Sz} = XRTArray{S, Sz, length(Sz)}
A = XRTArray(randn(Float32, 10, 10))
initvec = XRTArray(randn(Float32, 10))
maxsteps = 5

# Please excuse some of the messiness with manually wrapping
# the scalars below. That's be taken care of shortly.
function tsvd(A, initvec, ::Val{maxsteps}) where {maxsteps}
    m, n = size(A)

    # Initialize
    U  = zeros(similar_type(initvec, typeof(initvec), Size(maxsteps + 1)))
    V  = zeros(similar_type(initvec, typeof(initvec), Size(maxsteps)))
    αs = zeros(similar_type(initvec, Size(maxsteps)))
    βs = zeros(similar_type(initvec, Size(maxsteps)))

    nrmInit = norm(initvec)
    v  = A'initvec
    v /= nrmInit
    α  = norm(v)
    v /= α
    V  = setindex(V, v, 0)
    αs = setindex(αs, α, 0)

    uOld = initvec/nrmInit
    u  = A*v - α*uOld
    β  = norm(u)
    u /= β
    U  = setindex(U, uOld, 0)
    U  = setindex(U, u, 0)
    βs = setindex(βs, β, 0)

    j = XRTArray(1)
    while convert(Bool, j < XRTArray(maxsteps))
        # A'u
        v = A'u - β*v
        i = XRTArray(0)
        while convert(Bool, i <= j - XRTArray(1))
            ## reorthogonalize
            v -= V[i]*(V[i]'v)
            i += XRTArray(1)
        end

        α  = norm(v)

        v /= α
        αs = setindex(αs, α, j)
        V  = setindex(V, v, j)

        # A*v
        u  = A*v - α*u
        ## reorthogonalize
        i = XRTArray(0)
        while convert(Bool, i <= j)
            u -= U[i]*(U[i]'u)
            i += XRTArray(1)
        end

        β  = norm(u)
        u /= β
        βs = setindex(βs, β, j)
        U  = setindex(U, u, j + XRTArray(1))

        j += XRTArray(1)
    end
    return αs, βs, U, V
end

@tpu_compile tsvd(A, initvec, Val(5))
