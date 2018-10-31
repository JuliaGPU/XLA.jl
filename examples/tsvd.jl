using Revise

using XLA
using Metalhead
using Flux
using TensorFlow
using StaticArrays
using TSVD
using LinearAlgebra

pop!(Base.Multimedia.displays)


StaticArrays.similar_type(A::XRTArray{T}, s::Size{Sz}) where {T, Sz} = XRTArray{T, Sz, length(Sz)}
StaticArrays.similar_type(A::XRTArray{T}, ::Type{S}, s::Size{Sz}) where {T, S, Sz} = XRTArray{S, Sz, length(Sz)}
A = XRTArray(randn(10, 10))
initvec = XRTArray(randn(10))
maxsteps = 5


function test(A, initvec, ::Val{maxsteps}) where {maxsteps}
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
    V  = setindex(V, v, 1)
    αs = setindex(αs, α, 1)

    uOld = initvec/nrmInit
    u  = A*v - α*uOld
    β  = norm(u)
    u /= β
    U  = setindex(U, uOld, 1)
    U  = setindex(U, u, 2)
    βs = setindex(βs, β, 1)

    i = XRTArray(2)
    while convert(Bool, i <= XRTArray(maxsteps))
        # A'u
        v = A'u - β*v
        ## reorthogonalize
        v -= V[i]*(V[i]'v)

        α  = norm(v)
        v /= α
        αs = setindex(αs, α, i)
        V  = setindex(V, v, i)

        i += XRTArray(1)
    end
    return αs, βs, U, V
end
@tpu_compile test(A, initvec, Val(5))
