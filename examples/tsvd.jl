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

    i = 1
    j = 1

    v = A'u - β*v
    v -= V[i]*(V[i]'v)
    α  = norm(v)
    v /= α
    αs = setindex(αs, α, j)
    V  = setindex(V, v, j)
    # A*v
    u  = A*v - α*u
    ## reorthogonalize
    u -= U[i]*(U[i]'u)

    β  = norm(u)
    u /= β
    βs = setindex(βs, β, j)
    U  = setindex(U, u, j + 1)
end
XLA.code_typed_xla(test, Tuple{typeof(A), typeof(initvec), Val{5}})[1]
@tpu_compile test(A, initvec, Val(5))
