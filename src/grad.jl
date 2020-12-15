using Zygote
using ForwardDiff
using DiffRules
using SpecialFunctions
using NaNMath
using Zygote: @adjoint

ForwardDiff.can_dual(::Type{HLOArray{T,(),0}} where T) = true
#Zygote.isscalar(::HLOArray{<:Any,(),0}) = true

#Zygote.fill_similar_array(x::HLOArray{T, dims, N}, v::HLOScalar) where {T, dims, N} =
#    HloBroadcast((), dims)(v)
#Zygote.fill_similar_array(x::HLOArray{T, dims, N}, v::XLAScalar) where {T, dims, N} =
#    Zygote.fill_similar_array(x, convert(HLOArray{T, (), 0}, v))

sum_adjoint(xs::HLOArray, dims::Colon) = sum(xs), Δ->(Zygote.fill_similar_array(xs, Δ),)
@Base.pure function compute_sum_adj_dims(sz, dim)
    a = Int[]
    b = Int[]
    for i = 0:length(sz)-1
        if i + 1 == dim
            # omit it from both
        else
            push!(a, sz[i + 1])
            push!(b, i)
        end
    end
    tuple(a...), tuple(b...)
end
function sum_adjoint(xs::HLOArray{<:Any, rt_dims}, dim::Int) where rt_dims
    sum(xs, dims=dim), let vdim=Val(dim)
        Δ -> begin
            dims_dropped, all_but_dim = compute_sum_adj_dims(size(Δ), unval(vdim))
            (HloBroadcast(all_but_dim, rt_dims)(HloReshape(dims_dropped)(Δ)),)
        end
    end
end

@adjoint function sum(xs::HLOArray; dims = :)
  sum_adjoint(xs, dims)
end

using Base.Broadcast
using Base.Broadcast: Broadcasted, materialize

#=
function Zygote.broadcast_gradient(bc::Broadcasted{S}, ::Type{T}) where {S <: XLA.XLA.HLOArrayStyle, T}
    # XLA doesn't currently have multi-output kMap
    dest = let f = bc.f
        materialize(Broadcasted{S}((args...)->f(args...).value, bc.args, bc.axes))
    end
    grads = ntuple(length(bc.args)) do i
        let f = bc.f
            materialize(Broadcasted{S}((args...)->f(args...).partials.values[i], bc.args, bc.axes))
        end
    end
    dest, grads
end
=#

for (M, f, arity) in DiffRules.diffrules()
  arity == 1 || continue
  @eval begin
    @adjoint $M.$f(x::HLOScalar) = $M.$f(x),
      Δ -> (Δ * $(DiffRules.diffrule(M, f, :x)),)
  end
end

for (M, f, arity) in DiffRules.diffrules()
  arity == 2 || continue
  da, db = DiffRules.diffrule(M, f, :a, :b)
  @eval begin
    @adjoint $M.$f(a::HLOScalar, b::HLOScalar) = $M.$f(a, b),
      Δ -> (Δ * $da, Δ * $db)
    @adjoint $M.$f(a::Number, b::HLOScalar) = $M.$f(a, b),
      Δ -> (Δ * $da, Δ * $db)
    @adjoint $M.$f(a::HLOScalar, b::Number) = $M.$f(a, b),
      Δ -> (Δ * $da, Δ * $db)
  end
end

@adjoint HLOArray(a::Real) = HLOArray(a), Δ -> (Δ,)

# This is necessary, because even XRT scalars are AbstractArrays
Zygote.accum(a::HLOArray, b::HLOArray) = a+b

@adjoint function execute(args...)
    error("Zygote should not reach the HLO layer. Please define gradients for functions higher up.")
end

struct getindex_back{T, Ti}
    i::Ti
end

function (gb::getindex_back{T, Ti})(Δ) where {T, Ti}
    Δ′ = zero(T)
    Δ′ = Base.setindex(Δ′, Δ, gb.i...)
    (Δ′, ntuple(_->nothing, Val(length(Ti.parameters)))...)
end

@adjoint function getindex(xs::HLOArray, i...)
    xs[i...], getindex_back{typeof(xs), typeof(i)}(i)
end

@Base.pure function make_repeat_adjoint_window(folded_sz, dims_mapping)
    window = ntuple(length(folded_sz)) do i
        if (i - 1) in dims_mapping
            WindowDims(1, 1, 0, 0, 1, 1, false)
        else
            WindowDims(folded_sz[i], 1, 0, 0, 1, 1, false)
        end
    end
end

unval(v::Val{x}) where {x} = x

@adjoint function Base.repeat(A::HLOArray, dims::Integer...)
    repeat(A, dims...), let Asz=Val(size(A)), vdims=Val(dims)
        function (Δ)
            sz_A = unval(Asz)
            ddims = unval(vdims)
            dims_mapping, result_szs, final_sz = make_repeat_dims(sz_A, ddims)
            folded_back = XLA.HloReshape(result_szs)(Δ)
            summed = HloReduceWindow{typeof(+)}(
                make_repeat_adjoint_window(result_szs, dims_mapping)
            )(+, folded_back, HLOArray(zero(eltype(Δ))))
            (HloReshape(sz_A)(summed), ntuple(_->nothing, length(ddims))...)
        end
    end
end

#@Zygote.nograd count_summands
