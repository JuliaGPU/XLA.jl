using Zygote
using ForwardDiff
using DiffRules
using SpecialFunctions
using NaNMath
using Zygote: @adjoint

ForwardDiff.can_dual(::Type{XRTArray{T,(),0}} where T) = true
Zygote.isscalar(::XRTArray{<:Any,(),0}) = true
Zygote.fill_similar_array(x::XRTArray{T, dims, N}, v) where {T, dims, N} =
    HloBroadcast((), dims)(convert(XRTArray{T, (), 0}, v))

using Base.Broadcast
using Base.Broadcast: Broadcasted, materialize
function Zygote.broadcast_gradient(bc::Broadcasted{S}, ::Type{T}) where {S <: XLA.XLA.XRTArrayStyle, T}
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

for (M, f, arity) in DiffRules.diffrules()
  arity == 1 || continue
  @eval begin
    @adjoint $M.$f(x::XRTScalar) = $M.$f(x),
      Δ -> (Δ * $(DiffRules.diffrule(M, f, :x)),)
  end
end

for (M, f, arity) in DiffRules.diffrules()
  arity == 2 || continue
  da, db = DiffRules.diffrule(M, f, :a, :b)
  @eval begin
    @adjoint $M.$f(a::XRTScalar, b::XRTScalar) = $M.$f(a, b),
      Δ -> (Δ * $da, Δ * $db)
    @adjoint $M.$f(a::Number, b::XRTScalar) = $M.$f(a, b),
      Δ -> (Δ * $da, Δ * $db)
    @adjoint $M.$f(a::XRTScalar, b::Number) = $M.$f(a, b),
      Δ -> (Δ * $da, Δ * $db)
  end
end

@adjoint XRTArray(a::Real) = XRTArray(a), Δ -> (Δ,)

# This is necessary, because even XRT scalars are AbstractArrays
Zygote.accum(a::XRTArray, b::XRTArray) = a+b

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

@adjoint function getindex(xs::XRTArray, i...)
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

@adjoint function Base.repeat(A::XRTArray, dims::Integer...)
    repeat(A, dims...), let Asz=Val(size(A)), vdims=Val(dims)
        function (Δ)
            sz_A = unval(Asz)
            ddims = unval(vdims)
            dims_mapping, result_szs, final_sz = make_repeat_dims(sz_A, ddims)
            folded_back = XLA.HloReshape(result_szs)(Δ)
            summed = HloReduceWindow{typeof(+)}(
                make_repeat_adjoint_window(result_szs, dims_mapping)
            )(+, folded_back, XRTArray(typemin(eltype(x))))
            HloReshape(sz_A)(summed)
        end
    end
end

@Zygote.nograd count_summands
