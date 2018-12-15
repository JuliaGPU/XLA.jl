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
  end
end

@adjoint XRTArray(a::Real) = XRTArray(a), Δ -> (Δ,)

# This is necessary, because even XRT scalars are AbstractArrays
Zygote.accum(a::XRTArray, b::XRTArray) = a+b

onelike(x) = Int8(1)
onelike(x::XRTArray{T}) where T = XRTArray(one(T))

function grad(f, x)
  y, back = Zygote._forward(Zygote.Context{Nothing}(nothing), f, x)
  return back(onelike(x))[2]
end
