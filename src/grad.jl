using Zygote

Zygote.isscalar(::XRTArray{<:Any,(),0}) = true
Zygote.fill_similar_array(x::XRTArray{T, dims, N}, v) where {T, dims, N} =
    HloBroadcast((), dims)(XRTArray(convert(T, v)))

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
