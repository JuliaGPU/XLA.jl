using Flux

function make_onehot(::Type{T}, ::Val{alphabet_size}, labels::AbstractVector{S}) where {T, S, alphabet_size}
    bcast_labels = convert(XRTArray{T}, HloBroadcast((1,), (alphabet_size, length(labels)))(labels))
    iota = convert(XRTArray{T}, HloBroadcast((0,), (alphabet_size, length(labels)))(XRTArray(UInt32(1):UInt32(alphabet_size))))
    T.(iota .== bcast_labels)
end
@Zygote.nograd make_onehot
@Zygote.nograd Flux.OneHotMatrix

function Base.convert(::Type{<:XRTArray{T}}, m::Flux.OneHotMatrix) where {T}
    make_onehot(T, Val(size(m)[1]), m.data)
end
