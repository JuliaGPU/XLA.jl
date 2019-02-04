using Flux

function make_onehot(::Type{T}, ::Val{alphabet_size}, labels::AbstractVector{S}) where {T, S, alphabet_size}
    batch_size = length(labels)
    operand = zeros(XRTArray{T, (alphabet_size, batch_size), 2})
    updates = XLA.HloBroadcast((), (1, batch_size))(XRTArray(1f0))
    update_computation = (x, y) -> y
    scatter_indices = hcat(XRTArray(S(0):(S(batch_size-1))), labels)
    sdims = XLA.ScatterDimNums(
        #= update_window_dims =# (0,),
        #= inserted_window_dims =# (0,),
        #= scatter_dims_to_operand_dims =# (1, 0),
        #= index_vector_dim =# 1
    )
    return XLA.HloScatter{typeof(update_computation)}(sdims)(
        update_computation,
        operand,
        scatter_indices,
        updates
    )   
end
@Zygote.nograd make_onehot
@Zygote.nograd Flux.OneHotMatrix

function Base.convert(::Type{<:XRTArray{T}}, m::Flux.OneHotMatrix) where {T}
    make_onehot(T, Val(size(m)[1]), m.data)
end
