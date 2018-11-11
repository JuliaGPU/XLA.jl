using Revise

using XLA
using TensorFlow
using Metalhead
using Flux
using Zygote
using Statistics

include("resnet.jl")
include("tpu_batch_norm.jl")

map_to_tpu(x::Bool) = x
map_to_tpu(x::Real) = XRTArray(convert(Float32, x))
map_to_tpu(x::Chain) = ImmutableChain(map(map_to_tpu, x.layers)...)
map_to_tpu(x::AbstractArray) = XRTArray(Float32.(Flux.data(x)))
map_to_tpu(x) = Flux.mapchildren(map_to_tpu, x)
resnet_host = Flux.mapleaves(x->isa(x, AbstractArray) ? Float32.(Flux.data(x)) : x, resnet50())
resnet = map_to_tpu(resnet_host)

function my_derivative_with_loss(f, args...)
  y, back = Zygote._forward(Zygote.Context{Nothing}(nothing), f, args...)
  J = Δ -> Zygote.tailmemaybe(back(Δ))
  return y, J(1f0)[1]
end

function crossentropy(ŷ::AbstractVecOrMat, y::XRTArray{<:Any, Sz}) where {Sz}
    weight = 1f0 # one(eltype(y))
    z = sum(y .* log.(ŷ) .* weight)
    return z / XRTArray(1f0)
end

update_params(model::XRTArray, updates::XRTArray, η) = model + updates * η
update_params(model, updates::Nothing, η) = model
function update_params(model, updates, η)
    new_fields = ntuple(Val(nfields(model))) do i
        update_params(getfield(model, i), getfield(updates, i), η)
    end
    isa(model, Tuple) ?
        tuple(new_fields...) :
        typeof(model)(new_fields...)
end

function make_one_hot(data)
    operand = zero(XRTArray{Float32, (1000, length(data)), 2})
    updates = XLA.HloBroadcast((), (1, length(data)))(XRTArray(1f0))
    update_computation = (x,y)->y
    scatter_indices = hcat(XRTArray(0:(length(data)-1)), data)
    sdims = XLA.ScatterDimNums(
        #= update_window_dims =# (0,),
        #= inserted_window_dims =# (0,),
        #= scatter_dims_to_operand_dims =# (1, 0),
        #= index_vector_dim =# 1
    )
    XLA.HloScatter{typeof(update_computation)}(sdims)(
            update_computation,
            operand,
            scatter_indices,
            updates
        )
end

function epoch_loop(::Val{batch_size}, xrtic, nbatches, η) where {batch_size}
    while nbatches > XRTArray(0)
        # N.B: Ideally we'd have the labels be UInt16, but that doesn't work yet on TPUs
        (x, labels), _ = XLA.HloInfeed(Tuple{XRTArray{Float32, (224, 224, 3, batch_size), 4},
                                        XRTArray{UInt32, (batch_size,), 1}})(XLA.HloAfterAll()())
        y = make_one_hot(labels)
        loss, updates = my_derivative_with_loss(xrtic->crossentropy(xrtic(x), y), xrtic)
        xrtic = update_params(xrtic, updates, η)
        XLA.HloOutfeed()((loss,), XLA.HloAfterAll()())
        nbatches -= XRTArray(1)
    end
    return xrtic
end

sess = Session(Graph(); target="grpc://localhost:8470")

batch_size = 128
nbatches = 10_000
compld = @tpu_compile epoch_loop(Val(batch_size), resnet, XRTArray(nbatches), XRTArray(0.1f0))
