import NNlib: softmax, logsoftmax, ∇softmax, ∇logsoftmax
import Flux: logitcrossentropy


## Mapping utilities
# Convert scalars to single-element XRTArrays with eltype Float32:
map_to_tpu(x::Real) = XRTArray(convert(Float32, x))

# Convert arrays to XRTArrays with eltype Float32
map_to_tpu(x::AbstractArray) = XRTArray(Float32.(x))

# Strip off the TrackedArray coating to get at the data underneath
map_to_tpu(x::TrackedArray) = map_to_tpu(Flux.data(x))

# Turn Chain objects into ImmutableChain objects which store the computation within their type signature
map_to_tpu(x::Chain) = ImmutableChain(tuple(map(map_to_tpu, x.layers)...))

# Convert BatchNorm layers into TPUBatchNorm layers, passing all children straight through,
# except for the "active" child, which is not used by the TPUBatchNorm
map_to_tpu(x::BatchNorm) = TPUBatchNorm(map(map_to_tpu, Flux.children(x))[1:end-1]...)

# For all other objects, just map the children through `map_to_tpu`.
map_to_tpu(x) = Flux.mapchildren(map_to_tpu, x)



## Layer reimplementations
# Work around inference limitations to dynamically find the number of
# batches held within a tensor.
neg_nbatch(x) = XRTArray(Float32(-size(x, ndims(x))))
Zygote.@nograd neg_nbatch

function logitcrossentropy(logŷ::XRTArray, y::XRTArray)
    return sum(y .* logsoftmax(logŷ)) / neg_nbatch(y)
end

# Define softmax and its gradient for XRTArrays
function softmax(xs::XRTArray)
    ys = xs .- maximum(xs, dims=1)
    zs = exp.(ys)
    return zs ./ sum(zs, dims=1)
end
function ∇softmax(Δ, xs::XRTArray)
    sf = softmax(xs)
    return sf .* (Δ .- sum(Δ .* sf, dims = 1))
end
Zygote.@adjoint softmax(xs::XRTArray) = softmax(xs), Δ -> (∇softmax(Δ, xs),)

# Define logsoftmax and its gradient for XRTArrays
function logsoftmax(xs::XRTArray)
    ys = xs .- maximum(xs, dims=1)
    zs = sum(exp.(ys), dims=1)
    return ys .- log.(zs)
end
∇logsoftmax(Δ, xs::XRTArray) = ∇softmax(Δ ./ max.(eps(eltype(xs)),softmax(xs)), xs)
Zygote.@adjoint logsoftmax(xs::XRTArray) = logsoftmax(xs), Δ -> (∇logsoftmax(Δ, xs),)



## High-level TPU programs
# Read (x, y) in from an HloInfeed
function get_minibatch_data(::Val{batch_size}) where {batch_size}
    # Construct HloInfeed object
    infeed = XLA.HloInfeed(Tuple{
        XRTArray{UInt32, (224*224*batch_size,), 1},
        XRTArray{UInt32, (batch_size,), 1},
    })
    # Read in from the infeed
    (x, y), _ = infeed(XLA.HloAfterAll()())
    x = reshape(x, (224, 224, batch_size))

    # Do pixel unpacking/channel normalization
    x = unpack_pixels(x)

    # Convert labels to onehot represnetation
    y = make_onehot(y)
    return x, y
end


function evaluate(::Val{batch_size}, model, nbatches) where {batch_size}
    while nbatches > XRTArray(0)
        # Get next minibatch of data after preprocessing it
        x, y_unused = get_minibatch_data(Val(batch_size))

        # Calculate forward pass
        y = model(x)

        # Outfeed y
        y = reshape(y, (1,))
        XLA.HloOutfeed()((y,), XLA.HloAfterAll()())

        # Count down the batches
        nbatches -= XRTArray(1)
    end
    return y
end

# Main program entry point that creates  
function epoch_loop(::Val{batch_size}, model, nbatches, η, β) where {batch_size}
    # Initialize optimizer state as zeros
    flat_model = flatten_tuple(model)
    opt_state = Zygote.map(x -> (zero(x), zero(x), β), flat_model)
    opt_state = unflatten_opt_state(model, opt_state)

    # Run until nbatches is zero
    while nbatches > XRTArray(0)
        # Get next minibatch of data after preprocessing it on-device
        mb_data = get_minibatch_data(Val(batch_size))

        # Let block to magically fend off the inference demons
        loss, back = let x = mb_data[1], y = mb_data[2]
            # Calculate forward pass to get loss, and compile backwards pass
            # to get the updates to our model weights.
            Zygote._forward(
                Zygote.Context{Nothing}(nothing),
                model -> logitcrossentropy(model(x), y),
                model,
            )
        end

        # Evaluate the backwards pass.  Zygote automatically calculates
        # sensitivities upon `x` and `y`; we discard those via the tail()
        updates = Zygote.tailmemaybe(back(1f0))[1]

        # Cross-replica sum our model updates to mix-n-match across all tpus
        updates = unflatten_tuple(updates,
            XLA.HloCrossReplicaSum{typeof(+)}((), 0, "")(
                +,
                flatten_tuple(updates)...
            )
        )

        # Update parameters via our optimizer
        model, opt_state = update_params(model, updates, opt_state, η, β)

        # Outfeed the loss
        loss = reshape(loss, (1,))
        XLA.HloOutfeed()((loss,), XLA.HloAfterAll()())

        # Count down the batches
        nbatches -= XRTArray(1)
    end
    return model
end
