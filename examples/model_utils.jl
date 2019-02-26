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
