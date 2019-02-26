struct TPU_SGD
    # Learning rate; the only data this optimizer needs to bundle with itself
    η::XRTArray{Float32,(),0}
end

# Simplest update step in existence.
sgd_update!(model::XRTArray, Δ::XRTArray, η) = model - (Δ .* η)

# If this leaf node had no updates calculated for it, then skip out early.
sgd_update!(model, Δ::Nothing, η) = model

function sgd_update!(model, Δ, η)
    # Base condition; if we have reached a leaf node return the inputs unchanged.
    # Note that if `model` is an XRTArray, we will hit the override above that actually
    # updates the model rather than this generic update!(), same for if Δ is `nothing`.
    if nfields(model) == 0
        return model
    end
    
    # Recursively pass the fields of this model through the update machinery.  We use
    # this strange ntuple() do-block because we cannot perform any kind of mutation
    # (such as push!()'ing onto a list) and so we adopt this more functional-style of
    # programming.
    new_fields = ntuple(Val(nfields(model))) do i
        return sgd_update!(getfield(model, i), getfield(Δ, i), η)
    end
    
    # Return something of the same type as `model`, but with the new fields
    if isa(model, Tuple)
        return tuple(new_fields...)
    else
        return typeof(model)(new_fields...)
    end
end

# Main entry point for this optimizer's update steps
update!(opt::TPU_SGD, model, Δ) = sgd_update!(model, Δ, opt.η)





struct TPU_ADAM{T<:Tuple}
    # This `state` is a recursive named tuple that mimicks the structure
    # of the actual model weights, only for each parameter, this `state`
    # holds a tuple of state for that weight (mt, vt, βp)
    state::T
    η::XRTArray{Float32,(),0}
    β::Tuple{XRTArray{Float32,(),0},XRTArray{Float32,(),0}}
end

function TPU_ADAM(model::ImmutableChain, η, β)
    # Take in the model, flatten it into a tuple and map each parameter to (w1, w2, β)
    flat_model = XLA.flatten_tuple(model)
    opt_state = Zygote.map(x -> (zero(x), zero(x), β), flat_model)
    
    # We unflatten so that the optimizer state retains the same structure as the model,
    # so that our update!() function can simply walk the 
    opt_state = unflatten_opt_state(model, opt_state)
    return TPU_ADAM(opt_state, η, β)
end


function adam_update!(state::Tuple, model::XRTArray, Δ::XRTArray, η, β)
    # Destructure the state stored within `state`
    mt, vt, βp = state
    
    # Update `mt` and `vt` with statistics of `Δ`
    mt = β[1] .* mt .+ (XRTArray(1f0) .- β[1]) .* Δ
    vt = β[2] .* vt .+ (XRTArray(1f0) .- β[2]) .* Δ.^XRTArray(2f0)

    # Package these statistics up as the new state
    new_state = (mt, vt, βp .* β)
    
    # Calculate new weights by taking the statistics and subtracting them from the model
    vt_denom = sqrt.(vt ./ (XRTArray(1f0) .- βp[2])) .+ XRTArray(eps(Float32))
    Δ_model = mt ./ ((XRTArray(1f0) .- βp[1]) .* vt_denom)
    new_model = model .- η .* Δ_model
    
    # Return a tuple of updated (state, model)
    return (new_state, new_model)
end

# If this leaf node had no updates calculated for it, then skip out early.
adam_update!(state::Tuple, model, Δ::Nothing, η, β) = (state, model)

function adam_update!(state::Tuple, model, Δ, η, β)
    # Base condition; if we have reached a leaf node return the inputs unchanged.
    # Note that if `model` is an XRTArray, we will hit the override above that actually
    # updates the model rather than this generic update!(), same for if Δ is `nothing`.
    if nfields(model) == 0
        return (state, model)
    end

    # Recursively call update!(), delving into the state, model and updates,
    # which should have very similar structures.
    new_fields_and_state = ntuple(Val(nfields(model))) do i
        return adam_update!(getfield(state, i), getfield(model, i), getfield(Δ, i), η, β)
    end
    
    # Destructure combined (new_state, new_weights) return value from above.  We use
    # this strange ntuple() do-block syntax because we cannot push!() elements onto
    # lists and create tuples from them, we must program in a more functional style,
    # honoring the immutable structure of the program.
    new_state = ntuple(Val(nfields(model))) do i
        return new_fields_and_state[i][1]
    end
    new_fields = ntuple(Val(nfields(model))) do i
        return new_fields_and_state[i][2]
    end

    # Construct tuples that contain the model and state that we have updated.
    if isa(model, Tuple)
        return (new_state, new_fields)
    else
        return (tuple(new_state...), typeof(model)(new_fields...))
    end
end


# Entry point for optimizer update step; immediately start recursing over
# the values held within our optimizer state, and return (new_state, new_model)
function update!(opt::TPU_ADAM, model::ImmutableChain, Δ)
    new_opt_state, model = adam_update!(opt.state, model, Δ, opt.η, opt.β)
    return TPU_ADAM(new_opt_state, opt.η, opt.β), model
end





# Helper function to unflatten the optimizer state.  Very similar to unflatten_tuple(),
# but emits tuples and recurses slightly differently.
@generated function unflatten_opt_state(pattern, t)
    function expand_arg(pattern, idx)
        fields = Any[]
        for i = 1:fieldcount(pattern)
            fT = fieldtype(pattern, i)
            if fT === Nothing
                push!(fields, nothing)
                continue
            end
            if !(fT <: AbstractArray)
                arg, idx = expand_arg(fT, idx)
                push!(fields, arg)
            else
                push!(fields, Expr(:call, :getfield, :t, idx))
                idx += 1
            end
        end
        Expr(:call, tuple, fields...), idx
    end
    res = expand_arg(pattern, 1)[1]
    return res
end


