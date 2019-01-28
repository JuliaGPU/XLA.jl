struct ADAM
    # This `state` is a recursive named tuple that mimicks the structure
    # of the actual model weights, only for each parameter, this `state`
    # holds a tuple of state for that weight (mt, vt, βp)
    state::Tuple
	η
    β
end

function ADAM(model::ImmutableChain, η, β)
    # Take in the model, flatten it into a tuple and map each parameter to (w1, w2, β)
    flat_model = flatten_tuple(model)
    opt_state = Zygote.map(x -> (zero(x), zero(x), β), flat_model)
    
    # We unflatten so that the optimizer state retains the same structure as the model,
    # so that our update!() function can simply walk the 
    opt_state = unflatten_opt_state(tpu_model, opt_state)
    return ADAM(opt_state, η, β)
end


function update!(state::Tuple, model::XRTArray, Δ::XRTArray, η, β)
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
update!(state::Tuple, model, Δ::Nothing, η, β) = (state, model)

function update!(state::Tuple, model, Δ, η, β)
    # Base condition; if we have reached a leaf node return the inputs unchanged.
    # Note that if `model` is an XRTArray, we will hit the override above that actually
    # updates the model rather than this generic update!(), same for if Δ is `nothing`.
    if nfields(model) == 0
        return (opt_state, model)
    end

    # Recursively call update!(), delving into the state, model and updates,
    # which should have very similar structures.
    new_fields_and_state = ntuple(Val(nfields(model))) do i
        update!(getfield(state, i), getfield(model, i), getfield(Δ, i), η, β)
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
        return (tuple(new_state...), tuple(new_fields...))
    else
        return (tuple(new_state...), typeof(model)(new_fields...))
    end
end


# Entry point for optimizer update step; immediately start recursing over
# the values held within our optimizer state, and return (new_state, new_model)
update!(opt::ADAM, model::ImmutableChain, Δ) = update!(opt.state, model, Δ, opt.η, opt.β)





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


