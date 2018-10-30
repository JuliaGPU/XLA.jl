function emulate(op::HloRng, a, b)
    dynamic_not_implemented(op)
end

function emulate(op::HloBroadcast, arg)
    if ndims(arg) == 0
        XRTArray(fill(convert(Array, arg)[], op.result_shape...))
    else
        dynamic_not_implemented(op)
    end
end

function emulate(op::HloReshape, arg)
    XRTArray(reshape(convert(Array, arg), op.result_shape))
end

function emulate(op::HloTranspose, arg)
    aarg = convert(Array, arg)
    if ndims(aarg) != length(op.permutation)
        aarg = reshape(aarg, (size(aarg)...,
            (1 for _ in 1:(length(op.permutation)-ndims(aarg)))...))
    end
    perm = map(x->x+1, op.permutation)
    @show perm
    XRTArray(collect(permutedims(aarg, )))
end

# TODO: Fix these
function emulate(op::HloDot, a, b)
    dynamic_not_implemented(op)
end

function emulate(op, args...)
    dynamic_not_implemented(op)
end
