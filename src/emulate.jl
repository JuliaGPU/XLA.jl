function emulate(op::HloRng, a, b)
    XRTArray(rand(convert(Array, a)[]:convert(Array, b)[], op.shape...))
end

function emulate(op::HloBroadcast, arg)
    @assert ndims(arg) == 0
    XRTArray(fill(convert(Array, arg)[], op.result_shape...))
end

function emulate(op::HloReshape, arg)
    XRTArray(reshape(convert(Array, arg), op.result_shape))
end
