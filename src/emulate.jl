function emulate(op::HloRng, a, b)
    XRTArray(rand(convert(Array, a)[]:convert(Array, b)[], op.shape...))
end

function emulate(op::HloBroadcast, arg)
    if ndims(arg) == 0
        XRTArray(fill(convert(Array, arg)[], op.result_shape...))
    else
        # TODO: Fix this
        XRTArray(rand(shape_infer(op, typeof(arg))...))
    end
end

function emulate(op::HloReshape, arg)
    @Base.show (op, size(arg))
    XRTArray(reshape(convert(Array, arg), op.result_shape))
end


# TODO: Fix these
function emulate(op::HloDot, a, b)
    XRTArray(rand(shape_infer(op, typeof(a), typeof(b))...))
end

function emulate(op, args...)
    XRTArray(rand(shape_infer(op, map(typeof, args)...)...))
end
