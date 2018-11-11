function evaluate2(xrtic, x)
    loss, updates = my_derivative_with_loss(xrtic->sum(xrtic(x)), xrtic)
end

function bisect(n)
    start = ImmutableChain((resnet_host.layers[1:n]...,))
    y = start(rand(Float32, 224, 224, 3, 1))
    remainder = ImmutableChain((resnet.layers[n+1:end]...,))
    XLA.code_typed_xla(evaluate2, Tuple{typeof(remainder), typeof(XRTArray(y))})[1]
end

function bisect_one(n)
    start = ImmutableChain((resnet_host.layers[1:n]...,))
    y = start(rand(Float32, 224, 224, 3, 1))
    remainder = resnet.layers[n+1]
    XLA.code_typed_xla(evaluate2, Tuple{typeof(remainder), typeof(XRTArray(y))})[1]
end


function bisect_one_weak(n)
    start = ImmutableChain((resnet_host.layers[1:n]...,))
    y = start(rand(Float32, 224, 224, 3, 1))
    remainder = resnet.layers[n+1]
    code_typed(evaluate2, Tuple{typeof(remainder), typeof(XRTArray(y))}; params = Core.Compiler.CustomParams(typemax(UInt); aggressive_constant_propagation=true, ignore_all_inlining_heuristics=true))
end


function bisect_forward(n)
    start = ImmutableChain((resnet_host.layers[1:n]...,))
    y = start(rand(Float32, 224, 224, 3, 1))
    remainder = ImmutableChain((resnet.layers[n+1:end]...,))
    XLA.code_typed_xla(remainder, Tuple{typeof(XRTArray(y))})[1]
end


function bisect_residual(b, n)
    start = ImmutableChain((resnet_host.layers[1:b-1]...,))
    y = start(rand(Float32, 224, 224, 3, 1))
    host_block = resnet_host.layers[b]
    host_start = ImmutableChain((host_block.layers[1:n]...,))
    y = host_start(y)
    remainder = ImmutableChain((resnet.layers[b].layers[n+1:end]...,))
    XLA.code_typed_xla(evaluate2, Tuple{typeof(remainder), typeof(XRTArray(y))})[1]
end

bisect_residual(18, 2)
