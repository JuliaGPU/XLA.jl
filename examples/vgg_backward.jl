using Revise

using XLA
using Metalhead
using Flux
using TensorFlow

# Filter Dropout layers for now (the data-dependent condition is slightly
# harder to compile).
ic = ImmutableChain(filter(x->!isa(x, Flux.Dropout), VGG19().layers.layers)...)

# Connect to the TensorFlow distributed session (either xrt_server or TPU)
sess = Session(Graph(); target="grpc://localhost:8470")

# If on TPU and this is the first time connecting to a new TPU, run
# run(sess, TensorFlow.Ops.configure_distributed_tpu())

# Generate some random data (or use a real image here)
x = rand(Float32, 224, 224, 3, 1)

# @tpu ic(x)

# The ic is fairly large and takes a while to transfer over the network.
# Here we do the same manually, but only transfer the model once. This is
# useful when chaning the batch size (which uses the same data structure,
# but a different program).

xrt(ic) = Flux.mapleaves(x->isa(x, AbstractArray) ? XRTArray(x) : x, ic)
xrtic = xrt(ic)
ic_alloc = XRTAllocation(sess, XLA.struct_to_literal(xrtic))
xrtx = XRTArray(sess, x)

using Zygote
# Disable support for Param dicts
function my_derivative(f, args...)
  y, back = Zygote._forward(Zygote.Context{Nothing}(nothing), f, args...)
  J = Δ -> Zygote.tailmemaybe(back(Δ))
  return J(1)[1]
end

# The function we're going to differentiate. `sum` stands in for the loss function
# here.
function f(ic, x)
    my_derivative(ic -> sum(ic(x)), ic)
end

compld = @tpu_compile f(xrtic, xrtx)
run(compld, ic_alloc, XLA.gethandle!(sess, xrtx))
