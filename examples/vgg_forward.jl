using Revise
using XLA, TensorFlow, Flux, Metalhead

# For now. Media.jl has a bug
pop!(Base.Multimedia.displays)

# Filter Dropout layers for now (the data-dependent condition is slightly
# harder to compile).
ic = ImmutableChain(filter(x->!isa(x, Flux.Dropout), VGG19().layers.layers)...)

# Connect to the TensorFlow distributed session (either xrt_server or TPU)
sess = Session(Graph(); target="grpc://localhost:8470")

# If on TPU and this is the first time connecting to a new TPU, run
# run(sess, TensorFlow.Ops.configure_distributed_tpu())

# Generate some random data (or use a real image here)
x = rand(Float32, 224, 224, 3, 10);

function VGG_forward(sess, device)
    with_device(device) do
        xrt(ic) = Flux.mapleaves(x->isa(x, AbstractArray) ? XRTArray(x) : x, ic)
        xrtic = xrt(ic)
        ic_alloc = XRTAllocation(sess, XLA.struct_to_literal(xrtic))
        xrtx = XRTArray(sess, x)

        compld = @tpu_compile xrtic(xrtx)
        @time run(compld, ic_alloc, XLA.gethandle!(sess, xrtx))
    end
end

# Quick sanity tests
# with_device("/job:job/replica:1/task:1/device:XLA_GPU:1") do
#     xrtx = XRTArray(sess, x)
# end

# with_device("/job:job/replica:1/task:1/device:XLA_CPU:1") do
#     xrtx = XRTArray(sess, x)
# end

# Actually run VGG
VGG_forward(sess, "/job:job/replica:1/task:1/device:XLA_CPU:1")
VGG_forward(sess, "/job:job/replica:1/task:1/device:XLA_GPU:1")

