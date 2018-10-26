#!/usr/bin/env julia --project=..
using XLA, TensorFlow

# For now. Media.jl has a bug
pop!(Base.Multimedia.displays)

# Connect to the TensorFlow distributed session (In this case, assumed to be xrt_server)
sess = Session(Graph(); target="grpc://localhost:8470")


using Flux, Metalhead, BenchmarkTools, JLD2

"""
    prepare_model(model::Chain)

Pass in a Flux Chain model, do accelerator-independent work to move the model
to an ImmutableChain, ready to be lowered to XLA.
"""
function prepare_model(model::Chain)
    # Eliminate any dropout layers, and move it over to an immutable chain
    # (Dropout layers are hard to do on TPUs)
    ic = ImmutableChain(filter(x->!isa(x, Flux.Dropout), model.layers)...)
    
    # Next, map each array to an XRTArray
    ic = Flux.mapleaves(x->isa(x, AbstractArray) ? XRTArray(x) : x, ic)
end

struct CompiledModel
    allocation
    compiled_code
    sess
    device
end

"""
    compile_model_for_device(model::ImmutableChain, sess, device)

Compiles a model for a specific device and allocates space for it on that device.
"""
function compile_model_for_device(model::ImmutableChain, sess, device, batch_size)
    return with_device(device) do
        alloc = XRTAllocation(sess, XLA.struct_to_literal(model))
        compiled = @tpu_compile model(XRTArray(sess, randn(Float32, 224, 224, 3, batch_size)))
        return CompiledModel(alloc, compiled, sess, device)
    end
end

"""
    run(m::CompiledModel, args...)
"""
function XLA.run(m::CompiledModel, args...)
    # Translate any incoming args to XRTArrays if they aren't already,
    # moving arrays over from the CPU to the remote device
    transfer_array(x::XRTAllocation) = x
    function transfer_array(x::AbstractArray)
        return with_device(m.device) do
            return XLA.gethandle!(m.sess, XRTArray(m.sess, x))
        end
    end

    # Gotta Go Fast (TM)
    return run(m.compiled_code, m.allocation, transfer_array.(args)...)
end


model = prepare_model(Metalhead.VGG19().layers)

# We are going to precompile for all batch sizes we are interested in
CPU_device = "/job:job/replica:1/task:1/device:XLA_CPU:1"
GPU_device = "/job:job/replica:1/task:1/device:XLA_GPU:1"
batch_sizes = (1, 2, 4, 8, 16, 32, 64)

@info("Precompiling all CPU and GPU models:")
compiled_models = Dict("CPU" => Dict(), "GPU" => Dict())
for batch_size in batch_sizes
    print("$(batch_size) ")
    compiled_models["CPU"][batch_size] = compile_model_for_device(model, sess, CPU_device, batch_size)
    compiled_models["GPU"][batch_size] = compile_model_for_device(model, sess, GPU_device, batch_size)
end

# Allow benchmarking to go up to 60 seconds for each execution, so that we get
# good statistics for each call
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120

results = Dict("CPU" => Dict(), "GPU" => Dict())
for batch_size in batch_sizes
    x = randn(Float32, 224, 224, 3, batch_size)

    # Compile model for this particular batch size
    @info("[$batch_size CPU] Executing...")
    results["CPU"][batch_size] = @benchmark run($(compiled_models["CPU"][batch_size]), $x)
    display(results["CPU"][batch_size])
    
    @info("[$batch_size GPU] Executing...")
    results["GPU"][batch_size] = @benchmark run($(compiled_models["GPU"][batch_size]), $x)
    display(results["GPU"][batch_size])
end

jldopen("benchmark_results.jld2", "w") do f
    f["results"] = results
end
