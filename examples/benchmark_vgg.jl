# Invoke this as something like `julia --project=.. benchmark_vgg.jl`
# To run on CPU, have an `xrt_server` already running, I like to have mine bind to the default port 8470, and I run it on localhost.
# To run on GPU, have an `xrt_server` already running on a machine with GPUs.  I use the same `xrt_server` binary as the one I use
# for CPU work, and I also start it with only one CUDA device available so that it doesn't immediately grab all the memory on all
# GPUs available, as tensorflow is wont to do.  Example: `CUDA_VISIBLE_DEVICES=2 ~/.julia/dev/TensorFlow/deps/downloads/bin/xrt_server`
# To run on TPU, open up an SSH tunnel to a GCE instance that is on the same subnet as your TPU cloud instance, bouncing off of that
# server to access the TPU device from outside of Google's network.  Example: `ssh -L 8471:<tpu_internal_ip>:8470 <gce_external_ip>`
using XLA, TensorFlow

# For now. Media.jl has a bug
pop!(Base.Multimedia.displays)

# Connect to the TensorFlow distributed sessions (one for CPU, one for GPU, one for TPU)
cpu_sess = Session(Graph(); target="grpc://localhost:8470")
gpu_sess = cpu_sess
tpu_sess = Session(Graph(); target="grpc://localhost:8471")

# Try to get the associated device with each session; if one doesn't respond within 2 seconds, complain
# because we most likely have a missing xrt_server or TPU or something.
function wait_for_task_finish(t::Task, timeout::Float64)
    t_start = time()
    while !istaskdone(t)
        sleep(0.001)
        if time() - t_start > timeout
            return false
        end
    end
    return true
end
function get_device(sess, device_type)
    device = nothing
    t = @async begin
        devices = collect(TensorFlow.DeviceList(sess))
        device = first(filter(x -> x.device_type == device_type, devices))
    end
    if !wait_for_task_finish(t, 2.0)
        return nothing
    end
    inc_number(x::Number) = x + 1
    inc_number(x) = x
    function fixup_device(d::TensorFlow.Device)
        d.parts[:] .= [TensorFlow.DevicePart(p.kind, inc_number(p.index)) for p in d.parts]
        return d
    end
    fixup_device(d) = d
    return fixup_device(TensorFlow.Device(device.name))
end
    
cpu_device = get_device(cpu_sess, "XLA_CPU")
gpu_device = get_device(gpu_sess, "XLA_GPU")
tpu_device = get_device(tpu_sess, "TPU_SYSTEM")

using Flux, Metalhead, BenchmarkTools, JLD2, Printf

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
    return TensorFlow.as_default(sess.graph) do
        return with_device(device) do
            alloc = XRTAllocation(sess, XLA.struct_to_literal(model))
            compiled = @tpu_compile model(XRTArray(sess, randn(Float32, 224, 224, 3, batch_size)))
            return CompiledModel(alloc, compiled, sess, device)
        end
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
        return TensorFlow.as_default(m.sess.graph) do
            return with_device(m.device) do
                return XLA.gethandle!(m.sess, XRTArray(m.sess, x))
            end
        end
    end

    # Gotta Go Fast (TM)
    return TensorFlow.as_default(m.sess.graph) do
        run(m.compiled_code, m.allocation, transfer_array.(args)...)
    end
end


model = prepare_model(Metalhead.VGG19().layers)

# Allow benchmarking to go up to 60 seconds for each execution, so that we get
# good statistics for each call
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120
BenchmarkTools.DEFAULT_PARAMETERS.samples = 10
results = Dict()

for batch_size in (1, 2, 4, 8, 16, 32, 64)
    x = randn(Float32, 224, 224, 3, batch_size)

    for (device, sess) in [(cpu_device, cpu_sess), (gpu_device, gpu_sess), (tpu_device, tpu_sess)]
        # Skip nonexistent devices
        if device == nothing
            continue
        end
        device_type = device.parts[end].kind[2]

        if !haskey(results, device_type)
            results[device_type] = Dict()
        end

        @info "[$(device_type) $(batch_size)] Compiling..."
        compiled_model = compile_model_for_device(model, sess, device, batch_size)
    
        @info("[$(device_type) $(batch_size)] Executing...")
        results[device_type][batch_size] = @benchmark run($compiled_model, $x)
    
        # Save partial results
        jldopen("benchmark_results.jld2", "w") do f
            f["results"] = results
        end
    end
end

