const tf = TensorFlow

function current_device(sess)
    dvs = tf_graph(sess).op_context.devices
    isempty(dvs) ? nothing : dvs[end]
end

function run_external_optimizer(snp)
    p = open(`/home/keno/tensorflow/bazel-bin/tensorflow/compiler/xla/tools/external_optimizer`; read=true, write=true)
    buf = IOBuffer()
    writeproto(buf, snp)
    write(p, take!(buf))
    close(p.in)
    buf = IOBuffer(read(p))
    readproto(buf, HloSnapshot())
end

# Manually implement the xrt_compile op to plaster over 1.12 vs nightly
# differences. Nightly adds a second output.
function make_xrt_compile(graph)
    alloc = placeholder(String)
    desc = tf.NodeDescription(graph, "XRTCompile", tf.get_name("XRTCompile"))
    tf.add_input(desc, alloc)
    cc = tf.Tensor(tf.Operation(desc), 1)
    cc, alloc
end

mutable struct XRTCompilation
    sess
    device::Union{Nothing, TensorFlow.Device}
    # XLA Program Shape
    shape::ProgramShape
    # Julia Return Type
    rt::Type
    h::Int64
    global compile
    function compile(sess, comp::XLAComputation, rt::Type)
        buf = IOBuffer()
        # Enable this to proxy through an external XLA optimizer, to
        # try out optimizations before they're deployed to TPUs.
        #comp.hlo_snapshot = run_external_optimizer(comp.hlo_snapshot)
        writeproto(buf, comp)
        op, alloc = make_xrt_compile(tf_graph(sess))
        res = new(sess,
            current_device(sess),
            comp.hlo_snapshot.hlo.hlo_module.host_program_shape, rt,
            run(tf_session(sess), op, Dict(alloc => String(take!(buf)))))
        finalizer(close, res)
        res
    end
end
function Base.show(io::IO, comp::XRTCompilation)
    print(io, "XRTCompilation handle")
end

function Base.setproperty!(c::XRTCompilation, args...)
    error("XRTCompilation may not be modified")
end

function Base.close(c::XRTCompilation)
    if tf_session(c.sess).ptr == C_NULL
        # The session was closed so our resources are already freed
        Core.setfield!(c, :h, -1)
        return
    end
    f() = as_default(tf_graph(c.sess)) do
        try
            run(tf_session(c.sess), TensorFlow.Ops.xrt_release_compilation_handle(c.h))
        catch err
            # This runs as a finalizer. The error gets printed using the
            # C printer rather than the julia one. Transform the error
            # message to make sure it's readable
            isa(err, TensorFlow.TFException) || rethrow(err)
            error(string("TensorFlow error: ", string(err.status)))
        end
    end
    if c.device === nothing
        f()
    else
        with_device(f, c.device)
    end
    Core.setfield!(c, :h, -1)
end

function make_default_execution_config()
    return XRTExecutionConfig(rng_seed=rand(UInt32) | 0x01)
end

function make_run_op(com::XRTCompilation, inputs...; device=com.device, config=make_default_execution_config())
    iob = PipeBuffer();
    writeproto(iob, config)
    str = String(take!(iob))
    op = as_default(tf_graph(com.sess)) do
        handles = map(x->gethandle!(com.sess, x), Iterators.filter(x->representable(typeof(x)), inputs))
        # Make sure everything is on the same device
        @assert all(input->input.device == device, handles)
        str = TensorFlow.constant(str)
        _op() = TensorFlow.Ops.xrt_execute(com.h, str, collect(map(x->x.h, handles)))
        return device === nothing ? _op() : with_device(_op, device)
    end
end

mutable struct XRTAllocation
    sess
    device::Union{Nothing, TensorFlow.Device}
    h::Int64
    function XRTAllocation(sess, literal::LiteralProto)
        buf = IOBuffer()
        device = current_device(sess)
        device_ordinal = 0
        if device !== nothing && device.parts[end].kind[2] == "TPU"
            device_ordinal = device.parts[end].index - 1
        end
        writeproto(buf, XLAAllocation(
            device_ordinal = device_ordinal,
            value = literal))
        local alloc
        op = as_default(tf_graph(sess)) do
            alloc = placeholder(String)
            TensorFlow.Ops.xrt_allocate(alloc)
        end
        h = run(sess, op, Dict(alloc=>String(take!(buf))))
        #ccall(:jl_, Cvoid, (Any,), "Got allocation $(h)")
        res = new(sess,
            current_device(sess),
            h)
        finalizer(close, res)
        res
    end

    global _run
    function _run(com::XRTCompilation, inputs...; config=make_default_execution_config())
        op = make_run_op(com, inputs...; config=config)
        h = run(com.sess, op, Dict())
        #ccall(:jl_, Cvoid, (Any,), "Got allocation $(h)")
        res = @GC.preserve com inputs new(com.sess, com.device, h)
        finalizer(close, res)
        res
    end

    global run
    function run(com::XRTCompilation, inputs...; device=com.device, config=make_default_execution_config())
        if typeof(device) <: DeviceIndex
            device = tf_tpu_device(com.sess, device)
        end
        op = make_run_op(com, inputs...; device=device, config=config)
        h = @GC.preserve com inputs run(com.sess, op, Dict(); async=true)
        #ccall(:jl_, Cvoid, (Any,), "Got allocation $(h)")
        res = new(com.sess, com.device, h)
        finalizer(close, res)

        return if com.rt <: XRTArray
            T = convert(Type, XlaType(com.shape.result.element_type))
            dims = (com.shape.result.dimensions...,)
            XRTArray{T, dims, length(dims)}(res)
        else
            XRTRemoteStruct{com.rt}(res)
        end
    end

    global run_on_devices
    function run_on_devices(compilation_handle, args...; devices=all_tpu_devices(compilation_handle.sess))
		sess = compilation_handle.sess

		# Helper functions to prepare an argument for sending to the TPU
		prepare_arg(x::Number) = XRTArray(sess, x)
		prepare_arg(x::AbstractArray) = XRTArray(sess, x)
		prepare_arg(x::Tuple) = prepare_arg.(x)
		prepare_arg(x) = XRTRemoteStruct(sess, x)

		# For each TPU device
		worker_ops = Any[]
		all_remote_handles = Any[compilation_handle]
		for device in devices
			# Prepare arguments
			compilation_args = with_device(tf_tpu_device(sess, device)) do
				# Drop `Val`'s because we don't need to pass those in
				return prepare_arg.(filter(x -> !isa(x, Val), collect(args)))
			end
			push!(all_remote_handles, compilation_args)

			# Place an XRT execute op on the device
			push!(worker_ops, make_xrt_execute_on(device, compilation_handle, compilation_args...))
		end

        # Next, launch these worker ops
        task = @async begin
			rets = @GC.preserve all_remote_handles run(sess, worker_ops; async=true)

			# Wrap rets as XRTAllocations, then as XRTRemoteStructs
			rets = [new(sess, tf_tpu_device(sess, devices[idx]), rets[idx]) for idx in 1:length(devices)]
			return XRTRemoteStruct{compilation_handle.rt}.(rets)
		end

        return task
    end
end
function Base.setproperty!(c::XRTAllocation, args...)
    error("XRTAllocation may not be modified")
end

const tf = TensorFlow
global counter = 0
function Base.close(c::XRTAllocation)
     if tf_session(c.sess).ptr == C_NULL
        # The session was closed so our resources are already freed
        Core.setfield!(c, :h, -1)
        return
    end
    #ccall(:jl_, Cvoid, (Any,), "Start Releasing allocation $(c.h)")
    global counter
    counter += 1
    desc = tf.NodeDescription(tf_graph(c.sess), "Const", "ReleaseAllocationHandle$(counter)/Const")
    desc["value"] = TensorFlow.RawTensor(c.h)
    desc["dtype"] = typeof(c.h)
    cc = tf.Tensor(tf.Operation(desc))
    desc2 = tf.NodeDescription(tf_graph(c.sess), "XRTReleaseAllocationHandle", "ReleaseAllocationHandle$(counter)")
    tf.add_input(desc2, cc)
    c.device !== nothing && tf.set_device(desc2, c.device)
    #ccall(:jl_, Cvoid, (Any,), "Releasing allocation $(c.h)")
    try
        run(c.sess, tf.Tensor(tf.Operation(desc2)))
    catch err
        isa(err, TensorFlow.TFException) || rethrow(err)
        error(string("TensorFlow error: ", string(err.status)))
    end
    #ccall(:jl_, Cvoid, (Any,), "Done releasing allocation $(c.h)")
    Core.setfield!(c, :h, -1)
end
