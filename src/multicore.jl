struct TPUSession
    sess::TensorFlow.Session
    tpu_devices::Vector{DeviceIndex}
    topology::TopologyProto
end
function Base.show(io::IO, sess::TPUSession)
    print(io, "TPUSession(<$(length(sess.tpu_devices)) TPU chips in $(topo_string(sess.topology)) topology>)")
end

Base.run(sess::TPUSession, args...; kwargs...) = Base.run(tf_session(sess), args...; kwargs...)

const tf = TensorFlow
function TPUSession(grpc_endpoints::Vector{String})
    if length(grpc_endpoints) == 1
        sess = Session(Graph(); target="grpc://$(grpc_endpoints[1])")
    else
        job = tf.tensorflow.JobDef(
            name = "tpu_worker",
            tasks = Dict(pairs(grpc_endpoints)))

        cluster = tf.tensorflow.ClusterDef(job=[job])

        config = tf.tensorflow.ConfigProto()
        config.cluster_def = cluster

        sess = Session(Graph(), config; target="grpc://$(grpc_endpoints[1])")
    end
    devices = collect(TensorFlow.DeviceList(sess))
    tpu_devices = DeviceIndex[]
    for d in devices
        if d.device_type == "TPU"
            dev = TensorFlow.Device(d.name)
            push!(tpu_devices, DeviceIndex(dev.parts[3].index, dev.parts[4].index))
        end
    end
    topology = initialize_tpu!(sess)
    TPUSession(sess, tpu_devices, topology)
end
TPUSession(endpoint::String) = TPUSession([endpoint])

all_tpu_devices(sess::TPUSession) = sess.tpu_devices

function make_xrt_execute_on(device::DeviceIndex, com::XRTCompilation, inputs...)
    make_run_op(com, inputs...; device=tf_tpu_device(device))
end

function make_infeed_on(sess, device::DeviceIndex, args...)
    _make_infeed_op(sess,
        args...;
        device = tf_host_cpu_device(device),
        device_ordinal = device.index)
end

function make_outfeed_on(sess, device::DeviceIndex, args...)
    make_outfeed_op(sess,
        args...;
        device = tf_host_cpu_device(device),
        device_ordinal = device.index)
end

tf_session(sess) = sess
tf_session(sess::TPUSession) = sess.sess
tf_graph(sess) = tf_session(sess).graph
