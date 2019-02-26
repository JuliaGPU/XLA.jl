struct TPUSession
    sess::TensorFlow.Session
    job_name::String
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
        job_name = "tpu_worker"
        sess = Session(Graph(); target="grpc://$(grpc_endpoints[1])")
    else
        job_name = "tpu_job"
        job = tf.tensorflow.JobDef(
            name = job_name,
            tasks = Dict(x[1]-1=>x[2] for x in pairs(grpc_endpoints)))

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
    TPUSession(sess, job_name, tpu_devices, topology)
end
TPUSession(endpoint::String) = TPUSession([endpoint])

all_tpu_devices(sess::TPUSession) = sess.tpu_devices

function make_xrt_execute_on(device::DeviceIndex, com::XRTCompilation, inputs...)
    make_run_op(com, inputs...; device=tf_tpu_device(com.sess, device))
end

function make_infeed_on(sess, device::DeviceIndex, args...)
    _make_infeed_op(sess,
        args...;
        device = tf_host_cpu_device(sess, device),
        device_ordinal = device.index)
end

function make_outfeed_on(sess, device::DeviceIndex, args...)
    make_outfeed_op(sess,
        args...;
        device = tf_host_cpu_device(sess, device),
        device_ordinal = device.index)
end

tf_session(sess) = sess
tf_session(sess::TPUSession) = sess.sess
tf_graph(sess) = tf_session(sess).graph

job_name(sess::TPUSession) = sess.job_name

