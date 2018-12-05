function initialize_tpu!(sess)
    topo = run(sess, TensorFlow.Ops.configure_distributed_tpu())
    readproto(IOBuffer(topo), TopologyProto())
end

topo_string(topo::TopologyProto) = join(topo.mesh_shape, "x")

function permuted_coords(topo)
    permutedims(reshape(topo.device_coordinates, (length(topo.mesh_shape),
        topo.num_tpu_devices_per_task, topo.num_tasks)), (3,2,1))
end

function Base.show(io::IO, topo::TopologyProto)
    coordinates = permuted_coords(topo)
    print(io, topo_string(topo), " TPU Topology\n")
    for i = 1:topo.num_tasks
        println(" Task $(i-1):  ( x, y, core)")
        for j = 1:topo.num_tpu_devices_per_task
            println("  Device $(j-1): ", join(coordinates[i, j, :], "  "))
        end
    end
end

struct MeshIndex
    x::Int
    y::Int
    core::Int
end

struct DeviceIndex
    task::Int
    index::Int
end

function Base.getindex(topo::TopologyProto, idx::DeviceIndex)
    MeshIndex(permuted_coords(topo)[idx.task+1, idx.index+1, :]...)
end

function Base.getindex(topo::TopologyProto, idxs::Vector{DeviceIndex})
    coords = permuted_coords(topo)
    [MeshIndex(coords[idx.task+1, idx.index+1, :]...) for idx in idxs]
end


function tf_tpu_device(device::DeviceIndex)
    TensorFlow.Device("/job:tpu_worker/replica:1/task:$(device.task+1)/device:TPU:$(device.index+1)")
end

function tf_host_cpu_device(device::DeviceIndex)
    TensorFlow.Device("/job:tpu_worker/replica:1/task:$(device.task+1)/device:CPU:1")
end
