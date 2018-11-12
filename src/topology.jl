function initialize_tpu!(sess)
    topo = run(sess, TensorFlow.Ops.configure_distributed_tpu())
    readproto(IOBuffer(topo), TopologyProto())
end

function Base.show(io::IO, topo::TopologyProto)
    coordinates = permutedims(reshape(topo.device_coordinates, (length(topo.mesh_shape),
        topo.num_tpu_devices_per_task, topo.num_tasks)), (3,2,1))
    print(io, join(topo.mesh_shape, "x"), " TPU Topology\n")
    for i = 1:topo.num_tasks
        println(" Task $(i-1):  ( x, y, core)")
        for j = 1:topo.num_tpu_devices_per_task
            println("  Device $(j-1): ", join(coordinates[i, j, :], "  "))
        end
    end
end
