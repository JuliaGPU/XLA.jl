module XLA
    using ProtoBuf
    using TensorFlow

    import Base: run

    # Load generated xla protobuf definitions
    include(joinpath(dirname(@__FILE__), "gen","tensorflow.jl"))
    include(joinpath(dirname(@__FILE__), "gen","xla.jl"))
    include(joinpath(dirname(@__FILE__), "gen","xrt.jl"))
    using .xrt
    using .xla
    using .tensorflow.tpu
    const Shape = ShapeProto
    const ProgramShape = ProgramShapeProto

    include("literal.jl")
    include("xrt.jl")
    include("topology.jl")
    include("multicore.jl")
    include("hlo.jl")
    include("shape_infer.jl")
    include("execute.jl")
    include("emulate.jl")
    include("linalg.jl")
    include("explain.jl")
    include("compiler.jl")
    include("compiler_passes.jl")
    include("compiler_interface.jl")
    include("utils.jl")
    include("grad.jl")
    include("infeed.jl")

    function regen_proto()
        tf_path = "/home/keno/tensorflow/tensorflow/"
        xrt_proto = tf_path * "compiler/xrt/xrt.proto"
        topo_proto = tf_path * "contrib/tpu/proto/topology.proto"
        cd(@__DIR__) do
            run(ProtoBuf.protoc(`-I=/home/keno/tensorflow/ --julia_out=gen $xrt_proto $topo_proto`))
        end
    end

    export XRTArray, GenericHloOp, HloConstant, HloDot, HloSlice, ImmutableChain, XRTAllocation, XRTRemoteStruct

    # TPU Sessions/devices
    export TPUSession, all_tpu_devices

    export @tpu_compile, @tpu_dump, @tpu

    function __init__()
        TensorFlow.import_op("XRTCompile")
        TensorFlow.import_op("XRTExecute")
        TensorFlow.import_op("XRTAllocate")
        TensorFlow.import_op("XRTReadLiteral")
        TensorFlow.import_op("XRTReadLiteralAndRelease")
        TensorFlow.import_op("XRTReleaseAllocationHandle")
        TensorFlow.import_op("XRTReleaseCompilationHandle")
        TensorFlow.import_op("ConfigureDistributedTPU")
        TensorFlow.import_op("ShutdownDistributedTPU")
    end

end
