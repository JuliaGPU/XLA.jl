module XLA
    using ProtoBuf

    import Base: run

    # Load generated xla protobuf definitions
    include(joinpath(dirname(@__FILE__), "gen","tensorflow.jl"))
    include(joinpath(dirname(@__FILE__), "gen","xla.jl"))
    include(joinpath(dirname(@__FILE__), "gen","xrt.jl"))
    using .xrt
    using .xla
    const Shape = ShapeProto
    const Layout = LayoutProto
    const ProgramShape = ProgramShapeProto

    include("literal.jl")

    struct XRTAllocation
    end

    #include("xrt.jl")
    include("xrtarray.jl")
    #include("topology.jl")
    #include("multicore.jl")
    include("libtpu.jl")
    include("hlo.jl")
    include("shape_infer.jl")
    include("execute.jl")
    include("emulate.jl")
    include("linalg.jl")
    #include("explain.jl")
    include("compiler.jl")
    #include("compiler_passes.jl")
    #include("compiler_interface.jl")
    #include("utils.jl")
    #include("grad.jl")
    #include("infeed.jl")
    #include("flux.jl")
    include("abstractinterpret.jl")
    include("array.jl")

    function regen_proto()
        tf_path = "/home/keno/tensorflow/tensorflow/"
        xrt_proto = tf_path * "compiler/xrt/xrt.proto"
        #topo_proto = tf_path * "contrib/tpu/proto/topology.proto"
        cd(@__DIR__) do
            run(ProtoBuf.protoc(`-I=/home/keno/tensorflow/ --julia_out=gen $xrt_proto`))
        end
    end

    export XRTArray, GenericHloOp, HloConstant, HloDot, HloSlice, ImmutableChain, XRTAllocation, XRTRemoteStruct

    # TPU Sessions/devices
    export TPUSession, all_tpu_devices

    export @tpu_compile, @tpu_dump, @tpu
end
