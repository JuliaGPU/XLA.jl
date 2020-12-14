module XLA
    using ProtoBuf

    using XLA_Tools_jll

    import Base: run

    # Load generated xla protobuf definitions
    include("../gen/xrt/tensorflow.jl")
    include("../gen/xrt/xla.jl")
    include("../gen/xrt/xrt.jl")
    using .xrt
    using .xla
    const Shape = ShapeProto
    const Layout = LayoutProto
    const ProgramShape = ProgramShapeProto

    include("literal.jl")

    struct XRTAllocation
    end

    include("../lib/tpu/LibTPU.jl")
    using .LibTPU
    export LibTPU

    #include("xrt.jl")
    include("xrtarray.jl")
    #include("topology.jl")
    #include("multicore.jl")
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

    export XRTArray, GenericHloOp, HloConstant, HloDot, HloSlice, ImmutableChain, XRTAllocation, XRTRemoteStruct

    # TPU Sessions/devices
    export TPUSession, all_tpu_devices

    export @tpu_compile, @tpu_dump, @tpu
end
