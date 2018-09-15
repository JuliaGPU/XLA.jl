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

    include("literal.jl")
    include("xrt.jl")
    include("hlo.jl")
    include("execute.jl")
    include("linalg.jl")

    regen_proto() = run(ProtoBuf.protoc(`-I=/home/keno/tensorflow/ --julia_out=gen /home/keno/tensorflow/tensorflow/compiler/xrt/xrt.proto`))

    export XRTArray, GenericHloOp, HloConstant, HloDot, HloSlice

#=
    TensorFlow.import_op("XRTCompile")
    TensorFlow.import_op("XRTExecute")
    TensorFlow.import_op("XRTAllocate")
    TensorFlow.import_op("XRTReadLiteral")
    TensorFlow.import_op("XRTReadLiteralAndRelease")
    TensorFlow.import_op("XRTReleaseAllocationHandle")
    TensorFlow.import_op("XRTReleaseCompilationHandle")
=#

#= 
    using XLA: GenericHloOp, HloConstant
    comp = HloComputationProto(
        name = "comp",
        instructions = HloInstructionProto[ ],
        id = 0
    )
    root = HloInstructionProto(
        comp, GenericHloOp{:add, Float64, ()}(),
        HloInstructionProto.((comp,), (HloConstant(1.0), HloConstant(2.0)))...)
    comp.root_id = root.id
    pshape = ProgramShape(
        result = root.shape
    )

    config = XLAComputationConfig(
        program_shape = pshape
    )
    hlo_module = HloModuleProto(
        name = "test",
        computations = [ comp ],
        entry_computation_name = "op",
        entry_computation_id = 0,
        id = 0,
        program_shape = pshape,
    )
    hlo = HloProto(
        hlo_module = hlo_module
    )
    hlo_snap = HloSnapshot(
        hlo = hlo
    )
    xlac = XLAComputation(
        config=config,
        hlo_snapshot = hlo_snap
    )
    run(XLA.compile(sess, xlac))
=#

#=
    scalar_shape = Shape(
        element_type = PrimitiveType.F64,
        dimensions = [],
        layout = Layout(
            format = Format.DENSE,
            max_sparse_elements = 0
        )
    )

    pshape = ProgramShape(
        result = scalar_shape
    )

    config = XLAComputationConfig(
        program_shape = pshape
    )
#=
    instruction0 = HloInstructionProto(
        name = "const0",
        opcode = "constant",
        shape = scalar_shape,
        literal = LiteralProto(
            shape = scalar_shape,
            f64s = [1.0]),
        id = 0
    )
    instruction1 = HloInstructionProto(
        name = "const1",
        opcode = "constant",
        shape = scalar_shape,
        literal = LiteralProto(
            shape = scalar_shape,
            f64s = [2.0]),
        id = 1
    )
=#
    instruction2 = HloInstructionProto(
        name = "op",
        opcode = "parameter",
        shape = scalar_shape,
        parameter_number = 0,
        #operand_ids = [0, 1],
        id = 2
    )
    comp = HloComputationProto(
        name = "comp",
        root_id = 2,
        instructions = [ #= instruction0, instruction1, =# instruction2 ],
        id = 0
    )
    hlo_module = HloModuleProto(
        name = "test",
        computations = [ comp ],
        entry_computation_name = "op",
        entry_computation_id = 0,
        id = 0,
        program_shape = pshape,
    )
    hlo = HloProto(
        hlo_module = hlo_module
    )
    hlo_snap = HloSnapshot(
        hlo = hlo
    )
    xlac = XLAComputation(
        config=config,
        hlo_snapshot = hlo_snap
    )

    open("op_param_bounds.pb", "w") do f
        writeproto(f, hlo_snap)
    end
    #hlo2 = readproto(iob, XLAComputation)

    str = String(take!(iob))

    sess = Session(Graph(); target="grpc://localhost:8470")
    run(sess, TensorFlow.Ops.xrt_compile(str))
=#

#=
    econfig = XRTExecutionConfig()

=#

#=
    readproto(IOBuffer(run(sess, TensorFlow.Ops.xrt_read_literal(0))), LiteralProto())
=#

end
