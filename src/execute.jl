# Run a single op over XRTArrays
const XLAScalar = Union{Bool, Int8, Int16, Int32, Int64,
                        UInt8, UInt16, UInt32, UInt64,
                        Float16, Float32, Float64, Complex{Float32}}
const AnyXLA = Union{XRTArray, XLAScalar}

function build_computation(op::HloOp, args::AnyXLA...)
    comp = HloComputationProto(
        name = "comp",
        instructions = HloInstructionProto[ ],
        id = 0
    )
    root = HloInstructionProto(comp, op, map(i->(
        HloInstructionProto(comp, HloParameter{eltype(args[i]), size(args[i])}(i-1))
    ), 1:length(args))...)
    comp.root_id = root.id
    pshape = ProgramShape(
        parameters = collect(map(shape, args)),
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
end


function shape_infer(op::HloOp{opcode, T, Shape} where opcode, args::Type{<:AnyXLA}...) where {T, Shape}
    return (T, Shape, length(Shape))
end

function Base.run(xrt::XRTCompilation, args::AnyXLA...)
    run(xrt, map(a->a.storage, args)...)
end

function execute(op::HloOp, args::AnyXLA...)
    xrt = XLA.compile(args[1].storage.sess, build_computation(op, args...))
    run(xrt, args...)::XRTArray{shape_infer(op, map(typeof, args)...)...}
end

@noinline (op::GenericHloOp)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloCollapse)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloDot)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloReshape)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloBroadcast)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloConv)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloSlice)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloRng)(args::AnyXLA...) = execute(op, args...)
