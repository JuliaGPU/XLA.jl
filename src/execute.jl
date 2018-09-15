# Run a single op over XRTArrays
function build_computation(op::HloOp, args::XRTArray...)
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


function shape_infer(op::HloOp, args::Type{<:XRTArray}...)
    if isa(op, HloDot)
        return (Float32, (10,), 1)
    end
end

function Base.run(xrt::XRTCompilation, args::XRTArray...)
    run(xrt, map(a->a.storage, args)...)
end

function execute(op::HloOp, args::XRTArray...)
    xrt = XLA.compile(args[1].storage.sess, build_computation(op, args...))
    run(xrt, args...)::XRTArray{shape_infer(op, map(typeof, args)...)...}
end

@noinline (op::GenericHloOp)(args::XRTArray...) = execute(op, args...)
@noinline (op::HloDot)(args::XRTArray...) = execute(op, args...)
@noinline (op::HloSlice)(args::XRTArray...) = execute(op, args...)
