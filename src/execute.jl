# Run a single op over XRTArrays
function execute(op::HloOp, args::XRTArray...)
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
    run(XLA.compile(args[1].storage.sess, xlac), map(a->a.storage, args)...)
end
(op::GenericHloOp)(args::XRTArray...) = execute(op, args...)
(op::HloDot)(args::XRTArray...) = execute(op, args...)
(op::HloSlice)(args::XRTArray...) = execute(op, args...)
