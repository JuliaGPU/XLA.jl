# Run a single op over XRTArrays
const AnyXLA = Union{XRTArray, XLAScalar, HloToken}

function build_computation(op::HloOp, args::AnyXLA...)
    comp = HloComputationProto(
        name = "comp",
        instructions = HloInstructionProto[ ],
        id = 0
    )
    root = HloInstructionProto(comp, op, map(i->(
        HloInstructionProto(comp, HloParameter(eltype(args[i]), size(args[i]), i-1))
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
        host_program_shape = pshape,
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

function _execute(op::HloOp, args...)
    sess = nothing
    for x in args
        if x.storage.remotestorage !== nothing
            sess = x.storage.remotestorage.sess
            break
        end
    end
    if sess === nothing
        ret = emulate(op, args...)
    else
        xrt = XLA.compile(sess, build_computation(op, args...))
        ret = run(xrt, args...)
    end
    ret
end

function execute(op::HloOp, args...)
    # This acts as a temporary inference barrier. We don't benefit from looking
    # through this (since we use infer_rt to figure out the return type) and when
    # aggressive optimizations are on, compiling the execute code can be *very*
    # expensive.
    Base.invokelatest(_execute, op, args...)::infer_rt(op, map(typeof, args)...)
end

@noinline (op::GenericHloOp)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloDot)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloReshape)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloBroadcast)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloConv)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloSlice)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloRng)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloTranspose)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloRev)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloDynamicSlice)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloDynamicUpdateSlice)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloConcatenate)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloInfeed)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloOutfeed)(args...) = execute(op, args...)
@noinline (op::HloAfterAll)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloIota)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloPad)(args::AnyXLA...) = execute(op, args...)
@noinline (op::HloGather)(args::AnyXLA...) = execute(op, args...)

# This function is invoked via invokelatest which acts as an inference barrier.
# Thus statically, we get the type given by `infer_rt`, while dynamically we get
# this nice and informative error. TODO: Actually implement dynamic semantics for these
function dynamic_not_implemented(op)
    error("Dynamic semantics not implemented for HLO operation of type $(typeof(op))\n"*
          "Try compiled mode or implement dynamic semantics.")
end

@noinline function (m::HloReduceWindow{fT})(f::fT, arg::XRTArray, init::XRTArray) where {fT}
    Base.invokelatest(dynamic_not_implemented, m)::infer_rt(m, typeof(f), typeof(arg))
end

@noinline function (m::HloReduce{fT})(f::fT, arg::XRTArray, init::XRTArray) where {fT}
    Base.invokelatest(dynamic_not_implemented, m)::infer_rt(m, typeof(f), typeof(arg), typeof(init))
end

@noinline function (m::HloSelectAndScatter{T,S})(select::T, scatter::S, op::XRTArray, source::XRTArray, init::XRTArray) where {T,S}
    Base.invokelatest(dynamic_not_implemented, m)::infer_rt(m, T, S, typeof(op), typeof(source), typeof(init))
end

@noinline function (m::HloScatter{T})(update::T, args...) where {T}
    Base.invokelatest(dynamic_not_implemented, m)::infer_rt(m, T, map(typeof, args)...)
end

@noinline function (m::HloMap{fT})(f::fT, args::XRTArray...) where {fT}
    Base.invokelatest(dynamic_not_implemented, m)::infer_rt(m, typeof(f), map(typeof, args)...)
end

@noinline function (m::HloCrossReplicaSum)(f, args::AnyXLA...)
    xargs = args
    Base.invokelatest(dynamic_not_implemented, m)::infer_rt(m, typeof(f), typeof(args).parameters...)
end
