const Compiler = Core.Compiler
using .Compiler: ReturnNode, argextype, SSAValue, ⊑, widenconst
using InteractiveUtils
using ..XLA: HloParameter, HloOp, XLAComputationConfig, HloModuleProto, HloProto,
    HloSnapshot, XLAComputation, HloMap, HloGetTupleElement, XLAScalar,
    HloReduceWindow, HloReduce, HloTuple

function grab_ir(f, argtypes)
    method = @which f(W, x, b)
    sig = typeof((f, W, x, b))
    mi = Compiler.code_for_method(method, sig, Core.svec(), params.world)
    sv = Compiler.OptimizationState(mi, params)
    ir = Compiler.inflate_ir(ci, mi)
end

function _compile_to_xla!(computations, comp, ir, sv)
    arg_instrs = Vector{HloInstructionProto}(undef, length(ir.argtypes))
    xla_args = Type[]
    for i = 1:length(ir.argtypes)
        # TODO: We could represent this as a kConstant
        isa(ir.argtypes[i], Const) && continue
        if sv.linfo.def.isva && i == length(ir.argtypes)
            # To XLA, we express varargs as separate parameters, so gather all of
            # them and then put them in a tuple
            vargs = HloInstructionProto[]
            for T in ir.argtypes[i].parameters
                instr = HloInstructionProto(comp, "parameter")
                instr.shape = dtype_to_shape(T)
                instr.parameter_number = length(xla_args)
                push!(vargs, instr)
                push!(xla_args, T)
            end
            @Base.show ir.argtypes[i]
            arg_instrs[i] = HloInstructionProto(comp, HloTuple(), vargs...)
        elseif ir.argtypes[i] ⊑ XRTArray
            AT = widenconst(ir.argtypes[i])
            eltype = AT.parameters[1]
            dims = AT.parameters[2]
            arg_instrs[i] = HloInstructionProto(comp, HloParameter(eltype, dims, length(xla_args)))
            push!(xla_args, ir.argtypes[i])
        elseif representable(ir.argtypes[i])
            T = widenconst(ir.argtypes[i])
            instr = HloInstructionProto(comp, "parameter")
            instr.shape = dtype_to_shape(T)
            instr.parameter_number = length(xla_args)
            arg_instrs[i] = instr
            push!(xla_args, T)
        end
    end
    ssa_vals = Vector{HloInstructionProto}(undef, length(ir.stmts))
    function hlo_eval(arg)
        @show arg
        if isa(arg, Compiler.Argument)
            @Base.show arg_instrs[arg.n].parameter_number
            return arg_instrs[arg.n]
        elseif isa(arg, SSAValue)
            return ssa_vals[arg.id]
        else
            error()
        end
    end
    for (stmt_idx, stmt) in pairs(ir.stmts)
        isa(stmt, ReturnNode) && continue
        if isexpr(stmt, :new) || (isexpr(stmt, :call) && Compiler.is_known_call(stmt, tuple, ir, sv.sp))
            @Base.show stmt_idx
            ssa_vals[stmt_idx] = HloInstructionProto(comp, HloTuple(), map(hlo_eval, filter(x->x!=nothing, stmt.args[2:end]))...)
            continue
        end
        if isexpr(stmt, :call) && Compiler.is_known_call(stmt, getfield, ir, sv.sp)
            structT = widenconst(argextype(stmt.args[2], ir, sv.sp))
            elt = argextype(stmt.args[3], ir, sv.sp)
            isa(elt, Const) || error("non-constant structure indexing (1)")
            idx = Compiler.try_compute_fieldidx(structT, elt.val)
            if idx === nothing
                @Base.show stmt
                @Base.show stmt_idx
                @Base.show ir
                @Base.show length(ir.stmts)
            end
            idx !== nothing || error("non-constant structure indexing (2)")
            if !representable(fieldtype(structT, idx))
                continue
            end
            @Base.show stmt.args[2]
            tuple_idx = count(i->representable(fieldtype(structT, i)), 1:idx-1) + 1
            proto = HloInstructionProto(comp, HloGetTupleElement(tuple_idx - 1),
                hlo_eval(stmt.args[2]))
            ssa_vals[stmt_idx] = proto
            continue
        end
        if isa(stmt, GlobalRef) && isa(argextype(stmt, ir, sv.sp), Const)
            continue
        end
        if !isexpr(stmt, :invoke)
            @Base.show stmt
            Base.display(ir)
            error("Unrecognized expr")
        end
        hlo_inst = stmt.args[2]
        isa(hlo_inst, QuoteNode) && (hlo_inst = hlo_inst.value)
        if isa(hlo_inst, HloOp)
            if isa(hlo_inst, HloMap) || isa(hlo_inst, HloReduceWindow) || isa(hlo_inst, HloReduce)
                args = map(hlo_eval, stmt.args[4:end])
                proto = HloInstructionProto(comp, hlo_inst, args...)
                sig = Tuple{typeof(hlo_inst).parameters[1], (XRTArray{eltype(argextype(stmt.args[i], ir, sv.sp)), (), 0} for i = 4:length(stmt.args))...}
                argvals = nothing
                if sizeof(sig.parameters[1]) != 0
                    # If the function being called has non-zero information, we need to make sure it's a constant
                    # In the future, XLA will allow this.
                    fty = argextype(stmt.args[3], ir, sv.sp)
                    if !isa(fty, Const)
                        error("At the moment applied computations must not capture non-constant values")
                    end
                    argvals = Vector{Any}(undef, 1)
                    argvals[1] = fty.val
                end
                ir′, sv′ = code_typed_xla(sig; argvals=argvals)
                if sizeof(sig.parameters[1]) != 0
                    ir′.argtypes[1] = fty
                end
                comp′ = HloComputationProto(
                    name = "comp$(length(computations))",
                    instructions = HloInstructionProto[ ],
                    id = length(computations)
                )
                pushfirst!(computations, comp′)
                _compile_to_xla!(computations, comp′, ir′, sv′)
                proto.called_computation_ids = [comp′.id]
            else
                args = map(hlo_eval, stmt.args[3:end])
                proto = HloInstructionProto(comp, hlo_inst, args...)
            end
            ssa_vals[stmt_idx] = proto
        elseif isa(hlo_inst, Type) && hlo_inst <: XRTArray
            @assert isa(stmt.args[3], XLAScalar)
            ssa_vals[stmt_idx] = HloInstructionProto(comp, HloConstant(stmt.args[3]))
        end
    end
    ret = ir.stmts[end]
    @assert isa(ret, ReturnNode)
    comp.root_id = hlo_eval(ret.val).id
    xla_args
end

function compile_to_xla(ir, sv)
    @assert length(ir.cfg.blocks) == 1
    rt = argextype(ir.stmts[end].val, ir, sv.sp)
    computations = HloComputationProto[]
    comp = HloComputationProto(
        name = "comp",
        instructions = HloInstructionProto[ ],
        id = 0
    )
    push!(computations, comp)
    xla_args = _compile_to_xla!(computations, comp, ir, sv)

    pshape = ProgramShape(
        parameters = collect(map(dtype_to_shape, xla_args)),
        result = dtype_to_shape(rt)
    )

    config = XLAComputationConfig(
        program_shape = pshape
    )
    hlo_module = HloModuleProto(
        name = "test",
        computations = computations,
        entry_computation_name = "comp",
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

function representable(T)
    return sizeof(T) != 0
end

dtype_to_shape(T::Type{<:XRTArray}) = convert(Shape, T)
dtype_to_shape(T::Type{<:XLA.XLAScalar}) = dtype_to_shape(XRTArray{T, (), 0})
function dtype_to_shape(T::DataType)
    Shape(
        element_type = xla.PrimitiveType.TUPLE,
        tuple_shapes = collect(dtype_to_shape(fieldtype(T, i)) for i = 1:fieldcount(T) if representable(fieldtype(T, i)))
    )
end

struct_to_literal(A::XRTArray) = convert(LiteralProto, convert(Array, A))
struct_to_literal(x::XLA.XLAScalar) = struct_to_literal(XRTArray(x))
function struct_to_literal(x)
    T = typeof(x)
    LiteralProto(
        shape = dtype_to_shape(T),
        tuple_literals = collect(struct_to_literal(getfield(x, i)) for i = 1:fieldcount(T) if representable(fieldtype(T, i)))
    )
end
