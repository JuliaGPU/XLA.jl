const Compiler = Core.Compiler
using .Compiler: ReturnNode, argextype, SSAValue, ⊑, widenconst, userefs, LineInfoNode, GotoNode, IRCode, GotoIfNot
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

# A representation of HloIf that takes two IRCodes to represent the computations
struct _HloIf <: HloOp{:conditional}
    if_func::IRCode
    else_func::IRCode
end

Base.show(io::IO, hloif::_HloIf) = print(io, "_HloIf(<omitted>)")

function outline_bb(ir, bb_to_outline, merge_phi)
    block = ir.cfg.blocks[bb_to_outline]
    # For now disallow using any values outside the if
    @assert all(block.stmts) do idx
        urs = userefs(ir.stmts[idx])
        for op in urs
            val = op[]
            if isa(val, SSAValue)
                return false
            end
        end
        return true
    end
    outlined_ir = IRCode(Any[], Any[], Int32[], UInt8[], Compiler.CFG(BasicBlock[
        BasicBlock(Compiler.StmtRange(1, 0))
    ], Int64[]), LineInfoNode[], Any[], Any[], Core.svec())
    for idx in block.stmts
        stmt = ir.stmts[idx]
        isa(stmt, GotoNode) && continue
        # If we did have SSA values, we'd rename them here
        Compiler.append_node!(outlined_ir, ir.types[idx], copy(stmt), 0)
    end
    edge_idx = findfirst(==(bb_to_outline), merge_phi.edges)
    val_id = (merge_phi.values[edge_idx]::SSAValue).id
    Compiler.append_node!(outlined_ir, ir.types[val_id], ReturnNode(SSAValue(val_id - first(block.stmts) + 1)), 0)
    return outlined_ir
end

function outline_control_flow!(ir)
    # Simple version for experiments: Assume each branch of the if has
    # exactly one basic block
    if !isa(ir.stmts[ir.cfg.blocks[1].stmts[end]], GotoIfNot)
        return ir
    end
    split_block = 1
    while true
        divergence_blocks = copy(ir.cfg.blocks[split_block].succs)
        @assert length(divergence_blocks) == 2
        # For our simple experiments only allow a single block
        join_block = ir.cfg.blocks[divergence_blocks[1]].succs[1]
        @assert all(divergence_blocks) do block
            preds = ir.cfg.blocks[block].preds
            succs = ir.cfg.blocks[block].succs
            length(preds) == 1 && length(succs) == 1 && succs[1] == join_block
        end
        phi_node = ir.stmts[ir.cfg.blocks[join_block].stmts[1]]
        if_func = outline_bb(ir, divergence_blocks[1], phi_node)
        else_func = outline_bb(ir, divergence_blocks[2], phi_node)
        inst = _HloIf(if_func, else_func)
        condition = ir.stmts[ir.cfg.blocks[split_block].stmts[end]]
        @assert isa(condition, GotoIfNot)
        # Replace condition by direct branch to join block
        ir.stmts[ir.cfg.blocks[split_block].stmts[end]] = GotoNode(join_block)
        # Replace phi node by HloIf
        ## Look through the conversion to bool
        @assert isa(condition.cond, SSAValue)
        cond_stmt = ir.stmts[condition.cond.id]
        @assert isexpr(cond_stmt, :invoke)
        cond = cond_stmt.args[4]
        ir.stmts[ir.cfg.blocks[join_block].stmts[1]] = Expr(:call, inst, cond, nothing, nothing)
        # Delete divergence blocks
        for bbidx in divergence_blocks
            block = ir.cfg.blocks[bbidx]
            ir.stmts[block.stmts[1:end-1]] .= nothing
            ir.stmts[block.stmts[end]] = ReturnNode()
        end
        # Fix CFG
        let cb_succs = ir.cfg.blocks[split_block].succs, jb_preds = ir.cfg.blocks[join_block].preds
            empty!(cb_succs); empty!(jb_preds);
            push!(cb_succs, join_block); push!(jb_preds, split_block)
        end
        for bbidx in divergence_blocks
            block = ir.cfg.blocks[bbidx]
            empty!(block.preds); empty!(block.succs)
        end
        if !isa(ir.stmts[ir.cfg.blocks[join_block].stmts[end]], GotoIfNot)
            break
        end
        split_block = join_block
    end
    ir = Compiler.cfg_simplify!(ir)
    ir
end

function process_function_argument(argvals, sig, idx, ir, stmt, sparams)
    if sizeof(sig.parameters[idx]) != 0
        # If the function being called has non-zero information, we need to make sure it's a constant
        # In the future, XLA will allow this.
        fty = argextype(stmt.args[2+idx], ir, sparams)
        if !isa(fty, Const)
            error("At the moment applied computations must not capture non-constant values")
        end
        argvals = argvals === nothing ? Vector{Any}(undef, 1) : argvals
        argvals[1] = fty.val
    end
    argvals
end

function _compile_to_xla!(computations, comp, ir, sv)
    ir = outline_control_flow!(ir)
    arg_instrs = Vector{HloInstructionProto}(undef, length(ir.argtypes))
    xla_args = Type[]
    sparams = sv === nothing ? Core.svec() : sv.sp
    for i = 1:length(ir.argtypes)
        # TODO: We could represent this as a kConstant
        isa(ir.argtypes[i], Const) && continue
        if sv !== nothing && sv.linfo.def.isva && i == length(ir.argtypes)
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
        if isa(arg, Compiler.Argument)
            return arg_instrs[arg.n]
        elseif isa(arg, SSAValue)
            return ssa_vals[arg.id]
        else
            error()
        end
    end
    for (stmt_idx, stmt) in pairs(ir.stmts)
        isa(stmt, Nothing) && continue
        isa(stmt, ReturnNode) && continue
        if isexpr(stmt, :new) || (isexpr(stmt, :call) && Compiler.is_known_call(stmt, tuple, ir, sparams))
            ssa_vals[stmt_idx] = HloInstructionProto(comp, HloTuple(), map(hlo_eval, filter(x->x!=nothing, stmt.args[2:end]))...)
            continue
        end
        if isexpr(stmt, :call) && Compiler.is_known_call(stmt, getfield, ir, sparams)
            structT = widenconst(argextype(stmt.args[2], ir, sparams))
            elt = argextype(stmt.args[3], ir, sparams)
            isa(elt, Const) || error("non-constant structure indexing (1)")
            idx = Compiler.try_compute_fieldidx(structT, elt.val)
            idx !== nothing || error("non-constant structure indexing (2)")
            if !representable(fieldtype(structT, idx))
                continue
            end
            tuple_idx = count(i->representable(fieldtype(structT, i)), 1:idx-1) + 1
            proto = HloInstructionProto(comp, HloGetTupleElement(tuple_idx - 1),
                hlo_eval(stmt.args[2]))
            ssa_vals[stmt_idx] = proto
            continue
        end
        if isa(stmt, GlobalRef) && isa(argextype(stmt, ir, sparams), Const)
            continue
        end
        if !isexpr(stmt, :invoke) && !(isexpr(stmt, :call) && isa(stmt.args[1], _HloIf))
            @Base.show stmt
            Base.display(ir)
            error("Unrecognized expr")
        end
        hlo_inst = isexpr(stmt, :invoke) ? stmt.args[2] : stmt.args[1]
        isa(hlo_inst, QuoteNode) && (hlo_inst = hlo_inst.value)
        if isa(hlo_inst, HloOp)
            if isa(hlo_inst, HloMap) || isa(hlo_inst, HloReduceWindow) || isa(hlo_inst, HloReduce)
                args = map(hlo_eval, stmt.args[4:end])
                proto = HloInstructionProto(comp, hlo_inst, args...)
                sig = Tuple{typeof(hlo_inst).parameters[1], (XRTArray{eltype(argextype(stmt.args[i], ir, sparams)), (), 0} for i = 4:length(stmt.args))...}
                argvals = process_function_argument(nothing, sig, 1, ir, stmt, sparams)
                ir′, sv′ = code_typed_xla(sig; argvals=argvals)
                if sizeof(sig.parameters[1]) != 0
                    ir′.argtypes[1] = Const(argvals[1])
                end
                comp′ = HloComputationProto(
                    name = "comp$(length(computations))",
                    instructions = HloInstructionProto[ ],
                    id = length(computations)
                )
                pushfirst!(computations, comp′)
                _compile_to_xla!(computations, comp′, ir′, sv′)
                proto.called_computation_ids = [comp′.id]
            elseif isa(hlo_inst, HloSelectAndScatter)
                args = map(hlo_eval, stmt.args[5:end])
                proto = HloInstructionProto(comp, hlo_inst, args...)
                proto.called_computation_ids = Int64[]
                # Compile comparison function
                let
                    select_sig = Tuple{typeof(hlo_inst).parameters[1], (XRTArray{eltype(argextype(stmt.args[5], ir, sparams)), (), 0} for i = 1:2)...}
                    select_argvals = process_function_argument(nothing, select_sig, 1, ir, stmt, sparams)
                    ir′, sv′ = code_typed_xla(select_sig; argvals=select_argvals)
                    if sizeof(select_sig.parameters[1]) != 0
                        ir′.argtypes[1] = Const(select_argvals[1])
                    end
                    comp′ = HloComputationProto(
                        name = "comp$(length(computations))",
                        instructions = HloInstructionProto[ ],
                        id = length(computations)
                    )
                    pushfirst!(computations, comp′)
                    _compile_to_xla!(computations, comp′, ir′, sv′)
                    push!(proto.called_computation_ids, comp′.id)
                end
                # Compiler Scatter function
                let
                    scatter_sig = Tuple{typeof(hlo_inst).parameters[2], (XRTArray{eltype(argextype(stmt.args[5], ir, sparams)), (), 0} for i = 1:2)...}
                    scatter_argvals = process_function_argument(nothing, scatter_sig, 2, ir, stmt, sparams)
                    ir′, sv′ = code_typed_xla(scatter_sig; argvals=scatter_argvals)
                    if sizeof(scatter_sig.parameters[1]) != 0
                        ir′.argtypes[1] = Const(scatter_argvals[1])
                    end
                    comp′ = HloComputationProto(
                        name = "comp$(length(computations))",
                        instructions = HloInstructionProto[ ],
                        id = length(computations)
                    )
                    pushfirst!(computations, comp′)
                    _compile_to_xla!(computations, comp′, ir′, sv′)
                    push!(proto.called_computation_ids, comp′.id)
                end
            elseif isa(hlo_inst, _HloIf)
                @assert isexpr(stmt, :call)
                if any(x->isa(x, Nothing), stmt.args[2:end])
                    empty_tuple = HloInstructionProto(comp, HloTuple())
                end
                args = map(x->isa(x, Nothing) ? empty_tuple : hlo_eval(x), stmt.args[2:end])
                if_type = ir.types[stmt_idx]
                proto = HloInstructionProto(comp,
                    GenericHloOp{:conditional}(if_type.parameters[1], if_type.parameters[2]), args...)
                proto.called_computation_ids = Int64[]
                for ir′ in (hlo_inst.if_func, hlo_inst.else_func)
                    comp′ = HloComputationProto(
                        name = "comp$(length(computations))",
                        instructions = HloInstructionProto[ ],
                        id = length(computations)
                    )
                    instr = HloInstructionProto(comp′, "parameter")
                    instr.shape = dtype_to_shape(Tuple{})
                    instr.parameter_number = 0
                    pushfirst!(computations, comp′)
                    _compile_to_xla!(computations, comp′, ir′, nothing)
                    push!(proto.called_computation_ids, comp′.id)
                end
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
    xlac, rt
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
