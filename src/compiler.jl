const Compiler = Core.Compiler
using .Compiler: ReturnNode, argextype, SSAValue, ⊑, widenconst, userefs, LineInfoNode, GotoNode,
    IRCode, GotoIfNot, PiNode, PhiNode, foreachssa, block_for_inst, insert_node!,
    dominates, dominated, DominatedBlocks
using InteractiveUtils
using ..XLA: HloParameter, HloOp, XLAComputationConfig, HloModuleProto, HloProto,
    HloSnapshot, XLAComputation, HloMap, HloGetTupleElement, XLAScalar,
    HloReduceWindow, HloReduce, HloTuple

Base.iterate(d::Compiler.DominatedBlocks, state...) = Compiler.iterate(d, state...)

function grab_ir(f, argtypes)
    method = @which f(W, x, b)
    sig = typeof((f, W, x, b))
    mi = Compiler.code_for_method(method, sig, Core.svec(), params.world)
    sv = Compiler.OptimizationState(mi, params)
    ir = Compiler.inflate_ir(ci, mi)
end

macro check_ir(cond, args...)
    @assert length(args) <= 1
    reason = length(args) == 0 ? nothing : esc(args[1])
    quote
        if !$(esc(cond))
            throw(NotOffloadableError($(esc(:ir)), $(esc(:sv)), $(reason)))
        end
    end
end

# A representation of HloIf that takes two IRCodes to represent the computations
struct _HloIf <: HloOp{:conditional}
    if_func::IRCode
    else_func::IRCode
end

struct _HloWhile <: HloOp{:while}
    condition_func::IRCode
    body_func::IRCode
end

Base.show(io::IO, hloif::_HloIf) = print(io, "_HloIf(<omitted>)")
Base.show(io::IO, hloif::_HloWhile) = print(io, "_HloWhile(<omitted>)")

function outline_if_bb(ir, bb_to_outline, types, used_from_outside, merge_phi)
    block = ir.cfg.blocks[bb_to_outline]
    block = ir.cfg.blocks[bb_to_outline]
    outlined_ir = IRCode(Any[], Any[], Int32[], UInt8[], Compiler.CFG(BasicBlock[
        BasicBlock(Compiler.StmtRange(1, 0))
    ], Int64[]), LineInfoNode[], Any[], Any[], Core.svec())
    for i in 1:length(used_from_outside)
        Compiler.append_node!(outlined_ir, types[i], Expr(:call, Core.getfield, Compiler.Argument(2), i), 0)
    end
    for idx in block.stmts
        stmt = ir.stmts[idx]
        isa(stmt, GotoNode) && continue
        stmt′ = isa(stmt, Expr) ? copy(stmt) : stmt
        urs = userefs(stmt′)
        for op in urs
            val = op[]
            if isa(val, SSAValue)
                if val.id in block.stmts
                    op[] = SSAValue(val.id - first(block.stmts) + length(used_from_outside) + 1)
                else
                    op[] = SSAValue(findfirst(==(val), used_from_outside))
                end
            elseif isa(val, Compiler.Argument)
                op[] = SSAValue(findfirst(==(val), used_from_outside))
            end
        end
        stmt′ = urs[]
        Compiler.append_node!(outlined_ir, ir.types[idx], stmt′, 0)
    end
    edge_idx = findfirst(==(bb_to_outline), merge_phi.edges)
    val_id = (merge_phi.values[edge_idx]::SSAValue).id
    Compiler.append_node!(outlined_ir, ir.types[val_id], ReturnNode(SSAValue(val_id - first(block.stmts) + length(used_from_outside) + 1)), 0)
    return outlined_ir
end

function outline_loop_blocks(ir, domtree, bbs_to_outline, types, phi_nodes, used_from_outside, is_header=false)
    bbs = ir.cfg.blocks
    outlined_ir = IRCode(Any[], Any[], Int32[], UInt8[], Compiler.CFG(BasicBlock[
        BasicBlock(Compiler.StmtRange(1, 0))
    ], Int64[]), LineInfoNode[], Any[], Any[], Core.svec())
    ncarried_vals = length(phi_nodes)+length(used_from_outside)
    for i in 1:ncarried_vals
        Compiler.append_node!(outlined_ir, types[i], Expr(:call, Core.getfield, Compiler.Argument(2), i), 0)
    end
    bb_starts = Int[ncarried_vals+1]
    for bb_to_outline in bbs_to_outline
        block = bbs[bb_to_outline]
        for idx in block.stmts
            stmt = ir.stmts[idx]
            isa(stmt, GotoNode) && continue
            if isa(stmt, GotoIfNot)
                @assert is_header && bb_to_outline == bbs_to_outline[end] && idx == block.stmts[end]
            end
            if isa(stmt, PhiNode)
                continue
            end
            stmt′ = isa(stmt, Expr) ? copy(stmt) : stmt
            urs = userefs(stmt′)
            for op in urs
                val = op[]
                if isa(val, SSAValue)
                    val_bb = block_for_inst(ir.cfg, val.id)
                    bb_idx = findfirst(==(val_bb), bbs_to_outline)
                    if bb_idx !== nothing && !isa(ir.stmts[val.id], PhiNode)
                        def_block = bbs[val_bb]
                        block_local_id = val.id - first(def_block.stmts)
                        if bb_idx == 1 && is_header
                            block_local_id -= length(phi_nodes)
                        end
                        op[] = SSAValue(bb_starts[bb_idx] + block_local_id)
                    else
                        found = findfirst(e->e.first==val, phi_nodes)
                        if found == nothing
                            found = findfirst(==(val), used_from_outside) + length(phi_nodes)
                            @assert found !== nothing
                        end
                        op[] = SSAValue(found)
                    end
                elseif isa(val, Compiler.Argument)
                    op[] = SSAValue(findfirst(==(val), used_from_outside) + length(phi_nodes))
                end
            end
            stmt′ = urs[]
            Compiler.append_node!(outlined_ir, ir.types[idx], stmt′, 0)
        end
        push!(bb_starts, length(outlined_ir.stmts) + 1)
    end
    return (bb_starts, outlined_ir)
end

function outline_loop_header(ir, domtree, bb_to_outline, types, phi_nodes, used_from_outside)
    _, outlined_ir = outline_loop_blocks(ir, domtree, [bb_to_outline], types, phi_nodes, used_from_outside, true)
    branch = outlined_ir.stmts[end]
    outlined_ir.stmts[end] = ReturnNode(branch.cond)
    outlined_ir.types[end] = Bool
    return outlined_ir
end

function outline_loop_body(ir, domtree, bbs_to_outline, types, phi_nodes, used_from_outside)
    bbs = ir.cfg.blocks
    ncarried_vals = length(phi_nodes)+length(used_from_outside)
    # This is significant overkill, but I don't care for now. If all you have is a hammer...
    bbs_to_outline = sort(bbs_to_outline, lt=(a, b)->dominates(domtree, a, b))
    bb_starts, outlined_ir = outline_loop_blocks(ir, domtree, bbs_to_outline, types, phi_nodes, used_from_outside)
    # Reconstruct the loop carried tuple
    args = Any[]
    for (_, phi) in phi_nodes
        idx = findfirst(==(bbs_to_outline[end]), phi.edges)
        val = phi.values[idx]
        val_bb = block_for_inst(ir.cfg, val.id)
        bb_idx = findfirst(==(val_bb), bbs_to_outline)
        def_block = bbs[val_bb]
        block_local_id = val.id - first(def_block.stmts)
        push!(args, SSAValue(bb_starts[bb_idx] + block_local_id))
    end
    # The rest just gets carried through
    for i in 1:length(used_from_outside)
        push!(args, SSAValue(length(phi_nodes) + i))
    end
    tuple_type = Compiler.tuple_tfunc(types)
    Compiler.append_node!(outlined_ir, tuple_type, Expr(:call, Core.tuple, args...), 0)
    Compiler.append_node!(outlined_ir, tuple_type, ReturnNode(SSAValue(length(outlined_ir.stmts))), 0)
    return outlined_ir
end

function erase_block!(ir, bbidx)
    block = ir.cfg.blocks[bbidx]
    ir.stmts[block.stmts[1:end-1]] .= nothing
    ir.stmts[block.stmts[end]] = ReturnNode()
    empty!(ir.cfg.blocks[bbidx].preds)
    empty!(ir.cfg.blocks[bbidx].succs)
end

function NCA(domtree, nodes)
    nodes = copy(nodes)
    min_level = minimum(n->domtree.nodes[n].level, nodes)
    # Ascend all nodes that are not at the same level
    for i = 1:length(nodes)
        if domtree.nodes[nodes[i]].level != min_level
            nodes[i] = domtree.idoms[nodes[i]]
        end
    end
    # Ascend levels together until all the nodes are the same
    while !all(==(nodes[1]), nodes)
        for i = 1:length(nodes)
            nodes[i] = domtree.idoms[nodes[i]]
        end
    end
    nodes[1]
end

struct OutlineLoop
    header::Int
end

struct OutlineIf
    head::Int
    join::Int
end
entry(l::OutlineLoop) = l.header
entry(l::OutlineIf) = l.head

# Traverse the domtree and find all outlining regions
# TODO: There's much better ways to do this. For now
# I just want to get it to work at all
function analyze_regions(domtree, cfg)
    bbs = cfg.blocks
    todo = Any[]
    for i = 1:length(bbs)
        if length(bbs[i].preds) > 1
            # If we dominate one of the the predecessors, this is a backege,
            # and we outline it as a loop.
            # Else this is a conditional and we need to find the nearest common
            # ancestor of all predecessors in the dominator tree for outlining
            # as a conditional region.
            dominate_any = false
            backedges = Pair{Int, Int}[]
            for pred in bbs[i].preds
                if dominates(domtree, i, pred)
                    dominate_any = true
                    push!(backedges, pred=>i)
                    break
                end
            end
            if dominate_any
                push!(todo, OutlineLoop(i))
            else
                push!(todo, OutlineIf(NCA(domtree, bbs[i].preds), i))
            end
        end
    end
    sort(todo, lt=(a, b)->dominates(domtree, entry(b), entry(a)))
end

function outline_loop!(ir, sv, domtree, loop_header)
    bbs = ir.cfg.blocks
    loop_header_bb = ir.cfg.blocks[loop_header]
    loop_header_preds = loop_header_bb.preds
    @assert length(loop_header_preds) == 2

    # Map the loop
    if dominates(domtree, loop_header, loop_header_preds[1])
        loop_latch = loop_header_preds[1]
        entry_block = loop_header_preds[2]
    else
        loop_latch = loop_header_preds[2]
        entry_block = loop_header_preds[1]
    end
    @assert dominates(domtree, loop_header, loop_latch)
    loop_body_blocks = Int[loop_latch]
    worklist = Int[loop_latch]
    while !isempty(worklist)
        item = pop!(worklist)
        for pred in bbs[item].preds
            if pred != loop_header
                push!(loop_body_blocks, pred)
                push!(worklist, pred)
            end
        end
    end

    loop_cond = ir.stmts[loop_header_bb.stmts[end]]
    @assert isa(loop_cond, GotoIfNot)
    (bb1, bb2) = (loop_header + 1, loop_cond.dest)
    if dominates(domtree, bb2, loop_latch)
        exit_block = bb1
    else
        exit_block = bb2
    end

    # First identify everything that needs to be in our loop carry
    used_from_outside = Union{Compiler.Argument, SSAValue}[]
    used_outside = SSAValue[]
    phi_nodes = Pair{SSAValue,PhiNode}[]
    for bb in Iterators.flatten((loop_header, loop_body_blocks))
        stmt_range = bbs[bb].stmts
        for stmt_idx in stmt_range
            stmt = ir.stmts[stmt_idx]
            # We handle PhiNodes separately
            if isa(stmt, PhiNode)
                @assert bb == loop_header
                push!(phi_nodes, SSAValue(stmt_idx)=>stmt)
                continue
            end
            urs = userefs(stmt)
            for op in urs
                val = op[]
                if isa(val, SSAValue)
                    # If it's in our current basic block, we don't care
                    val.id in stmt_range && continue
                    # Else find out which bb it's in
                    val_bb = block_for_inst(ir.cfg, val.id)
                    # TODO: If something from the loop header (other than a loop
                    # carried phi) is used in the loop latch it needs to be duplicated.
                    val_bb == loop_header && continue
                    val_bb in loop_body_blocks && continue
                elseif !isa(val, Compiler.Argument)
                    continue
                end
                push!(used_from_outside, val)
            end
        end
    end
    unique!(used_from_outside)
    for bb in dominated(domtree, exit_block)
        stmt_range = ir.cfg.blocks[bb].stmts
        for stmt_idx in stmt_range
            stmt = ir.stmts[stmt_idx]
            foreachssa(stmt) do val
                val_bb = block_for_inst(ir.cfg, val.id)
                (val_bb == loop_header) || return
                push!(used_outside, val)
            end
        end
    end
    types = Any[]
    ssa_order = Any[]
    for (val, _) in phi_nodes
        push!(ssa_order, val)
        push!(types, argextype(val, ir, sv.sp))
    end
    for val in used_from_outside
        push!(ssa_order, val)
        push!(types, argextype(val, ir, sv.sp))
    end
    # We put these in a tuple in the order (phi_nodes..., used_from_outside...)
    # Replace Phi nodes by getfield calls
    tuple_T = Compiler.tuple_tfunc(types)
    cond_ir = outline_loop_header(ir, domtree, loop_header, types, phi_nodes, used_from_outside)
    append!(cond_ir.argtypes, [nothing, tuple_T])
    Compiler.verify_ir(cond_ir)
    body_ir = outline_loop_body(ir, domtree, loop_body_blocks, types, phi_nodes, used_from_outside)
    append!(body_ir.argtypes, [nothing, tuple_T])
    Compiler.verify_ir(body_ir)

    # Now erase the blocks we outlined
    erase_block!.((ir,), [loop_header; loop_body_blocks])
    # Fix the CFG
    entry_succs, exit_preds = ir.cfg.blocks[entry_block].succs, ir.cfg.blocks[exit_block].preds
    empty!(entry_succs); empty!(exit_preds)
    push!(entry_succs, exit_block); push!(exit_preds, entry_block)
    # Insert a GotoNode at the end of the entry block
    insert_node!(ir, bbs[entry_block].stmts[end], Any, GotoNode(exit_block), true)
    # Assemble the initial execution tuple
    initial_args = Any[]
    for (_, node) in phi_nodes
        push!(initial_args, node.values[findfirst(==(entry_block), node.edges)])
    end
    append!(initial_args, used_from_outside)
    first_exit_pos = bbs[exit_block].stmts[1]
    initial = insert_node!(ir, first_exit_pos, tuple_T,
        Expr(:call, Core.tuple, initial_args...), false)
    # Insert our while node at the top of the exit_block
    while_return = insert_node!(ir, first_exit_pos, tuple_T,
        Expr(:call, _HloWhile(cond_ir, body_ir), initial), false)
    # Insert destructuring calls
    used_outside_replacement = Any[]
    for u in used_outside
        idx = findfirst(==(u), ssa_order)
        push!(used_outside_replacement,
            insert_node!(ir, first_exit_pos, types[idx],
                Expr(:call, Core.getfield, while_return, idx), false))
    end
    # Replace any outside uses of loop values
    for bb in dominated(domtree, exit_block)
        stmt_range = ir.cfg.blocks[bb].stmts
        for stmt_idx in stmt_range
            stmt = ir.stmts[stmt_idx]
            urs = userefs(stmt)
            for op in urs
                val = op[]
                if isa(val, SSAValue) && val in used_outside
                    op[] = used_outside_replacement[findfirst(==(val), used_outside)]
                end
            end
            ir.stmts[stmt_idx] = urs[]
        end
    end
    ir = Compiler.compact!(ir)
    Compiler.verify_ir(ir)
    ir
end


function outline_if!(ir, sv, domtree, split_block, join)
    # Simple version for experiments: Assume each branch of the if has
    # exactly one basic block
    @assert isa(ir.stmts[ir.cfg.blocks[split_block].stmts[end]], GotoIfNot)
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

    all_used_from_outside = Vector{Any}[]
    for bb in divergence_blocks
        # TODO: Allow each branch to have multiple BBS
        used_from_outside = Vector{Any}()
        stmt_range = ir.cfg.blocks[bb].stmts
        for stmt_idx in stmt_range
            stmt = ir.stmts[stmt_idx]
            urs = userefs(stmt)
            for op in urs
                val = op[]
                if isa(val, SSAValue)
                    # If it's in our current basic block, we don't care
                    val.id in stmt_range && continue
                elseif !isa(val, Compiler.Argument)
                    continue
                end
                push!(used_from_outside, val)
            end
        end
        unique!(used_from_outside)
        push!(all_used_from_outside, used_from_outside)
    end
    if_types = Any[argextype(arg, ir, sv.sp) for arg in all_used_from_outside[1]]
    if_tuple_T = Compiler.tuple_tfunc(if_types)
    if_func = outline_if_bb(ir, divergence_blocks[1], if_types, all_used_from_outside[1], phi_node)
    else_types = Any[argextype(arg, ir, sv.sp) for arg in all_used_from_outside[2]]
    else_tuple_T = Compiler.tuple_tfunc(else_types)
    else_func = outline_if_bb(ir, divergence_blocks[2], else_types, all_used_from_outside[2], phi_node)
    append!(if_func.argtypes, [nothing, if_tuple_T])
    append!(else_func.argtypes, [nothing, else_tuple_T])
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
    #
    insert_pos = ir.cfg.blocks[join_block].stmts[1]
    if_tuple = insert_node!(ir, insert_pos, if_tuple_T, Expr(:call, Core.tuple, all_used_from_outside[1]...))
    else_tuple = insert_node!(ir, insert_pos, else_tuple_T, Expr(:call, Core.tuple, all_used_from_outside[2]...))
    ir.stmts[insert_pos] = Expr(:call, inst, cond, if_tuple, else_tuple)
    # Delete divergence blocks
    erase_block!.((ir,), divergence_blocks)
    # Fix CFG
    let cb_succs = ir.cfg.blocks[split_block].succs, jb_preds = ir.cfg.blocks[join_block].preds
        empty!(cb_succs); empty!(jb_preds);
        push!(cb_succs, join_block); push!(jb_preds, split_block)
    end
    for bbidx in divergence_blocks
        block = ir.cfg.blocks[bbidx]
        empty!(block.preds); empty!(block.succs)
    end
    ir = Compiler.compact!(ir)
    Compiler.verify_ir(ir)
    ir
end

function outline_control_flow!(ir, sv)
    bbs = ir.cfg.blocks
    length(bbs) == 1 && return ir

    domtree = Compiler.construct_domtree(ir.cfg);

    todo = analyze_regions(domtree, ir.cfg)
    while !isempty(todo)
        item = popfirst!(todo)
        if isa(item, OutlineLoop)
            ir = outline_loop!(ir, sv, domtree, item.header)
        else
            ir = outline_if!(ir, sv, domtree, item.head, item.join)
        end
    end
    Compiler.verify_ir(ir)
    ir = Compiler.cfg_simplify!(ir)
    Compiler.verify_ir(ir)
    ir = Compiler.compact!(ir)
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
    ir = outline_control_flow!(ir, sv)
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
            isassigned(ssa_vals, arg.id) || @show arg
            return ssa_vals[arg.id]
        else
            error()
        end
    end
    for (stmt_idx, stmt) in pairs(ir.stmts)
        isa(stmt, Nothing) && continue
        isa(stmt, ReturnNode) && continue
        @check_ir !isa(stmt, PhiNode) "Unsupported control flow found in function"
        if isa(stmt, PiNode)
            continue
            #ssa_vals[stmt_idx] = hlo_eval(stmt.args[1])
        end
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
        if !isexpr(stmt, :invoke) && !(isexpr(stmt, :call) && (isa(stmt.args[1], _HloIf) || isa(stmt.args[1], _HloWhile)))
            @Base.show stmt
            Base.display(ir)
            error("Unrecognized expr")
        end
        hlo_inst = isexpr(stmt, :invoke) ? stmt.args[2] : stmt.args[1]
        hlo_inst = argextype(hlo_inst, ir, sparams)
        hlo_inst = hlo_inst.val
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
                args = map(hlo_eval, stmt.args[2:end])
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
                    # Having a paremeter instruction is required, so if we don't
                    # otherwise have it, add it manually
                    if !representable(widenconst(ir′.argtypes[2]))
                        instr = HloInstructionProto(comp′, "parameter")
                        instr.shape = dtype_to_shape(Tuple{})
                        instr.parameter_number = 0
                    end
                    pushfirst!(computations, comp′)
                    _compile_to_xla!(computations, comp′, ir′, nothing)
                    push!(proto.called_computation_ids, comp′.id)
                end
            elseif isa(hlo_inst, _HloWhile)
                initial = hlo_eval(stmt.args[2])
                while_type = ir.types[stmt_idx]
                tuple_T = hlo_inst.condition_func.argtypes[2]
                shape = dtype_to_shape(tuple_T)
                proto = HloInstructionProto(comp,
                    GenericHloOp{:while}(Float32, ()), initial,
                    shape = shape)
                proto.called_computation_ids = Int64[]
                for ir′ in (hlo_inst.body_func, hlo_inst.condition_func)
                    comp′ = HloComputationProto(
                        name = "comp$(length(computations))",
                        instructions = HloInstructionProto[ ],
                        id = length(computations)
                    )
                    if !representable(widenconst(ir′.argtypes[2]))
                        instr = HloInstructionProto(comp′, "parameter")
                        instr.shape = dtype_to_shape(Tuple{})
                        instr.parameter_number = 0
                    end
                    pushfirst!(computations, comp′)
                    _compile_to_xla!(computations, comp′, ir′, nothing)
                    push!(proto.called_computation_ids, comp′.id)
                end
            else
                args = map(hlo_eval, stmt.args[3:end])
                shape = Shape(shape_infer(hlo_inst, map(arg->argextype(arg, ir, sparams), stmt.args[3:end])...)...)
                proto = HloInstructionProto(comp, hlo_inst, args...; shape=shape)
            end
            ssa_vals[stmt_idx] = proto
        elseif isa(hlo_inst, Type) && hlo_inst <: XRTArray
            ty = argextype(stmt.args[3], ir, sparams)
            if isa(ty, Const) && isa(ty.val, XLAScalar)
                ssa_vals[stmt_idx] = HloInstructionProto(comp, HloConstant(ty.val))
            else
                # Allow treating an XRTArray as a scalar inside another XRTArray
                # This is a no-op at the HLO level
                ty = widenconst(ty)
                @assert ty <: XRTArray && (hlo_inst <: XRTArray{<:Any, (), 0})
                ssa_vals[stmt_idx] = hlo_eval(stmt.args[3])
            end
        elseif hlo_inst == Base.convert
            ty = argextype(stmt.args[3], ir, sparams)
            arg = argextype(stmt.args[4], ir, sparams)
            if arg <: XRTArray{<:Any, (), 0} && isa(ty, Const) && ty.val == eltype(arg)
                # Allow conversions from a scalar array to its element
                # type and codegen it as a no-op.
                ssa_vals[stmt_idx] = hlo_eval(stmt.args[4])
            else
                @show stmt
                error("Unrecognized convert")
            end
        else
            @show stmt
            error("Unrecognized")
        end
    end
    ret = ir.stmts[end]
    @assert isa(ret, ReturnNode)
    comp.root_id = hlo_eval(ret.val).id
    xla_args
end

struct NotOffloadableError
    ir
    sv
    reason
end

function Base.showerror(io::IO, e::NotOffloadableError)
    println(io, "The specified function is not offloadable. The offending IR was:")
    Base.IRShow.show_ir(stdout, e.ir; verbose_linetable=true)
    if isa(e.reason, String)
        println(io, e.reason)
    elseif isa(e.reason, Function)
        e.reason(io, e.ir, e.sv)
    end
end

function synthesize_bt_for_line(ir, line)
    entry = ir.lines[line]
    frames = Base.StackFrame[]
    while entry !== 0
        lin = ir.linetable[entry]
        push!(frames, Base.StackFrame(lin.method,  lin.file, lin.line, nothing, false, lin.inlined_at != 0, C_NULL))
        entry = lin.inlined_at
    end
    frames
end

function analyze_unreachable(io::IO, ir, sv)
    print(io, "The called function was guaranteed to error. Analyzing...\n")
    rv = ir.stmts[end]
    @assert isa(rv, ReturnNode)
    # Generally the call before the ReturnNode will be the offending one
    before = ir.stmts[end - 1]
    if ir.types[end - 1] !== Bottom || !isexpr(before, :call)
        println(io, "This IR looks odd. Prior statement was not a call that was guaranteed to error")
        return
    end
    # See if this is a method error
    f = argextype(before.args[1], ir, sv.sp)
    isa(f, Const) || return println(io, "Expected function to be a constant")
    f = f.val
    args = Tuple{[widenconst(argextype(before.args[i], ir, sv.sp)) for i = 2:length(before.args)]...}
    args = Base.to_tuple_type(args)
    tt = Base.signature_type(f, args)
    m = ccall(:jl_gf_invoke_lookup, Any, (Any, UInt), tt, sv.params.world)
    if m === nothing
        synthetic_bt = synthesize_bt_for_line(ir, length(ir.stmts)-1)
        showerror(io, MethodError(f, args, sv.params.world), synthetic_bt)
    end
end

function typeinf_code_slottypes(mi::Core.Compiler.MethodInstance, run_optimizer::Bool, params::Core.Compiler.Params)
    ccall(:jl_typeinf_begin, Cvoid, ())
    result = Core.Compiler.InferenceResult(mi)
    frame = Core.Compiler.InferenceState(result, false, params)
    frame === nothing && return (nothing, Any)
    Core.Compiler.typeinf(frame)
    ccall(:jl_typeinf_end, Cvoid, ())
    frame.inferred || return (nothing, Any)
    return (frame.src, result.result, frame.slottypes)
end


function explain_suboptimal_inference(sig, bt=Base.StackFrame[])
    params = Core.Compiler.CustomParams(typemax(UInt); aggressive_constant_propagation=true)
    methds = Base._methods_by_ftype(sig, 1, params.world)
    x = methds[1]
    meth = Base.func_for_method_checked(x[3], sig.parameters[2:end])
    mi = Compiler.code_for_method(meth, sig, x[2], params.world)
    (ci, ty, slottypes) = typeinf_code_slottypes(mi, false, params)
    ir = Compiler.inflate_ir(ci, Compiler.spvals_from_meth_instance(mi),
                             Compiler.matching_cache_argtypes(mi, nothing)[1])
    empty!(ir.argtypes); append!(ir.argtypes, slottypes)
    sv = Compiler.OptimizationState(mi, params)
    for (idx, (stmt, type)) in enumerate(zip(ir.stmts, ir.types))
        if type === Any || isa(type, UnionAll)
            if isexpr(stmt, :(=))
                stmt = stmt.args[2]
            end
            if isexpr(stmt, :call)
                println("Found uninferred call to $(stmt.args[1]), recursing")
                append!(bt, synthesize_bt_for_line(ir, idx))
                atys = Any[]
                for i = 1:length(stmt.args)
                    ty = widenconst(argextype(stmt.args[i], ir, sv.sp, slottypes))
                    if ty === Any
                        if isa(stmt.args[i], GlobalRef)
                            printstyled("Global variable ($(stmt.args[i])) was referenced.", color=:bold)
                            Base.show_backtrace(stdout, reverse(bt))
                            return
                        end
                        printstyled("Call argument was `Any` (analysis was unable to determine a precise reason. This should be improved).", color=:bold)
                    end
                    push!(atys, ty)
                end
                if atys[1] == typeof(Core.getfield)
                    if atys[2] == Core.Box
                        printstyled("A closure `Box` was introduced. See Julia issue #15276.", color=:bold)
                        Base.show_backtrace(stdout, reverse(bt))
                        return
                    end
                    printstyled("Unkown destructuring.", color=:bold)
                    Base.show_backtrace(stdout, reverse(bt))
                    return
                end
                nsig = Tuple{atys...}
                return explain_suboptimal_inference(nsig, bt)
            end
        end
    end
end

function compile_to_xla(ir, sv)
    # TODO: Since all HLO operations are essentially effect free, we could just have
    # a compilation result that throws this error.
    @check_ir isa(ir.stmts[end], ReturnNode) && isdefined(ir.stmts[end], :val) analyze_unreachable
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
