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
    retval_val = (merge_phi.values[edge_idx]::SSAValue)
    val_id = retval_val.id
    retval_outside = findfirst(==(retval_val), used_from_outside)
    retval_val = retval_outside === nothing ?
        SSAValue(val_id - first(block.stmts) + length(used_from_outside) + 1) :
        SSAValue(retval_outside)
    Compiler.append_node!(outlined_ir, ir.types[val_id], ReturnNode(retval_val), 0)
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
        # It's possible for a phi node to reference a previous phi
        # node, in which case we can handle that
        carried_phi = findfirst(x->x[1] == val, phi_nodes)
        if carried_phi !== nothing
            push!(args, SSAValue(carried_phi))
            continue
        end
        val_bb = block_for_inst(ir.cfg, val.id)
        bb_idx = findfirst(==(val_bb), bbs_to_outline)
        @assert bb_idx !== nothing
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
        function process_val(val)
            if isa(val, SSAValue)
                # If it's in our current basic block, we don't care
                val.id in stmt_range && return
            elseif !isa(val, Compiler.Argument)
                return
            end
            push!(used_from_outside, val)
        end
        for stmt_idx in stmt_range
            stmt = ir.stmts[stmt_idx]
            urs = userefs(stmt)
            for op in urs
                val = op[]
                process_val(val)
            end
        end
        process_val(phi_node.values[findfirst(==(bb), phi_node.edges)])
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
    Compiler.verify_ir(if_func)
    Compiler.verify_ir(else_func)
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
