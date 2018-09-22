"""
A verify dumb CFG simplification pass that simply goes through and merges any basic blocks
that only have a single edge between them.
"""

using NotInferenceDontLookHere
const Compiler = NI

using Base.Meta
using .Compiler: IRCode, IncrementalCompact, process_node!, BasicBlock, Const, Bottom, quoted,
    stmt_effect_free
Base.size(r::Compiler.StmtRange) = Compiler.size(r)
Base.getindex(r::Compiler.StmtRange, i) = Compiler.getindex(r, i)
function cfg_simplify!(ir::IRCode)
    bbs = ir.cfg.blocks
    merge_into = zeros(Int64, length(bbs))
    merged_succ = zeros(Int64, length(bbs))
    # Walk the CFG at from the entry block and aggressively combine blocks
    for (idx, bb) in enumerate(bbs)
        if length(bb.succs) == 1
            succ = bb.succs[1]
            if length(bbs[succ].preds) == 1
                merge_into[succ] = idx
                merged_succ[idx] = succ
            end
        end
    end
    max_bb_num = 1
    bb_rename_succ = zeros(Int64, length(bbs))
    for i = 1:length(bbs)
        if merge_into[i] != 0
            bb_rename_succ[i] = -1
            continue
        end
        bb_rename_succ[i] = max_bb_num
        max_bb_num += 1
    end
    bb_rename_pred = zeros(Int64, length(bbs))
    for i = 1:length(bbs)
        if merged_succ[i] != 0
            bb_rename_pred[i] = -1
            continue
        end
        bbnum = i
        while merge_into[bbnum] != 0
            bbnum = merge_into[bbnum]
        end
        bb_rename_pred[i] = bb_rename_succ[bbnum]
    end
    result_bbs = [findfirst(j->i==j, bb_rename_succ) for i = 1:max_bb_num-1]
    result_bbs_lengths = zeros(Int, max_bb_num-1)
    compact = IncrementalCompact(ir, true)
    compact.bb_rename_succ = bb_rename_succ
    compact.bb_rename_pred = bb_rename_pred
    for (idx, orig_bb) in enumerate(result_bbs)
        ms = orig_bb
        while ms != 0
            result_bbs_lengths[idx] += length(bbs[ms].stmts)
            ms = merged_succ[ms]
        end
    end
    bb_starts = [1; 1 .+ cumsum(result_bbs_lengths)]
    # Look at the original successor
    function compute_succs(i)
        orig_bb = result_bbs[i]
        while merged_succ[orig_bb] != 0
            orig_bb = merged_succ[orig_bb]
        end
        map(i->bb_rename_succ[i], bbs[orig_bb].succs)
    end

    function compute_preds(i)
        orig_bb = result_bbs[i]
        preds = bbs[orig_bb].preds
        map(preds) do pred
            while merge_into[pred] != 0
                pred = merge_into[pred]
            end
            bb_rename_succ[pred]
        end
    end
    compact.result_bbs = [BasicBlock(
        Compiler.StmtRange(bb_starts[i], i+1 > length(bb_starts) ? length(compact.result) : bb_starts[i+1]-1),
        compute_preds(i), compute_succs(i)) for i = 1:length(result_bbs)]
    result_idx = 1
    for (idx, orig_bb) in enumerate(result_bbs)
        ms = orig_bb
        while ms != 0
            for i in bbs[ms].stmts
                stmt = ir.stmts[i]
                compact.result[compact.result_idx] = nothing
                compact.result_types[compact.result_idx] = ir.types[i]
                compact.result_lines[compact.result_idx] = ir.lines[i]
                compact.result_flags[compact.result_idx] = ir.flags[i]
                process_node!(compact, compact.result_idx, stmt, i, i, ms, true)
                # We always increase the result index to ensure a predicatable
                # placement of the resulting nodes.
                compact.result_idx += 1
            end
            ms = merged_succ[ms]
        end
    end
    @assert Base.sum(result_bbs_lengths) == length(compact.result)

    compact.active_result_bb = length(bb_starts)
    # Compute the new CFG

    Compiler.finish(compact)
end

function agressive_new_tfunc(stmt, t, ir, sv)
    isconst = true
    args = Vector{Any}(undef, length(stmt.args) - 1)
    for i = 2:length(stmt.args)
        at = Compiler.argextype(stmt.args[i], ir, sv.sp)
        if at === Bottom
            t = Bottom
            isconst = false
            break
        elseif at isa Const
            if !(at.val isa fieldtype(t, i - 1))
                t = Bottom
                isconst = false
                break
            end
            args[i-1] = at.val
        else
            isconst = false
        end
    end
    if isconst
        t = Const(ccall(:jl_new_structv, Any, (Any, Ptr{Cvoid}, UInt32), t, args, length(args)))
    end
    t
end

function refine_types!(ir, sv)
    for i = 1:length(ir.stmts)
        stmt = ir.stmts[i]
        (isexpr(stmt, :call) || isexpr(stmt, :new)) || continue
        arg1t = Compiler.argextype(stmt.args[1], ir, sv.sp)
        # This is a more agressive version of the tfunc inference we do for :new
        if isexpr(stmt, :new)
            if !isa(arg1t, Const) || arg1t.val.mutable
                continue
            end
            t = agressive_new_tfunc(stmt, arg1t.val, ir, sv)
            ir.types[i] = t
            if isa(t, Const) && stmt_effect_free(stmt, t, ir, ir.spvals)
                ir.stmts[i] = t.val
            end
            continue
        end
        f = Compiler.singleton_type(arg1t)
        f === nothing && continue
        if isa(f, Core.Builtin)
            argtypes = Any[Compiler.argextype(stmt.args[i], ir, sv.sp) for i in 2:length(stmt.args)]
            rt = Compiler.builtin_tfunction(f, argtypes, nothing, sv.params)
            ir.types[i] = rt
        end
    end
    ir
end

function pre_xla_passes(ir::IRCode, )
    ir = refine_types!(ir, sv)
    ir = NI.run_passes(ir, sv)
    domtree = Compiler.construct_domtree(ir.cfg)
    ir = Compiler.domsort_ssa!(ir, domtree)
    ir = NI.run_passes(ir, sv)
end
