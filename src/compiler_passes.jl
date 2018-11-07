using Base.Meta
using .Compiler: IRCode, IncrementalCompact, process_node!, BasicBlock, Const, Bottom, quoted,
    stmt_effect_free

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
        if isexpr(stmt, :static_parameter)
            @assert isa(ir.types[i], Const)
            ir.stmts[i] = ir.types[i].val
            continue
        end
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

function run_xla_embedding_passes!(ir, sv)
    Compiler.verify_ir(ir);
    ir = Compiler.compact!(ir, true)
    Compiler.verify_ir(ir);
    domtree = Compiler.construct_domtree(ir.cfg);
    ir = Compiler.domsort_ssa!(ir, domtree);
    Compiler.verify_ir(ir);
    ir = Compiler.cfg_simplify!(ir)
    Compiler.verify_ir(ir);
    ir = Compiler.compact!(ir, true)
    ir = refine_types!(ir, sv)
    ir = Compiler.ssa_inlining_pass!(ir, ir.linetable, sv)
    ir = Compiler.compact!(ir, true)
    for i = 1:3
        domtree = Compiler.construct_domtree(ir.cfg);
        ir = Compiler.domsort_ssa!(ir, domtree)
        Compiler.verify_ir(ir);
        ir = Compiler.cfg_simplify!(ir)
        ir = Compiler.compact!(ir, true)
    end
    ir = Compiler.compact!(ir)
    ir = Compiler.compact!(ir)
    Compiler.verify_ir(ir);
    ir, sv
end
