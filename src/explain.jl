struct NotOffloadableError
    ir
    sv
    reason
end

function Base.showerror(io::IO, e::NotOffloadableError)
    println(io, "The specified function is not offloadable. The offending IR was:")
    Base.IRShow.show_ir(io, e.ir; verbose_linetable=false)
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
    meth = Base.func_for_method_checked(x[3], Tuple{sig.parameters[2:end]...})
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
                    printstyled("Unknown destructuring.", color=:bold)
                    Base.show_backtrace(stdout, reverse(bt))
                    return
                end
                nsig = Tuple{atys...}
                return explain_suboptimal_inference(nsig, bt)
            end
        end
    end
end
