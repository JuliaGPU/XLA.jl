struct OriginalArgument
    n::Int
end

function peval_expr(atype, inline_argmapping)
    world = typemax(UInt)
    min_valid = UInt[typemin(UInt)]
    max_valid = UInt[typemax(UInt)]
    meth = Base._methods_by_ftype(atype, 1, world, min_valid, max_valid)
    if meth === false || length(meth) != 1
        error("Broadcastee method match failed")
    end
    sparams = Any[meth[1][2]...]
    inline_ir, inline_linetable = method_match_to_ir(meth, Tuple{atype.parameters[2:end]...})
    inline_ir = partial_evaluation_inlining_pass!(inline_ir, inline_linetable, inline_argmapping, sparams)
    inline_ir, meth, sparams, inline_linetable
end

"""
Like a regular inling pass, but does partial evaluation along the way.
"""
function partial_evaluation_inlining_pass!(ir, linetable, argmapping, sparams=Any[])
    function static_eval(arg)
        if isa(arg, Compiler.Argument)
            return haskey(argmapping, arg.n) ? argmapping[arg.n] : OriginalArgument(arg.n)
        elseif isa(arg, GlobalRef)
            return getfield(arg.mod, arg.name)
        elseif isa(arg, QuoteNode)
            return arg.value
        elseif isa(arg, SSAValue)
            return static_eval(ir[arg])
        elseif isexpr(arg, :static_parameter)
            return sparams[arg.args[1]]
        else
            return arg
        end
    end
    todo = Tuple{Int, Tuple{Bool, Bool, Bool, Int}, Method, Vector{Any}, Vector{LineInfoNode}, Compiler.IRCode, Bool}[]
    world = ccall(:jl_get_world_counter, UInt, ())
    for (idx, stmt) in pairs(ir.stmts)
        if isa(stmt, GlobalRef)
            ir[SSAValue(idx)] = static_eval(stmt)
            continue
        end
        isa(stmt, Expr) || continue
        # See if all arguments are statically known
        is_completely_static = all(stmt.args[1:end]) do arg
            while true
                if isa(arg, Compiler.Argument)
                    if !haskey(argmapping, arg.n)
                        return false
                    else
                        arg = argmapping[arg.n]
                    end
                elseif isa(arg, OriginalArgument)
                    return false
                elseif isa(arg, GlobalRef)
                    return true
                elseif isa(arg, SSAValue)
                    arg = ir[arg]
                elseif isa(arg, Expr)
                    return false
                else
                    return true
                end
            end
        end
        if isexpr(stmt, :new)
            if is_completely_static
                args = map(static_eval, stmt.args)
                ir[SSAValue(idx)] = eval(Expr(:new, args...))
            end
            continue
        end
        isexpr(stmt, :call) || continue
        eargs = stmt.args
        isempty(eargs) && continue
        arg1 = eargs[1]

        res = Compiler.exprtype_func(arg1, ir)
        res !== nothing || continue
        (f, ft) = res

        if is_completely_static
            val = static_eval(arg1)
            args = map(static_eval, eargs[2:end])
            ir[SSAValue(idx)] = val(args...)
        else
            isa(f, Core.Builtin) && continue

            # If we've reached a broadcast, partially evalute the closure
            # with respect to any captured arguments
            if typeof(f) == typeof(broadcast)
                broadcast_arg = static_eval(stmt.args[2])
                atype = Tuple{typeof(broadcast_arg), map(arg->eltype(Compiler.widenconst(Compiler.exprtype(arg, ir, ir.mod))), eargs[3:end])...}
                inline_ir, meth, sparams, inline_linetable = peval_expr(atype, Dict(1=>broadcast_arg))
                stmt.args[2] = inline_ir
                continue
            end
            # Don't recurse into the internals of operations intercepted by xla
            if typeof(f) == typeof(*) || typeof(f) == typeof(+) || typeof(f) == typeof(max) || typeof(f) == typeof(min) || typeof(f) == typeof(sin) || typeof(f) == typeof(cos)
                continue
            end
            inline_argmapping = Dict(i=>static_eval(eargs[i]) for i = 1:length(eargs))
            atype = Tuple{map(arg->Compiler.widenconst(Compiler.exprtype(arg, ir, ir.mod)), eargs)...}
            inline_ir, meth, sparams, inline_linetable = peval_expr(atype, inline_argmapping)

            push!(todo, (idx, (false, false, false, false), meth[1][3], sparams, inline_linetable, inline_ir, Compiler.linear_inline_eligible(inline_ir)))
        end
    end
    domtree = Compiler.construct_domtree(ir.cfg)
    if !isempty(todo)
        ir = Compiler.batch_inline!(todo, ir, domtree, linetable, nothing)
    end
    return ir
end

#=
m = Chain2(
  Dense(28^2, 32, relu),
  Dense(32, 10))
ir, linetable = @grab_ir m(xf)
ir = Compiler.compact!(ir)
ir = partial_evaluation_inlining_pass!(ir, linetable, Dict(1=>m))
ir = Compiler.compact!(ir)
builder = icxx"""new xla::ComputationBuilder($client, "test");"""
generate_xla_from(builder, ir, typemax(UInt), ["m","xf"], [xla_shapeof(xf)])
=#
