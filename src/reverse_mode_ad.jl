"""
    Source-to-Source Reverse-Mode AD

Turns a function like

function foo(a, b, c)
    d = f(a, b)
    z = g(c, d)
    z
end

into a function that computes the gradient
using reverse mode automatic differentiation:

function ∇foo(∇z, foo, a, b, c)
    d = f(a, b)
    z = g(c, d)
    ∇g, ∇c, ∇d = ∇(g)(∇z, g, c, d)
    ∇f, ∇a, ∇b = ∇(f)(∇d, f, a, b)
    (∇a, ∇b, ∇c)
end
"""
function reverse_mode_ad! end

using .Compiler: NewSSAValue, Argument

# z = a * b
@inline ∇star(∇z, _, a, b) = (0., ∇z*b, ∇z*a)
∇(::Type{typeof(*)}) = ∇star
@inline ∇plus(∇z, _, a, b) = (0., ∇z, ∇z)
∇(::Type{typeof(+)}) = ∇plus
@inline ∇sin(∇z, _, a) = (0., ∇z*cos(a))
∇(::Type{typeof(sin)}) = ∇sin
∇(x) = nothing

Compiler.code_typed(∇star, Tuple{Float64, Function, Int64, Float64})

function reverse_mode_ad!(ir)
    new_stmts = Vector{Any}()
    new_types = Vector{Any}()
    retval_gradient = 1.0
    arg_gradients = Any[0.0 for i = 1:length(ir.argtypes)]
    gradients = Any[0.0 for i = 1:length(ir.stmts)]
    NewSSAValue(x) = SSAValue(length(ir.stmts) + x -1)
    function add_gradient!(gradients, new_stmts, new_types, i, val)
        if gradients[i] == 0.0
            gradients[i] = val
        else
            push!(new_stmts, Expr(:call, +, gradients[i], val))
            push!(new_types, Float64)
            gradients[i] = NewSSAValue(length(new_stmts))
        end
    end
    for i = (length(ir.stmts):-1:1)
        stmt = ir[SSAValue(i)]
        if isa(stmt, Compiler.ReturnNode)
            add_gradient!(gradients, new_stmts, new_types, stmt.val.id, retval_gradient)
        elseif isexpr(stmt, :call)
            res = Compiler.exprtype_func(stmt.args[1], ir)
            res !== nothing || error("Function of unkown type")
            (f, ft) = res
            reverse_mode_deriv = ∇(typeof(f))
            reverse_mode_deriv === nothing && error("No derivative for call $stmt")
            push!(new_stmts, Expr(:call, reverse_mode_deriv,
                    gradients[i], stmt.args[1:end]...))
            push!(new_types, NTuple{length(stmt.args[1:end]), Float64})
            new_call = NewSSAValue(length(new_stmts))
            for (fieldidx, arg) in enumerate(stmt.args[1:end])
                isa(arg, Union{SSAValue, Argument}) || continue
                push!(new_stmts, Expr(:call, getfield, new_call, fieldidx))
                push!(new_types, Float64)
                if isa(arg, SSAValue)
                    add_gradient!(gradients, new_stmts, new_types, arg.id, NewSSAValue(length(new_stmts)))
                elseif isa(arg, Compiler.Argument)
                    add_gradient!(arg_gradients, new_stmts, new_types, arg.n, NewSSAValue(length(new_stmts)))
                end
            end
        end
    end
    push!(new_stmts, Expr(:call, tuple, arg_gradients...))
    push!(new_types, NTuple{length(arg_gradients), Float64})
    push!(new_stmts, Compiler.ReturnNode(NewSSAValue(length(new_stmts))))
    push!(new_types, new_types[end])
    # Remove old return node
    (s, t, l) = pop!(ir.stmts), pop!(ir.types), pop!(ir.lines)
    append!(ir.stmts, new_stmts)
    append!(ir.types, new_types)
    append!(ir.lines, [1 for i = 1:length(new_stmts)])
    #push!(ir.stmts, s); push!(ir.types, t); push!(ir.lines, l)
    ir.cfg.blocks[1] = Compiler.BasicBlock(ir.cfg.blocks[1],
        Compiler.StmtRange(1, length(ir.stmts)))
    ir
end

eval(Compiler, quote
    function ssa_inlining_pass!(ir::IRCode, domtree, linetable, sv)
        # Go through the function, perfoming simple ininlingin (e.g. replacing call by constants
        # and analyzing legality of inlining).
        @timeit "analysis" todo = assemble_inline_todo!(ir, domtree, linetable, sv)
        isempty(todo) && return ir
        # Do the actual inlining for every call we identified
        @timeit "execution" ir = batch_inline!(todo, ir, domtree, linetable, sv)
    return ir
end
function maybe_make_invoke!(ir, idx, @nospecialize(etype), atypes::Vector{Any}, sv,
                    @nospecialize(atype_unlimited), isinvoke, isapply, @nospecialize(invoke_data))
    nu = countunionsplit(atypes)
    nu > 1 && return # TODO: The old optimizer did union splitting here. Is this the right place?
    ex = Expr(:invoke)
    linfo = spec_lambda(atype_unlimited, sv, invoke_data)
    linfo === nothing && return
    add_backedge!(linfo, sv)
    argexprs = ir[SSAValue(idx)].args
    argexprs = rewrite_exprargs((node, typ)->insert_node!(ir, idx, typ, node), arg->exprtype(arg, ir, ir.mod),
        isinvoke, isapply, argexprs)
    pushfirst!(argexprs, linfo)
    ex.typ = etype
    ex.args = argexprs
    ir[SSAValue(idx)] = ex
end
function spec_lambda(@nospecialize(atype), sv, @nospecialize(invoke_data))
    if invoke_data === nothing
        return ccall(:jl_get_spec_lambda, Any, (Any, UInt), atype, sv.params.world)
    else
        invoke_data = invoke_data::InvokeData
        atype <: invoke_data.types0 || return nothing
        return ccall(:jl_get_invoke_lambda, Any, (Any, Any, Any, UInt),
                     invoke_data.mt, invoke_data.entry, atype, sv.params.world)
    end
end
end)

struct FakeOptimizationState2
    params::Compiler.Params
end
Compiler.add_backedge!(::MethodInstance, ::FakeOptimizationState2) = nothing

function do_reverse_mode_ad!(ir)
    ir = Compiler.compact!(Compiler.compact!(reverse_mode_ad!(ir)))
    sv = eval(Expr(:new, NI.OptimizationState))
    sv.params = NI.Params(typemax(UInt))
    ir = partial_evaluation_inlining_pass!(ir, linetable, Dict())
    domtree = Compiler.construct_domtree(ir.cfg)
    ir = Compiler.compact!(Compiler.getfield_elim_pass!(ir, domtree))
    ir
end

function f(x, y)
    a = x * y
    b = sin(x)
    z = a + b
end

ir, linetable = @grab_ir f(1, 1);
ir = Compiler.compact!(ir);
@show ir;
@show do_reverse_mode_ad!(ir);
