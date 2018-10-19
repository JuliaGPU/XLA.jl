function code_typed_argtys(@nospecialize(types=Tuple); optimize=true, params=nothing, constvals=nothing)
    ccall(:jl_is_in_pure_context, Bool, ()) && error("code reflection cannot be used from generated functions")
    types = Base.to_tuple_type(types)
    asts = []
    world = ccall(:jl_get_world_counter, UInt, ())
    params = params === nothing ? Core.Compiler.Params(world) : params
    for x in Base._methods_by_ftype(types, -1, world)
        meth = Base.func_for_method_checked(x[3], types.parameters[2:end])
        argtypes = nothing
        if constvals !== nothing
            argtypes = Vector(undef, length(types.parameters)+1)
            for i = 1:length(types.parameters)+1
                argtypes[i] = isassigned(constvals, i) ?
                    Core.Compiler.Const(constvals[i]) : (
                        i == 1 ? typeof(f) : types.parameters[i-1])
            end
        end
        (code, ty) = Core.Compiler.typeinf_code(meth, x[1], x[2], optimize, params, argtypes)
        code === nothing && error("inference not successful") # inference disabled?
        push!(asts, code => ty)
    end
    return asts
end

function code_typed_xla(f, argtypes::Type)
    @Base.show (f, argtypes)
    argvals = Vector{Any}(undef, length(argtypes.parameters) + 1)
    argvals[1] = f
    params = Compiler.CustomParams(typemax(UInt); aggressive_constant_propagation=true, ignore_all_inlining_heuristics=true)
    ci = (Compiler == Core.Compiler ? Base.code_typed : NI.code_typed)(f, argtypes, argvals; params=params)[1].first
    Base.display(ci)
    method = which(f, argtypes)
    (metharg, methsp) = ccall(:jl_type_intersection_with_env, Any, (Any, Any),
                              Compiler.argtypes_to_type(Any[typeof(f), argtypes.parameters...]),
                              method.sig)
    mi = Compiler.code_for_method(method, metharg, methsp, params.world);
    sv = Compiler.OptimizationState(mi, params);
    ir = Compiler.inflate_ir(ci, mi);
    run_xla_embedding_passes!(ir, sv)
end

function code_typed_xla(sig::Type; argvals=nothing)
    params = Compiler.CustomParams(typemax(UInt); aggressive_constant_propagation=true, ignore_all_inlining_heuristics=true)
    ci = code_typed_argtys(sig; params=params, constvals=argvals)[1].first
    methods = Base._methods_by_ftype(sig, -1, params.world)
    method = ccall(:jl_gf_invoke_lookup, Any, (Any, UInt), sig, params.world).func
    (metharg, methsp) = ccall(:jl_type_intersection_with_env, Any, (Any, Any),
                              Compiler.argtypes_to_type(Any[sig.parameters...]),
                              method.sig)
    mi = Compiler.code_for_method(method, metharg, methsp, params.world);
    sv = Compiler.OptimizationState(mi, params);
    ir = Compiler.inflate_ir(ci, mi);
    run_xla_embedding_passes!(ir, sv)
end

function Base.dump(c::XLAComputation)
   mktemp() do f, io
       writeproto(io, c.hlo_snapshot)
       close(io)
       run(`/home/keno/tensorflow/bazel-bin/tensorflow/compiler/xla/tools/dumped_computation_to_text $f`)
   end
   nothing
end

macro tpu_dump(expr)
    @assert isexpr(expr, :call)
    quote
        let f = $(esc(expr.args[1]))
            ir, sv = code_typed_xla(f, Base.typesof($(map(esc, expr.args[2:end])...)))
            Base.display(ir)
            Base.dump(XLA.compile_to_xla(ir, sv))
        end
    end
end

macro tpu_dump_file(file, expr)
    @assert isexpr(expr, :call)
    quote
        let f = $(esc(expr.args[1]))
            ir, sv = code_typed_xla(f, Base.typesof($(map(esc, expr.args[2:end])...)))
            c = XLA.compile_to_xla(ir, sv)
            open($(esc(file)), "w") do io
                writeproto(io, c.hlo_snapshot)
            end
        end
    end
end

macro tpu_compile(expr)
    @assert isexpr(expr, :call)
    quote
        let f = $(esc(expr.args[1]))
            ir, sv = code_typed_xla(f, Base.typesof($(map(esc, expr.args[2:end])...)))
            Base.display(ir)
            compld = XLA.compile($(esc(:sess)), XLA.compile_to_xla(ir, sv))
            compld
        end
    end
end
