function code_typed_argtys(@nospecialize(types=Tuple); optimize=true, params=nothing, constvals=nothing)
    ccall(:jl_is_in_pure_context, Bool, ()) && error("code reflection cannot be used from generated functions")
    types = Base.to_tuple_type(types)
    asts = []
    world = ccall(:jl_get_world_counter, UInt, ())
    params = params === nothing ? Core.Compiler.Params(world) : params
    for x in Base._methods_by_ftype(types, -1, world)
        meth = Base.func_for_method_checked(x[3], Tuple{types.parameters[2:end]...})
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
    argvals = Vector{Any}(undef, length(argtypes.parameters) + 1)
    argvals[1] = f
    params = Compiler.CustomParams(typemax(UInt); aggressive_constant_propagation=true, ignore_all_inlining_heuristics=true)
    ci = (Compiler == Core.Compiler ? Base.code_typed : NI.code_typed)(f, argtypes, argvals; params=params)[1].first
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
    cds = code_typed_argtys(sig; params=params, constvals=argvals)
    if length(cds) == 0
        return nothing
    end
    ci = cds[1].first
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
       run(`$(joinpath(dirname(dirname(pathof(TensorFlow))), "deps", "downloads", "bin", "dumped_computation_to_text")) $f`)
   end
   nothing
end

macro tpu_dump(expr)
    @assert isexpr(expr, :call)
    quote
        let f = $(esc(expr.args[1]))
            ir, sv = code_typed_xla(f, Base.typesof($(map(esc, expr.args[2:end])...)))
            Base.dump(XLA.compile_to_xla(ir, sv)[1])
        end
    end
end

macro tpu_dump_file(file, expr)
    @assert isexpr(expr, :call)
    quote
        let f = $(esc(expr.args[1]))
            ir, sv = code_typed_xla(f, Base.typesof($(map(esc, expr.args[2:end])...)))
            c, rt = XLA.compile_to_xla(ir, sv)
            open($(esc(file)), "w") do io
                writeproto(io, c.hlo_snapshot)
            end
        end
    end
end

make_options(sess::Session) = nothing
function make_options(;sess::TPUSession=global_session(), devices=nothing)
    replica_device_coords = devices === nothing ? nothing : sess.topology[devices]
    (sess, replica_device_coords)
end

adjust_type(T) = T
adjust_type(::Type{XRTRemoteStruct{T}}) where {T} = T

macro tpu_compile(args...)
    expr = args[end] # The expression to compile
    kwargs = Expr[]
    for i in 1:length(args)-1
        x = args[i]
        if x isa Expr && x.head == :(=) # Keyword given of the form "foo=bar"
            push!(kwargs, x)
        else
            return Expr(:call, :error, "@tpu_compile expects only one non-keyword argument")
        end
    end
    opts = Expr(:call, make_options,
            (Expr(:kw, esc(kw.args[1]), esc(kw.args[2])) for kw in kwargs)...)
    @assert isexpr(expr, :call)
    quote
        let f = $(esc(expr.args[1]))
            argtypes = Tuple{map(adjust_type, Base.typesof($(map(esc, expr.args[2:end])...)).parameters)...}
            ir, sv = code_typed_xla(f, argtypes)
            sess, replica_device_coords = $opts
            try
                XLA.compile(sess, XLA.compile_to_xla(ir, sv; replica_device_coords=replica_device_coords)...)
            catch err
                @warn("Compilation failed; attempting to explain suboptimal inference:")
                XLA.explain_suboptimal_inference(Tuple{typeof(f), argtypes.parameters...})
                rethrow(err)
            end
        end
    end
end


macro tpu(args...)
    expr = args[end]
    @assert isexpr(expr, :call)
    quote
        the_args = tuple($(map(esc,expr.args)...))
        handle = @tpu_compile $(args[1:end-1]...) $(Expr(:call, (Expr(:call, getfield, :the_args, i) for i = 1:length(expr.args))...))
        run(handle, the_args...)
    end
end
