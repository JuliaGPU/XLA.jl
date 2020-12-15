function code_typed_xla(f, argtypes::Type; kwargs...)
    sig = Base.signature_type(f, argtypes)
    code_typed_xla(sig; kwargs...)
end

function code_typed_xla(sig::Type; argvals=nothing)
    interp = XLA.GPUInterpreter(Base.get_world_counter())
    matches = Core.Compiler.findall(sig, Core.Compiler.method_table(interp); limit=1)
    @assert Core.Compiler.length(matches) == 1
    if argvals == nothing
        mi = Core.Compiler.specialize_method(Core.Compiler.getindex(matches, 1))
    else
        match = Core.Compiler.getindex(matches, 1)
        argtypes = Any[match.spec_types.parameters...]  # FIXME: relation to sig?
        @assert length(argvals) == length(argtypes)-1
        for i = 1:length(argvals)
            if isassigned(argvals, i)
                argtypes[i+1] = Core.Compiler.Const(argvals[i])
            end
        end
        mi = Core.Compiler.specialize_method(match.method, argtypes, match.sparams)
        # FIXME: doesn't work
        #        https://github.com/JuliaLang/julia/pull/29261/files
    end
    Core.Compiler.typeinf_ext_toplevel(interp, mi)
    ci = ci_cache_lookup(mi, interp.world, interp.world)
    ir = Core.Compiler.inflate_ir(ci.inferred, mi)

    sv = Compiler.OptimizationState(mi, Core.Compiler.OptimizationParams(interp), interp)
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

#make_options(sess::Session) = nothing
#=
function make_options(;sess::TPUSession=global_session(), devices=nothing)
    replica_device_coords = devices === nothing ? nothing : sess.topology[devices]
    (sess, replica_device_coords)
end
=#

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
