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
    sig = Tuple{typeof(f), argtypes.parameters...}
    mi = Compiler.code_for_method(method, metharg, methsp, params.world);
    sv = Compiler.OptimizationState(mi, params);
    ir = Compiler.inflate_ir(ci, mi);
    run_xla_embedding_passes!(ir, sv)
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
