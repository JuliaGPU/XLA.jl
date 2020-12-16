using Test

using XLA
using .LibTPU

function main(dims=(128,128); T=Int32)
    # compile

    # FIXME: redefining `f` with a dirty cache (e.g. using Revise) results in
    #        badly-inferred code. For example, using `x.+Int32(1)` first yields:
    # 12 1 ─ %1 = invoke HLOArray{Int32, (), 0}(1::Int32)::HLOArray{Int32, (), 0}
    #    │   %2 = invoke $(QuoteNode(XLA.HloBroadcast{0, 2}((), (128, 128))))(%1::HLOArray{Int32, (), 0})::HLOArray{Int32, (128, 128), 2}
    #    │   %3 = invoke $(QuoteNode(XLA.HloMap{typeof(+)}()))(+::typeof(+), _2::HLOArray{Int32, (128, 128), 2}, %2::Vararg{HLOArray{Int32, (128, 128), 2}, N} where N)::HLOArray{Int32, (128, 128), 2
    #    └──      return %3
    #
    #        redefining to `x.+Int32(2)` (without clearing the cache)
    #        introduces an extra conditional:
    #
    # 17 1 ─ %1 = Base.not_int(false)::Bool
    #    └──      goto #3 if not %1
    #    2 ─ %3 = invoke HLOArray{Int32, (), 0}(2::Int32)::HLOArray{Int32, (), 0}
    #    3 ┄ %4 = φ (#2 => %3, #1 => 2)::Union{Int32, HLOArray{Int32, (), 0}}
    #    │   %5 = π (%4, HLOArray{Int32, (), 0})
    #    │   %6 = invoke $(QuoteNode(XLA.HloBroadcast{0, 2}((), (128, 128))))(%5::HLOArray{Int32, (), 0})::HLOArray{Int32, (128, 128), 2}
    #    │   %7 = invoke $(QuoteNode(XLA.HloMap{typeof(+)}()))(+::typeof(+), _2::HLOArray{Int32, (128, 128), 2}, %6::Vararg{HLOArray{Int32, (128, 128), 2}, N} where N)::HLOArray{Int32, (128, 128), 2}
    #    └──      return %7
    empty!(XLA.CI_CACHE)

    # FIXME: the above IR triggers a compiler error in control flow outlining

    # FIXME: compiler error when using `x->x.+T(1)`

    # FIXME: runtime error when using `x->x.+1`

    # FIXME: compiler error using `x->broadcast(y->y == Int32(0) ? Int32(1) : Int32(2), x)`

    f = x->x.+x
    tt = Tuple{HLOArray{T, dims, length(dims)}}

    ir, sv = XLA.code_typed_xla(f, tt)
    println("Generated Julia IR:\n", ir)
    hlo_module_group, rt = XLA.compile_to_xla(ir, nothing)
    println("Generated HLO modules:\n")
    XLA.code_hlo(hlo_module_group)

    compiler = TpuCompiler()
    allocator = TpuDeviceMemoryAllocator(platform(), executor())
    executable = compile!(compiler, hlo_module_group, [[executor()]], allocator)[]


    # allocate

    x = rand(T, dims)
    y = zero(x)

    d_x = TPUArray(x)


    # execute

    buffers = [
        # TODO: support for donating the input
        TpuMaybeOwningDeviceMemory(d_x.mem, false, 0, allocator)
    ]

    inputs = [
        TpuExecutionInput(
            TpuMaybeOwningDeviceMemoryShapeTree(d_x.host_shape, buffers), [], d_x.host_shape
        )
    ]

    options = TpuExecutableRunOptions(allocator, default_stream(), nothing; run_id=5)
    output = execute_async!(executable, options, inputs)


    # verify

    copyto!(y, output.result; stream=default_stream(), manager=transfer_manager())
    @test y == f(x)
end

isinteractive() || main()
