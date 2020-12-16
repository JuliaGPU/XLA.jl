using Test

using XLA
using .LibTPU

function main(dims=(128,128); T=Int32)
    # compile

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
