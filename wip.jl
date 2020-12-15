using Test

using XLA
using .LibTPU

function main(dims=(128,128); T=Int32)
    # compile

    x = rand(T, dims)
    y = zero(x)

    f = x->x*x
    tt = Tuple{HLOArray{T, dims, length(dims)}}
    sig = Base.signature_type(f, tt)

    interp = XLA.GPUInterpreter(Base.get_world_counter())
    ir = XLA.compile_sig(interp, sig)
    println("Generated Julia IR:\n", ir)
    hlo_module_group, rt = XLA.compile_to_xla(ir, nothing)
    println("Generated HLO modules:\n")
    XLA.code_hlo(hlo_module_group)

    compiler = TpuCompiler()
    allocator = TpuDeviceMemoryAllocator(platform(), executor())
    executable = compile!(compiler, hlo_module_group, [[executor()]], allocator)[]


    # allocate

    xs = XLA.Shape(T, dims)

    # FIXME: deal with array padding
    # https://cloud.google.com/tpu/docs/performance-guide#consequences_of_tiling
    ys = with_status() do status
        ys = Ref{LibTPU.XLA_Shape}(xs)  # does not work with an #undef ref
        LibTPU.XlaShapeToTpuPaddedShape(Ref{LibTPU.XLA_Shape}(xs), ys, status)
        convert(XLA.Shape, ys[])
    end
    if ys != xs
        error("Array shape $xs will get padded on the TPU to $ys")
    end

    d_x = TPUArray(x)

    size = shape_size(compiler, xs)
    @assert size == sizeof(d_x)


    # execute

    buffers = [
        # TODO: support for donating the input
        TpuMaybeOwningDeviceMemory(d_x.mem, false, 0, allocator)
    ]

    inputs = [
        TpuExecutionInput(
            TpuMaybeOwningDeviceMemoryShapeTree(xs, buffers), [], xs
        )
    ]

    options = TpuExecutableRunOptions(allocator, default_stream(), nothing; run_id=5)
    output = execute_async!(executable, options, inputs)
    output_mem = unsafe_load(output.result.bases)


    # verify

    synchronize()

    # TODO: create arrays from execution output
    unsafe_copyto!(pointer(y), output_mem, size; executor=executor())

    @test y == x*x
end

isinteractive() || main()
