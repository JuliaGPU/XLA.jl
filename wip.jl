using Test

using XLA
using .LibTPU

function main(dims=(128,128); T=Int32)
    # compile

    x = rand(T, dims)
    y = zero(x)

    f = x->x*x
    tt = Tuple{XRTArray{T, dims, length(dims)}}
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

    xs = convert(LibTPU.XLA_Shape, XLA.Shape(T, dims))

    size = shape_size(compiler, xs)
    @assert size == sizeof(y)
    mem = allocate!(executor(), UInt64(size), 0)

    buffers = [
        TpuMaybeOwningDeviceMemory(mem, false, 0, allocator)
    ]

    unsafe_copyto!(mem, Ptr{UInt8}(pointer(x)), UInt64(size); exec=executor())


    # execute

    inputs = [
        TpuExecutionInput(
            TpuMaybeOwningDeviceMemoryShapeTree(xs, buffers), [], xs
        )
    ]

    options = TpuExecutableRunOptions(allocator, global_stream(), nothing; run_id=5)
    output = execute_async!(executable, options, inputs)
    output_mem = unsafe_load(output.result.bases)


    # verify

    synchronize()

    unsafe_copyto!(Ptr{UInt8}(pointer(y)), output_mem, UInt64(size); exec=executor())

    @test y == x*x
end

isinteractive() || main()
