using Test

using XLA
using .LibTPU

const p = TpuPlatform()
initialize!(p)

function main(dims=(128,128); T=Int32)
    # init

    global p

    exec = TpuExecutor(p)
    initialize!(exec)


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
    println("Generated HLO modules:\n", hlo_module_group)

    compiler = TpuCompiler()
    allocator = TpuDeviceMemoryAllocator(p, exec)
    executable = compile!(compiler, hlo_module_group, [[exec]], allocator)[]


    # allocate

    xs = convert(LibTPU.XLA_Shape, XLA.Shape(T, dims))

    size = shape_size(compiler, xs)
    size == sizeof(y) || # FIXME: try (4,4) or (64,64)
        error("XLA tensor size $size doesn't match Julia's $(sizeof(y))-byte array")
    mem = allocate!(exec, UInt64(size), 0)

    buffers = [
        TpuMaybeOwningDeviceMemory(mem, false, 0, allocator)
    ]

    unsafe_copyto!(mem, Ptr{UInt8}(pointer(x)), UInt64(size); exec)


    # execute

    stream = TpuStream(exec)

    inputs = [
        TpuExecutionInput(
            TpuMaybeOwningDeviceMemoryShapeTree(xs, buffers), [], xs
        )
    ]

    options = TpuExecutableRunOptions(allocator, stream, nothing; run_id=5)
    output = execute_async!(executable, options, inputs)
    output_mem = unsafe_load(output.result.bases)



    # verify

    # FIXME: LibTPU.TpuExecutor_SynchronizeAllActivity(exec)
    with_status() do status
        LibTPU.TpuExecutor_BlockHostUntilDone(exec, stream, status)
    end

    unsafe_copyto!(Ptr{UInt8}(pointer(y)), output_mem, UInt64(size); exec)

    @test y == x*x
end

isinteractive() || main()
