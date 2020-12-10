using Revise
using XLA, ProtoBuf

using .LibTPU
using .LibTPU: SE_DeviceMemoryAllocator, SE_ExecutableRunOptions,
               SE_ExecutionInput, SE_MaybeOwningDeviceMemory,
               XLA_MaybeOwningDeviceMemoryShapeTree

p = TpuPlatform()
initialize!(p)
exec = TpuExecutor(p)
initialize!(exec)
compiler = TpuCompiler()

TT = XRTArray{Float32, (1024, 1024), 2}
interp = XLA.GPUInterpreter(Base.get_world_counter())
ir = XLA.compile_sig(interp, Tuple{typeof(identity), TT})
hlo_module_group, rt = XLA.compile_to_xla(ir, nothing)

stream = TpuStream(exec)

allocator = SE_DeviceMemoryAllocator(p, exec)
executable = compile!(compiler, hlo_module_group, [[exec]], allocator)[]

xs = convert(LibTPU.XLA_Shape, XLA.Shape(Float32, (1024, 1024)))
size = shape_size(compiler, xs)
mem = allocate!(exec, UInt64(size), 0)

x = ones(Float32, 1024, 1024)

ptrs = SE_MaybeOwningDeviceMemory[
    SE_MaybeOwningDeviceMemory(mem, false, 0, allocator)
]

unsafe_copyto_async!(mem, Ptr{UInt8}(Base.unsafe_convert(Ptr{Float32}, x)), UInt64(size); exec, stream)
inputs = SE_ExecutionInput[
    SE_ExecutionInput(
        XLA_MaybeOwningDeviceMemoryShapeTree(xs,
            Base.unsafe_convert(Ptr{SE_MaybeOwningDeviceMemory}, ptrs)
        ),
        C_NULL, 0, xs
    )#=,
    SE_ExecutionInput(
        XLA_MaybeOwningDeviceMemoryShapeTree(xs,
            Base.unsafe_convert(Ptr{SE_MaybeOwningDeviceMemory}, ptrs)
        ),
        C_NULL, 0, xs
    ),=#
]
output = execute_async!(executable, SE_ExecutableRunOptions(allocator, stream, nothing; run_id=5), inputs)
output_mem = unsafe_load(output.result.bases)

y = zeros(Float32, 1024, 1024)
unsafe_copyto_async!(Ptr{UInt8}(Base.unsafe_convert(Ptr{Float32}, y)), output_mem, UInt64(size); exec, stream)
LibTPU.TpuExecutor_SynchronizeAllActivity(exec)

