using Test
using XLA

using .LibTPU
using .LibTPU: SE_DeviceMemoryAllocator, SE_ExecutableRunOptions,
               SE_ExecutionInput, SE_MaybeOwningDeviceMemory,
               XLA_MaybeOwningDeviceMemoryShapeTree

# init

p = TpuPlatform()
initialize!(p)

exec = TpuExecutor(p)
initialize!(exec)


# compile

f = identity
tt = XRTArray{Float32, (1024, 1024), 2}

interp = XLA.GPUInterpreter(Base.get_world_counter())
ir = XLA.compile_sig(interp, Tuple{typeof(f), tt})
hlo_module_group, rt = XLA.compile_to_xla(ir, nothing)

compiler = TpuCompiler()
allocator = SE_DeviceMemoryAllocator(p, exec)
executable = compile!(compiler, hlo_module_group, [[exec]], allocator)[]


# allocate

xs = convert(LibTPU.XLA_Shape, XLA.Shape(Float32, (1024, 1024)))
size = shape_size(compiler, xs)
mem = allocate!(exec, UInt64(size), 0)

x = rand(Float32, 1024, 1024)
y = similar(x)

ptrs = SE_MaybeOwningDeviceMemory[
    SE_MaybeOwningDeviceMemory(mem, false, 0, allocator)
]

stream = TpuStream(exec)

unsafe_copyto_async!(mem, Ptr{UInt8}(Base.unsafe_convert(Ptr{Float32}, x)), UInt64(size); exec, stream)
inputs = SE_ExecutionInput[
    SE_ExecutionInput(
        XLA_MaybeOwningDeviceMemoryShapeTree(xs,
            Base.unsafe_convert(Ptr{SE_MaybeOwningDeviceMemory}, ptrs)
        ),
        C_NULL, 0, xs
    )
]


# execute

output = execute_async!(executable, SE_ExecutableRunOptions(allocator, stream, nothing; run_id=5), inputs)
output_mem = unsafe_load(output.result.bases)


# verify

unsafe_copyto_async!(Ptr{UInt8}(Base.unsafe_convert(Ptr{Float32}, y)), output_mem, UInt64(size); exec, stream)
LibTPU.TpuExecutor_SynchronizeAllActivity(exec)

@test y == x
