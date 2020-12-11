using Test

using XLA
using .LibTPU

# init

p = TpuPlatform()
initialize!(p)

exec = TpuExecutor(p)
initialize!(exec)


# compile

N = 4
M = 4
T = Int32

f = x->x
tt = Tuple{XRTArray{T, (N, M), 2}}
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

xs = convert(LibTPU.XLA_Shape, XLA.Shape(T, (N, M)))
size = shape_size(compiler, xs)
mem = allocate!(exec, UInt64(size), 0)

x = rand(T, N, M)
y = zero(x)

buffers = [
    TpuMaybeOwningDeviceMemory(mem, false, 0, allocator)
]

stream = TpuStream(exec)

unsafe_copyto_async!(mem, Ptr{UInt8}(pointer(x)), UInt64(size); exec, stream)
inputs = [
    TpuExecutionInput(
        TpuMaybeOwningDeviceMemoryShapeTree(xs, buffers), [], xs
    )
]


# execute

options = TpuExecutableRunOptions(allocator, stream, nothing; run_id=5)
output = execute_async!(executable, options, inputs)
output_mem = unsafe_load(output.result.bases)


# verify

unsafe_copyto_async!(Ptr{UInt8}(pointer(y)), output_mem, UInt64(size); exec, stream)
LibTPU.TpuExecutor_SynchronizeAllActivity(exec)

@test y == x
