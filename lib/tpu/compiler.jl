# compiler

export TpuCompiler, run_backend!, compile!, shape_size

mutable struct TpuCompiler
    handle::Ptr{Tpu_Compiler}
    function TpuCompiler()
        handle = TpuCompiler_New()
        compiler = new(handle)
        finalizer(TpuCompiler_Free, compiler)
    end
end

Base.unsafe_convert(::Type{Ptr{Tpu_Compiler}}, x::TpuCompiler) = x.handle


# executable

mutable struct TpuExecutable
    handle::Ptr{SE_Executable}
    function TpuExecutable(handle::Ptr{Cvoid})
        executable = new(handle)
        finalizer(TpuExecutable_Free, executable)
    end
end

Base.unsafe_convert(::Type{Ptr{SE_Executable}}, x::TpuExecutable) = x.handle


# driver

TpuSerializedProto() = TpuSerializedProto(C_NULL, 0)

function run_backend!(compiler::TpuCompiler, mod::XLA.HloModuleProto,
                      exec::TpuExecutor, allocator::TpuDeviceMemoryAllocator)
    buf = IOBuffer()
    writeproto(buf, mod)
    module_serialized = take!(buf)
    @GC.preserve exec module_serialized begin
        modr = Ref(XLA_HloModule(
            TpuSerializedProto(Base.unsafe_convert(Ptr{UInt8}, module_serialized), sizeof(module_serialized)),
            XLA_HloModuleConfig(
            0, 1, 1, 0, true, false, TpuSerializedProto(), true,
            XLA_ComputationLayout(0, C_NULL, XLA_Shape(11, Int64List(), BoolList(),
            C_NULL, 0, XLA_Layout())))
        ))
        result = Ref{Ptr{Cvoid}}()
        with_status() do s
            TpuCompiler_RunBackend(compiler, modr, exec, allocator, result, s)
        end
        return TpuExecutable(result[])
    end
end

function compile!(compiler::TpuCompiler, module_group::XLA.HloModuleGroupProto,
                  execs::Vector{Vector{TpuExecutor}},
                  allocator::TpuDeviceMemoryAllocator)
    buf = IOBuffer()
    writeproto(buf, module_group)
    module_group_serialized = take!(buf)
    confs = map(module_group.hlo_modules) do hlo_module
        XLA_HloModuleConfig(
            0, 1, 1, 0, true, false, TpuSerializedProto(), true,
            convert(XLA_ComputationLayout, hlo_module.host_program_shape))
    end
    @GC.preserve execs module_group_serialized confs begin
        exec_ptrs = [[Base.unsafe_convert(Ptr{SE_StreamExecutor}, exec) for exec in exec_list] for exec_list in execs]
        @GC.preserve exec_ptrs begin
            s_list = SE_StreamExecutorList[SE_StreamExecutorList(
                Base.unsafe_convert(Ptr{Ptr{Cvoid}}, ptr_list),
                length(ptr_list)
            ) for ptr_list in exec_ptrs]

            modr = Ref(XLA_HloModuleGroup(
                TpuSerializedProto(Base.unsafe_convert(Ptr{UInt8}, module_group_serialized), sizeof(module_group_serialized)),
                Base.unsafe_convert(Ptr{XLA_HloModuleConfig}, confs)
            ))
            results = Vector{Ptr{Cvoid}}(undef, length(module_group.hlo_modules))
            with_status() do s
                TpuCompiler_Compile(compiler, modr, s_list, length(execs),
                                    allocator, results, s)
            end
            return map(TpuExecutable, results)
        end
    end
end

function shape_size(compiler::TpuCompiler, shape::Shape)
    rshape = Ref{XLA_Shape}(shape)
    TpuCompiler_ShapeSize(compiler, rshape)
end


# execution

export TpuExecutableRunOptions, TpuExecutionInput, execute_async!

struct TpuExecutableRunOptions
    handle::SE_ExecutableRunOptions

    allocator::TpuDeviceMemoryAllocator
    stream::TpuStream
    host_to_device_stream::Union{TpuStream, Nothing}

    function TpuExecutableRunOptions(allocator::TpuDeviceMemoryAllocator,
                                     stream::TpuStream,
                                     host_to_device_stream::Union{TpuStream, Nothing};
                                     device_ordinal=0, rng_seed=0, run_id=0, launch_id=0)
        handle = SE_ExecutableRunOptions(
            Base.unsafe_convert(SE_DeviceMemoryAllocator, allocator),
            device_ordinal,
            Base.unsafe_convert(Ptr{SE_Stream}, stream),
            host_to_device_stream === nothing ? C_NULL :
                Base.unsafe_convert(Ptr{SE_Stream}, host_to_device_stream),
            TpuSerializedProto(), rng_seed, run_id, launch_id
        )
        new(handle, allocator, stream, host_to_device_stream)
    end
end

Base.cconvert(T::Type{Ptr{SE_ExecutableRunOptions}}, x::TpuExecutableRunOptions) =
    Ref(x.handle)

Base.unsafe_convert(T::Type{SE_ExecutableRunOptions}, x::TpuExecutableRunOptions) =
    x.handle

struct TpuExecutionInput
    handle::SE_ExecutionInput

    shape_tree::TpuMaybeOwningDeviceMemoryShapeTree
    unowned_indices::Vector{Cint}

    function TpuExecutionInput(shape_tree, unowned_indices::Vector, dynamic_shape::Shape)
        unowned_indices = convert(Vector{Cint}, unowned_indices)
        handle = SE_ExecutionInput(
            Base.unsafe_convert(XLA_MaybeOwningDeviceMemoryShapeTree, shape_tree),
            isempty(unowned_indices) ? C_NULL : pointer(unowned_indices),
            length(unowned_indices),
            dynamic_shape)
        new(handle, shape_tree, unowned_indices)
    end
end

Base.cconvert(::Type{Ptr{SE_ExecutionInput}}, x::TpuExecutionInput) = Ref(x.handle)

SE_ExecutionOutput() = SE_ExecutionOutput(XLA_ShapedBuffer(), C_NULL, 0, C_NULL, 0)

function execute_async!(executable::TpuExecutable, options::TpuExecutableRunOptions,
                        arguments::Vector{TpuExecutionInput})
    output = Ref(SE_ExecutionOutput())
    with_status() do s
        TpuExecutable_ExecuteAsyncOnStream(executable, options, arguments,
                                           length(arguments), C_NULL, output, s)
    end
    output[]
end
