## utils

macro new_with_finalizer(new_sym)
    if isa(new_sym, Symbol)
        call = :($(new_sym)())
    else
        call = new_sym
    end
    esc(quote
        handle = $call
        ret = new(handle)
        finalizer(close, ret)
        ret
    end)
end


## initialization

let initialized = false
    global initialize_tpu_runtime
    function initialize_tpu_runtime()
        if !initialized
            args = ["--install_signal_handlers=0"; split(get(ENV, "JULIA_LIBTPU_ARGS", ""))]
            @ccall libtpu.TfTpu_Initialize(true::Bool, length(args)::Cint, args::Ptr{Cstring})::Cvoid
            initialized = true
        end
    end
end


## status

export TpuStatus, with_status

mutable struct TpuStatus
    handle::Ptr{TF_Status}

    function TpuStatus()
        @new_with_finalizer(TpuStatus_New)
    end

    function TpuStatus(code, msg)
        @new_with_finalizer begin
            TpuStatus_Create(code, msg)
        end
    end
end

Base.unsafe_convert(::Type{Ptr{TF_Status}}, x::TpuStatus) = x.handle

function with_status(f)
    status = TpuStatus()
    rv = f(status)
    if !TpuStatus_Ok(status)
        throw(status)
    end
    close(status)
    rv
end

function Base.show(io::IO, status::TpuStatus)
    print(io, "TpuStatus(")
    if TpuStatus_Ok(status)
        print(io, "OK)")
    else
        print(io, "not OK")
        print(io, ", code ", TpuStatus_Code(status), ")")
        print(io, ": ", unsafe_string(TpuStatus_Message(status)))
    end
end


## platform

export TpuPlatform, initialize!

mutable struct TpuPlatform
    handle::Ptr{SE_Platform}
    function TpuPlatform()
        initialize_tpu_runtime()
        @new_with_finalizer(TpuPlatform_New)
    end
end

Base.unsafe_convert(::Type{Ptr{SE_Platform}}, x::TpuPlatform) = x.handle

function initialize!(p::TpuPlatform)
    status = Ref{Ptr{SE_Platform}}(C_NULL)
    with_status() do s
        TpuPlatform_Initialize(p, 0, C_NULL, C_NULL, s)
    end
end


## executor config

export TpuStreamExecutorConfig, set_ordinal!

mutable struct TpuStreamExecutorConfig
    handle::Ptr{SE_StreamExecutorConfig}
    function TpuStreamExecutorConfig()
        @new_with_finalizer(TpuStreamExecutorConfig_Default)
    end
end

Base.unsafe_convert(::Type{Ptr{SE_StreamExecutorConfig}}, x::TpuStreamExecutorConfig) = x.handle

set_ordinal!(sec::TpuStreamExecutorConfig, o::Integer) =
    TpuStreamExecutorConfig_SetOrdinal(sec, o)



## executor

export TpuExecutor

mutable struct TpuExecutor
    handle::Ptr{SE_StreamExecutor}
    function TpuExecutor(platform::TpuPlatform; config = TpuStreamExecutorConfig())
        @new_with_finalizer with_status() do s
            TpuPlatform_GetExecutor(platform, config, s)
        end
    end
end

Base.unsafe_convert(::Type{Ptr{SE_StreamExecutor}}, x::TpuExecutor) = x.handle

function initialize!(e::TpuExecutor, ordinal = 0, options = SEDeviceOptions(UInt32(0)))
    with_status() do s
        TpuExecutor_Init(e, ordinal, options, s)
    end
end

# allocations

export allocate!, deallocate!

SE_DeviceMemoryBase() = SE_DeviceMemoryBase(C_NULL, 0, 0)

function allocate!(e::TpuExecutor, size::UInt64, memory_space::Int64)
    TpuExecutor_Allocate(e, size, memory_space)
end

function deallocate!(e::TpuExecutor, mem::SE_DeviceMemoryBase)
    rmem = Ref{SE_DeviceMemoryBase}(mem)
    TpuExecutor_Deallocate(e, rmem)
end

# device options

mutable struct SEDeviceOptions
    handle::Ptr{SE_DeviceOptions}
    function SEDeviceOptions(flags::Cuint)
        TpuExecutor_NewDeviceOptions(flags)
    end
end


## compiler

export TpuCompiler, run_backend!, compile!, shape_size

mutable struct TpuCompiler
    handle::Ptr{Tpu_Compiler}
    function TpuCompiler()
        @new_with_finalizer(TpuCompiler_New)
    end
end

Base.unsafe_convert(::Type{Ptr{Tpu_Compiler}}, x::TpuCompiler) = x.handle

# allocator

export TpuDeviceMemoryAllocator

function device_allocate(ctx::Ptr{Cvoid}, device_ordinal::Cint, size::UInt64,
    retry_on_failure::Bool, memory_space::Int64, result::Ptr{SE_ScopedDeviceMemory},
    status::Ptr{Cvoid})

    executor = unsafe_pointer_to_objref(ctx)::TpuExecutor
    mem = allocate!(executor, size, memory_space)
    unsafe_store!(result, SE_ScopedDeviceMemory(mem, device_ordinal))
    nothing
end

function device_deallocate(ctx::Ptr{Cvoid}, base::Ptr{SE_ScopedDeviceMemory},
    device_ordinal::Cint, status::Ptr{Cvoid})
    error("Unimplemented")
end

struct TpuDeviceMemoryAllocator
    handle::SE_DeviceMemoryAllocator

    platform::TpuPlatform
    executor::TpuExecutor

    function TpuDeviceMemoryAllocator(platform::TpuPlatform, executor::TpuExecutor)
        handle = SE_DeviceMemoryAllocator(
            Base.unsafe_convert(Ptr{SE_Platform}, platform),
            pointer_from_objref(executor),
            @cfunction(device_allocate, Cvoid, (Ptr{Cvoid}, Cint, UInt64, Bool, Int64, Ptr{SE_ScopedDeviceMemory}, Ptr{Cvoid})),
            @cfunction(device_deallocate, Cvoid, (Ptr{Cvoid}, Cint, UInt64, Bool, Int64, Ptr{SE_ScopedDeviceMemory}, Ptr{Cvoid})),
        )
        new(handle, platform, executor)
    end
end

Base.unsafe_convert(T::Type{SE_DeviceMemoryAllocator}, x::TpuDeviceMemoryAllocator) =
    x.handle

Base.cconvert(::Type{Ptr{SE_DeviceMemoryAllocator}}, x::TpuDeviceMemoryAllocator) =
    Ref(x.handle)



# executable

mutable struct TpuExecutable
    handle::Ptr{SE_Executable}
    function TpuExecutable(handle::Ptr{Cvoid})
        @new_with_finalizer(begin
            handle
        end)
    end
end

Base.unsafe_convert(::Type{Ptr{SE_Executable}}, x::TpuExecutable) = x.handle

# layout

XLA_Layout() = XLA_Layout(0, Int64List(), TileList(), 0, 0)

function Base.convert(::Type{XLA_Layout}, layout::Layout)
    fields = layout.__protobuf_jl_internal_values
    XLA_Layout(layout.format,
        haskey(fields, :minor_to_major) ? convert(Int64List, layout.minor_to_major) : Int64List(),
        haskey(fields, :tiles) ? convert(TileList, layout.tiles) : TileList(),
        haskey(fields, :element_size_in_bits) ? layout.element_size_in_bits : 0,
        haskey(fields, :memory_space) ? layout.memory_space : 0)
end

# shape

XLA_Shape() = XLA_Shape(0, Int64List(), BoolList(), (), 0, XLA_Layout())

function Base.convert(::Type{XLA_Shape}, shape::Shape)
    fields = shape.__protobuf_jl_internal_values
    tuple_shapes = ()
    if haskey(fields, :tuple_shapes) && length(shape.tuple_shapes) !== 0
        tuple_shapes = tuple(map(shape.tuple_shapes) do tshape
            convert(XLA_Shape, tshape)
        end...)
    end
    XLA_Shape(shape.element_type,
        haskey(fields, :dimensions) ? convert(Int64List, shape.dimensions) : Int64List(),
        haskey(fields, :is_dynamic_dimension) ? convert(BoolList, shape.is_dynamic_dimension) : BoolList(),
        tuple_shapes, length(tuple_shapes),
        haskey(fields, :layout) ? convert(XLA_Layout, shape.layout) : XLA_Layout()
    )
end

# computation layout

function Base.convert(::Type{XLA_ComputationLayout}, pshape::ProgramShape)
    parameters = nothing
    if length(pshape.parameters) != 0
        parameters = map(pshape.parameters) do shape
            convert(XLA_Shape, shape)
        end
    end
    XLA_ComputationLayout(
        parameters === nothing ? 0 : length(parameters),
        parameters === nothing ? C_NULL : pointer(parameters),
        pshape.result,
        parameters
    )
end

# driver

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
        exec_ptrs = [[Base.unsafe_convert(Ptr{Cvoid}, exec) for exec in exec_list] for exec_list in execs]
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


## stream

export TpuStream

mutable struct TpuStream
    handle::Ptr{SE_Stream}
    function TpuStream(parent::TpuExecutor)
        @new_with_finalizer(TpuStream_New(parent))
    end
end

Base.unsafe_convert(::Type{Ptr{SE_Stream}}, x::TpuStream) = x.handle


## memory copying

export TpuMaybeOwningDeviceMemory, TpuMaybeOwningDeviceMemoryShapeTree, unsafe_copyto_async!

struct TpuMaybeOwningDeviceMemory
    handle::SE_MaybeOwningDeviceMemory

    allocator::TpuDeviceMemoryAllocator

    function TpuMaybeOwningDeviceMemory(memory::SE_DeviceMemoryBase, owned::Bool,
                                        device_ordinal::Int, allocator::TpuDeviceMemoryAllocator)
        handle = SE_MaybeOwningDeviceMemory(memory, owned, device_ordinal,
            Base.unsafe_convert(SE_DeviceMemoryAllocator, allocator))
        new(handle, allocator)
    end
end

Base.cconvert(T::Type{Ptr{SE_MaybeOwningDeviceMemory}}, x::TpuMaybeOwningDeviceMemory) =
    Ref(x.handle)

Base.unsafe_convert(T::Type{SE_MaybeOwningDeviceMemory}, x::TpuMaybeOwningDeviceMemory) =
    x.handle

struct TpuMaybeOwningDeviceMemoryShapeTree
    handle::XLA_MaybeOwningDeviceMemoryShapeTree

    buffers::Vector{TpuMaybeOwningDeviceMemory}
    ptrs::Vector{SE_MaybeOwningDeviceMemory}

    function TpuMaybeOwningDeviceMemoryShapeTree(shape::Shape, buffers::Vector{TpuMaybeOwningDeviceMemory})
        ptrs = [Base.unsafe_convert(SE_MaybeOwningDeviceMemory, buffer) for buffer in buffers]
        handle = XLA_MaybeOwningDeviceMemoryShapeTree(shape, pointer(ptrs))
        new(handle, buffers, ptrs)
    end
end

Base.cconvert(T::Type{Ptr{XLA_MaybeOwningDeviceMemoryShapeTree}},
              x::TpuMaybeOwningDeviceMemoryShapeTree) =
    Ref(x.handle)

Base.unsafe_convert(T::Type{XLA_MaybeOwningDeviceMemoryShapeTree},
                    x::TpuMaybeOwningDeviceMemoryShapeTree) =
    x.handle

function unsafe_copyto_async!(dst::SE_DeviceMemoryBase, src::Ptr{UInt8}, size::Csize_t; exec::TpuExecutor, stream::TpuStream)
    rdst = Ref{SE_DeviceMemoryBase}(dst)
    TpuExecutor_MemcpyFromHost(exec, stream, rdst, src, size)
end

function unsafe_copyto_async!(dst::Ptr{UInt8}, src::SE_DeviceMemoryBase, size::Csize_t; exec::TpuExecutor, stream::TpuStream)
    rsrc = Ref{SE_DeviceMemoryBase}(src)
    TpuExecutor_MemcpyToHost(exec, stream, dst, rsrc, size)
end

function Base.unsafe_copyto!(dst::SE_DeviceMemoryBase, src::Ptr{UInt8}, size::Csize_t; exec::TpuExecutor)
    rdst = Ref{SE_DeviceMemoryBase}(dst)
    with_status() do status
        TpuExecutor_SynchronousMemcpyFromHost(exec, rdst, src, size, status)
    end
end

function Base.unsafe_copyto!(dst::Ptr{UInt8}, src::SE_DeviceMemoryBase, size::Csize_t; exec::TpuExecutor)
    rsrc = Ref{SE_DeviceMemoryBase}(src)
    with_status() do status
        TpuExecutor_SynchronousMemcpyToHost(exec, dst, rsrc, size, status)
    end
end


## execution

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

XLA_ShapedBuffer() = XLA_ShapedBuffer(XLA_Shape(), 0, C_NULL, 0)

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


################################################################################

TpuSerializedProto() = TpuSerializedProto(C_NULL, 0)

Base.zero(::Type{XLA_Tile}) = XLA_Tile(Int64List())

for (T, sym) in
    ((TpuStatus, :TpuStatus_Free),
     (TpuPlatform, :TpuPlatform_Free),
     (TpuExecutor, :TpuExecutor_Free),
     (TpuStreamExecutorConfig, :TpuStreamExecutorConfig_Free),
     (SEDeviceOptions, :TpuExecutor_FreeDeviceOptions),
     (TpuCompiler, :TpuCompiler_Free),
     (TpuExecutable, :TpuExecutable_Free),
     (TpuStream, :TpuStream_Free))
    Base.unsafe_convert(::Type{Ptr{Cvoid}}, h::T) = h.handle
    @eval function Base.close(h::$T)
        $(sym)(h)
        h.handle = C_NULL
    end
end
