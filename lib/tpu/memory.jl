# device allocations

export SE_DeviceMemoryBase, allocate!, deallocate!, memory_usage

SE_DeviceMemoryBase() = SE_DeviceMemoryBase(C_NULL, 0, 0)

function allocate!(e::TpuExecutor, size::Integer, memory_space::Integer)
    TpuExecutor_Allocate(e, size, memory_space)
end

function deallocate!(e::TpuExecutor, mem::SE_DeviceMemoryBase)
    rmem = Ref{SE_DeviceMemoryBase}(mem)
    TpuExecutor_Deallocate(e, rmem)
end

function memory_usage(e::TpuExecutor)
    free = Ref{Int64}()
    total = Ref{Int64}()
    TpuExecutor_DeviceMemoryUsage(e, free, total)
    (free=free[], total=total[])
end


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


# (un)owned memory

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


# raw memory copying

function unsafe_copyto_async!(dst::SE_DeviceMemoryBase, src::Ptr, size::Integer;
                              executor::TpuExecutor, stream::TpuStream)
    rdst = Ref{SE_DeviceMemoryBase}(dst)
    TpuExecutor_MemcpyFromHost(executor, stream, rdst, src, size)
end

function unsafe_copyto_async!(dst::Ptr, src::SE_DeviceMemoryBase, size::Integer;
                              executor::TpuExecutor, stream::TpuStream)
    rsrc = Ref{SE_DeviceMemoryBase}(src)
    TpuExecutor_MemcpyToHost(executor, stream, dst, rsrc, size)
end

function Base.unsafe_copyto!(dst::SE_DeviceMemoryBase, src::Ptr, size::Integer;
                             executor::TpuExecutor)
    rdst = Ref{SE_DeviceMemoryBase}(dst)
    with_status() do status
        TpuExecutor_SynchronousMemcpyFromHost(executor, rdst, src, size, status)
    end
end

function Base.unsafe_copyto!(dst::Ptr, src::SE_DeviceMemoryBase, size::Integer;
                             executor::TpuExecutor)
    rsrc = Ref{SE_DeviceMemoryBase}(src)
    with_status() do status
        TpuExecutor_SynchronousMemcpyToHost(executor, dst, rsrc, size, status)
    end
end


# transfer manager

export TpuTransferManager, host_to_device_shape, compact_layout, bytesize_requirement

mutable struct TpuTransferManager
    handle::Ptr{XLA_TransferManager}

    function TpuTransferManager()
        handle = TpuTransferManager_New()
        manager = new(handle)
        finalizer(TpuTransferManager_Free, manager)
    end
end

Base.unsafe_convert(::Type{Ptr{XLA_TransferManager}}, x::TpuTransferManager) =
    x.handle

# Returns the shape of the on-device representation for the given shape on
# the host. This is intended for use with ShapedBuffer where buffers are
# pre-allocated by the host, e.g. TransferLiteralToDevice, without the user
# needing to consider device-specific behaviors.
function host_to_device_shape(manager::TpuTransferManager, host_shape::Shape)::Shape
    device_shape = Ref{XLA_Shape}(host_shape)
    TpuTransferManager_HostShapeToDeviceShape(manager, Ref{XLA_Shape}(host_shape), device_shape)
    return device_shape[]
end

# Chooses a compact layout for 'host_shape', ignoring any existing layout on
# 'host_shape'. What "reasonable" means is left up to the backend. The
# intended use case is to choose a layout that avoids excessive padding on
# devices that have tiled memory architectures.
# The default implementation always picks a default (major-to-minor) layout.
# Fails if 'shape' cannot be represented by the device.
function compact_layout(manager::TpuTransferManager, host_shape::Shape)::Shape
    output = Ref{XLA_Shape}(host_shape)
    with_status() do status
        TpuTransferManager_ChooseCompactLayoutForShape(manager, Ref{XLA_Shape}(host_shape), output, status)
    end
    return output[]
end

# Determines the byte size requirement for the given shape on the underlying
# architecture. This will be used to allocate an appropriately sized memory
# region for a host-to-device transfer.
function bytesize_requirement(manager::TpuTransferManager, shape::Shape)
    TpuTransferManager_GetByteSizeRequirement(manager, Ref{XLA_Shape}(shape))
end

## integration with copyto!

# TODO: can we do general copyto!, with offsets, etc? raw memcpy requires us
#       handling the layout, which the transfer manager normally takes care of.

function TransferLiteralFromDeviceCallback(ctx, status)
    status_refptr = convert(Ptr{Ptr{TF_Status}}, ctx)
    unsafe_store!(status_refptr, status)
    return
end

function Base.copyto!(dest::XLA_ShapedBuffer, src::Array{T};
                      stream::TpuStream,
                      manager::TpuTransferManager) where T
    dest_shape = convert(Shape, dest.on_device_shape)
    src_shape = Shape(T, size(src))
    @boundscheck dest_shape.dimensions == src_shape.dimensions || throw(BoundsError())

    buffers = [pointer(src)]
    sizes = Csize_t[sizeof(src)]

    GC.@preserve src buffers sizes begin
        src_literal = XLA_Literal(pointer(buffers), pointer(sizes), 1, src_shape)

        with_status() do status
            TpuTransferManager_TransferLiteralToDeviceAsync(
                    manager, stream, Ref(src_literal), Ref(dest), status)
        end
    end

    return dest
end

function Base.copyto!(dest::Array{T}, src::XLA_ShapedBuffer;
                      stream::TpuStream,
                      manager::TpuTransferManager) where {T}
    dest_shape = Shape(T, size(dest))
    src_shape = convert(Shape, src.on_device_shape)
    @boundscheck dest_shape.dimensions == src_shape.dimensions || throw(BoundsError())

    buffers = [pointer(dest)]
    sizes = Csize_t[sizeof(dest)]
    dest_literal = XLA_Literal(pointer(buffers), pointer(sizes), 1, src_shape)

    GC.@preserve buffers sizes begin
        callback = @cfunction(TransferLiteralFromDeviceCallback,
                              Cvoid, (Ptr{Cvoid}, Ptr{TF_Status}))

        status_ref = Ref{Ptr{TF_Status}}(C_NULL)
        TpuTransferManager_TransferLiteralFromDevice(manager, stream,
                                                     Ref(src), Ref(dest_literal),
                                                     callback, status_ref)
        wait(stream)
        while status_ref[] == C_NULL
            # spin a while until the callback finishes
            # (this can be non-instantaneous on a heavily-loaded system).
            sleep(0.1)  # or the callback never happens
        end
        status = TpuStatus(status_ref[])
        is_ok(status) || throw(status)
    end

    return dest
end
