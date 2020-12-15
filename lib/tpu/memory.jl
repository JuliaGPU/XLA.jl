
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


# memory copying

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
