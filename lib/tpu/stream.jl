
export TpuStream, allocated

mutable struct TpuStream
    handle::Ptr{SE_Stream}

    executor::TpuExecutor

    function TpuStream(executor::TpuExecutor)
        # XXX: why is New and Allocate separate? should we keep that?
        handle = TpuStream_New(executor)
        stream = new(handle, executor)
        TpuExecutor_AllocateStream(executor, stream)
        finalizer(stream) do obj
            TpuExecutor_DeallocateStream(executor, stream)
            TpuStream_Free(stream)
        end
        stream
    end
end

Base.unsafe_convert(::Type{Ptr{SE_Stream}}, x::TpuStream) = x.handle

allocated(stream::TpuStream) = TpuStream_Status(stream)

function Base.wait(stream::TpuStream)
    with_status() do status
        LibTPU.TpuExecutor_BlockHostUntilDone(stream.executor, stream, status)
    end
end
