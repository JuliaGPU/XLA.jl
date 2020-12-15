export TpuStatus, with_status

mutable struct TpuStatus
    handle::Ptr{TF_Status}

    function TpuStatus()
        handle = TpuStatus_New()
        status = new(handle)
        finalizer(TpuStatus_Free, status)
    end

    function TpuStatus(code, msg)
        handle = TpuStatus_Create(code, msg)
        status = new(handle)
        finalizer(TpuStatus_Free, status)
    end
end

Base.unsafe_convert(::Type{Ptr{TF_Status}}, x::TpuStatus) = x.handle

function with_status(f)
    status = TpuStatus()
    rv = f(status)
    if !TpuStatus_Ok(status)
        throw(status)
    end
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
