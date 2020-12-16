export TpuStatus, with_status, is_ok

mutable struct TpuStatus
    handle::Ptr{TF_Status}

    function TpuStatus()
        handle = TpuStatus_New()
        status = new(handle)
        finalizer(TpuStatus_Free, status)
        return status
    end

    function TpuStatus(code, msg)
        handle = TpuStatus_Create(code, msg)
        status = new(handle)
        finalizer(TpuStatus_Free, status)
        return status
    end

    function TpuStatus(handle::Ptr{TF_Status}, managed=true)
        status = new(handle)
        managed && finalizer(TpuStatus_Free, status)
        return status
    end
end

Base.unsafe_convert(::Type{Ptr{TF_Status}}, x::TpuStatus) = x.handle

is_ok(status::TpuStatus) = TpuStatus_Ok(status)

function with_status(f)
    status = TpuStatus()
    rv = f(status)
    if !is_ok(status)
        throw(status)
    end
    rv
end

function Base.show(io::IO, status::TpuStatus)
    print(io, "TpuStatus(")
    if is_ok(status)
        print(io, "OK)")
    else
        print(io, "not OK")
        print(io, ", code ", TpuStatus_Code(status), ")")
        print(io, ": ", unsafe_string(TpuStatus_Message(status)))
    end
end
