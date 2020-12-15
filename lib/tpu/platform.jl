export TpuPlatform, initialize!, initialized, visible_device_count, memory_limit

mutable struct TpuPlatform
    handle::Ptr{SE_Platform}

    function TpuPlatform()
        handle = TpuPlatform_New()
        platform = new(handle)
        finalizer(TpuPlatform_Free, platform)
    end
end

Base.unsafe_convert(::Type{Ptr{SE_Platform}}, x::TpuPlatform) = x.handle

function initialize!(p::TpuPlatform)
    status = Ref{Ptr{SE_Platform}}(C_NULL)
    with_status() do s
        TpuPlatform_Initialize(p, 0, C_NULL, C_NULL, s)
    end
end

initialized(p::TpuPlatform) = TpuPlatform_Initialized(p)

visible_device_count(p::TpuPlatform) = TpuPlatform_VisibleDeviceCount(p)

memory_limit(p::TpuPlatform) = TpuPlatform_TpuMemoryLimit(p)
