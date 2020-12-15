export TpuDeviceDescription

mutable struct TpuDeviceDescription
    handle::Ptr{SE_DeviceDescription}

    function TpuDeviceDescription(executor::TpuExecutor)
        handle = TpuDeviceDescription_New()
        description = new(handle)
        with_status() do status
            TpuExecutor_CreateDeviceDescription(executor, description, status)
        end
        finalizer(TpuDeviceDescription_Free, description)
    end
end

Base.unsafe_convert(::Type{Ptr{SE_DeviceDescription}}, x::TpuDeviceDescription) = x.handle

function Base.getproperty(description::TpuDeviceDescription, name::Symbol)
    if hasfield(SE_DeviceDescription, name)
        actual_description = unsafe_load(description.handle)
        value = getfield(actual_description, name)
        if value isa Cstring
            unsafe_string(value)
        else
            value
        end
    else
        getfield(description, name)
    end
end

function Base.show(io::IO, description::TpuDeviceDescription)
    print(io, "TpuDeviceDescription($(description.name), vendor=$(description.device_vendor))")
end
