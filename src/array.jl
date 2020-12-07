using GPUArrays

export TPUArray, StaticTPUArray

abstract type AbstractTPUArray{T, N} <: AbstractGPUArray{T, N}; end

# TPUArray with dynamic dimensions
mutable struct TPUArray{T, N} <: AbstractTPUArray{T, N}
    mem::SE_DeviceMemoryBase
    dims::Dims{N}

    executor::TpuExecutor
    function TPUArray{T, N}(mem::SE_DeviceMemoryBase, dims::Dims{N}, executor) where {T,N}
        arr = new{T,N}(mem, dims, executor)
        finalizer(unsafe_free!, arr)
        arr
    end
end

# TPUArray with static dimensions
mutable struct StaticTPUArray{T, N, D} <: AbstractTPUArray{T, N}
    mem::SE_DeviceMemoryBase

    executor::TpuExecutor
    function StaticTPUArray{T, N, D}(mem::SE_DeviceMemoryBase, executor) where {T,N,D}
        arr = new{T,N, D}(mem, executor)
        finalizer(unsafe_free!, arr)
        arr
    end
end
(::Type{StaticTPUArray{T, N}})(mem::SE_DeviceMemoryBase, dims::Dims{N}, executor) where {T,N} =
    StaticTPUArray{T, N, dims}(mem, executor)

function (AT::Type{<:AbstractTPUArray{T,N}})(::UndefInitializer, dims::Dims{N}; executor) where {T,N}
    mem = allocate!(executor, UInt64(prod(dims) * sizeof(T)), 0)
    AT(mem, dims, executor)
end

# TODO: Could this be in GPUArrays.jl
Base.convert(::Type{TPUArray{T,N}}, x::TPUArray{T,N}) where {T,N} = x
Base.convert(::Type{StaticTPUArray{T,N,D}}, x::StaticTPUArray{T,N,D}) where {T,N,D} = x
Base.similar(a::AbstractTPUArray{T,N}) where {T,N} = typeof(a)(undef, size(a); executor=a.executor)

Base.similar(a::TPUArray{T}, dims::Base.Dims{N}) where {T,N} = TPUArray{T,N}(undef, dims; executor=a.executor)
Base.similar(a::TPUArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} = TPUArray{T,N}(undef, dims; executor=a.executor)

# XXX: BAD!!
Base.similar(::Type{TPUArray{T}}, dims::Dims{N}) where {T,N} = TPUArray{T,N}(undef, dims; executor=Core.Main.exec)

Base.elsize(::Type{<:AbstractTPUArray{T}}) where {T} = sizeof(T)
Base.size(x::AbstractTPUArray) = x.dims
Base.sizeof(x::AbstractTPUArray) = Base.elsize(x) * length(x)

function unsafe_free!(xs::AbstractTPUArray)
    deallocate!(xs.executor, xs.mem)
    xs.mem = SE_DeviceMemoryBase()
    nothing
end

function Base.copyto!(dest::Array{T}, doffs::Integer, src::AbstractTPUArray{T}, soffs::Integer, n::Integer) where T
    n==0 && return dest
    @boundscheck checkbounds(dest, doffs)
    @boundscheck checkbounds(dest, doffs+n-1)
    @boundscheck checkbounds(src, soffs)
    @boundscheck checkbounds(src, soffs+n-1)
    @assert soffs == 1 # For now, until we figure out how to ask for offsets
    @GC.preserve dest unsafe_copyto!(Ptr{UInt8}(pointer(dest, doffs)),
        src.mem, UInt64(n*Base.elsize(dest)); exec=src.executor)
    return dest
end

function Base.copyto!(dest::AbstractTPUArray{T}, doffs::Integer, src::Array{T}, soffs::Integer, n::Integer) where T
    n==0 && return dest
    @boundscheck checkbounds(dest, doffs)
    @boundscheck checkbounds(dest, doffs+n-1)
    @boundscheck checkbounds(src, soffs)
    @boundscheck checkbounds(src, soffs+n-1)
    @assert doffs == 1 # For now, until we figure out how to ask for offsets
    @GC.preserve dest unsafe_copyto!(dest.mem, Ptr{UInt8}(pointer(src, doffs)),
        UInt64(n*Base.elsize(dest)); exec=dest.executor)
    return dest
end

