# host array

export TPUArray, StaticTPUArray

abstract type AbstractTPUArray{T, N} <: AbstractGPUArray{T, N}; end

# TPUArray with dynamic dimensions
mutable struct TPUArray{T, N} <: AbstractTPUArray{T, N}
    mem::SE_DeviceMemoryBase
    dims::Dims{N}

    host_shape::Shape
    device_shape::Shape

    executor::TpuExecutor

    function TPUArray{T, N}(::UndefInitializer, dims::Dims{N};
                            executor=executor()) where {T,N}
        # determine the shape of the data, and the shapes of the buffer
        # (when on the host and device, respectively)
        shape = shape = XLA.Shape(T, dims)
        host_shape = compact_layout(transfer_manager(), shape)
        device_shape = host_to_device_shape(transfer_manager(), shape)

        # allocate a device buffer
        compact_size = bytesize_requirement(transfer_manager(), host_shape)
        mem = allocate!(executor, compact_size, 0)

        arr = new{T,N}(mem, dims, host_shape, device_shape, executor)
        finalizer(unsafe_free!, arr)
    end
end

# TPUArray with static dimensions
mutable struct StaticTPUArray{T, N, D} <: AbstractTPUArray{T, N}
    mem::SE_DeviceMemoryBase

    executor::TpuExecutor

    function StaticTPUArray{T, N, D}(::UndefInitializer;
                                     executor=executor()) where {T,N,D}
        mem = allocate!(executor, prod(D) * sizeof(T), 0)
        arr = new{T,N,D}(mem, executor)
        finalizer(unsafe_free!, arr)
    end
end

function unsafe_free!(xs::AbstractTPUArray)
    deallocate!(xs.executor, xs.mem)
    xs.mem = SE_DeviceMemoryBase()
    nothing
end


## convenience constructors

StaticTPUArray{T,N}(::UndefInitializer, dims::Dims{N}; kwargs...) where {T,N} =
    StaticTPUArray{T, N, dims}(undef; kwargs...)

for AT in (:TPUArray, :StaticTPUArray)
    @eval begin
        # XXX: not sure how to express this generically in terms of AbstractTPUArray

        # type and dimensionality specified, accepting dims as series of Ints
        $AT{T,N}(::UndefInitializer, dims::Integer...) where {T,N} = $AT{T,N}(undef, dims)

        # type but not dimensionality specified
        $AT{T}(::UndefInitializer, dims::Dims{N}) where {T,N} = $AT{T,N}(undef, dims)
        $AT{T}(::UndefInitializer, dims::Integer...) where {T} =
            $AT{T}(undef, convert(Tuple{Vararg{Int}}, dims))

        # empty vector constructor
        $AT{T,1}() where {T} = $AT{T,1}(undef, 0)
    end
end

Base.similar(a::TPUArray{T}, dims::Base.Dims{N}) where {T,N} =
    TPUArray{T,N}(undef, dims; executor=a.executor)
Base.similar(a::TPUArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} =
    TPUArray{T,N}(undef, dims; executor=a.executor)
Base.similar(::Type{TPUArray{T}}, dims::Dims{N}) where {T,N} =
    TPUArray{T,N}(undef, dims; executor=executor())

Base.similar(a::AbstractTPUArray{T,N}) where {T,N} =
    typeof(a)(undef, size(a); executor=a.executor)


## array interface

Base.elsize(::Type{<:AbstractTPUArray{T}}) where {T} = sizeof(T)

Base.size(x::AbstractTPUArray) = x.dims
Base.sizeof(x::AbstractTPUArray) = Base.elsize(x) * length(x)


## interop with other arrays

@inline function (AT::Type{<:AbstractTPUArray{T,N}})(xs::AbstractArray{<:Any,N};
    executor=executor()) where {T,N}
  A = AT(undef, size(xs); executor)
  copyto!(A, convert(Array{T}, xs))
  return A
end

for AT in (:TPUArray, :StaticTPUArray)
    @eval begin
        # underspecified constructors
        $AT{T}(xs::AbstractArray{S,N}) where {T,N,S} = $AT{T,N}(xs)
        (::Type{$AT{T,N} where T})(x::AbstractArray{S,N}) where {S,N} = $AT{S,N}(x)
        $AT(A::AbstractArray{T,N}) where {T,N} = $AT{T,N}(A)
    end
end

# idempotency
(::Type{AT})(xs::AT) where {T,N,AT<:AbstractTPUArray{T,N}} = xs


## conversions

# TODO: Could this be in GPUArrays.jl
Base.convert(::Type{T}, x::T) where T <: AbstractTPUArray = x


## interop with CPU arrays

# TODO: copy into larger array

function Base.copyto!(dest::Array{T}, src::AbstractTPUArray{T}) where T
    isempty(src) && return dest
    @boundscheck size(src) == size(dest) || throw(BoundsError)

    src_bases = [src.mem]
    GC.@preserve src_bases begin
        dest_buffer = LibTPU.XLA_ShapedBuffer(src.device_shape, 0, pointer(src_bases), 1)
        copyto!(dest, src_buffer; stream, manager)
    end

    return dest
end

function Base.copyto!(dest::AbstractTPUArray{T}, src::Array{T};
                      stream=default_stream(),
                      manager=transfer_manager()) where T
    isempty(src) && return dest
    @boundscheck size(src) == size(dest) || throw(BoundsError)

    dest_bases = [dest.mem]
    GC.@preserve dest_bases begin
        dest_buffer = LibTPU.XLA_ShapedBuffer(dest.device_shape, 0, pointer(dest_bases), 1)
        copyto!(dest_buffer, src; stream, manager)
    end

    return dest
end
