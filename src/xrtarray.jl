
mutable struct XRTStorage{T, N}
    localstorage::Union{Array{T, N}, Nothing}
    remotestorage::Union{XRTAllocation, Nothing}
    function XRTStorage{T, N}(localstorage::Union{Array{T, N}, Nothing},
                              remotestorage::Union{XRTAllocation, Nothing}) where {T, N}
        @assert localstorage !== nothing || remotestorage !== nothing
        new{T, N}(localstorage, remotestorage)
    end
    function XRTStorage{T, N}(a::Array{T, N}) where {T, N}
        new{T, N}(a, nothing)
    end
    function XRTStorage{T, N}(a::XRTAllocation) where {T, N}
        new{T, N}(nothing, a)
    end
end

struct XRTArray{T, Dims, N} <: AbstractArray{T, N}
    storage::XRTStorage{T, N}
    @noinline function XRTArray{T, Dims, N}(a::Array{T, N}) where {T, Dims, N}
        @assert size(a) == Dims
        @assert length(Dims) == N
        new{T, Dims, N}(XRTStorage{T, N}(a))
    end
    @noinline function XRTArray{T, (), 0}(a::T) where {T<:XLAScalar}
        XRTArray{T, (), 0}(fill(a))
    end
    @noinline function XRTArray{T, (), 0}(a::T) where {T<:XRTArray}
        # Can share storage
        storage = a.storage.remotestorage !== nothing ?
            XRTStorage{T, 0}(a.storage.remotestorage) :
            XRTStorage{T, 0}(fill(a))
        new{T, (), 0}(storage)
    end
    @noinline function XRTArray{T, Dims, N}(h::XRTAllocation) where {T, Dims, N}
        @assert length(Dims) == N
        new{T, Dims, N}(XRTStorage{T, N}(h))
    end
end

function read_literal(rs::XRTAllocation)
    sess, dev, h = rs.sess, rs.device, rs.h
    op() = as_default(tf_graph(sess)) do
        run(sess, TensorFlow.Ops.xrt_read_literal(h))::String
    end
    literal = dev === nothing ? op() : with_device(op, dev)
    readproto(IOBuffer(literal), LiteralProto())
end

function Base.convert(::Type{Array}, A::XRTArray)
    if A.storage.localstorage !== nothing
        return copy(A.storage.localstorage)
    end
    rs = A.storage.remotestorage
    lit = read_literal(rs)
    A.storage.localstorage = convert(Array, lit)
    return copy(A.storage.localstorage)
end

@noinline function Base.convert(::Type{T}, A::XRTArray{T, (), 0}) where {T}
    convert(Array, A)[]::T
end

function gethandle!(sess, A::XRTArray)
    if A.storage.remotestorage !== nothing
        @assert A.storage.remotestorage.sess === sess
        return A.storage.remotestorage
    end
    A.storage.remotestorage = XRTAllocation(sess, convert(LiteralProto, A.storage.localstorage))
    A.storage.remotestorage
end
function gethandle!(sess, x::XRTAllocation)
    @assert sess == x.sess
    return x
end

const XRTMatrix{T, Dims} = XRTArray{T, Dims, 2} where {T, Dims}
const XRTVector{T, Dims} = XRTArray{T, Dims, 1} where {T, Dims}
XRTArray(A::AbstractArray) = XRTArray(collect(A)::Array)
function XRTArray(a::Array{T}) where {T}
    XRTArray{T, size(a), ndims(a)}(a)
end
function XRTArray(a::XLAScalar)
    XRTArray{typeof(a), (), 0}(a)
end
function XRTArray(a::Tuple{XLAScalar})
    XRTArray{typeof(a[1]), (), 0}(a[1])
end
function XRTArray(a::Tuple{XRTArray})
    XRTArray{typeof(a[1]), (), 0}(a[1])
end
XRTArray(sess, A::AbstractArray) = XRTArray(sess, collect(A)::Array)
XRTArray(sess, a::XLAScalar) = XRTArray(sess, fill(a))
function XRTArray(sess, a::Array{T}) where {T}
    ret = XRTArray{T, size(a), ndims(a)}(a)
    gethandle!(sess, ret)
    ret
end

Base.eltype(A::Type{<:XRTArray{T}}) where {T} = T
Base.size(A::Type{<:XRTArray{T, Dims}} where T) where {Dims} = Dims
Base.size(A::Type{<:XRTArray{T, Dims}} where T, i) where {Dims} = i <= length(Dims) ? Dims[i] : 1
@inline function Base.axes(A::Type{<:XRTArray{<:Any, Dims}}, d) where {Dims}
    d <= length(Dims) ? axes(A)[d] : Base.OneTo(1)
end
Base.length(A::Type{<:XRTArray}) = prod(size(A))

Base.eltype(A::XRTArray{T}) where {T} = T
Base.size(A::XRTArray{T, Dims}) where {T, Dims} = Dims
Base.size(A::XRTArray{T, Dims}, i) where {T, Dims} = i <= length(Dims) ? Dims[i] : 1
@inline function Base.axes(A::XRTArray{<:Any, Dims}, d) where {Dims}
    d <= length(Dims) ? axes(A)[d] : Base.OneTo(1)
end

import .xla: Shape
function Shape(::Type{XRTArray{T, Dims, N}} where N) where {T, Dims}
    Shape(T, Dims)
end
Shape(T::Type, SHP::Tuple) = Shape(convert(XlaType, T), SHP)
function Shape(::Type{<:XRTArray{T, SHP1}}, SHP2::Tuple) where {T, SHP1}
    # Map arrays of arrays to higher rank arrays
    Shape(T, (SHP1..., SHP2...))
end
function Shape(XLAT::XlaType, SHP::Tuple)
    perm = sortperm(collect(SHP); rev=true)
    Shape(
        element_type = XLAT.which,
        dimensions = Int64[SHP...],
        layout = Layout(
            format = Format.DENSE,
            minor_to_major = perm .- 1
        )
    )
end

Base.convert(::Type{Shape}, AT::Type{XRTArray{T, Dims, N}} where N) where {T, Dims} =
    Shape(AT)

shape(A::XRTArray) = Shape(typeof(A))

Base.isempty(A::XRTArray{T, Dims}) where {T, Dims} = prod(Dims) == 0

Base.print_array(io::IO, A::XRTArray) = Base.print_array(io, convert(Array, A))
Base.show_vector(io::IO, A::XRTArray, opn='[', cls=']') =
    Base.show_vector(io, convert(Array, A), opn, cls)
Base._show_nonempty(io::IO, A::XRTArray, prefix::String) =
    Base._show_nonempty(io, convert(Array, A), prefix)
Base._show_nonempty(io::IO, A::XRTMatrix, prefix::String) =
    Base._show_nonempty(io, convert(Array, A), prefix)

# References a remote struct. We currently have a tension between
# local structs of remote data and a remote reference to a struct.
# Ideally we'd unify them eventually.
struct XRTRemoteStruct{T}
    storage::XRTAllocation
end
function gethandle!(sess, A::XRTRemoteStruct)
    @assert sess == A.storage.sess
    A.storage
end
Base.show(io::IO, x::XRTRemoteStruct{T}) where {T} =
    println("Remote $(T)")

struct_to_literal(A::XRTArray) = convert(LiteralProto, convert(Array, A))
struct_to_literal(x::XLA.XLAScalar) = struct_to_literal(XRTArray(x))
function struct_to_literal(x)
    T = typeof(x)
    LiteralProto(
        shape = dtype_to_shape(T),
        tuple_literals = collect(struct_to_literal(getfield(x, i)) for i = 1:fieldcount(T) if representable(fieldtype(T, i)))
    )
end

function literal_to_struct(::Type{<:XRTArray}, x::LiteralProto)
    # TODO: We could instead request the remote to create this for
    # us. Otherwise we're doing a round trip here. Good enough for now.
    XRTArray(convert(Array, x))
end

function literal_to_struct(::Type{<:XLA.XLAScalar}, x::LiteralProto)
    # TODO: We could instead request the remote to create this for
    # us. Otherwise we're doing a round trip here. Good enough for now.
    convert(Array, x)[]
end

function literal_to_struct(T::Type, x::LiteralProto)
    reconstructed_fields = Any[]
    cur_proto_field = 1
    for i = 1:fieldcount(T)
        fT = fieldtype(T, i)
        if representable(fT)
            push!(reconstructed_fields,
                literal_to_struct(fT, x.tuple_literals[cur_proto_field]))
            cur_proto_field += 1
        else
            push!(reconstructed_fields, fT.instance)
        end
    end
    eval(Expr(:new, T, reconstructed_fields...))
end

function XRTRemoteStruct{T}(sess, x::T) where {T}
    XRTRemoteStruct{T}(XRTAllocation(sess, struct_to_literal(x)))
end
XRTRemoteStruct(sess, x::T) where {T} = XRTRemoteStruct{T}(sess, x)

function Base.convert(::Type{T}, strct::XRTRemoteStruct{T}) where {T}
    proto = read_literal(strct.storage)
    literal_to_struct(T, proto)
end

Base.fetch(strct::XRTRemoteStruct{T}) where {T} = convert(T, strct)
