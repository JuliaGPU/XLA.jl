
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

struct HLOArray{T, Dims, N} <: AbstractArray{T, N}
    storage::XRTStorage{T, N}
    @noinline function HLOArray{T, Dims, N}(a::Array{T, N}) where {T, Dims, N}
        @assert size(a) == Dims
        @assert length(Dims) == N
        new{T, Dims, N}(XRTStorage{T, N}(a))
    end
    @noinline function HLOArray{T, (), 0}(a::T) where {T<:XLAScalar}
        HLOArray{T, (), 0}(fill(a))
    end
    @noinline function HLOArray{T, (), 0}(a::T) where {T<:HLOArray}
        # Can share storage
        storage = a.storage.remotestorage !== nothing ?
            XRTStorage{T, 0}(a.storage.remotestorage) :
            XRTStorage{T, 0}(fill(a))
        new{T, (), 0}(storage)
    end
    @noinline function HLOArray{T, Dims, N}(h::XRTAllocation) where {T, Dims, N}
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

function Base.convert(::Type{Array}, A::HLOArray)
    if A.storage.localstorage !== nothing
        return copy(A.storage.localstorage)
    end
    rs = A.storage.remotestorage
    lit = read_literal(rs)
    A.storage.localstorage = convert(Array, lit)
    return copy(A.storage.localstorage)
end

@noinline function Base.convert(::Type{T}, A::HLOArray{T, (), 0}) where {T}
    convert(Array, A)[]::T
end

function gethandle!(sess, A::HLOArray)
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

const HLOMatrix{T, Dims} = HLOArray{T, Dims, 2} where {T, Dims}
const HLOVector{T, Dims} = HLOArray{T, Dims, 1} where {T, Dims}
HLOArray(A::AbstractArray) = HLOArray(collect(A)::Array)
function HLOArray(a::Array{T}) where {T}
    HLOArray{T, size(a), ndims(a)}(a)
end
function HLOArray(a::XLAScalar)
    HLOArray{typeof(a), (), 0}(a)
end
function HLOArray(a::Tuple{XLAScalar})
    HLOArray{typeof(a[1]), (), 0}(a[1])
end
function HLOArray(a::Tuple{HLOArray})
    HLOArray{typeof(a[1]), (), 0}(a[1])
end
HLOArray(sess, A::AbstractArray) = HLOArray(sess, collect(A)::Array)
HLOArray(sess, a::XLAScalar) = HLOArray(sess, fill(a))
function HLOArray(sess, a::Array{T}) where {T}
    ret = HLOArray{T, size(a), ndims(a)}(a)
    gethandle!(sess, ret)
    ret
end

Base.eltype(A::Type{<:HLOArray{T}}) where {T} = T
Base.size(A::Type{<:HLOArray{T, Dims}} where T) where {Dims} = Dims
Base.size(A::Type{<:HLOArray{T, Dims}} where T, i) where {Dims} = i <= length(Dims) ? Dims[i] : 1
@inline function Base.axes(A::Type{<:HLOArray{<:Any, Dims}}, d) where {Dims}
    d <= length(Dims) ? axes(A)[d] : Base.OneTo(1)
end
Base.length(A::Type{<:HLOArray}) = prod(size(A))

Base.eltype(A::HLOArray{T}) where {T} = T
Base.size(A::HLOArray{T, Dims}) where {T, Dims} = Dims
Base.size(A::HLOArray{T, Dims}, i) where {T, Dims} = i <= length(Dims) ? Dims[i] : 1
@inline function Base.axes(A::HLOArray{<:Any, Dims}, d) where {Dims}
    d <= length(Dims) ? axes(A)[d] : Base.OneTo(1)
end

function Shape(::Type{HLOArray{T, Dims, N}} where N) where {T, Dims}
    Shape(T, Dims)
end
Shape(T::Type, SHP::Tuple) = Shape(convert(XlaType, T), SHP)
function Shape(::Type{<:HLOArray{T, SHP1}}, SHP2::Tuple) where {T, SHP1}
    # Map arrays of arrays to higher rank arrays
    Shape(T, (SHP1..., SHP2...))
end
function Shape(XLAT::XlaType, SHP::Tuple)
    perm = sortperm(collect(SHP); rev=true)
    Shape(
        element_type = XLAT.which,
        dimensions = Int64[SHP...],
        is_dynamic_dimension = Bool[false for i = 1:length(SHP)],
        layout = Layout(
            format = Format.DENSE,
            minor_to_major = perm .- 1
        )
    )
end

Base.convert(::Type{Shape}, AT::Type{HLOArray{T, Dims, N}} where N) where {T, Dims} =
    Shape(AT)

shape(A::HLOArray) = Shape(typeof(A))

Base.isempty(A::HLOArray{T, Dims}) where {T, Dims} = prod(Dims) == 0

Base.print_array(io::IO, A::HLOArray) = Base.print_array(io, convert(Array, A))
Base.show_vector(io::IO, A::HLOArray, opn='[', cls=']') =
    Base.show_vector(io, convert(Array, A), opn, cls)
Base._show_nonempty(io::IO, A::HLOArray, prefix::String) =
    Base._show_nonempty(io, convert(Array, A), prefix)
Base._show_nonempty(io::IO, A::HLOMatrix, prefix::String) =
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

struct_to_literal(A::HLOArray) = convert(LiteralProto, convert(Array, A))
struct_to_literal(x::XLA.XLAScalar) = struct_to_literal(HLOArray(x))
function struct_to_literal(x)
    T = typeof(x)
    LiteralProto(
        shape = dtype_to_shape(T),
        tuple_literals = collect(struct_to_literal(getfield(x, i)) for i = 1:fieldcount(T) if representable(fieldtype(T, i)))
    )
end

function literal_to_struct(::Type{<:HLOArray}, x::LiteralProto)
    # TODO: We could instead request the remote to create this for
    # us. Otherwise we're doing a round trip here. Good enough for now.
    HLOArray(convert(Array, x))
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
